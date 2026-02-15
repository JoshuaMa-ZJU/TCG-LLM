import os
# 在导入 torch / unsloth 之前禁用 torch.compile / torchdynamo，避免 FX fake tensor 形状检查报错
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_OPTIMIZER", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import json
import torch
import random
import numpy as np
from glob import glob
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset
# 先导入 TRL / Transformers / PEFT，避免 Unsloth 替换 GRPOTrainer 的实现
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor, AutoTokenizer
from peft import PeftModel
# 再尝试导入 Unsloth，仅用于模型加载与 LoRA 封装，加速可选

try:
    from unsloth import FastLanguageModel
    _HAS_UNSLOTH = True
except Exception:
    _HAS_UNSLOTH = False
# 轻量 CNN 融合模型（可选，用于生成先验候选并插入到提示）
try:
    from multimodal_cnn_cross_attention import CycloneFusionModel, FusionConfig as FusionCfgLite
    _HAS_CNN_FUSER = True
except Exception:
    CycloneFusionModel = None
    FusionCfgLite = None
    _HAS_CNN_FUSER = False

# Prefix encoder (可选)
try:
    from prefix_injector import make_prefix_encoder_from_config
    _HAS_PREFIX = True
except Exception:
    _HAS_PREFIX = False

# --- 环境与缓存 ---
hf_cache_dir = "/root/autodl-tmp"
os.environ.setdefault("HF_HOME", hf_cache_dir)
os.makedirs(hf_cache_dir, exist_ok=True)


@dataclass
class ScriptConfig:
    # 路径
    data_folder: str = "/root/autodl-tmp/tc_processed_data"
    docs_folder: str = "/root/autodl-tmp/tc_processed_docs"
    output_dir: str = "/root/autodl-tmp/cyclone_detector_grpo_qwen3_fast/llava"
    cache_dir: str = hf_cache_dir

    # 模型
    model_name: str = "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit"
    load_in_4bit: bool = True
    use_flash_attn2: bool = True
    model_family: str = "llava"  # 'qwen' | 'llava'

    # LoRA（可选）
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # GRPO 超参
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2  # 固定图像尺寸后允许 batch>1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7
    warmup_steps: int = 50
    logging_steps: int = 50   # 稀疏日志
    save_steps: int = 0       # 我们只在结束时手动保存
    max_completion_length: int = 512
    num_generations: int = 2
    beta: float = 0.04

    # 数据
    train_split: float = 4/5
    max_length: int = 2048
    doc_max_chars: int = 1200

    # 新增：统一图像尺寸 -> 固定视觉 token 数量，支持 batch>1；LLM 输入改为 512x512
    image_size: Tuple[int, int] = (512, 512)
    force_fixed_image_size: bool = True
    allow_batch_gt1_if_fixed_image: bool = True
    use_original_image_size: bool = False  # 是否使用原始图像尺寸（主要用于LLaVA）

    # 加速选项
    fast_mode: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True

    seed: int = 42

    # 是否预测24小时后的气旋
    predict_24h: bool = False  # 是否预测未来24小时的气旋

    # 奖励函数可调超参
    scale_factor_km: float = 100.0  # 距离衰减尺度，km
    weight_count_current: float = 0.5  # 调整为0.5，因为移除了24h预测
    weight_pos_current: float = 0.5    # 调整为0.5，因为移除了24h预测

    # Few-shot 与 CoT 配置（与 SFT 脚本对齐）
    use_few_shot: bool = False
    few_shot_num: int = 1
    few_shot_sampling: str = "random"  # "random" | "head"
    few_shot_doc_max_chars: int = 800
    few_shot_seed: int = 1234
    few_shot_with_images: bool = True

    use_cot: bool = True
    cot_instruction: str = (
        "Please follow these steps to reason before answering:\n"
        "1. Analyze the current input data to identify patterns of tropical cyclone occurrence, and determine whether tropical cyclones are forming globally.\n"
        "2. If any are present, focus on locating (a) minima of geopotential height, (b) minima of sea surface temperature, and (c) the spiral center of the cloud system in the satellite imagery, to pinpoint the cyclone eye location(s).\n"
        "3. Based on the current satellite image and environmental fields, identify all current tropical cyclones and their precise locations."
    )

    # 可选：从 SFT 载入已有 LoRA 适配器，继续在 GRPO 上训练
    initial_adapter_path: str = ""
    #/root/autodl-tmp/cyclone_detector_model/Qwen3-VL-8B-Instruct_QLoRA-fast_cot-on_fs-1_cnnfeat-off_cnnprefix-on_pfx4#
    # 轻量 CNN 先验配置（与 SFT fast 对齐）
    use_cnn_features: bool = False
    cnn_feature_ckpt: str = "/root/autodl-tmp/cnn_cross_attn_model/best.pt"          # 例如 /root/autodl-tmp/cnn_cross_attn_model/best.pt
    cnn_device: str = "cuda"            # 轻量模型所用设备
    cnn_exist_threshold: float = 0.5     # 过滤存在性概率阈值
    cnn_feature_topk: int = 8            # 每类最多注入的候选数
    cnn_image_size: Tuple[int,int] = (256,256)
    cnn_append_mode: str = "json"       # 'json' | 'text'
    cnn_image_channels: int = 1          # 单通道支持

    # Prefix KV 注入（基于 CNN 融合特征）
    use_cnn_feature_prefix: bool = False
    prefix_len: int = 4
    prefix_target_layers: int = 0  # 0=全部层
    prefix_share_across_layers: bool = True
    # 是否在提示中加入 12h 历史上下文块
    use_prev12h_context: bool = True


class CycloneGRPO_DatasetFast(Dataset):
    """GRPO 用加速数据集：
    - 统一图像尺寸（固定视觉 token 数）
    - 文档缓存，避免重复 IO
    - 其余与原逻辑一致
    """
    def __init__(self, data_files: List[str], docs_folder: str, processor, max_length: int = 2048,
                 image_size: Tuple[int, int] = (384, 384),
                 predict_24h: bool = False,
                 use_few_shot: bool = False, few_shot_num: int = 2, few_shot_sampling: str = "random",
                 few_shot_doc_max_chars: int = 800, few_shot_seed: int = 1234,
                 few_shot_with_images: bool = False,
                 use_cot: bool = False, cot_instruction: str = "",
                 model_family: str = "qwen",
                 doc_max_chars: int = 1200):
        self.data_files = data_files
        self.docs_folder = docs_folder
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        self.predict_24h = predict_24h
        # few-shot / CoT
        self.use_few_shot = use_few_shot
        self.few_shot_num = few_shot_num
        self.few_shot_sampling = few_shot_sampling
        self.few_shot_doc_max_chars = few_shot_doc_max_chars
        self.few_shot_seed = few_shot_seed
        self.few_shot_with_images = few_shot_with_images
        self.use_cot = use_cot
        self.cot_instruction = cot_instruction or ""
        self.model_family = model_family
        self.doc_max_chars = doc_max_chars
        # 从script_cfg获取use_original_image_size，如果没有则默认为False（GRPO通常使用固定尺寸）
        self.use_original_image_size = getattr(getattr(processor, '_script_cfg', None), 'use_original_image_size', False)
        # 文档缓存
        self._doc_cache: Dict[Tuple[str, int], str] = {}
        # CNN 先验
        self._script_cfg = getattr(processor, '_script_cfg', None)
        self._cnn_ready = False
        self._cnn_model = None
        self._cnn_cfg = None
        self._feat_cache: Dict[str,str] = {}
        self._prefix_encoder = None
        need_cnn = (self._script_cfg and _HAS_CNN_FUSER and (
            getattr(self._script_cfg, 'use_cnn_features', False) or getattr(self._script_cfg, 'use_cnn_feature_prefix', False)
        ))
        if need_cnn:
            ckpt = getattr(self._script_cfg, 'cnn_feature_ckpt', '')
            if ckpt and os.path.exists(ckpt):
                try:
                    ckpt_obj = torch.load(ckpt, map_location='cpu')
                    cfg_dict = ckpt_obj.get('cfg', {})
                    lite_cfg = FusionCfgLite()
                    for k,v in cfg_dict.items():
                        if hasattr(lite_cfg, k):
                            setattr(lite_cfg, k, v)
                    # 覆盖通道数
                    lite_cfg.image_channels = getattr(self._script_cfg, 'cnn_image_channels', getattr(lite_cfg,'image_channels',1))
                    lite_cfg.device = getattr(self._script_cfg, 'cnn_device', 'cuda')
                    vocab_size = 128
                    st = ckpt_obj.get('model_state', {})
                    for name,tensor in st.items():
                        if name.endswith('text_enc.emb.weight'):
                            vocab_size = tensor.shape[0]; break
                    model = CycloneFusionModel(lite_cfg, vocab_size=vocab_size)
                    model.load_state_dict(st)
                    model.to(lite_cfg.device)
                    model.eval()
                    self._cnn_cfg = lite_cfg
                    self._cnn_model = model
                    self._cnn_ready = True
                    if getattr(self._script_cfg, 'use_cnn_feature_prefix', False) and _HAS_PREFIX:
                        llm_cfg = getattr(self.processor, '_llm_config', None)
                        if llm_cfg is not None:
                            try:
                                z_dim = int(2 * getattr(lite_cfg, 'd_model', 256))
                                self._prefix_encoder = make_prefix_encoder_from_config(
                                    llm_cfg,
                                    z_dim=z_dim,
                                    prefix_len=getattr(self._script_cfg, 'prefix_len', 4),
                                    target_layers=(getattr(self._script_cfg, 'prefix_target_layers', 0) or None),
                                    share_across_layers=getattr(self._script_cfg, 'prefix_share_across_layers', True),
                                ).to(lite_cfg.device)
                                print(f"[grpo-prefix] encoder ready (prefix_len={getattr(self._script_cfg,'prefix_len',4)})")
                            except Exception as e:
                                print(f"[grpo-prefix] failed to init prefix encoder: {e}")
                    print(f"[grpo-cnn] loaded prior model: {ckpt}")
                except Exception as e:
                    print(f"[grpo-cnn] failed to load {ckpt}: {e}")
            else:
                print("[grpo-cnn] checkpoint not found; skip CNN prior.")

    def __len__(self):
        return len(self.data_files)

    def _to_pil_image(self, arr: np.ndarray) -> Image.Image:
        if hasattr(arr, 'filled'):
            arr = arr.filled(0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        vmin, vmax = np.percentile(arr, [2, 98])
        if vmax == vmin:
            img = np.zeros_like(arr, dtype=np.uint8)
        else:
            img = np.clip((arr - vmin) / (vmax - vmin), 0, 1) * 255
        img = (255 - img).astype(np.uint8)
        pil = Image.fromarray(img).convert('RGB')
        if self.image_size and isinstance(self.image_size, (tuple, list)) and len(self.image_size) == 2:
            pil = pil.resize(self.image_size, Image.BICUBIC)
        return pil

    def _to_cnn_tensor(self, arr: np.ndarray, size: Tuple[int,int]) -> torch.Tensor:
        if hasattr(arr,'filled'): arr = arr.filled(0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        vmin, vmax = np.percentile(arr, [2,98])
        if vmax == vmin:
            norm = np.zeros_like(arr, dtype=np.float32)
        else:
            norm = np.clip((arr - vmin)/(vmax - vmin),0,1)
        desired_ch = 3
        if self._cnn_cfg is not None and hasattr(self._cnn_cfg,'image_channels'):
            desired_ch = int(getattr(self._cnn_cfg,'image_channels',3))
        if desired_ch == 1:
            pil = Image.fromarray((norm*255).astype(np.uint8)).resize(size, Image.BICUBIC)
            arrf = np.array(pil, dtype=np.float32)/255.0
            return torch.from_numpy(arrf).unsqueeze(0)
        pil = Image.fromarray((norm*255).astype(np.uint8)).convert('RGB').resize(size, Image.BICUBIC)
        return torch.tensor(np.array(pil), dtype=torch.float32).permute(2,0,1)/255.0

    def _cnn_features_block(self, base: str, md_text: str, img_tensor: torch.Tensor) -> str:
        if base in self._feat_cache: return self._feat_cache[base]
        if not self._cnn_ready or self._cnn_model is None or self._cnn_cfg is None: return ""
        try:
            base_chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            extra = list(getattr(self._cnn_cfg,'vocab_extra_chars',"#*_{}[]()<>/\\-:+.,;\n \t"))
            vocab = sorted(set(base_chars + extra))
            vocab = ["<pad>","<unk>"] + vocab
            char2idx = {c:i for i,c in enumerate(vocab)}
            ids = [char2idx.get(ch,1) for ch in md_text[: getattr(self,'doc_max_chars',1200)]] or [1]
            text_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            text_mask = (text_ids != 0)
            img = img_tensor.unsqueeze(0)
            dev = self._cnn_cfg.device
            with torch.no_grad():
                out = self._cnn_model(img.to(dev), text_ids.to(dev), text_mask.to(dev))
                # 优先支持带不确定性的原始 flat 输出 (mean + logvar)
                cur_raw = out.get('current_pos_flat_raw')
                fut_raw = out.get('future_pos_flat_raw')
                if cur_raw is not None:
                    cur_raw = cur_raw[0].view(self._cnn_cfg.max_current, 4).cpu()
                    cur_mean = cur_raw[:, :2]
                    cur_logvar = cur_raw[:, 2:]
                else:
                    cur_mean = out['current_pos_flat'][0].view(self._cnn_cfg.max_current,2).cpu()
                    cur_logvar = None
                # 仅在predict_24h=True时获取24小时预测数据
                fut_mean = None
                fut_logvar = None
                if self.predict_24h:
                    fut_raw = out.get('future_pos_flat_raw')
                    if fut_raw is not None:
                        fut_raw = fut_raw[0].view(self._cnn_cfg.max_future, 4).cpu()
                        fut_mean = fut_raw[:, :2]
                        fut_logvar = fut_raw[:, 2:]
                    else:
                        fut = out.get('future_pos_flat', None)
                        if fut is not None:
                            fut_mean = fut[0].view(self._cnn_cfg.max_future,2).cpu()
                            fut_logvar = None
                cur_exist = (torch.sigmoid(out['cur_exist_logits'][0]) > getattr(self._script_cfg,'cnn_exist_threshold',0.5)).cpu()
            cur_list=[]
            for i in range(self._cnn_cfg.max_current):
                if cur_exist[i]:
                    lat = float(cur_mean[i,0].item()*90.0); lon=float(cur_mean[i,1].item()*180.0)
                    entry = {"lat":lat,"lon":lon,"p":1.0}
                    if cur_logvar is not None:
                        # 提供 1-sigma 标准差，单位为度
                        lat_sigma = float(torch.exp(0.5*cur_logvar[i,0]).item() * 90.0)
                        lon_sigma = float(torch.exp(0.5*cur_logvar[i,1]).item() * 180.0)
                        entry["lat_sigma"] = lat_sigma
                        entry["lon_sigma"] = lon_sigma
                    cur_list.append(entry)
            k = getattr(self._script_cfg,'cnn_feature_topk',8)
            cur_list = cur_list[:k]
            
            # 仅在predict_24h=True时处理24小时数据
            if self.predict_24h:
                fut_exist = (torch.sigmoid(out['fut_exist_logits'][0]) > getattr(self._script_cfg,'cnn_exist_threshold',0.5)).cpu()
                fut_list = []
                for i in range(self._cnn_cfg.max_future):
                    if fut_exist[i]:
                        lat = float(fut_mean[i,0].item()*90.0); lon=float(fut_mean[i,1].item()*180.0)
                        entry = {"lat":lat,"lon":lon,"p":1.0}
                        if fut_logvar is not None:
                            lat_sigma = float(torch.exp(0.5*fut_logvar[i,0]).item() * 90.0)
                            lon_sigma = float(torch.exp(0.5*fut_logvar[i,1]).item() * 180.0)
                            entry["lat_sigma"] = lat_sigma
                            entry["lon_sigma"] = lon_sigma
                        fut_list.append(entry)
                fut_list = fut_list[:k]
                block = {"preprocessed_features": {"current_candidates": cur_list, "future_candidates": fut_list,
                          "counts_estimate": {"current": len(cur_list), "future": len(fut_list)}}}
            else:
                block = {"preprocessed_features": {"current_candidates": cur_list,
                          "counts_estimate": {"current": len(cur_list)}}}
            text_block = ("```json\n" + json.dumps(block, ensure_ascii=False, indent=2) + "\n```") if getattr(self._script_cfg,'cnn_append_mode','json')=='json' else str(block)
            self._feat_cache[base] = text_block
            return text_block
        except Exception:
            return ""

    def _build_prefix_kv(self, md_text: str, img_tensor: torch.Tensor):
        if not (self._cnn_ready and self._prefix_encoder is not None):
            return None
        try:
            base_chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            extra = list(getattr(self._cnn_cfg,'vocab_extra_chars',"#*_{}[]()<>/\\-:+.,;\n \t"))
            vocab = sorted(set(base_chars + extra))
            vocab = ["<pad>","<unk>"] + vocab
            char2idx = {c:i for i,c in enumerate(vocab)}
            ids = [char2idx.get(ch,1) for ch in md_text[: getattr(self,'doc_max_chars',1200)]] or [1]
            text_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            text_mask = (text_ids != 0)
            img = img_tensor.unsqueeze(0)
            dev = self._cnn_cfg.device
            with torch.no_grad():
                out = self._cnn_model(img.to(dev), text_ids.to(dev), text_mask.to(dev))
                fused = out.get('fused_vec', None)
            if fused is None:
                return None
            pkv = self._prefix_encoder.build_prefix_kv(fused)
            layers_cpu = []
            for (k,v) in pkv:
                layers_cpu.append((k.detach().cpu(), v.detach().cpu()))
            return tuple(layers_cpu)
        except Exception:
            return None

    def _read_file(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return "（无可用的附加说明文档）"

    def _extract_prev12h_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        count_keys = [
            'past_cyclones_12h_count',
        ]
        pos_keys = [
            'past_cyclones_12h_positions',
        ]
        c = None
        for k in count_keys:
            if k in data:
                try:
                    c = int(data.get(k, 0)); break
                except Exception:
                    pass
        plist = None
        for k in pos_keys:
            if k in data:
                plist = data.get(k, []); break
        out = {
            'prev12h_tc_count': int(c) if c is not None else 0,
            'prev12h_tcs': self._normalize_positions(plist or []),
        }
        return out

    def _prev12h_context_text(self, data: Dict[str, Any]) -> str:
        info = self._extract_prev12h_info(data)
        try:
            note = (
                "Historical context (12h prior):\n"
                "Note: This block describes the situation 12 hours earlier and is provided for reference only. "
                "Do NOT copy it to the final output JSON. Your output must describe the CURRENT state" + (" and the NEXT 24h prediction" if self.predict_24h else "") + ".\n"
            )
            return note + "```text\n" + json.dumps(info, ensure_ascii=False, indent=2) + "\n```\n\n"
        except Exception:
            return ""

    def _cached_doc(self, path: str, max_chars: int) -> str:
        key = (path, max_chars)
        if key in self._doc_cache:
            return self._doc_cache[key]
        text = self._read_file(path)
        if isinstance(text, str) and len(text) > max_chars:
            text = text[: max_chars] + "\n..."
        self._doc_cache[key] = text
        return text

    def _load_doc(self, npy_path: str) -> str:
        base = os.path.splitext(os.path.basename(npy_path))[0]
        path = os.path.join(self.docs_folder, f"{base}.md")
        return self._cached_doc(path, getattr(self, 'doc_max_chars', 1200))

    def _load_doc_by_path(self, path: str) -> str:
        return self._cached_doc(path, getattr(self, 'few_shot_doc_max_chars', 800))

    def _normalize_positions(self, items):
        out = []
        if not items:
            return out
        for it in items:
            try:
                if isinstance(it, dict) and 'lat' in it and 'lon' in it:
                    lat = float(it['lat']); lon = float(it['lon'])
                    out.append({"lat": lat, "lon": lon})
                elif isinstance(it, (list, tuple)) and len(it) == 2:
                    lat = float(it[0]); lon = float(it[1])
                    out.append({"lat": lat, "lon": lon})
            except Exception:
                continue
        return out

    def _build_response_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cur_src = (
            data.get('current_tcs')
            or data.get('tc_positions')
            or data.get('current_tc_positions')
            or []
        )
        result = {
            "current_tc_count": int(data.get('tc_count', 0)),
            "current_tcs": self._normalize_positions(cur_src),
        }
        # 仅在predict_24h=True时添加24小时预测数据
        if self.predict_24h:
            fut_src = (
                data.get('new_24h_tcs')
                or data.get('new_cyclones_24h_positions')
                or data.get('new_24h_tc_positions')
                or data.get('future_tcs')
                or []
            )
            result["new_24h_tc_count"] = int(data.get('new_cyclones_24h_count', 0))
            result["new_24h_tcs"] = self._normalize_positions(fut_src)
        return result

    def _build_few_shot_block(self, current_idx: int) -> str:
        if not self.use_few_shot or self.few_shot_num <= 0:
            return ""
        candidate_indices = [i for i in range(len(self.data_files)) if i != current_idx]
        if not candidate_indices:
            return ""
        rng = np.random.RandomState(self.few_shot_seed + current_idx)
        if self.few_shot_sampling == "head":
            chosen = candidate_indices[: self.few_shot_num]
        else:
            size = min(self.few_shot_num, len(candidate_indices))
            chosen = list(rng.choice(candidate_indices, size=size, replace=False))
        examples_txt = []
        for j, ex_idx in enumerate(chosen, 1):
            ex_path = self.data_files[ex_idx]
            try:
                ex_data = np.load(ex_path, allow_pickle=True).item()
            except Exception:
                continue
            ex_base = os.path.splitext(os.path.basename(ex_path))[0]
            ex_md_path = os.path.join(self.docs_folder, f"{ex_base}.md")
            ex_md = self._load_doc_by_path(ex_md_path)
            ex_answer = self._build_response_data(ex_data)
            ex_block = (
                f"Example {j}:\n"
                f"Markdown:\n{ex_md}\n"
                f"Answer:\n```json\n{json.dumps(ex_answer, ensure_ascii=False, indent=2)}\n```\n"
            )
            examples_txt.append(ex_block)
        if not examples_txt:
            return ""
        return "Few-shot solved examples (image omitted; markdown + answer shown):\n\n" + "\n".join(examples_txt) + "\n"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.data_files[idx]
        data = np.load(path, allow_pickle=True).item()

        img_arr = data.get('image', data.get('img', None))
        if img_arr is None:
            raise ValueError(f"No image found in {path}")
        image = self._to_pil_image(img_arr)

        md_text = self._load_doc(path)
        # 可选 CNN 先验
        cnn_block = ""
        prefix_kv = None
        if self._script_cfg and getattr(self._script_cfg,'use_cnn_features',False):
            try:
                cnn_tensor = self._to_cnn_tensor(img_arr, getattr(self._script_cfg,'cnn_image_size',(256,256)))
                base = os.path.splitext(os.path.basename(path))[0]
                cnn_block = self._cnn_features_block(base, md_text, cnn_tensor)
                if getattr(self._script_cfg, 'use_cnn_feature_prefix', False):
                    prefix_kv = self._build_prefix_kv(md_text, cnn_tensor)
            except Exception:
                cnn_block = ""

        use_image_fewshot = self.use_few_shot and self.few_shot_with_images
        if not use_image_fewshot:
            few_shot_block = self._build_few_shot_block(idx)
            user_text_parts = [
                "You are an AI assistant specialized in tropical cyclogenesis detection and forecasting. "
                "Using the satellite image together with the Markdown note below, identify the number and positions of current tropical cyclones" +
                (", and predict the number and positions of cyclones that will newly form within the next 24 hours. " if self.predict_24h else ". ") +
                "Return the result strictly in the following JSON format.",
            ]
            if self.use_cot and self.cot_instruction:
                user_text_parts.append("\n\nCoT instruction: " + self.cot_instruction)
            # 可选：12h 先验上下文（受配置控制）
            prev12h_ctx = self._prev12h_context_text(data) if (self._script_cfg and getattr(self._script_cfg, 'use_prev12h_context', True)) else ""
            if prev12h_ctx:
                user_text_parts.append("\n\n" + prev12h_ctx)

            if cnn_block:
                user_text_parts.append("\n\nPreprocessed features (from a lightweight CNN encoder):\n" + cnn_block + "\n")
                # 若含不确定性字段，追加使用说明
                if ('lat_sigma' in cnn_block) or ('lon_sigma' in cnn_block):
                    user_text_parts.append(
                        "Notes on candidates: lat_sigma/lon_sigma indicate 1-sigma uncertainty in degrees. "
                        "Prefer candidates with smaller sigma when conflicts occur, and you may discard candidates with excessively large sigma.\n\n"
                    )
            if few_shot_block:
                user_text_parts.append("\n\n" + few_shot_block)
            format_text = "\nAdditional context (Markdown):\n" + md_text + "\n\nFormat requirements:\n```json\n{\n  \"current_tc_count\": int,\n"
            if self.predict_24h:
                format_text += "  \"new_24h_tc_count\": int,\n  \"current_tcs\": [ {\"lat\": float, \"lon\": float} ],\n  \"new_24h_tcs\": [ {\"lat\": float, \"lon\": float} ]\n}\n```\n"
            else:
                format_text += "  \"current_tcs\": [ {\"lat\": float, \"lon\": float} ]\n}\n```\n"
            format_text += "Output only the JSON block above. Do not include any extra text."
            user_text_parts.append(format_text)
            user_text = "".join(user_text_parts)
            if getattr(self, 'model_family', 'qwen') == 'llava':
                prompt_text = "\n".join([
                    "USER: <image>",
                    user_text,
                    "ASSISTANT:",
                ])
                # LLaVA: 使用原始图像尺寸时，禁用截断以避免图像token数量不匹配
                use_truncation = not getattr(self, 'use_original_image_size', False)
                proc_out = self.processor(
                    text=prompt_text, 
                    images=[image], 
                    return_tensors='pt', 
                    truncation=use_truncation, 
                    max_length=self.max_length if use_truncation else None
                )
            else:
                # Qwen: 使用原来的设置，保持 truncation=True
                messages = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
                ]
                prompt_text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                proc_out = self.processor(text=prompt_text, images=[image], return_tensors='pt', truncation=True, max_length=self.max_length)
        else:
            messages = []
            images_list = []
            candidate_indices = [i for i in range(len(self.data_files)) if i != idx]
            rng = np.random.RandomState(self.few_shot_seed + idx)
            if self.few_shot_sampling == "head":
                chosen = candidate_indices[: self.few_shot_num]
            else:
                size = min(self.few_shot_num, len(candidate_indices))
                chosen = list(rng.choice(candidate_indices, size=size, replace=False)) if size > 0 else []
            for ex_idx in chosen:
                try:
                    ex_path = self.data_files[ex_idx]
                    ex_data = np.load(ex_path, allow_pickle=True).item()
                    ex_img = self._to_pil_image(ex_data['image'])
                    ex_base = os.path.splitext(os.path.basename(ex_path))[0]
                    ex_md_path = os.path.join(self.docs_folder, f"{ex_base}.md")
                    ex_md = self._load_doc_by_path(ex_md_path)
                    ex_answer = self._build_response_data(ex_data)
                except Exception:
                    continue
                ex_prev = self._prev12h_context_text(ex_data) if (self._script_cfg and getattr(self._script_cfg, 'use_prev12h_context', True)) else ""
                ex_user_text = (
                    "Use the image and the Markdown note below, and return the JSON strictly in the specified schema.\n\n"
                    + (f"CoT instruction: {self.cot_instruction}\n\n" if self.use_cot and self.cot_instruction else "")
                    + (ex_prev + "\n" if ex_prev else "")
                    + "Additional context (Markdown):\n" + ex_md +
                    "\n\nFormat requirements:\n```json\n{\n"
                    + "  \"current_tc_count\": int,\n"
                )
                if self.predict_24h:
                    ex_user_text += (
                        "  \"new_24h_tc_count\": int,\n"
                        + "  \"current_tcs\": [ {\"lat\": float, \"lon\": float} ],\n"
                        + "  \"new_24h_tcs\": [ {\"lat\": float, \"lon\": float} ]\n"
                        + "}\n```\n"
                    )
                else:
                    ex_user_text += (
                        "  \"current_tcs\": [ {\"lat\": float, \"lon\": float} ]\n"
                        + "}\n```\n"
                    )
                ex_user_text += "Output only the JSON block above."
                messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ex_user_text}]})
                images_list.append(ex_img)
                messages.append({"role": "assistant", "content": f"```json\n{json.dumps(ex_answer, ensure_ascii=False, indent=2)}\n```"})

            cur_user_text = (
                "You are an AI assistant specialized in tropical cyclone detection and forecasting. "
                + ("Using the image and the Markdown note below, identify the number and positions of current tropical cyclones, and predict the number and positions within 24 hours. " if self.predict_24h else "Using the image and the Markdown note below, identify the number and positions of current tropical cyclones. ")
                + "Return strictly in the JSON schema.\n\n"
                + (f"CoT instruction: {self.cot_instruction}\n\n" if self.use_cot and self.cot_instruction else "")
                + (self._prev12h_context_text(data) if (self._script_cfg and getattr(self._script_cfg, 'use_prev12h_context', True)) else "")
                + ("Preprocessed features (from a lightweight CNN encoder):\n" + cnn_block + "\n\n" if cnn_block else "")
                + ("Notes on candidates: lat_sigma/lon_sigma indicate 1-sigma uncertainty in degrees. Prefer candidates with smaller sigma when conflicts occur, and you may discard candidates with excessively large sigma.\n\n" if (cnn_block and (('lat_sigma' in cnn_block) or ('lon_sigma' in cnn_block))) else "")
                + "Additional context (Markdown):\n" + md_text +
                "\n\nFormat requirements:\n```json\n{\n"
                + "  \"current_tc_count\": int,\n"
            )
            if self.predict_24h:
                cur_user_text += (
                    "  \"new_24h_tc_count\": int,\n"
                    + "  \"current_tcs\": [ {\"lat\": float, \"lon\": float} ],\n"
                    + "  \"new_24h_tcs\": [ {\"lat\": float, \"lon\": float} ]\n"
                    + "}\n```\n"
                    + "Additional rules: current_tc_count must equal len(current_tcs); new_24h_tc_count must equal len(new_24h_tcs); "
                )
            else:
                cur_user_text += (
                    "  \"current_tcs\": [ {\"lat\": float, \"lon\": float} ]\n"
                    + "}\n```\n"
                    + "Additional rules: current_tc_count must equal len(current_tcs); "
                )
            cur_user_text += "Output only the JSON block above."
            messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": cur_user_text}]})
            images_list.append(image)

            if getattr(self, 'model_family', 'qwen') == 'llava':
                parts = []
                for m in messages:
                    if m["role"] == "user":
                        txt = m["content"][1]["text"] if isinstance(m.get("content"), list) else str(m.get("content"))
                        parts.append("USER: <image>\n" + txt)
                parts.append("ASSISTANT:")
                prompt_text = "\n".join(parts)
                # LLaVA 多图 few-shot：禁止截断，避免 image token 与 images 数量不一致
                proc_out = self.processor(text=prompt_text, images=images_list, return_tensors='pt', padding=False, truncation=False)
            else:
                # Qwen 多图 few-shot：使用原来的设置，保持 truncation=True
                try:
                    prompt_text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    prompt_text = "\n\n".join([m["content"][1]["text"] if isinstance(m.get("content"), list) else str(m.get("content")) for m in messages])
                proc_out = self.processor(text=prompt_text, images=images_list, return_tensors='pt', truncation=True, max_length=self.max_length)

        item: Dict[str, Any] = {}
        for k, v in proc_out.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                item[k] = v.squeeze(0)
            else:
                item[k] = v
        item['raw_ground_truth'] = data.get('label', data.get('ground_truth', None))
        item['prompt'] = prompt_text if 'prompt_text' in locals() else ""
        item['meta_path'] = path
        if prefix_kv is not None:
            item['past_key_values'] = prefix_kv
        return item


# ---------- Reward functions ----------
SCALE_FACTOR_KM = 100.0
WEIGHT_COUNT_CURRENT = 0.5  # 调整为0.5，因为移除了24h预测
WEIGHT_POS_CURRENT = 0.5    # 调整为0.5，因为移除了24h预测


def _to_latlon_list(items):
    out = []
    if not items:
        return out
    for it in items:
        try:
            if isinstance(it, dict) and 'lat' in it and 'lon' in it:
                lat = float(it['lat']); lon = float(it['lon'])
                if np.isfinite(lat) and np.isfinite(lon):
                    out.append((lat, lon))
            elif isinstance(it, (list, tuple)) and len(it) == 2:
                lat = float(it[0]); lon = float(it[1])
                if np.isfinite(lat) and np.isfinite(lon):
                    out.append((lat, lon))
        except Exception:
            continue
    return out


def _haversine_km(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _position_score_km(pred_items, gt_items, scale_factor_km: float = SCALE_FACTOR_KM) -> float:
    pred_pts = _to_latlon_list(pred_items)
    gt_pts = _to_latlon_list(gt_items)

    if not gt_pts:
        return 1.0 if not pred_pts else 0.0
    if not pred_pts:
        return 0.0

    dist = np.zeros((len(pred_pts), len(gt_pts)), dtype=float)
    for r, (plat, plon) in enumerate(pred_pts):
        for c, (glat, glon) in enumerate(gt_pts):
            dist[r, c] = _haversine_km(plat, plon, glat, glon)

    matched_gt = set()
    total_reward = 0.0
    for r in range(len(pred_pts)):
        best_c = -1
        best_d = float('inf')
        for c in range(len(gt_pts)):
            if c in matched_gt:
                continue
            d = dist[r, c]
            if d < best_d:
                best_d = d
                best_c = c
        if best_c != -1:
            matched_gt.add(best_c)
            total_reward += float(np.exp(-best_d / max(1e-6, scale_factor_km)))

    denom = max(len(pred_pts), len(gt_pts))
    return total_reward / max(1, denom)


def format_reward(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    for completion in completions:
        text = completion[0] if isinstance(completion, (list, tuple)) else completion
        try:
            json_str = text.split("```json")[-1].split("```")[0].strip()
            json.loads(json_str)
            rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def accuracy_reward(completions: List[str], ground_truth: List[Any] = None, **kwargs) -> List[float]:
    ground_truth = ground_truth or []
    rewards = []

    for i, completion in enumerate(completions):
        text = completion[0] if isinstance(completion, (list, tuple)) else completion
        gt = None
        try:
            gt = ground_truth[i]
            if isinstance(gt, str):
                gt = json.loads(gt)
        except Exception:
            gt = None
        try:
            json_str = text.split("```json")[-1].split("```")[0].strip()
            pred = json.loads(json_str)
        except Exception:
            rewards.append(0.0)
            continue
        # 确保 pred 是字典类型
        if not isinstance(pred, dict):
            rewards.append(0.0)
            continue
        if gt is None or not isinstance(gt, dict):
            rewards.append(0.1 if isinstance(pred.get('current_tc_count'), int) else 0.0)
            continue

        cfg = kwargs.get('config', None)
        w_cc = getattr(cfg, 'weight_count_current', WEIGHT_COUNT_CURRENT) if cfg else WEIGHT_COUNT_CURRENT
        w_pc = getattr(cfg, 'weight_pos_current', WEIGHT_POS_CURRENT) if cfg else WEIGHT_POS_CURRENT
        scale_km = getattr(cfg, 'scale_factor_km', SCALE_FACTOR_KM) if cfg else SCALE_FACTOR_KM

        score = 0.0
        # 当前气旋数量奖励
        if pred.get('current_tc_count') == gt.get('current_tc_count'):
            score += w_cc

        # 当前气旋位置奖励
        cur_pred = pred.get('current_tcs', []) or pred.get('current_tc_positions', []) or []
        cur_gt = gt.get('current_tcs', []) or gt.get('current_tc_positions', []) or []
        cur_pos_score = _position_score_km(cur_pred, cur_gt, scale_km)
        score += w_pc * cur_pos_score

        # 24小时预测已移除，不再计算相关奖励

        rewards.append(float(score))

    return rewards


def combined_reward(completions: List[str], **kwargs) -> List[float]:
    gts = kwargs.get('ground_truth', [])
    cfg = kwargs.get('config', None)
    fmt = format_reward(completions)
    acc = accuracy_reward(completions, ground_truth=gts, config=cfg)
    out = []
    for f, a in zip(fmt, acc):
        out.append(f + a if f > 0 else 0.0)
    return out


def make_combined_reward(cfg: ScriptConfig):
    def _combined(completions: List[str], **kwargs) -> List[float]:
        gts = kwargs.get('ground_truth', [])
        fmt = format_reward(completions)
        acc = accuracy_reward(completions, ground_truth=gts, config=cfg)
        return [f + a if f > 0 else 0.0 for f, a in zip(fmt, acc)]
    return _combined


# ---------- Model loader ----------
def load_model_and_processor(config: ScriptConfig):
    if _HAS_UNSLOTH:
        print("[info] loading with unsloth.FastLanguageModel.from_pretrained")
        extra_kwargs = {}
        if getattr(config, 'use_flash_attn2', False):
            extra_kwargs["attn_implementation"] = "flash_attention_2"
        model, processor = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_length,
            load_in_4bit=config.load_in_4bit,
            token=os.getenv('HF_TOKEN'),
            trust_remote_code=True,
            cache_dir=config.cache_dir,
            **extra_kwargs,
        )
        try:
            tok = getattr(processor, 'tokenizer', None)
            if tok is not None:
                pad_id = getattr(tok, 'pad_token_id', None)
                if pad_id is None or (isinstance(pad_id, int) and pad_id < 0):
                    eos_id = getattr(tok, 'eos_token_id', None) or 0
                    try:
                        tok.pad_token_id = eos_id
                    except Exception:
                        pass
                try:
                    processor.pad_token_id = tok.pad_token_id
                except Exception:
                    pass
            else:
                if not hasattr(processor, 'pad_token_id'):
                    processor.pad_token_id = 0
        except Exception:
            if not hasattr(processor, 'pad_token_id'):
                processor.pad_token_id = 0
        # 不在加载阶段直接附加已有 LoRA，先让 unsloth.get_peft_model 处理，再在 main 中加载权重，避免二次嵌套导致形状异常
        # 将模型 config 暴露给 processor
        try:
            processor._llm_config = model.config
        except Exception:
            pass
        return model, processor

    print("[info] loading with transformers AutoProcessor (fallback)")
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True, cache_dir=config.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True, cache_dir=config.cache_dir)
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_id is None or (isinstance(pad_id, int) and pad_id < 0):
        eos_id = getattr(tokenizer, 'eos_token_id', None) or 0
        try:
            tokenizer.pad_token_id = eos_id
        except Exception:
            pass
    if not hasattr(processor, 'pad_token_id'):
        try:
            processor.pad_token_id = tokenizer.pad_token_id
        except Exception:
            processor.pad_token_id = 0

    from transformers import AutoModelForCausalLM
    if getattr(config, 'use_flash_attn2', False):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                cache_dir=config.cache_dir,
                attn_implementation="flash_attention_2",
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(config.model_name, trust_remote_code=True, cache_dir=config.cache_dir)
            try:
                model.config.attn_implementation = "flash_attention_2"
            except Exception:
                pass
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, trust_remote_code=True, cache_dir=config.cache_dir)
    try:
        processor._llm_config = model.config
    except Exception:
        pass
    return model, processor


def smart_data_collator(batch: List[Dict[str, Any]]):
    out: Dict[str, Any] = {}
    keys = set()
    for sample in batch:
        keys.update(sample.keys())
    passthrough_keys = ('raw_ground_truth', 'meta_path', 'prompt')
    for k in keys:
        vals = [b[k] for b in batch if k in b]
        if len(vals) == 0:
            continue
        if k in passthrough_keys:
            out[k] = vals
            continue
        if k == 'past_key_values':
            # 合并每个样本的 per-layer KV 到 batch 维
            L = len(vals[0])
            merged = []
            for l in range(L):
                ks=[]; vs=[]
                for s in range(len(vals)):
                    k_s, v_s = vals[s][l]
                    ks.append(k_s)
                    vs.append(v_s)
                k_b = torch.cat(ks, dim=0)
                v_b = torch.cat(vs, dim=0)
                merged.append((k_b, v_b))
            out[k] = tuple(merged)
            continue
        if isinstance(vals[0], torch.Tensor):
            shapes = [tuple(v.shape) for v in vals]
            if all(s == shapes[0] for s in shapes):
                out[k] = torch.stack(vals, dim=0)
            else:
                max_dims = [max(dim[d] for dim in shapes) for d in range(len(shapes[0]))]
                pad_value = -100 if k == 'labels' else 0
                padded = vals[0].new_full((len(vals),) + tuple(max_dims), pad_value)
                for i, v in enumerate(vals):
                    slices = (i,)
                    for d in range(v.dim()):
                        slices += (slice(0, v.shape[d]),)
                    padded[slices] = v
                out[k] = padded
        else:
            out[k] = vals
    return out


def main():
    cfg = ScriptConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    try:
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    model, processor = load_model_and_processor(cfg)
    # 将脚本配置附加到 processor，便于数据集读取 CNN 先验相关开关
    try:
        processor._script_cfg = cfg
    except Exception:
        pass

    if _HAS_UNSLOTH:
        print("[info] Applying LoRA adapter with unsloth.get_peft_model")
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            target_modules=cfg.lora_target_modules,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
            max_seq_length=cfg.max_length,
        )
        # 若提供了 initial_adapter_path，则在 LoRA 结构建立后加载权重
        if getattr(cfg, 'initial_adapter_path', ""):
            adapter_path = cfg.initial_adapter_path
            if os.path.isdir(adapter_path):
                try:
                    print(f"[info] Loading initial LoRA weights into fresh adapter from: {adapter_path}")
                    # 注意：如果 model 或其 base 已经带有 peft_config，再次调用
                    # PeftModel.from_pretrained 会导致在模型中存在多个 adapter（nested adapters），
                    # 这会触发 PEFT 的警告并可能引发训练/形状异常。为避免重复包装，先检测并跳过。
                    base_for_loading = model.get_base_model() if hasattr(model, 'get_base_model') else model
                    if hasattr(model, 'peft_config') or hasattr(base_for_loading, 'peft_config'):
                        print("[info] Model already has a PEFT adapter (peft_config detected). Skipping PeftModel.from_pretrained to avoid nested adapters.")
                    else:
                        # 仅在目标基础模型没有 peft_config 时才包装并加载适配器
                        loaded = PeftModel.from_pretrained(base_for_loading, adapter_path, is_trainable=True)
                        model = loaded
                except Exception as e:
                    print(f"[warn] Failed loading initial adapter weights: {e}")

    if not hasattr(processor, 'pad_token_id'):
        try:
            processor.pad_token_id = processor.tokenizer.pad_token_id
        except Exception:
            processor.pad_token_id = 0

    all_files = sorted(glob(os.path.join(cfg.data_folder, "*.npy")))
    if len(all_files) == 0:
        raise RuntimeError(f"No data files found in {cfg.data_folder}")
    split = int(len(all_files) * cfg.train_split)
    train_files = all_files[:split]

    train_dataset = CycloneGRPO_DatasetFast(
        train_files,
        cfg.docs_folder,
        processor,
        max_length=cfg.max_length,
        image_size=cfg.image_size,
        predict_24h=cfg.predict_24h,
        use_few_shot=cfg.use_few_shot,
        few_shot_num=cfg.few_shot_num,
        few_shot_sampling=cfg.few_shot_sampling,
        few_shot_doc_max_chars=cfg.few_shot_doc_max_chars,
        few_shot_seed=cfg.few_shot_seed,
        few_shot_with_images=cfg.few_shot_with_images,
        use_cot=cfg.use_cot,
        cot_instruction=cfg.cot_instruction,
        model_family=cfg.model_family,
        doc_max_chars=cfg.doc_max_chars,
    )

    # fast_mode 下调生成成本
    max_completion_length = cfg.max_completion_length
    num_generations = cfg.num_generations
    per_device_train_batch_size = cfg.per_device_train_batch_size
    if getattr(cfg, 'fast_mode', False):
        max_completion_length = min(max_completion_length, 256)
        # GRPO 至少需要 2 个采样，fast 模式下将其压低到 2（而非 1）以满足约束
        if num_generations < 2:
            num_generations = 2
        # 若未固定图像尺寸，则保守使用 batch=1
        if not (getattr(cfg, 'force_fixed_image_size', True) and getattr(cfg, 'allow_batch_gt1_if_fixed_image', True)):
            per_device_train_batch_size = 1

    # 额外保险：无论是否 fast_mode，都强制保证 >=2，以免外部配置传入 1 导致报错
    if num_generations < 2:
        print(f"[guard] Adjusting num_generations from {num_generations} to 2 (GRPO requires >=2).")
        num_generations = 2

    # 如果固定图像尺寸开关关闭且 batch>1，则强制改为 1（视觉 token 不一致风险）
    if (getattr(cfg, 'model_family', 'qwen') == 'qwen' and
        (not getattr(cfg, 'force_fixed_image_size', True) or not getattr(cfg, 'allow_batch_gt1_if_fixed_image', True)) and
        per_device_train_batch_size > 1):
        print(f"[guard] Forcing per_device_train_batch_size from {per_device_train_batch_size} to 1 (dynamic visual tokens).")
        cfg.gradient_accumulation_steps *= per_device_train_batch_size
        per_device_train_batch_size = 1
        print(f"[guard] gradient_accumulation_steps -> {cfg.gradient_accumulation_steps}")

    grpo_args = GRPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=max(cfg.save_steps, 0),
        max_completion_length=max_completion_length,
        num_generations=num_generations,
        beta=cfg.beta,
        bf16=True,
        remove_unused_columns=False,
        report_to="tensorboard",
        dataloader_num_workers=getattr(cfg, 'dataloader_num_workers', 0),
        dataloader_pin_memory=getattr(cfg, 'dataloader_pin_memory', False),
        dataloader_persistent_workers=getattr(cfg, 'dataloader_persistent_workers', False),
        # 关闭 torch.compile 以绕过当前 matmul 形状追踪/FX fake tensor 报错，后续可再按需开启
        torch_compile=False,
    )

    reward_fn = make_combined_reward(cfg)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=processor,
    )

    print("开始 GRPO 微调（加速版，Qwen3-VL-8B-Instruct）")
    trainer.train()

    model_short_name = cfg.model_name.split('/')[-1]
    method_tag = "GRPO-fast"
    cot_tag = "on" if cfg.use_cot else "off"
    fs_num = cfg.few_shot_num if cfg.use_few_shot else 0
    save_dir = os.path.join(cfg.output_dir, f"{model_short_name}_{method_tag}_cot-{cot_tag}_fs-{fs_num}")
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    try:
        processor.save_pretrained(save_dir)
    except Exception:
        pass
    print("训练结束，模型已保存到", save_dir)


if __name__ == '__main__':
    main()
