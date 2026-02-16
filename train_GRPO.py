# =============================================================================
# train_GRPO.py — GRPO RL Fine-tuning with
# Quality-based Reward Shaping (TCG-LLM)
#
# After SFT, this script further fine-tunes the model using Group Relative
# Policy Optimization (GRPO), a reinforcement learning algorithm based on
# within-group relative comparison.  Unlike SFT which imitates training data,
# GRPO uses reward functions to unify multiple optimization objectives and
# enables exploration of diverse strategies.
#
# Reward components:
#   (1) Format Reward: valid JSON check
#   (2) Count Reward:  |pred_count – gt_count| normalized to [0,1]
#   (3) Position Reward: exp(-d / Scale_pos), localization with exponential decay
#   (4) Fine-grained Reward: TP(+1) / FP(-0.5) / FN(-0.8) via Hungarian matching
#   (5) Quality Shaping: γ·Q(s') – Q(s), online-learned quality function
# =============================================================================

import os
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_OPTIMIZER", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import json
import math
import torch
import random
import numpy as np
from glob import glob
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor, AutoTokenizer, TrainerCallback
from peft import PeftModel
from scipy.optimize import linear_sum_assignment

try:
    from unsloth import FastLanguageModel
    _HAS_UNSLOTH = True
except Exception:
    _HAS_UNSLOTH = False
try:
    from cnn_encoders import CycloneFusionModel, FusionConfig as FusionCfgLite
    _HAS_CNN_FUSER = True
except Exception:
    CycloneFusionModel = None
    FusionCfgLite = None
    _HAS_CNN_FUSER = False

try:
    from prefix_injector import make_prefix_encoder_from_config
    _HAS_PREFIX = True
except Exception:
    _HAS_PREFIX = False

hf_cache_dir = "/root/autodl-tmp"
os.environ.setdefault("HF_HOME", hf_cache_dir)
os.makedirs(hf_cache_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# ScriptConfig — GRPO training hyperparameters.
# GRPO settings: lr=5e-5, num_generations=8, batch_size=16 (effective),
# epochs=2, beta=0.01, Scale_pos=100km, gamma=0.95, alpha=0.01.
# Reward weights: w_c=0.3, w_p=0.3, w_f=0.2, w_q=0.2.
# Only QLoRA weights are updated; pretrained VLM weights are frozen.
# ---------------------------------------------------------------------------
@dataclass
class ScriptConfig:
    data_folder: str = "/root/autodl-tmp/TCDLD/image"
    docs_folder: str = "/root/autodl-tmp/TCDLD/image_docs"
    label_folder: str = "/root/autodl-tmp/TCDLD/label"
    gph_folder: str = "/root/autodl-tmp/TCDLD/gph"
    gph_docs_folder: str = "/root/autodl-tmp/TCDLD/gph_docs"
    use_gph: bool = True
    sst_folder: str = "/root/autodl-tmp/TCDLD/sst"
    sst_docs_folder: str = "/root/autodl-tmp/TCDLD/sst_docs"
    use_sst: bool = True
    output_dir: str = "/root/autodl-tmp/GRPO/qwen"
    cache_dir: str = hf_cache_dir

    model_name: str = "unsloth/Qwen3-VL-8B-Instruct"
    load_in_4bit: bool = True
    use_flash_attn2: bool = True
    model_family: str = "qwen"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    logging_steps: int = 20
    save_steps: int = 100
    max_completion_length: int = 512
    num_generations: int = 8
    beta: float = 0.01

    train_split: float = 4/5
    max_length: int = 2048
    doc_max_chars: int = 1200

    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True

    seed: int = 42

    # Position reward exponential decay scale (Scale_pos = 100 km)
    scale_factor_km: float = 100.0
    # Reward component weights (w_c, w_p, w_f, w_q)
    weight_count_current: float = 0.3
    weight_pos_current: float = 0.3
    
    # Fine-grained detection reward: penalize FP and FN differently
    use_finegrained_detection_reward: bool = True
    reward_tp: float = 1.0      # TP reward
    penalty_fp: float = -0.5    # FP penalty (false alarm)
    penalty_fn: float = -0.8    # FN penalty (missed detection, more harmful)
    finegrained_reward_weight: float = 0.2  # w_f
    
    # Quality-based reward shaping: online-learned quality function
    use_quality_based_shaping: bool = True
    value_shaping_weight: float = 0.2       # w_q
    value_gamma: float = 0.95               # discount factor gamma
    quality_estimator_type: str = "heuristic"
    value_learning_rate: float = 0.01       # quality function learning rate alpha
    quality_count_weight: float = 0.4       # weight of count in Q(s)
    quality_position_weight: float = 0.6    # weight of position in Q(s)

    use_cot: bool = True
    cot_instruction: str = (
        "Please follow these steps to reason before answering:\n"
        "(1) Analyze the current input data to identify patterns of tropical cyclone occurrence, and determine whether tropical cyclones are forming globally.\n"
        "(2) If any are present, focus on locating (a) minima of geopotential height, (b) cold-core locations of sea surface temperature, and (c) the spiral center of the cloud system in the satellite imagery, to pinpoint the cyclone eye location(s).\n"
        "(3) Based on the current satellite image and environmental fields, identify all current tropical cyclones and their precise locations."
    )

    initial_adapter_path: str = ""
    cnn_feature_ckpt: str = "/root/autodl-tmp/cnn_encoders/best.pt"
    cnn_device: str = "cuda"

    use_cnn_feature_prefix: bool = True
    prefix_len: int = 128
    prefix_target_layers: int = 0
    prefix_share_across_layers: bool = True
    
    train_prefix_encoder: bool = True
    prefix_encoder_lr: float = 1e-4
    prefix_encoder_weight_decay: float = 0.01


# =====================================================================
# CycloneGRPO_DatasetFast — GRPO Dataset for TCG-LLM.
# Similar to the SFT dataset but returns prompt-only input (no labels)
# since GRPO generates multiple completions and evaluates them via reward.
# =====================================================================
class CycloneGRPO_DatasetFast(Dataset):
    def __init__(self, data_files: List[str], docs_folder: str, processor, max_length: int = 2048,
                 gph_folder: str = "", gph_docs_folder: str = "", use_gph: bool = False,
                 sst_folder: str = "", sst_docs_folder: str = "", use_sst: bool = False,
                 use_cot: bool = False, cot_instruction: str = "",
                 model_family: str = "qwen",
                 doc_max_chars: int = 1200):
        self.data_files = data_files
        self.docs_folder = docs_folder
        self.gph_folder = gph_folder
        self.gph_docs_folder = gph_docs_folder
        self.use_gph = use_gph
        self.sst_folder = sst_folder
        self.sst_docs_folder = sst_docs_folder
        self.use_sst = use_sst
        self.processor = processor
        self.max_length = max_length
        self.model_family = model_family
        self.use_cot = use_cot
        self.cot_instruction = cot_instruction or ""
        self.doc_max_chars = doc_max_chars
        self._doc_cache: Dict[Tuple[str, int], str] = {}
        self._script_cfg = getattr(processor, '_script_cfg', None)
        self._cnn_ready = False
        self._cnn_model = None
        self._cnn_cfg = None
        self._prefix_encoder = None
        need_cnn = (self._script_cfg and _HAS_CNN_FUSER and
            getattr(self._script_cfg, 'use_cnn_feature_prefix', False)
        )
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
                    lite_cfg.image_channels = getattr(self._script_cfg, 'cnn_image_channels', getattr(lite_cfg,'image_channels',1))
                    lite_cfg.device = getattr(self._script_cfg, 'cnn_device', 'cuda')
                    
                    checkpoint_has_gph = False
                    if 'model_state' in ckpt_obj:
                        model_state = ckpt_obj['model_state']
                        checkpoint_has_gph = any(k.startswith('gph_enc.') for k in model_state.keys())
                    
                    if checkpoint_has_gph and self.use_gph:
                        if not hasattr(lite_cfg, 'use_gph') or not lite_cfg.use_gph:
                            lite_cfg.use_gph = True
                        if not hasattr(lite_cfg, 'gph_channels') or lite_cfg.gph_channels != 6:
                            lite_cfg.gph_channels = 6
                    elif not checkpoint_has_gph:
                        lite_cfg.use_gph = False
                        print(f"[grpo-cnn] Checkpoint does not contain GPH modules. Disabling GPH to match checkpoint structure.")
                    else:
                        lite_cfg.use_gph = False
                    
                    vocab_size = 128
                    st = ckpt_obj.get('model_state', {})
                    for name,tensor in st.items():
                        if name.endswith('text_enc.emb.weight'):
                            vocab_size = tensor.shape[0]; break
                    model = CycloneFusionModel(lite_cfg, vocab_size=vocab_size)
                    try:
                        model.load_state_dict(st, strict=True)
                    except RuntimeError as e:
                        error_str = str(e)
                        if "Missing key(s)" in error_str or "size mismatch" in error_str or "Unexpected key(s)" in error_str:
                            print(f"[grpo-cnn] Warning: Strict loading failed ({error_str[:100]}). Attempting partial load (strict=False)...")
                            try:
                                missing_keys, unexpected_keys = model.load_state_dict(st, strict=False)
                                if missing_keys:
                                    print(f"[grpo-cnn] Missing keys (will use random init): {len(missing_keys)} keys")
                                    if len(missing_keys) <= 10:
                                        for mk in missing_keys:
                                            print(f"  - {mk}")
                                    else:
                                        for mk in missing_keys[:5]:
                                            print(f"  - {mk}")
                                        print(f"  ... and {len(missing_keys)-5} more")
                                if unexpected_keys:
                                    print(f"[grpo-cnn] Unexpected keys (ignored, may be from delayed-init modules): {len(unexpected_keys)} keys")
                                    if len(unexpected_keys) <= 10:
                                        for uk in unexpected_keys:
                                            print(f"  - {uk}")
                                    else:
                                        for uk in unexpected_keys[:5]:
                                            print(f"  - {uk}")
                                        print(f"  ... and {len(unexpected_keys)-5} more")
                            except Exception as e2:
                                print(f"[grpo-cnn] Failed to load checkpoint even with strict=False: {e2}")
                                raise
                        else:
                            raise
                    model.to(lite_cfg.device)
                    model.eval()
                    self._cnn_cfg = lite_cfg
                    self._cnn_model = model
                    self._cnn_ready = True
                    if getattr(self._script_cfg, 'use_cnn_feature_prefix', False) and _HAS_PREFIX:
                        llm_cfg = getattr(self.processor, '_llm_config', None)
                        if llm_cfg is not None:
                            try:
                                d_model = getattr(lite_cfg, 'd_model', 256)
                                z_dim = int(3 * d_model)
                                self._prefix_encoder = make_prefix_encoder_from_config(
                                    llm_cfg,
                                    z_dim=z_dim,
                                    prefix_len=getattr(self._script_cfg, 'prefix_len', 128),
                                    target_layers=(getattr(self._script_cfg, 'prefix_target_layers', 0) or None),
                                    share_across_layers=getattr(self._script_cfg, 'prefix_share_across_layers', True),
                                ).to(lite_cfg.device)
                                
                                train_prefix = getattr(self._script_cfg, 'train_prefix_encoder', False)
                                if train_prefix:
                                    for param in self._prefix_encoder.parameters():
                                        param.requires_grad = True
                                    self._prefix_encoder.train()
                                    print(f"[grpo-prefix] encoder ready (prefix_len={getattr(self._script_cfg,'prefix_len',128)}, trainable=True)")
                                else:
                                    for param in self._prefix_encoder.parameters():
                                        param.requires_grad = False
                                    self._prefix_encoder.eval()
                                    print(f"[grpo-prefix] encoder ready (prefix_len={getattr(self._script_cfg,'prefix_len',128)}, trainable=False)")
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
        return pil

    def _to_cnn_tensor(self, arr: np.ndarray) -> torch.Tensor:
        if hasattr(arr, 'filled'): arr = arr.filled(0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        vmin, vmax = np.percentile(arr, [2, 98])
        if vmax == vmin:
            norm = np.zeros_like(arr, dtype=np.float32)
        else:
            norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1).astype(np.float32)
        desired_ch = 1
        if self._cnn_cfg is not None and hasattr(self._cnn_cfg, 'image_channels'):
            desired_ch = int(getattr(self._cnn_cfg, 'image_channels', 1))
        if desired_ch == 1:
            return torch.from_numpy(norm).unsqueeze(0)
        norm_rgb = np.stack([norm]*3, axis=0)
        return torch.from_numpy(norm_rgb)

    def _build_prefix_kv(self, base_name: str, md_text: str, img_tensor: torch.Tensor, requires_grad: bool = False):
        """Generate physics-aware KV prefixes for GRPO training."""
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
            
            gph_tensor = None
            if self.use_gph and hasattr(self._cnn_cfg, 'use_gph') and getattr(self._cnn_cfg, 'use_gph', False):
                try:
                    gph_path1 = os.path.join(self.gph_folder, f"{base_name}_gph.npy")
                    gph_path2 = os.path.join(self.gph_folder, f"{base_name}.npy")
                    gph_path = gph_path1 if os.path.exists(gph_path1) else (gph_path2 if os.path.exists(gph_path2) else None)
                    if gph_path and os.path.exists(gph_path):
                        gph_arr = np.load(gph_path, allow_pickle=True)
                        if hasattr(gph_arr, 'filled'):
                            gph_arr = gph_arr.filled(0)
                        gph_arr = np.nan_to_num(gph_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                        
                        if len(gph_arr.shape) == 3:
                            if gph_arr.shape[2] <= gph_arr.shape[0] and gph_arr.shape[2] <= gph_arr.shape[1]:
                                if gph_arr.shape[2] == getattr(self._cnn_cfg, 'gph_channels', 6):
                                    gph_arr = np.transpose(gph_arr, (2, 0, 1))
                            normalized_levels = []
                            for level in range(gph_arr.shape[0]):
                                level_data = gph_arr[level]
                                vmin, vmax = np.percentile(level_data, [2, 98])
                                if vmax == vmin:
                                    norm_level = np.zeros_like(level_data, dtype=np.float32)
                                else:
                                    norm_level = np.clip((level_data - vmin) / (vmax - vmin), 0, 1)
                                normalized_levels.append(norm_level)
                            gph_arr = np.stack(normalized_levels, axis=0)
                            
                            expected_levels = getattr(self._cnn_cfg, 'gph_channels', 6)
                            if gph_arr.shape[0] != expected_levels:
                                if gph_arr.shape[0] > expected_levels:
                                    gph_arr = gph_arr[:expected_levels]
                                else:
                                    last_level = gph_arr[-1:]
                                    padding = np.repeat(last_level, expected_levels - gph_arr.shape[0], axis=0)
                                    gph_arr = np.concatenate([gph_arr, padding], axis=0)
                            
                            gph_tensor = torch.tensor(gph_arr, dtype=torch.float32).unsqueeze(0)
                except Exception as e:
                    gph_tensor = None
            
            sst_tensor = None
            if self.use_sst and hasattr(self._cnn_cfg, 'use_sst') and getattr(self._cnn_cfg, 'use_sst', False):
                try:
                    sst_base = base_name.replace('_image', '_sst')
                    sst_path1 = os.path.join(self.sst_folder, f"{sst_base}.npy")
                    sst_path2 = os.path.join(self.sst_folder, f"{base_name.replace('_image', '')}_sst.npy")
                    sst_path = sst_path1 if os.path.exists(sst_path1) else (sst_path2 if os.path.exists(sst_path2) else None)
                    if sst_path and os.path.exists(sst_path):
                        sst_arr = np.load(sst_path, allow_pickle=True)
                        if hasattr(sst_arr, 'filled'):
                            sst_arr = sst_arr.filled(0)
                        sst_arr = np.nan_to_num(sst_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                        if len(sst_arr.shape) == 2:
                            vmin, vmax = np.percentile(sst_arr, [2, 98])
                            if vmax == vmin:
                                sst_arr = np.zeros_like(sst_arr, dtype=np.float32)
                            else:
                                sst_arr = np.clip((sst_arr - vmin) / (vmax - vmin), 0, 1).astype(np.float32)
                            sst_arr = np.expand_dims(sst_arr, axis=0)
                            sst_tensor = torch.tensor(sst_arr, dtype=torch.float32).unsqueeze(0)
                except Exception:
                    sst_tensor = None
            
            if requires_grad:
                out = self._cnn_model(img.to(dev), text_ids.to(dev), text_mask.to(dev),
                                     gph_images=gph_tensor.to(dev) if gph_tensor is not None else None,
                                     sst_images=sst_tensor.to(dev) if sst_tensor is not None else None)
                fused = out.get('fused_vec', None)
                if fused is None:
                    return None
                pkv = self._prefix_encoder.build_prefix_kv(fused)
                return pkv
            else:
                with torch.no_grad():
                    out = self._cnn_model(img.to(dev), text_ids.to(dev), text_mask.to(dev),
                                         gph_images=gph_tensor.to(dev) if gph_tensor is not None else None,
                                         sst_images=sst_tensor.to(dev) if sst_tensor is not None else None)
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
            return "(No supplementary documentation available)"

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
        doc_base = base.replace('_image', '')
        path = os.path.join(self.docs_folder, f"{doc_base}.md")
        return self._cached_doc(path, self.doc_max_chars)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.data_files[idx]
        data = np.load(path, allow_pickle=True).item()

        img_arr = data.get('image', data.get('img', None))
        if img_arr is None:
            raise ValueError(f"No image found in {path}")
        image = self._to_pil_image(img_arr)

        md_text = self._load_doc(path)
        prefix_kv = None
        cnn_tensor = None
        base_for_prefix = os.path.splitext(os.path.basename(path))[0]
        if self._script_cfg and getattr(self._script_cfg, 'use_cnn_feature_prefix', False) and self._cnn_ready:
            try:
                cnn_tensor = self._to_cnn_tensor(img_arr)
                train_prefix = getattr(self._script_cfg, 'train_prefix_encoder', False)
                if train_prefix:
                    prefix_kv = self._build_prefix_kv(base_for_prefix, md_text, cnn_tensor, requires_grad=True)
                else:
                    prefix_kv = self._build_prefix_kv(base_for_prefix, md_text, cnn_tensor, requires_grad=False)
            except Exception:
                prefix_kv = None

        user_text_parts = [
            "You are an AI assistant specialized in tropical cyclogenesis detection and localization.\n\n"
            "Your task is to use the satellite image, geopotential height data and sea surface temperature data "
            "together with the Markdown notes corresponding to each data to obtain the current TC numbers and positions.\n\n",
        ]
        if self.use_cot and self.cot_instruction:
            user_text_parts.append(self.cot_instruction + "\n\n")
        if self.use_gph and self.gph_docs_folder:
            base = os.path.splitext(os.path.basename(path))[0]
            gph_base = base.replace('_image', '_gph')
            gph_doc_path1 = os.path.join(self.gph_docs_folder, f"{gph_base}.md")
            gph_doc_path2 = os.path.join(self.gph_docs_folder, f"{base.replace('_image', '')}_gph.md")
            gph_doc = None
            if os.path.exists(gph_doc_path1):
                gph_doc = self._cached_doc(gph_doc_path1, self.doc_max_chars)
            elif os.path.exists(gph_doc_path2):
                gph_doc = self._cached_doc(gph_doc_path2, self.doc_max_chars)
            if gph_doc:
                user_text_parts.append("\nGPH (Geopotential Height) data context (Markdown):\n" + gph_doc + "\n")
        if self.use_sst and self.sst_docs_folder:
            base = os.path.splitext(os.path.basename(path))[0]
            sst_base = base.replace('_image', '_sst')
            sst_doc_path1 = os.path.join(self.sst_docs_folder, f"{sst_base}.md")
            sst_doc_path2 = os.path.join(self.sst_docs_folder, f"{base.replace('_image', '')}_sst.md")
            sst_doc = None
            if os.path.exists(sst_doc_path1):
                sst_doc = self._cached_doc(sst_doc_path1, self.doc_max_chars)
            elif os.path.exists(sst_doc_path2):
                sst_doc = self._cached_doc(sst_doc_path2, self.doc_max_chars)
            if sst_doc:
                user_text_parts.append("\nSST (Sea Surface Temperature) data context (Markdown):\n" + sst_doc + "\n")
        user_text_parts.append("\nSatellite image data context (Markdown):\n" + md_text + "\n\n")
        user_text_parts.append(
            "Return the result strictly in the following JSON format:\n"
            "\"current_tc_count\": int, "
            "\"current_tcs\": [ {\"lat\": float, \"lon\": float} ]"
        )
        user_text = "".join(user_text_parts)
        if self.model_family == 'llava':
            prompt_text = "\n".join([
                "USER: <image>",
                user_text,
                "ASSISTANT:",
            ])
            proc_out = self.processor(
                text=prompt_text,
                images=[image],
                return_tensors='pt',
                padding=False,
                truncation=False,
            )
        else:
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
            ]
            prompt_text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            proc_out = self.processor(text=prompt_text, images=[image], return_tensors='pt', truncation=True, max_length=self.max_length)

        item: Dict[str, Any] = {}
        for k, v in proc_out.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                item[k] = v.squeeze(0)
            else:
                item[k] = v
        item['raw_ground_truth'] = data.get('label', data.get('ground_truth', None))
        item['meta_path'] = path
        item['prompt'] = prompt_text if 'prompt_text' in locals() else ""
        
        if self._script_cfg and hasattr(self._script_cfg, 'label_folder'):
            label_info = load_label_file(path, self._script_cfg.label_folder)
            if label_info:
                item['label_info'] = label_info
            else:
                item['label_info'] = None
        else:
            item['label_info'] = None
        
        if getattr(self._script_cfg, 'train_prefix_encoder', False) and getattr(self._script_cfg, 'use_cnn_feature_prefix', False):
            if cnn_tensor is not None:
                item['_cnn_tensor'] = cnn_tensor
                item['_md_text'] = md_text
                item['_base_name'] = base_for_prefix
            item['past_key_values'] = None
        elif prefix_kv is not None:
            item['past_key_values'] = prefix_kv
        
        return item


SCALE_FACTOR_KM = 100.0
WEIGHT_COUNT_CURRENT = 0.3
WEIGHT_POS_CURRENT = 0.3


def msw_to_intensity_category(msw_knots: float) -> int:
    if msw_knots < 34:
        return 1
    elif msw_knots < 48:
        return 2
    elif msw_knots < 64:
        return 3
    elif msw_knots < 85:
        return 4
    elif msw_knots < 105:
        return 5
    else:
        return 6


def extract_basin_from_filename(filename: str) -> str:
    try:
        basename = os.path.splitext(os.path.basename(filename))[0]
        parts = basename.split('_')
        if len(parts) >= 3:
            return parts[2]
        return "unknown"
    except Exception:
        return "unknown"


def load_label_file(data_path: str, label_folder: str) -> dict:
    try:
        basename = os.path.splitext(os.path.basename(data_path))[0]
        if basename.endswith('_image'):
            label_basename = basename.replace('_image', '_label')
        else:
            label_basename = None
        
        if label_basename:
            label_path = os.path.join(label_folder, f"{label_basename}.npy")
            if os.path.exists(label_path):
                label_data = np.load(label_path, allow_pickle=True).item()
                basin = extract_basin_from_filename(label_path)
                return {
                    'basin': basin,
                    'tc_count': label_data.get('tc_count', 0),
                    'tc_positions': label_data.get('tc_positions', []),
                    'tc_sids': label_data.get('tc_sids', []),
                    'tc_msw': label_data.get('tc_msw', [])
                }
        
        from glob import glob as glob_func
        search_base = basename.replace('_image', '')
        pattern = os.path.join(label_folder, f"{search_base}*label.npy")
        matching_files = glob_func(pattern)
        
        if matching_files:
            label_path = matching_files[0]
            label_data = np.load(label_path, allow_pickle=True).item()
            basin = extract_basin_from_filename(label_path)
            return {
                'basin': basin,
                'tc_count': label_data.get('tc_count', 0),
                'tc_positions': label_data.get('tc_positions', []),
                'tc_sids': label_data.get('tc_sids', []),
                'tc_msw': label_data.get('tc_msw', [])
            }
        else:
            return {}
    except Exception as e:
        return {}


# =====================================================================
# QualityEstimator — Online-learned quality function Q(s).
# State s = [basin, gt_count, pred_count, intensity_category].
# Q(s') = 0.4 * r_count + 0.6 * r_position
# r_quality = gamma * Q(s') - Q(s)
# Online update via exponential moving average:
#   Q_new = (1 - alpha) * Q_old + alpha * (r_observed - Q_old)
#
# This transforms sparse terminal rewards into dense step-wise intermediate
# signals, enabling the model to identify the direction and magnitude of
# improvement more effectively, thereby accelerating convergence.
# =====================================================================
class QualityEstimator:
    def __init__(self, value_type: str = "heuristic", learning_rate: float = 0.01,
                 count_weight: float = 0.4, position_weight: float = 0.6):
        self.value_type = value_type
        self.learning_rate = learning_rate
        self.count_weight = count_weight
        self.position_weight = position_weight
        self.value_cache: Dict[str, float] = {}
        self.value_counts: Dict[str, int] = {}
    
    def estimate_value(self, pred: Dict[str, Any], gt: Dict[str, Any], 
                       scale_km: float = 100.0, match_threshold: float = 300.0) -> float:
        if self.value_type == "heuristic":
            return self._heuristic_value(pred, gt, scale_km, match_threshold)
        else:
            return self._heuristic_value(pred, gt, scale_km, match_threshold)
    
    def _heuristic_value(self, pred: Dict[str, Any], gt: Dict[str, Any],
                        scale_km: float, match_threshold: float) -> float:
        """Compute heuristic quality: Q(s') = w_count * r_count + w_pos * r_position."""
        if not isinstance(pred, dict) or not isinstance(gt, dict):
            return 0.0
        
        pred_count = pred.get('current_tc_count', 0)
        gt_count = gt.get('current_tc_count', 0)
        r_count = 1.0 if pred_count == gt_count else 0.0
        
        cur_pred = pred.get('current_tcs', []) or pred.get('current_tc_positions', []) or []
        cur_gt = gt.get('current_tcs', []) or gt.get('current_tc_positions', []) or []
        pred_pts = _to_latlon_list(cur_pred)
        gt_pts = _to_latlon_list(cur_gt)
        
        if not gt_pts:
            r_position = 1.0 if not pred_pts else 0.0
        elif not pred_pts:
            r_position = 0.0
        else:
            tp, fp, fn, matched_distances = _match_tp_fp_fn(pred_pts, gt_pts, match_threshold)
            total_gt = len(gt_pts)
            
            if total_gt == 0:
                r_position = 1.0 if len(pred_pts) == 0 else 0.0
            else:
                tp_ratio = tp / total_gt if total_gt > 0 else 0.0
                fp_penalty = fp / max(1, len(pred_pts)) * 0.5
                fn_penalty = fn / total_gt * 0.8
                
                r_position = tp_ratio - fp_penalty - fn_penalty
                r_position = max(0.0, min(1.0, r_position))
                
                if matched_distances:
                    avg_distance = np.mean(matched_distances)
                    distance_bonus = float(np.exp(-avg_distance / max(1e-6, scale_km)))
                    r_position = r_position * 0.7 + distance_bonus * 0.3
        
        quality = self.count_weight * r_count + self.position_weight * r_position
        return max(0.0, min(1.0, quality))
    
    def update_value(self, state_key: str, observed_reward: float):
        """Online update of the quality function:
        Q_new = Q_old + alpha * (r_observed - Q_old)
        For a newly observed state, initialize Q = r_observed."""
        if state_key not in self.value_cache:
            self.value_cache[state_key] = observed_reward
            self.value_counts[state_key] = 1
        else:
            Q_old = self.value_cache[state_key]
            count = self.value_counts[state_key]
            Q_new = Q_old + self.learning_rate * (observed_reward - Q_old)
            self.value_cache[state_key] = Q_new
            self.value_counts[state_key] = count + 1


_global_quality_estimator: Optional[QualityEstimator] = None


def get_quality_estimator(cfg) -> QualityEstimator:
    global _global_quality_estimator
    if _global_quality_estimator is None:
        estimator_type = getattr(cfg, 'quality_estimator_type', 'heuristic') if cfg else 'heuristic'
        learning_rate = getattr(cfg, 'value_learning_rate', 0.01) if cfg else 0.01
        count_weight = getattr(cfg, 'quality_count_weight', 0.4) if cfg else 0.4
        position_weight = getattr(cfg, 'quality_position_weight', 0.6) if cfg else 0.6
        _global_quality_estimator = QualityEstimator(
            value_type=estimator_type,
            learning_rate=learning_rate,
            count_weight=count_weight,
            position_weight=position_weight
        )
    return _global_quality_estimator


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


# Haversine formula: great-circle distance (km) between two lat/lon points.
# Used for position reward computation and Hungarian matching.
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ---------------------------------------------------------------------------
# Hungarian algorithm matching for TP/FP/FN computation.
# Uses scipy.optimize.linear_sum_assignment to find optimal pred→gt matching
# that minimizes total Haversine distance.  Matches within threshold_km (300 km)
# are counted as TP; unmatched predictions are FP; unmatched ground truth are FN.
# ---------------------------------------------------------------------------
def _match_tp_fp_fn(pred_pts, gt_pts, threshold_km: float = 300.0):
    if not pred_pts and not gt_pts:
        return 0, 0, 0, []
    if not gt_pts:
        return 0, len(pred_pts), 0, []
    if not pred_pts:
        return 0, 0, len(gt_pts), []
    
    dist_matrix = np.zeros((len(pred_pts), len(gt_pts)), dtype=float)
    for r, (plat, plon) in enumerate(pred_pts):
        for c, (glat, glon) in enumerate(gt_pts):
            dist_matrix[r, c] = _haversine_km(plat, plon, glat, glon)
    
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    matched_distances = []
    tp = 0
    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] <= threshold_km:
            tp += 1
            matched_distances.append(dist_matrix[r, c])
    
    fp = len(pred_pts) - tp
    fn = len(gt_pts) - tp
    
    return tp, fp, fn, matched_distances


# Position reward: r_position = exp(-d / Scale_pos)
# Uses Hungarian algorithm matching to pair predicted and ground-truth TCs.
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

    row_ind, col_ind = linear_sum_assignment(dist)
    
    total_reward = 0.0
    for r, c in zip(row_ind, col_ind):
        total_reward += float(np.exp(-dist[r, c] / max(1e-6, scale_factor_km)))

    denom = max(len(pred_pts), len(gt_pts))
    return total_reward / max(1, denom)


# Format Reward (component 1): checks valid JSON output.
# Returns 1.0 if format is correct, 0.0 otherwise.
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


# ---------------------------------------------------------------------------
# Combined reward function — the core of GRPO reward shaping.
# Computes the overall reward r_all by aggregating five components:
#   r_all = w_c * r_count + w_p * r_position + w_f * r_fine + w_q * r_quality
#         (only when r_format = 1; otherwise r_all = 0)
# The quality function Q(s) is updated online after each reward computation.
# ---------------------------------------------------------------------------
def accuracy_reward(completions: List[str], ground_truth: List[Any] = None, **kwargs) -> List[float]:
    ground_truth = ground_truth or []
    label_info_list = kwargs.get('label_info', [])
    rewards = []
    
    cfg = kwargs.get('config', None)
    use_finegrained = getattr(cfg, 'use_finegrained_detection_reward', True) if cfg else True
    use_quality_shaping = getattr(cfg, 'use_quality_based_shaping', True) if cfg else True
    value_shaping_weight = getattr(cfg, 'value_shaping_weight', 0.2) if cfg else 0.2
    value_gamma = getattr(cfg, 'value_gamma', 0.95) if cfg else 0.95
    match_threshold = getattr(cfg, 'scale_factor_km', SCALE_FACTOR_KM) * 3.0
    if hasattr(cfg, 'match_threshold_km'):
        match_threshold = cfg.match_threshold_km
    
    quality_estimator = None
    if use_quality_shaping and cfg:
        quality_estimator = get_quality_estimator(cfg)
    
    for i, completion in enumerate(completions):
        text = completion[0] if isinstance(completion, (list, tuple)) else completion
        
        try:
            json_str = text.split("```json")[-1].split("```")[0].strip()
            pred = json.loads(json_str)
        except Exception:
            rewards.append(0.0)
            continue
        if not isinstance(pred, dict):
            rewards.append(0.0)
            continue
        
        gt = None
        try:
            gt = ground_truth[i]
            if isinstance(gt, str):
                gt = json.loads(gt)
        except Exception:
            gt = None
        if gt is None or not isinstance(gt, dict):
            rewards.append(0.1 if isinstance(pred.get('current_tc_count'), int) else 0.0)
            continue

        w_cc = getattr(cfg, 'weight_count_current', WEIGHT_COUNT_CURRENT) if cfg else WEIGHT_COUNT_CURRENT
        w_pc = getattr(cfg, 'weight_pos_current', WEIGHT_POS_CURRENT) if cfg else WEIGHT_POS_CURRENT
        scale_km = getattr(cfg, 'scale_factor_km', SCALE_FACTOR_KM) if cfg else SCALE_FACTOR_KM

        # --- Count Reward: r_count = 1 - |pred - gt| / max(gt, 1) ---
        score = 0.0
        
        pred_count_val = int(pred.get('current_tc_count', 0))
        gt_count_val = int(gt.get('current_tc_count', 0))
        r_count = max(0.0, 1.0 - abs(pred_count_val - gt_count_val) / max(gt_count_val, 1))
        score += w_cc * r_count

        cur_pred = pred.get('current_tcs', []) or pred.get('current_tc_positions', []) or []
        cur_gt = gt.get('current_tcs', []) or gt.get('current_tc_positions', []) or []
        pred_pts = _to_latlon_list(cur_pred)
        gt_pts = _to_latlon_list(cur_gt)
        
        # --- Position Reward: exponential decay of Haversine distance ---
        cur_pos_score = _position_score_km(cur_pred, cur_gt, scale_km)
        score += w_pc * cur_pos_score
        
        # --- Fine-grained Reward: TP/FP/FN via Hungarian matching ---
        if use_finegrained:
            tp, fp, fn, _ = _match_tp_fp_fn(pred_pts, gt_pts, match_threshold)
            reward_tp = getattr(cfg, 'reward_tp', 1.0) if cfg else 1.0
            penalty_fp = getattr(cfg, 'penalty_fp', -0.5) if cfg else -0.5
            penalty_fn = getattr(cfg, 'penalty_fn', -0.8) if cfg else -0.8
            finegrained_weight = getattr(cfg, 'finegrained_reward_weight', 0.2) if cfg else 0.2
            
            total_gt = len(gt_pts)
            if total_gt > 0:
                finegrained_score = (tp * reward_tp + fp * penalty_fp + fn * penalty_fn) / (total_gt * reward_tp)
                finegrained_score = max(0.0, min(1.0, finegrained_score))
                score += finegrained_weight * finegrained_score
        
        # --- Quality Shaping Reward: r_quality = gamma * Q(s') - Q(s) ---
        if use_quality_shaping and quality_estimator is not None:
            basin = "unknown"
            intensity = 0
            
            label_info = None
            if i < len(label_info_list):
                label_info = label_info_list[i]
            
            if label_info and isinstance(label_info, dict):
                basin = label_info.get('basin', 'unknown')
                tc_msw_list = label_info.get('tc_msw', [])
                if tc_msw_list:
                    avg_msw = np.mean(tc_msw_list)
                    intensity = msw_to_intensity_category(avg_msw)
                else:
                    intensity = 0
            
            # State key: s = [basin, gt_count, pred_count, intensity]
            gt_count = gt.get('current_tc_count', 0)
            pred_count = pred.get('current_tc_count', 0)
            state_key = f"{basin}_{gt_count}_{pred_count}_{intensity}"
            
            Q_s = quality_estimator.value_cache.get(state_key, 0.0)  # Q(s): previous quality
            
            Q_s_prime = quality_estimator.estimate_value(pred, gt, scale_km, match_threshold)  # Q(s'): current quality
            
            r_quality = value_gamma * Q_s_prime - Q_s  # reward shaping
            
            raw_task_reward = score  # save raw task score before adding quality shaping
            score = score + value_shaping_weight * r_quality  # add weighted quality shaping
            
            quality_estimator.update_value(state_key, raw_task_reward)  # online update Q(s) with raw task reward

        rewards.append(float(score))

    return rewards


# Factory: create a reward function closure with access to ScriptConfig.
def make_combined_reward(cfg: ScriptConfig):
    def _combined(completions: List[str], **kwargs) -> List[float]:
        gts = kwargs.get('ground_truth', [])
        label_info = kwargs.get('label_info', [])
        return accuracy_reward(completions, ground_truth=gts, config=cfg, label_info=label_info)
    
    return _combined


# Load Qwen3-VL-8B with optional LoRA adapter (from SFT checkpoint).
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


# Collate function: batch samples with variable-length padding and KV prefix merging.
def smart_data_collator(batch: List[Dict[str, Any]]):
    out: Dict[str, Any] = {}
    keys = set()
    for sample in batch:
        keys.update(sample.keys())
    passthrough_keys = ('raw_ground_truth', 'meta_path', 'prompt', 'label_info')
    passthrough_keys = passthrough_keys + ('_cnn_tensor', '_md_text', '_base_name')
    for k in keys:
        vals = [b[k] for b in batch if k in b]
        if len(vals) == 0:
            continue
        if k in passthrough_keys:
            out[k] = vals
            continue
        if k == 'past_key_values':
            valid_vals = [v for v in vals if v is not None]
            if valid_vals:
                L = len(valid_vals[0])
                merged = []
                for l in range(L):
                    ks=[]; vs=[]
                    for s in range(len(valid_vals)):
                        k_s, v_s = valid_vals[s][l]
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
    try:
        processor._script_cfg = cfg
    except Exception:
        pass

    if _HAS_UNSLOTH:
        if getattr(cfg, 'initial_adapter_path', ""):
            adapter_path = cfg.initial_adapter_path
            if os.path.isdir(adapter_path):
                try:
                    print(f"[info] Loading existing LoRA adapter from: {adapter_path}")
                    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
                    print(f"[info] Successfully loaded LoRA adapter from {adapter_path}")
                    if hasattr(model, 'peft_config'):
                        peft_config = list(model.peft_config.values())[0] if model.peft_config else None
                        if peft_config:
                            print(f"[info] Loaded adapter config: r={peft_config.r}, alpha={peft_config.lora_alpha}, "
                                  f"target_modules={peft_config.target_modules if hasattr(peft_config, 'target_modules') else 'N/A'}")
                except Exception as e:
                    print(f"[warn] Failed loading initial adapter weights: {e}")
                    print(f"[info] Creating new LoRA adapter with specified config instead...")
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
            else:
                print(f"[warn] initial_adapter_path not found: {adapter_path}. Creating new LoRA adapter...")
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
        else:
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

    if not hasattr(processor, 'pad_token_id'):
        try:
            processor.pad_token_id = processor.tokenizer.pad_token_id
        except Exception:
            processor.pad_token_id = 0

    all_files = sorted(glob(os.path.join(cfg.data_folder, "*_image.npy")))
    if len(all_files) == 0:
        raise RuntimeError(f"No data files found in {cfg.data_folder}")
    split = int(len(all_files) * cfg.train_split)
    train_files = all_files[:split]

    train_dataset = CycloneGRPO_DatasetFast(
        train_files,
        cfg.docs_folder,
        processor,
        max_length=cfg.max_length,
        gph_folder=cfg.gph_folder,
        gph_docs_folder=cfg.gph_docs_folder,
        use_gph=cfg.use_gph,
        sst_folder=cfg.sst_folder,
        sst_docs_folder=cfg.sst_docs_folder,
        use_sst=cfg.use_sst,
        use_cot=cfg.use_cot,
        cot_instruction=cfg.cot_instruction,
        model_family=cfg.model_family,
        doc_max_chars=cfg.doc_max_chars,
    )

    max_completion_length = min(cfg.max_completion_length, 256)
    num_generations = max(cfg.num_generations, 2)
    per_device_train_batch_size = cfg.per_device_train_batch_size

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
        torch_compile=False,
    )

    reward_fn = make_combined_reward(cfg)
    
    # ---------------------------------------------------------------
    # Cooperative Training — same as in SFT but for GRPO.
    # Jointly optimize LoRA weights + prefix encoder with separate LRs.
    # ---------------------------------------------------------------
    prefix_encoder = None
    cnn_model = None
    cnn_cfg = None
    if getattr(cfg, 'use_cnn_feature_prefix', False) and getattr(cfg, 'train_prefix_encoder', False):
        if hasattr(train_dataset, '_prefix_encoder'):
            prefix_encoder = train_dataset._prefix_encoder
        if hasattr(train_dataset, '_cnn_model'):
            cnn_model = train_dataset._cnn_model
        if hasattr(train_dataset, '_cnn_cfg'):
            cnn_cfg = train_dataset._cnn_cfg
        
        if prefix_encoder is not None:
            print(f"[CooperativeTraining] Prefix encoder detected, enabling cooperative training")
            print(f"  Prefix encoder LR: {getattr(cfg, 'prefix_encoder_lr', 1e-4)}")
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=processor,
    )
    
    if prefix_encoder is not None and getattr(cfg, 'train_prefix_encoder', False):
        def create_cooperative_optimizer():
            lora_params = []
            prefix_params = []
            other_params = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'lora' in name.lower():
                        lora_params.append(param)
                    else:
                        other_params.append(param)
            
            prefix_params = [p for p in prefix_encoder.parameters() if p.requires_grad]
            
            param_groups = []
            
            if lora_params:
                param_groups.append({
                    'params': lora_params,
                    'lr': cfg.learning_rate,
                    'weight_decay': getattr(cfg, 'weight_decay', 0.01),
                })
            
            if prefix_params:
                param_groups.append({
                    'params': prefix_params,
                    'lr': getattr(cfg, 'prefix_encoder_lr', 1e-4),
                    'weight_decay': getattr(cfg, 'prefix_encoder_weight_decay', 0.01),
                })
            
            if other_params:
                param_groups.append({
                    'params': other_params,
                    'lr': cfg.learning_rate,
                    'weight_decay': getattr(cfg, 'weight_decay', 0.01),
                })
            
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            
            print(f"[CooperativeTraining] Creating cooperative optimizer:")
            print(f"  LoRA params: {len(lora_params)}")
            print(f"  Prefix encoder params: {len(prefix_params)}")
            print(f"  Other params: {len(other_params)}")
            
            return optimizer
        
        trainer.create_optimizer = create_cooperative_optimizer
        
        class GradientClipCallback(TrainerCallback):
            def __init__(self, prefix_encoder):
                self.prefix_encoder = prefix_encoder
            
            def on_step_end(self, args, state, control, model=None, **kwargs):
                if self.prefix_encoder is not None:
                    lora_params = [p for n, p in model.named_parameters() 
                                  if 'lora' in n.lower() and p.requires_grad and p.grad is not None]
                    if lora_params:
                        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                    
                    prefix_params = [p for p in self.prefix_encoder.parameters() 
                                   if p.requires_grad and p.grad is not None]
                    if prefix_params:
                        torch.nn.utils.clip_grad_norm_(prefix_params, max_norm=1.0)
        
        trainer.add_callback(GradientClipCallback(prefix_encoder))
        print(f"[CooperativeTraining] Gradient clipping callback added")
        
        try:
            callback_path = os.path.join(os.path.dirname(__file__), 'prefix_generation_callback.py')
            if os.path.exists(callback_path):
                from prefix_generation_callback import PrefixGenerationCallback
                device = next(model.parameters()).device if hasattr(model, 'parameters') and next(model.parameters(), None) is not None else "cuda"
                prefix_callback = PrefixGenerationCallback(
                    prefix_encoder=prefix_encoder,
                    cnn_model=cnn_model,
                    cnn_cfg=cnn_cfg,
                    device=str(device),
                )
                trainer.add_callback(prefix_callback)
                print(f"[CooperativeTraining] Prefix KV generation callback added")
            else:
                print(f"[CooperativeTraining] Warning: prefix_generation_callback.py not found, will use Prefix KV generated during data loading")
        except Exception as e:
            print(f"[CooperativeTraining] Warning: Failed to import PrefixGenerationCallback: {e}")
            print(f"[CooperativeTraining] Will use Prefix KV generated during data loading (gradients may be lost)")
    
    print("Starting GRPO fine-tuning (accelerated, Qwen3-VL-8B-Instruct)")
    trainer.train()

    model_short_name = cfg.model_name.split('/')[-1]
    method_tag = "GRPO-fast"
    cot_tag = "on" if cfg.use_cot else "off"
    gph_tag = "gph-on" if getattr(cfg, 'use_gph', False) else "gph-off"
    sst_tag = "sst-on" if getattr(cfg, 'use_sst', False) else "sst-off"
    save_dir = os.path.join(cfg.output_dir, f"{model_short_name}_{method_tag}_cot-{cot_tag}_{gph_tag}_{sst_tag}")
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    try:
        processor.save_pretrained(save_dir)
    except Exception:
        pass
    print("Training complete, model saved to", save_dir)


if __name__ == '__main__':
    main()
