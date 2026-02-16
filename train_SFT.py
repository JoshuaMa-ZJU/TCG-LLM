# =============================================================================
# train_SFT.py — Physics-aware SFT Training
#
# Supervised fine-tuning (SFT) of Qwen3-VL-8B using QLoRA PEFT for basin-scale
# tropical cyclogenesis (TCG) detection and localization.  Physics-aware CNN
# encoder features are injected as KV prefixes into VLM self-attention layers
# (via prefix_injector.py), allowing the model to leverage physically
# informative representations during adaptation.
#
# Key components:
#   - QLoRA (r=16, alpha=32) on Qwen3-VL-8B-Instruct
#   - Physics-aware prefix injection from CNN encoders (768-dim → 128 prefix tokens)
#   - Cooperative training: jointly optimize LoRA weights + prefix encoder
#   - Multimodal input: satellite image + GPH + SST + textual descriptors
#   - CoT-style prompting for multi-step meteorological reasoning
# =============================================================================

import os
import json
import re
import math
import torch
import numpy as np
import torch.nn as nn
from unsloth import FastLanguageModel
from glob import glob
from PIL import Image
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from torch.utils.data import Dataset

try:
    from cnn_encoders import CycloneFusionModel, FusionConfig as FusionCfgLite
    _HAS_CNN_FUSER = True
except Exception:
    CycloneFusionModel = None
    FusionCfgLite = None
    _HAS_CNN_FUSER = False

from transformers import (
    AutoProcessor,
    AutoConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)

try:
    from prefix_injector import make_prefix_encoder_from_config
    _HAS_PREFIX = True
except Exception:
    _HAS_PREFIX = False

hf_cache_dir = "/root/autodl-tmp"
os.environ["HF_HOME"] = hf_cache_dir
os.makedirs(hf_cache_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# ScriptConfig — Training hyperparameters.
# For QLoRA SFT: r=16, alpha=32, dropout=0.05, batch_size=16 (effective),
# epochs=3, lr=1.5e-4.  Physics-aware prefix: prefix_len=128, shared across
# all VLM layers.  The pretrained Qwen3-VL-8B weights are frozen; only
# QLoRA weights and the prefix encoder are updated.
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
    output_dir: str = "/root/autodl-tmp/output/"
    cache_dir: str = "/root/autodl-tmp/cache/"

    model_name: str = "unsloth/Qwen3-VL-8B-Instruct"
    use_flash_attn2: bool = True

    load_in_4bit: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1.5e-4
    warmup_steps: int = 100
    logging_steps: int = 20
    save_steps: int = 100
    eval_steps: int = 100

    train_split: float = 0.8
    max_length: int = 2048
    doc_max_chars: int = 1200

    dataloader_num_workers: int = 8
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True

    monitor_during_training: bool = True
    monitor_strategy: str = "steps"

    enable_early_stopping: bool = False
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.0

    seed: int = 42
    fp16: bool = False
    bf16: bool = True

    cnn_feature_ckpt: Optional[str] = '/root/autodl-tmp/cnn_encoders/best.pt'
    cnn_device: str = "cuda"

    use_cnn_feature_prefix: bool = True
    prefix_len: int = 128
    prefix_target_layers: int = 0
    prefix_share_across_layers: bool = True
    
    train_prefix_encoder: bool = True
    prefix_encoder_lr: float = 1.5e-4
    prefix_encoder_weight_decay: float = 0.01

    # CoT (Chain-of-Thought) prompting.
    # Encourages decomposing the TCG problem into intermediate reasoning steps:
    # (1) identify TC patterns, (2) locate GPH minima / SST cold cores /
    # cloud spiral centers, (3) determine all TC positions.
    use_cot: bool = True
    cot_instruction: str = (
        "Please follow these steps to reason before answering:\n"
        "(1) Analyze the current input data to identify patterns of tropical cyclone occurrence, and determine whether tropical cyclones are forming globally.\n"
        "(2) If any are present, focus on locating (a) minima of geopotential height, (b) cold-core locations of sea surface temperature, and (c) the spiral center of the cloud system in the satellite imagery, to pinpoint the cyclone eye location(s).\n"
        "(3) Based on the current satellite image and environmental fields, identify all current tropical cyclones and their precise locations."
    )


# Extract basin identifier from filename (e.g., "20240914_0000_WP_image" → "WP").
def extract_basin_from_filename(filename: str) -> str:
    try:
        basename = os.path.splitext(os.path.basename(filename))[0]
        parts = basename.split('_')
        if len(parts) >= 3:
            return parts[2]
        return "unknown"
    except Exception:
        return "unknown"


# Load ground-truth label file containing tc_count, tc_positions, tc_sids, tc_msw.
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
        
        search_base = basename.replace('_image', '')
        pattern = os.path.join(label_folder, f"{search_base}*label.npy")
        matching_files = glob(pattern)
        
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
# CycloneDatasetFast — SFT Dataset for TCG-LLM.
# Builds multimodal training samples: each sample consists of a satellite
# image (PIL), textual statistical descriptors (GPH/SST/image docs),
# ground-truth labels, and optionally physics-aware KV prefixes generated
# by the CNN encoders + FeaturePrefixEncoder.
# =====================================================================
class CycloneDatasetFast(Dataset):
    def __init__(self, data_files: List[str], docs_folder: str, processor, max_length: int,
                 gph_folder: str = "", gph_docs_folder: str = "", use_gph: bool = False,
                 sst_folder: str = "", sst_docs_folder: str = "", use_sst: bool = False,
                 use_cot: bool = False, cot_instruction: str = "",
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
        self.use_cot = use_cot
        self.cot_instruction = cot_instruction or ""
        self.doc_max_chars = doc_max_chars
        self._doc_cache: Dict[Tuple[str, int], str] = {}

        # ----- Physics-aware CNN prefix initialization -----
        # If use_cnn_feature_prefix is enabled, load the pretrained CNN encoder
        # checkpoint and build a FeaturePrefixEncoder to generate KV prefixes
        # for VLM self-attention injection during training.
        self._cnn_ready = False
        self._cnn_cfg = None
        self._cnn_model = None
        self._prefix_encoder = None
        if getattr(processor, '_script_cfg', None) is not None:
            self._script_cfg = processor._script_cfg
        else:
            self._script_cfg = None
        need_cnn = (self._script_cfg and _HAS_CNN_FUSER and
            getattr(self._script_cfg, 'use_cnn_feature_prefix', False)
        )
        if need_cnn:
            ckpt = getattr(self._script_cfg, 'cnn_feature_ckpt', None)
            if ckpt and os.path.exists(ckpt):
                try:
                    ckpt_obj = torch.load(ckpt, map_location='cpu')
                    cfg_dict = ckpt_obj.get('cfg', {})
                    lite_cfg = FusionCfgLite()
                    for k, v in cfg_dict.items():
                        if hasattr(lite_cfg, k):
                            setattr(lite_cfg, k, v)
                    
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
                        print(f"[cnn-preproc] Checkpoint does not contain GPH modules. Disabling GPH to match checkpoint structure.")
                    else:
                        lite_cfg.use_gph = False
                    
                    lite_cfg.device = self._script_cfg.cnn_device
                    self._cnn_cfg = lite_cfg
                    vocab_size = 128
                    if 'model_state' in ckpt_obj:
                        st = ckpt_obj['model_state']
                        for name, tensor in st.items():
                            if name.endswith('text_enc.emb.weight'):
                                vocab_size = tensor.shape[0]
                                break
                    model = CycloneFusionModel(self._cnn_cfg, vocab_size=vocab_size)
                    try:
                        model.load_state_dict(ckpt_obj['model_state'], strict=True)
                    except RuntimeError as e:
                        error_str = str(e)
                        if "Missing key(s)" in error_str or "size mismatch" in error_str or "Unexpected key(s)" in error_str:
                            print(f"[cnn-preproc] Warning: Strict loading failed ({error_str[:100]}). Attempting partial load (strict=False)...")
                            try:
                                missing_keys, unexpected_keys = model.load_state_dict(ckpt_obj['model_state'], strict=False)
                                if missing_keys:
                                    print(f"[cnn-preproc] Missing keys (will use random init): {len(missing_keys)} keys")
                                    if len(missing_keys) <= 10:
                                        for mk in missing_keys:
                                            print(f"  - {mk}")
                                    else:
                                        for mk in missing_keys[:5]:
                                            print(f"  - {mk}")
                                        print(f"  ... and {len(missing_keys)-5} more")
                                if unexpected_keys:
                                    print(f"[cnn-preproc] Unexpected keys (ignored): {len(unexpected_keys)} keys")
                                    if len(unexpected_keys) <= 10:
                                        for uk in unexpected_keys:
                                            print(f"  - {uk}")
                                    else:
                                        for uk in unexpected_keys[:5]:
                                            print(f"  - {uk}")
                                        print(f"  ... and {len(unexpected_keys)-5} more")
                            except Exception as e2:
                                print(f"[cnn-preproc] Failed to load checkpoint even with strict=False: {e2}")
                                raise
                        else:
                            raise
                    model.to(lite_cfg.device)
                    model.eval()
                    self._cnn_model = model
                    self._cnn_ready = True
                    print(f"[cnn-preproc] loaded: {ckpt}")
                    if getattr(self._script_cfg, 'use_cnn_feature_prefix', False):
                        try:
                            llm_cfg = getattr(self.processor, '_llm_config', None)
                            if llm_cfg is not None:
                                d_model = getattr(self._cnn_cfg, 'd_model', 256)
                                z_dim = int(3 * d_model)
                                self._prefix_encoder = make_prefix_encoder_from_config(
                                    llm_cfg,
                                    z_dim=z_dim,
                                    prefix_len=getattr(self._script_cfg, 'prefix_len', 128),
                                    target_layers=(getattr(self._script_cfg, 'prefix_target_layers', 0) or None),
                                    share_across_layers=getattr(self._script_cfg, 'prefix_share_across_layers', True),
                                ).to(self._cnn_cfg.device)
                                
                                train_prefix = getattr(self._script_cfg, 'train_prefix_encoder', False)
                                if train_prefix:
                                    for param in self._prefix_encoder.parameters():
                                        param.requires_grad = True
                                    self._prefix_encoder.train()
                                    print(f"[prefix] Prefix encoder ready (trainable=True).")
                                else:
                                    for param in self._prefix_encoder.parameters():
                                        param.requires_grad = False
                                    self._prefix_encoder.eval()
                                    print(f"[prefix] Prefix encoder ready (trainable=False).")
                            else:
                                print("[prefix] LLM config not found on processor; prefix disabled.")
                        except Exception as e:
                            self._prefix_encoder = None
                            print(f"[prefix] Failed to build prefix encoder, prefix disabled: {e}")
                except Exception as e:
                    print(f"[cnn-preproc] failed to load {ckpt}: {e}")
            else:
                print("[cnn-preproc] checkpoint not provided or not found; skipping CNN features.")

    def __len__(self):
        return len(self.data_files)

    def _load_data(self, path: str) -> Dict[str, Any]:
        return np.load(path, allow_pickle=True).item()

    def _read_file(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""

    def _cached_doc(self, path: str, max_chars: int) -> str:
        key = (path, max_chars)
        if key in self._doc_cache:
            return self._doc_cache[key]
        text = self._read_file(path)
        if not text:
            text = "No supplementary data available."
        if isinstance(text, str) and len(text) > max_chars:
            text = text[: max_chars] + "\n..."
        self._doc_cache[key] = text
        return text

    def _to_pil_image(self, img_array: np.ndarray) -> Image.Image:
        """Convert satellite image array to PIL Image for VLM processor.
        Applies percentile normalization (2-98%) and inverts for cloud visualization."""
        if hasattr(img_array, 'filled'):
            img_array = img_array.filled(0)
        img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0)
        vmin, vmax = np.percentile(img_array, [2, 98])
        if vmax == vmin:
            img_normalized = np.zeros_like(img_array, dtype=np.uint8)
        else:
            img_normalized = np.clip((img_array - vmin) / (vmax - vmin), 0, 1) * 255
        img_colored = 255 - img_normalized.astype(np.uint8)
        pil = Image.fromarray(img_colored).convert('RGB')
        return pil

    def _to_cnn_tensor(self, img_array: np.ndarray) -> torch.Tensor:
        """Convert satellite image array to normalized tensor for CNN encoder."""
        if hasattr(img_array, 'filled'):
            img_array = img_array.filled(0)
        img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        vmin, vmax = np.percentile(img_array, [2, 98])
        if vmax == vmin:
            norm = np.zeros_like(img_array, dtype=np.float32)
        else:
            norm = np.clip((img_array - vmin) / (vmax - vmin), 0, 1).astype(np.float32)
        desired_ch = 1
        if self._cnn_cfg is not None and hasattr(self._cnn_cfg, 'image_channels'):
            try:
                desired_ch = int(getattr(self._cnn_cfg, 'image_channels', 1))
            except Exception:
                desired_ch = 1
        if desired_ch == 1:
            return torch.from_numpy(norm).unsqueeze(0)
        else:
            norm_rgb = np.stack([norm]*3, axis=0)
            return torch.from_numpy(norm_rgb)

    def _load_and_preprocess_gph(self, base_name: str) -> Optional[torch.Tensor]:
        """Load and normalize GPH data (6 pressure levels) for the CNN GPH Encoder."""
        gph_tensor = None
        if self.use_gph and hasattr(self._cnn_cfg, 'use_gph') and getattr(self._cnn_cfg, 'use_gph', False):
            try:
                base = base_name.replace('_image', '_gph')
                gph_path1 = os.path.join(self.gph_folder, f"{base}.npy")
                gph_path2 = os.path.join(self.gph_folder, f"{base_name.replace('_image', '')}_gph.npy")
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
                                norm_level = np.clip((level_data - vmin) / (vmax - vmin), 0, 1).astype(np.float32)
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
            except Exception:
                gph_tensor = None
        
        return gph_tensor
    
    def _load_and_preprocess_sst(self, base_name: str) -> Optional[torch.Tensor]:
        """Load and normalize SST data for the CNN SST Encoder."""
        sst_tensor = None
        if self.use_sst and hasattr(self._cnn_cfg, 'use_sst') and getattr(self._cnn_cfg, 'use_sst', False):
            try:
                base = base_name.replace('_image', '_sst')
                sst_path1 = os.path.join(self.sst_folder, f"{base}.npy")
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
        
        return sst_tensor
    
    def _build_prefix_kv(self, base_name: str, data: Dict[str, Any], image_tensor: torch.Tensor, md_text: str, requires_grad: bool = False):
        """Generate physics-aware KV prefixes.
        Runs the image (+ GPH + SST) through the CNN encoder to obtain fused_vec,
        then maps it to past_key_values via the FeaturePrefixEncoder."""
        if not (self._cnn_ready and self._prefix_encoder is not None):
            return None
        try:
            base_chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            extra = list(getattr(self._cnn_cfg, 'vocab_extra_chars', "#*_{}[]()<>/\\-:+.,;\n \t"))
            vocab = sorted(set(base_chars + extra))
            vocab = ["<pad>", "<unk>"] + vocab
            char2idx = {c:i for i,c in enumerate(vocab)}
            ids = [char2idx.get(ch, 1) for ch in md_text[: getattr(self, 'doc_max_chars', 1200)]] or [1]
            text_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            text_mask = (text_ids != 0)
            img = image_tensor.unsqueeze(0)
            dev = self._cnn_cfg.device
            
            gph_tensor = self._load_and_preprocess_gph(base_name)
            
            sst_tensor = self._load_and_preprocess_sst(base_name)
            
            if requires_grad:
                img_dev = img.to(dev)
                text_ids_dev = text_ids.to(dev)
                text_mask_dev = text_mask.to(dev)
                gph_images_dev = gph_tensor.to(dev) if gph_tensor is not None else None
                sst_images_dev = sst_tensor.to(dev) if sst_tensor is not None else None
                out = self._cnn_model(img_dev, text_ids_dev, text_mask_dev, 
                                     gph_images=gph_images_dev, sst_images=sst_images_dev)
                fused = out.get('fused_vec', None)
                if fused is None:
                    return None
                pkv = self._prefix_encoder.build_prefix_kv(fused)
                return pkv
            else:
                with torch.no_grad():
                    img_dev = img.to(dev)
                    text_ids_dev = text_ids.to(dev)
                    text_mask_dev = text_mask.to(dev)
                    gph_images_dev = gph_tensor.to(dev) if gph_tensor is not None else None
                    sst_images_dev = sst_tensor.to(dev) if sst_tensor is not None else None
                    out = self._cnn_model(img_dev, text_ids_dev, text_mask_dev, 
                                         gph_images=gph_images_dev, sst_images=sst_images_dev)
                    fused = out.get('fused_vec', None)
                if fused is None:
                    return None
                pkv = self._prefix_encoder.build_prefix_kv(fused)
                pkv_cpu = []
                for (k, v) in pkv:
                    pkv_cpu.append((k.detach().cpu(), v.detach().cpu()))
                return tuple(pkv_cpu)
        except Exception:
            return None

    def _normalize_positions(self, items):
        out = []
        if not items:
            return out
        for it in items:
            try:
                if isinstance(it, dict) and 'lat' in it and 'lon' in it:
                    out.append({"lat": float(it['lat']), "lon": float(it['lon'])})
                elif isinstance(it, (list, tuple)) and len(it) == 2:
                    out.append({"lat": float(it[0]), "lon": float(it[1])})
            except Exception:
                continue
        return out

    def _build_response_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cur_src = (data.get('tc_positions') or data.get('current_tcs') or data.get('current_tc_positions') or [])
        result = {
            "current_tc_count": int(data.get('tc_count', 0)),
            "current_tcs": self._normalize_positions(cur_src),
        }
        return result

    def _load_doc_by_stem(self, base_name: str, max_chars: int) -> str:
        doc_base = base_name.replace('_image', '')
        doc_path = os.path.join(self.docs_folder, f"{doc_base}.md")
        return self._cached_doc(doc_path, max_chars)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Build a single training sample with multimodal input and labels.
        Constructs the VLM prompt with: system + task + CoT + GPH/SST/image docs
        + format constraint."""
        npy_path = self.data_files[idx]
        data = self._load_data(npy_path)
        image = self._to_pil_image(data['image'])
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        doc_content = self._load_doc_by_stem(base_name, self.doc_max_chars)

        # Compose the VLM user prompt (System + Task + CoT + Data + Format)
        user_text = (
            "You are an AI assistant specialized in tropical cyclogenesis detection and localization.\n\n"
            "Your task is to use the satellite image, geopotential height data and sea surface temperature data "
            "together with the Markdown notes corresponding to each data to obtain the current TC numbers and positions.\n\n"
        )
        if self.use_cot and self.cot_instruction:
            user_text += self.cot_instruction + "\n\n"
        if self.use_gph:
            gph_doc_path1 = os.path.join(self.gph_docs_folder, f"{base_name}_gph.md")
            gph_doc_path2 = os.path.join(self.gph_docs_folder, f"{base_name}.md")
            gph_doc = None
            if os.path.exists(gph_doc_path1):
                gph_doc = self._cached_doc(gph_doc_path1, self.doc_max_chars)
            elif os.path.exists(gph_doc_path2):
                gph_doc = self._cached_doc(gph_doc_path2, self.doc_max_chars)
            if gph_doc:
                user_text += "GPH (Geopotential Height) data context (Markdown):\n" + gph_doc + "\n\n"
        if self.use_sst:
            sst_doc_path1 = os.path.join(self.sst_docs_folder, f"{base_name}_sst.md")
            sst_doc_path2 = os.path.join(self.sst_docs_folder, f"{base_name}.md")
            sst_doc = None
            if os.path.exists(sst_doc_path1):
                sst_doc = self._cached_doc(sst_doc_path1, self.doc_max_chars)
            elif os.path.exists(sst_doc_path2):
                sst_doc = self._cached_doc(sst_doc_path2, self.doc_max_chars)
            if sst_doc:
                user_text += "SST (Sea Surface Temperature) data context (Markdown):\n" + sst_doc + "\n\n"
        user_text += "Satellite image data context (Markdown):\n" + doc_content + "\n\n"
        user_text += (
            "Return the result strictly in the following JSON format:\n"
            "\"current_tc_count\": int, \"current_tcs\": [ {\"lat\": float, \"lon\": float} ]"
        )
        # Build ground-truth JSON response for SFT target
        response_data = self._build_response_data(data)
        response = f"```json\n{json.dumps(response_data, indent=2, ensure_ascii=False)}\n```"

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
            {"role": "assistant", "content": response},
        ]
        prompt_text = self.processor.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full_text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_tokens = self.processor(text=prompt_text, add_special_tokens=True, return_tensors="pt").input_ids
        inputs = self.processor(
            text=full_text,
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
        )
        # Tokenize: mask prompt tokens in labels (only train on assistant response)
        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = prompt_tokens.shape[1] - 1

        labels[:prompt_len] = -100  # mask prompt tokens
        padding_token_id = self.processor.tokenizer.pad_token_id
        if padding_token_id is not None:
            labels[labels == padding_token_id] = -100
        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        inputs["labels"] = labels

        try:
            if "image_grid_thw" in inputs and isinstance(inputs["image_grid_thw"], torch.Tensor):
                g = inputs["image_grid_thw"]
                if g.dim() == 1:
                    if g.shape[0] == 2:
                        g = torch.tensor([g[0].item(), g[1].item(), g[1].item()], dtype=g.dtype)
                    if g.shape[0] == 3:
                        g = g.unsqueeze(0)
                    inputs["image_grid_thw"] = g
                elif g.dim() == 2 and g.shape[1] == 2:
                    h_col = g[:, 1].unsqueeze(-1)
                    g = torch.cat([g, h_col], dim=1)
                    inputs["image_grid_thw"] = g
        except Exception:
            pass

        out_item = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate physics-aware KV prefix if CNN feature injection is enabled
        pkv = None
        if getattr(self._script_cfg, 'use_cnn_feature_prefix', False) and self._cnn_ready and self._prefix_encoder is not None:
            try:
                cnn_tensor = self._to_cnn_tensor(data['image'])
                train_prefix = getattr(self._script_cfg, 'train_prefix_encoder', False)
                if train_prefix:
                    pkv = self._build_prefix_kv(base_name, data, cnn_tensor, doc_content, requires_grad=True)
                    out_item['_cnn_tensor'] = cnn_tensor
                    out_item['_md_text'] = doc_content
                    out_item['_base_name'] = base_name
                else:
                    pkv = self._build_prefix_kv(base_name, data, cnn_tensor, doc_content, requires_grad=False)
            except Exception:
                pkv = None

        if pkv is not None:
            out_item['past_key_values'] = pkv
        
        if self._script_cfg and hasattr(self._script_cfg, 'label_folder'):
            label_info = load_label_file(npy_path, self._script_cfg.label_folder)
            if label_info:
                out_item['label_info'] = label_info
            else:
                out_item['label_info'] = None
        else:
            out_item['label_info'] = None
        
        return out_item


# ---------------------------------------------------------------------------
# Load Qwen3-VL-8B with QLoRA PEFT.
# Uses unsloth for efficient 4-bit quantized loading and LoRA application.
# ---------------------------------------------------------------------------
def load_model(config: ScriptConfig):
    extra_kwargs = {}
    if getattr(config, "use_flash_attn2", False):
        try:
            import flash_attn  # noqa: F401
            extra_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using FlashAttention 2.")
        except Exception:
            print("FlashAttention 2 not available. Falling back to default attention.")

    model, processor = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
        token=os.getenv("HF_TOKEN"),
        trust_remote_code=True,
        cache_dir=config.cache_dir,
        **extra_kwargs,
    )
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.lora_target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        max_seq_length=config.max_length,
    )
    def _pick_base_transformer_config(m):
        cand = []
        try:
            cand.append(getattr(getattr(getattr(m, 'base_model', None), 'model', None), 'config', None))
        except Exception:
            pass
        try:
            cand.append(getattr(getattr(m, 'base_model', None), 'config', None))
        except Exception:
            pass
        try:
            cand.append(getattr(getattr(m, 'model', None), 'config', None))
        except Exception:
            pass
        try:
            cand.append(getattr(m, 'config', None))
        except Exception:
            pass
        def _has_fields(cfg):
            if cfg is None:
                return False
            has_hid = hasattr(cfg, 'hidden_size') or hasattr(cfg, 'n_embd')
            has_heads = hasattr(cfg, 'num_attention_heads') or hasattr(cfg, 'n_head')
            has_layers = hasattr(cfg, 'num_hidden_layers') or hasattr(cfg, 'n_layer')
            return bool(has_hid and has_heads and has_layers)
        for cfg in cand:
            if _has_fields(cfg):
                return cfg
        for cfg in cand:
            if cfg is not None:
                return cfg
        return None

    try:
        base_cfg = _pick_base_transformer_config(model)
        def _has_req(cfg):
            if cfg is None:
                return False
            return (
                hasattr(cfg, 'hidden_size') or hasattr(cfg, 'n_embd')
            ) and (
                hasattr(cfg, 'num_attention_heads') or hasattr(cfg, 'n_head')
            ) and (
                hasattr(cfg, 'num_hidden_layers') or hasattr(cfg, 'n_layer')
            )
        if not _has_req(base_cfg):
            try:
                auto_cfg = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True, cache_dir=config.cache_dir)
                base_cfg = auto_cfg
                print("[prefix] Using AutoConfig.from_pretrained for LLM config.")
            except Exception as e:
                print(f"[prefix] AutoConfig fallback failed: {e}")
        processor._llm_config = base_cfg if base_cfg is not None else getattr(model, 'config', None)
        processor._llm_model_name = config.model_name
    except Exception:
        processor._llm_config = getattr(model, 'config', None)
    return model, processor


def _extract_json_blocks(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    fence = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fence:
        return fence[0]
    start = text.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0) if m else ""


def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def _angular_distance_deg(lat1, lon1, lat2, lon2) -> float:
    r = math.pi / 180.0
    phi1, phi2 = lat1 * r, lat2 * r
    dphi = (lat2 - lat1) * r
    dl = (lon2 - lon1) * r
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return c * (180.0 / math.pi)


# Hungarian algorithm-based matching for multi-TC localization evaluation.
# Computes mean minimum angular distance between predicted and ground-truth TC centers.
def _mean_min_distance_deg(gt_list, pred_list) -> float:
    from scipy.optimize import linear_sum_assignment

    if not gt_list or not pred_list:
        return 0.0

    n_gt, n_pred = len(gt_list), len(pred_list)
    dist_matrix = np.zeros((n_gt, n_pred), dtype=float)
    for i, g in enumerate(gt_list):
        glat, glon = float(g.get('lat', 0.0)), float(g.get('lon', 0.0))
        for j, p in enumerate(pred_list):
            plat, plon = float(p.get('lat', 0.0)), float(p.get('lon', 0.0))
            dist_matrix[i, j] = _angular_distance_deg(glat, glon, plat, plon)
    
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    matched_dists = [dist_matrix[r, c] for r, c in zip(row_ind, col_ind)]
    return float(sum(matched_dists) / max(1, len(matched_dists)))


def main():
    cfg = ScriptConfig()
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    try:
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    print("Loading model...")
    model, processor = load_model(cfg)
    try:
        processor._script_cfg = cfg
    except Exception:
        pass

    print("Preparing dataset...")
    all_files = sorted(glob(os.path.join(cfg.data_folder, '*_image.npy')))
    if len(all_files) == 0:
        raise RuntimeError(f"No image files found in {cfg.data_folder}. Expected format: YYYYMMDD_HHMM_BASIN_image.npy")
    split_idx = int(len(all_files) * cfg.train_split)
    train_files, eval_files = all_files[:split_idx], all_files[split_idx:]
    print(f"Train: {len(train_files)} | Val: {len(eval_files)}")

    train_dataset = CycloneDatasetFast(
        train_files, cfg.docs_folder, processor, cfg.max_length,
        gph_folder=cfg.gph_folder, gph_docs_folder=cfg.gph_docs_folder,
        use_gph=cfg.use_gph,
        sst_folder=cfg.sst_folder, sst_docs_folder=cfg.sst_docs_folder,
        use_sst=cfg.use_sst,
        use_cot=cfg.use_cot, cot_instruction=cfg.cot_instruction,
        doc_max_chars=cfg.doc_max_chars,
    )
    eval_dataset = CycloneDatasetFast(
        eval_files, cfg.docs_folder, processor, cfg.max_length,
        gph_folder=cfg.gph_folder, gph_docs_folder=cfg.gph_docs_folder,
        use_gph=cfg.use_gph,
        sst_folder=cfg.sst_folder, sst_docs_folder=cfg.sst_docs_folder,
        use_sst=cfg.use_sst,
        use_cot=cfg.use_cot, cot_instruction=cfg.cot_instruction,
        doc_max_chars=cfg.doc_max_chars,
    )

    if cfg.per_device_train_batch_size != 1:
        print(f"[guard] Forcing per_device_train_batch_size from {cfg.per_device_train_batch_size} to 1 (allow original image size).")
        cfg.gradient_accumulation_steps *= cfg.per_device_train_batch_size
        cfg.per_device_train_batch_size = 1
        print(f"[guard] gradient_accumulation_steps -> {cfg.gradient_accumulation_steps}")

    if cfg.monitor_during_training:
        if getattr(cfg, 'monitor_strategy', 'steps') == 'steps':
            evaluation_strategy = "steps"
            save_strategy = "steps"
            logging_steps = max(50, cfg.logging_steps)
            eval_steps = max(10, cfg.eval_steps)
            save_steps = max(10, cfg.save_steps)
        else:
            evaluation_strategy = "epoch"
            save_strategy = "epoch"
            logging_steps = max(50, cfg.logging_steps)
            eval_steps = 0
            save_steps = 0
    else:
        evaluation_strategy = "no"
        save_strategy = "steps"
        logging_steps = max(50, cfg.logging_steps)
        eval_steps = 0
        save_steps = max(1000, cfg.save_steps)

    load_best_model = (evaluation_strategy != "no")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="tensorboard",
        dataloader_num_workers=getattr(cfg, 'dataloader_num_workers', 0),
        dataloader_pin_memory=getattr(cfg, 'dataloader_pin_memory', False),
        dataloader_persistent_workers=getattr(cfg, 'dataloader_persistent_workers', False),
    )

    def smart_data_collator(features):
        batch: Dict[str, Any] = {}
        keys = set()
        for f in features:
            keys.update(f.keys())
        passthrough_keys = ('_cnn_tensor', '_md_text', '_base_name', 'label_info')
        for k in keys:
            vals = [f[k] for f in features if k in f]
            if len(vals) == 0:
                continue
            if k in passthrough_keys:
                batch[k] = vals
                continue
            if k == 'image_grid_thw' and isinstance(vals[0], torch.Tensor):
                merged_list = []
                total_imgs = 0
                for idx, t in enumerate(vals):
                    g = t
                    orig_shape = tuple(g.shape)
                    if g.dim() == 0:
                        g = g.view(1)
                    if g.dim() == 1:
                        if g.shape[0] == 2:
                            g = torch.tensor([g[0].item(), g[1].item(), g[1].item()], dtype=g.dtype)
                        if g.shape[0] == 3:
                            g = g.unsqueeze(0)
                    if g.dim() == 2 and g.shape[1] == 2:
                        g = torch.cat([g, g[:, 1:2]], dim=1)
                    if g.dim() != 2 or g.shape[1] != 3:
                        continue
                    merged_list.append(g)
                    total_imgs += g.shape[0]
                if len(merged_list) == 0:
                    batch[k] = torch.stack(vals, dim=0)
                else:
                    batch[k] = torch.cat(merged_list, dim=0)
                continue
            if k == 'past_key_values':
                valid_vals = [v for v in vals if v is not None]
                if valid_vals:
                    L = len(valid_vals[0])
                    stacked = []
                    for l in range(L):
                        ks = []
                        vs = []
                        for s in range(len(valid_vals)):
                            k_s, v_s = valid_vals[s][l]
                            ks.append(k_s)
                            vs.append(v_s)
                        k_b = torch.cat(ks, dim=0)
                        v_b = torch.cat(vs, dim=0)
                        stacked.append((k_b, v_b))
                    batch[k] = tuple(stacked)
                continue
            if isinstance(vals[0], torch.Tensor):
                shapes = [tuple(v.shape) for v in vals]
                if all(s == shapes[0] for s in shapes):
                    batch[k] = torch.stack(vals, dim=0)
                else:
                    max_dims = [max(s[d] for s in shapes) for d in range(len(shapes[0]))]
                    pad_value = -100 if k == "labels" else 0
                    out = vals[0].new_full((len(vals),) + tuple(max_dims), pad_value)
                    for i, v in enumerate(vals):
                        slices = (i,)
                        for d in range(v.dim()):
                            slices += (slice(0, v.shape[d]),)
                        out[slices] = v
                    batch[k] = out
            else:
                batch[k] = vals
        return batch

    def preprocess_logits_for_metrics(logits, labels):
        import transformers
        
        if isinstance(logits, tuple):
            for item in logits:
                if isinstance(item, (transformers.cache_utils.DynamicCache, 
                                    transformers.cache_utils.StaticCache,
                                    transformers.cache_utils.Cache)):
                    continue
                if isinstance(item, torch.Tensor) and item.dim() >= 2:
                    logits = item
                    break
            else:
                if len(logits) > 0:
                    first_item = logits[0]
                    if not isinstance(first_item, (transformers.cache_utils.DynamicCache,
                                                  transformers.cache_utils.StaticCache,
                                                  transformers.cache_utils.Cache)):
                        logits = first_item
                    else:
                        if len(logits) > 1:
                            logits = logits[1]
        elif isinstance(logits, dict):
            logits_dict = logits
            logits = logits_dict.get('logits', None)
            if logits is None:
                for v in logits_dict.values():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        logits = v
                        break
        if not isinstance(logits, torch.Tensor):
            if hasattr(logits, 'logits'):
                logits = logits.logits
            elif hasattr(logits, '__getitem__'):
                try:
                    if isinstance(logits, (list, tuple)):
                        logits = logits[0]
                except Exception:
                    pass
            if not isinstance(logits, torch.Tensor):
                raise ValueError(f"Unable to extract logits tensor from {type(logits)}. Value: {logits}")
        return torch.argmax(logits, dim=-1)

    tok = processor.tokenizer

    # Compute evaluation metrics: TC count MAE and position MAE (degrees).
    # Uses Hungarian algorithm for optimal pred→gt matching in multi-TC cases.
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        B = labels.shape[0]
        cur_cnt_abs = []
        cur_pos_deg = []
        for i in range(B):
            label_ids = labels[i]
            pred_ids = preds[i]
            mask = (label_ids != -100)
            if not np.any(mask):
                continue
            gt_text = tok.decode(label_ids[mask].tolist(), skip_special_tokens=True)
            pr_text = tok.decode(pred_ids[mask].tolist(), skip_special_tokens=True)
            gt_json = _safe_json_loads(_extract_json_blocks(gt_text)) or {}
            pr_json = _safe_json_loads(_extract_json_blocks(pr_text)) or {}
            gt_cur = int(gt_json.get('current_tc_count', 0)) if isinstance(gt_json.get('current_tc_count', 0), (int, float)) else 0
            pr_cur = int(pr_json.get('current_tc_count', 0)) if isinstance(pr_json.get('current_tc_count', 0), (int, float)) else 0
            cur_cnt_abs.append(abs(pr_cur - gt_cur))
            gt_cur_list = gt_json.get('current_tcs', []) or []
            pr_cur_list = pr_json.get('current_tcs', []) or []
            try:
                cur_pos_deg.append(_mean_min_distance_deg(gt_cur_list, pr_cur_list))
            except Exception:
                cur_pos_deg.append(0.0)
        def _mean(x):
            return float(sum(x) / max(1, len(x))) if x else 0.0
        return {
            "cur_count_mae": _mean(cur_cnt_abs),
            "cur_pos_mae_deg": _mean(cur_pos_deg),
        }

    callbacks = []
    if evaluation_strategy != "no" and getattr(cfg, 'enable_early_stopping', True):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience,
                                               early_stopping_threshold=cfg.early_stopping_threshold))

    # NumericFocusTrainer: custom Trainer with robust logit extraction
    # for evaluation, handling various output formats (tuple/dict/cache).
    class NumericFocusTrainer(Trainer):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            has_labels = "labels" in inputs
            inputs = self._prepare_inputs(inputs)
            
            eval_inputs = {k: v for k, v in inputs.items()}
            eval_inputs['use_cache'] = False
            
            original_cache_impl = None
            try:
                if hasattr(model, 'config') and hasattr(model.config, 'cache_implementation'):
                    original_cache_impl = getattr(model.config, 'cache_implementation', None)
                    try:
                        delattr(model.config, 'cache_implementation')
                    except (AttributeError, TypeError):
                        setattr(model.config, 'cache_implementation', None)
            except Exception:
                pass
            
            with torch.no_grad():
                try:
                    outputs = model(**eval_inputs)
                finally:
                    if original_cache_impl is not None:
                        try:
                            setattr(model.config, 'cache_implementation', original_cache_impl)
                        except Exception:
                            pass
                
                logits = None
                loss = None
                
                if isinstance(outputs, tuple):
                    for item in outputs:
                        if isinstance(item, torch.Tensor) and item.dim() >= 2:
                            logits = item
                            break
                        elif isinstance(item, torch.Tensor) and item.dim() == 1:
                            if loss is None:
                                loss = item
                    
                    if logits is None and len(outputs) > 0:
                        first_item = outputs[0]
                        if isinstance(first_item, torch.Tensor) and first_item.dim() >= 2:
                            logits = first_item
                        elif hasattr(first_item, 'logits'):
                            logits = first_item.logits
                        if hasattr(first_item, 'loss') and loss is None:
                            loss = first_item.loss
                            
                elif isinstance(outputs, dict):
                    logits = outputs.get("logits", None)
                    loss = outputs.get("loss", None)
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    loss = getattr(outputs, 'loss', None)
                
                if logits is None:
                    return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
                
                if has_labels and loss is None:
                    loss_fct = nn.CrossEntropyLoss()
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs["labels"][..., 1:].contiguous()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                if not isinstance(logits, torch.Tensor):
                    if hasattr(logits, 'logits'):
                        logits = logits.logits
                    else:
                        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
                
                return (loss, logits, None) if has_labels else (logits, None)

    TrainerCls = NumericFocusTrainer

    trainer = TrainerCls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if evaluation_strategy != "no" else None,
        data_collator=smart_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
        processing_class=processor,
    )
    
    # ---------------------------------------------------------------
    # Cooperative Training Setup
    # When physics-aware prefix injection is enabled with trainable prefix
    # encoder, we create a cooperative optimizer that jointly updates:
    #   (1) LoRA adapter weights (main VLM adaptation)
    #   (2) Prefix encoder weights (physics-aware feature mapping)
    # with separate learning rates and gradient clipping.
    # ---------------------------------------------------------------
    prefix_encoder = None
    if getattr(cfg, 'use_cnn_feature_prefix', False) and getattr(cfg, 'train_prefix_encoder', False):
        if hasattr(train_dataset, '_prefix_encoder'):
            prefix_encoder = train_dataset._prefix_encoder
        
        if prefix_encoder is not None:
            print(f"[CooperativeTraining] Prefix encoder detected, enabling cooperative training")
            print(f"  Prefix encoder LR: {getattr(cfg, 'prefix_encoder_lr', 1e-4)}")
            
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
                        'weight_decay': 0.01,
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
                        'weight_decay': 0.01,
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
                import os
                callback_path = os.path.join(os.path.dirname(__file__), 'prefix_generation_callback.py')
                if os.path.exists(callback_path):
                    from prefix_generation_callback import PrefixGenerationCallback
                    cnn_model = getattr(train_dataset, '_cnn_model', None)
                    cnn_cfg = getattr(train_dataset, '_cnn_cfg', None)
                    if cnn_model is not None and cnn_cfg is not None:
                        try:
                            device = next(model.parameters()).device
                            device = str(device) if device else "cuda"
                        except:
                            device = "cuda"
                        
                        prefix_callback = PrefixGenerationCallback(
                            prefix_encoder=prefix_encoder,
                            cnn_model=cnn_model,
                            cnn_cfg=cnn_cfg,
                            device=device,
                        )
                        trainer.add_callback(prefix_callback)
                        print(f"[CooperativeTraining] Prefix KV generation callback added")
            except Exception as e:
                print(f"[CooperativeTraining] Warning: Failed to add Prefix KV generation callback: {e}")
                print(f"[CooperativeTraining] Will use Prefix KV generated during data loading")

    print("Starting fine-tuning (accelerated)...")
    trainer.train()

    print("Training complete. Saving final model...")
    model_short_name = cfg.model_name.split('/')[-1]
    finetune_method = "QLoRA-fast"
    cot_tag = "on" if getattr(cfg, "use_cot", False) else "off"
    cnn_pref_tag = "cnnprefix-on" if getattr(cfg, 'use_cnn_feature_prefix', False) else "cnnprefix-off"
    gph_tag = "gph-on" if getattr(cfg, 'use_gph', False) else "gph-off"
    sst_tag = "sst-on" if getattr(cfg, 'use_sst', False) else "sst-off"
    extra_bits = []
    if getattr(cfg, 'use_cnn_feature_prefix', False):
        extra_bits.append(f"pfx{getattr(cfg,'prefix_len',128)}")
    extra_suffix = ("_" + "_".join(extra_bits)) if extra_bits else ""
    model_folder_name = f"{model_short_name}_{finetune_method}_cot-{cot_tag}_{cnn_pref_tag}_{gph_tag}_{sst_tag}{extra_suffix}"
    final_path = os.path.join(cfg.output_dir, model_folder_name)
    os.makedirs(final_path, exist_ok=True)
    trainer.save_model(final_path)
    try:
        processor.save_pretrained(final_path)
    except Exception:
        pass
    print(f"Model saved to: {final_path}")


if __name__ == "__main__":
    main()
