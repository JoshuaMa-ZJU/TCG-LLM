# =============================================================================
# cnn_encoders.py — Physics-aware CNN Encoders + Transformer JSON Decoder
# (TCG-LLM)
#
# This module implements the physics-motivated visual encoders that extract
# TC-specific physically meaningful features from satellite imagery, GPH, and
# SST data.  The encoder outputs are fused via cross-attention and a Fusion
# Transformer, then decoded into structured JSON by an autoregressive decoder.
# The fused feature vector (768-dim = 3×256) is also exported for prefix
# injection into the VLM during physics-aware fine-tuning.
# =============================================================================

import os
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob


# ---------------------------------------------------------------------------
# Output vocabulary for the JSON Decoder.  The decoder generates JSON text
# character-by-character from a compact vocabulary covering digits, brackets,
# punctuation, and lowercase letters needed for the output format:
#   {"current_tc_count": int, "current_tcs": [{"lat": float, "lon": float}]}
# ---------------------------------------------------------------------------
def _build_output_vocab():
    special = ['<pad>', '<bos>', '<eos>']
    chars = sorted(set(
        list(' ",-.:0123456789[]_abcdefghijklmnopqrstuvwxyz{}')
    ))
    vocab = special + chars
    return vocab, {c: i for i, c in enumerate(vocab)}


OUTPUT_VOCAB, OUTPUT_C2I = _build_output_vocab()
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


def encode_json_string(s: str) -> List[int]:
    ids = [BOS_ID]
    for ch in s:
        ids.append(OUTPUT_C2I.get(ch, OUTPUT_C2I.get(' ', PAD_ID)))
    ids.append(EOS_ID)
    return ids


def decode_token_ids(ids: List[int]) -> str:
    chars = []
    for i in ids:
        if i == BOS_ID:
            continue
        if i == EOS_ID:
            break
        if i == PAD_ID:
            continue
        if 0 <= i < len(OUTPUT_VOCAB):
            chars.append(OUTPUT_VOCAB[i])
    return ''.join(chars)


def make_target_json(tc_count: int, tc_positions, max_current: int = 8) -> str:
    tcs = []
    if isinstance(tc_positions, np.ndarray):
        tc_positions = tc_positions.tolist()
    for pos in (tc_positions or [])[:max_current]:
        if isinstance(pos, dict):
            lat = round(float(pos.get('lat', 0)), 2)
            lon = round(float(pos.get('lon', 0)), 2)
        elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
            lat = round(float(pos[0]), 2)
            lon = round(float(pos[1]), 2)
        else:
            continue
        tcs.append({"lat": lat, "lon": lon})
    result = {"current_tc_count": int(tc_count), "current_tcs": tcs}
    return json.dumps(result, separators=(', ', ': '), ensure_ascii=True)


# ---------------------------------------------------------------------------
# FusionConfig — Hyperparameters for standalone CNN encoder training.
# In the ablation study ("CNN encoders + Transformer"), these
# encoders are trained independently with batch_size=32, lr=5e-4, 100 epochs
# and early stopping.  The resulting weights serve as initialization for
# subsequent physics-aware SFT / GRPO fine-tuning.
# ---------------------------------------------------------------------------
@dataclass
class FusionConfig:
    data_folder: str = "/root/autodl-tmp/TCDLD/image"
    docs_folder: str = "/root/autodl-tmp/TCDLD/image_docs"
    gph_folder: str = "/root/autodl-tmp/TCDLD/gph"
    gph_docs_folder: str = "/root/autodl-tmp/TCDLD/gph_docs"
    sst_folder: str = "/root/autodl-tmp/TCDLD/sst"
    sst_docs_folder: str = "/root/autodl-tmp/TCDLD/sst_docs"
    label_folder: str = "/root/autodl-tmp/TCDLD/label"
    output_dir: str = "/root/autodl-tmp/TCDLD/cnn_encoders"

    use_gph: bool = True
    use_sst: bool = True
    gph_channels: int = 6

    image_channels: int = 1

    d_model: int = 256
    n_heads: int = 8
    dropout: float = 0.1
    embedding_dim: int = 256
    transformer_layers: int = 2
    fusion_transformer_layers: int = 2
    decoder_layers: int = 3
    decoder_heads: int = 4
    max_output_len: int = 512
    max_current: int = 8

    max_doc_chars: int = 1500
    vocab_extra_chars: str = "#*_{}[]()<>/\\-:+.,;\n \t"

    seed: int = 42
    train_split: float = 0.8
    batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    num_epochs: int = 100
    lr: float = 5e-4
    warmup_steps: int = 500
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    num_workers: int = 4
    pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    monitor_during_training: bool = True
    monitor_strategy: str = "steps"
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3
    enable_early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    fp16: bool = False
    bf16: bool = True

    label_smoothing: float = 0.0


# ---------------------------------------------------------------------------
# Gradient-aware feature enhancement module.
# All three visual encoders share this common preprocessing step, which
# computes Sobel gradient maps (horizontal/vertical) and a Laplacian map
# to highlight edges and anomalies—useful for detecting TC eye-walls,
# rainband boundaries, and GPH/SST spatial gradients.
# Output: original channels + [grad_x, grad_y, gradient_magnitude, laplacian]
# ---------------------------------------------------------------------------
def _compute_gradient_channels(x: torch.Tensor) -> torch.Tensor:
    ch0 = x[:, 0:1]  # use first channel for gradient computation
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    lap_k   = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx  = F.conv2d(ch0, sobel_x, padding=1)   # horizontal gradient
    gy  = F.conv2d(ch0, sobel_y, padding=1)   # vertical gradient
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)  # gradient magnitude
    lap = F.conv2d(ch0, lap_k, padding=1)     # Laplacian (edge/anomaly)
    return torch.cat([x, gx, gy, mag, lap], dim=1)


# ---------------------------------------------------------------------------
# Channel Attention & Spatial Attention (CBAM)
# Incorporated in all visual encoders to focus on channels and spatial regions
# most relevant to TCG detection (e.g., the TC eye, low-GPH center, SST cold core).
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.mlp = nn.Sequential(nn.Linear(ch, mid, bias=False), nn.ReLU(True), nn.Linear(mid, ch, bias=False))

    def forward(self, x):
        b, c, _, _ = x.size()
        att = torch.sigmoid(self.mlp(x.mean(dim=[2, 3])) + self.mlp(x.amax(dim=[2, 3]))).view(b, c, 1, 1)
        return x * att


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        att = torch.sigmoid(self.conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * att


class CBAM(nn.Module):
    def __init__(self, ch, reduction=8, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(ch, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.sa(self.ca(x))


# Squeeze-and-Excitation block — channel attention used in the SST Encoder
class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch // reduction, 4)), nn.ReLU(True),
            nn.Linear(max(ch // reduction, 4), ch), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x.mean(dim=[2, 3])).view(x.size(0), -1, 1, 1)


# =====================================================================
# Image Encoder — Satellite imagery encoder.
# Adopts multi-scale convolutions (3×3, 5×5, 7×7) to capture both
# small-scale eye features and large-scale rainband patterns, thereby
# leveraging both global context and fine-grained details.
# CBAM is applied to focus on spatially and channel-wise relevant regions.
# Output: d_model-dimensional feature vector.
# =====================================================================
class ImageEncoder(nn.Module):
    def __init__(self, in_ch=1, d_model=256):
        super().__init__()
        stem_in = in_ch + 4  # original channel + 4 gradient channels
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(3, 2, 1),
        )
        # Multi-scale convolution branches (3×3, 5×5, 7×7)
        self.b3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.b5 = nn.Sequential(nn.Conv2d(64, 64, 5, padding=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.b7 = nn.Sequential(nn.Conv2d(64, 64, 7, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.ms_fuse = nn.Sequential(nn.Conv2d(192, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))  # fuse 3 branches
        self.deep = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.cbam = CBAM(256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, d_model)

    def forward(self, x):
        x = _compute_gradient_channels(x)  # append gradient channels
        x = self.stem(x)  # downsample with 3×3 stem conv
        x = self.ms_fuse(torch.cat([self.b3(x), self.b5(x), self.b7(x)], 1))  # multi-scale fusion
        x = self.cbam(self.deep(x))  # deep convs + CBAM attention
        return self.fc(self.pool(x).flatten(1))  # global pool → d_model vector


# =====================================================================
# GPH Encoder — Geopotential Height encoder.
# Processes GPH data at 6 isobaric levels (200, 300, 500, 700, 850, 1000 hPa).
# Uses 3D convolutions to capture the baroclinic instability and vertical wind
# shear characteristic of TCG, followed by a multi-level self-attention fusion
# module to capture both intra-level and inter-level information.
# Output: d_model-dimensional feature vector.
# =====================================================================
class GPHEncoder(nn.Module):
    def __init__(self, n_levels=6, d_model=256):
        super().__init__()
        self.n_levels = n_levels
        # Per-level 2D CNN to extract spatial features at each pressure level
        self.level_convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
        ) for _ in range(n_levels)])
        self.level_sa = nn.ModuleList([SpatialAttention(7) for _ in range(n_levels)])  # per-level spatial attention
        # 3D convolutions to capture vertical structure across pressure levels
        self.conv3d = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1, bias=False), nn.BatchNorm3d(32), nn.ReLU(True),
            nn.Conv3d(32, 64, 3, padding=1, bias=False), nn.BatchNorm3d(64), nn.ReLU(True),
        )
        # Multi-level self-attention to fuse intra-level and inter-level information
        self.self_attn = nn.MultiheadAttention(64, 4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(64)
        self.fc = nn.Sequential(nn.Linear(64, d_model), nn.ReLU(True))

    def forward(self, x):
        feats = []
        for i in range(self.n_levels):
            lv = _compute_gradient_channels(x[:, i:i+1])  # gradient enhancement per level
            feats.append(self.level_sa[i](self.level_convs[i](lv)))  # 2D CNN + spatial attention
        stk = torch.stack(feats, dim=2)  # [B, C, n_levels, H, W]
        x3 = self.conv3d(stk).mean(dim=[3, 4]).permute(0, 2, 1)  # 3D conv → pool spatial → [B, n_levels, C]
        att, _ = self.self_attn(x3, x3, x3)  # self-attention across pressure levels
        return self.fc(self.norm(x3 + att).mean(dim=1))  # mean-pool levels → d_model vector


# =====================================================================
# Cold-Core Detector — Sub-module of the SST Encoder.
# Motivated by the SST cooling near TC centers (cold wakes) induced by
# oceanic upwelling under strong winds, this module explicitly computes
# local minima (background_avg – raw) to capture thermodynamic signatures
# of ocean-atmosphere interaction and identify potential TC centers.
# =====================================================================
class ColdCoreDetector(nn.Module):
    def __init__(self, pool_size=15):
        super().__init__()
        self.bg_pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2)  # local background average
        self.attn_conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 1, 1), nn.Sigmoid(),  # attention map highlighting cold-core regions
        )

    def forward(self, sst_raw, feat):
        cold = self.bg_pool(sst_raw) - sst_raw  # cold anomaly = background – actual (positive where colder)
        cold_rs = F.interpolate(cold, feat.shape[2:], mode='bilinear', align_corners=False)
        sst_rs  = F.interpolate(sst_raw, feat.shape[2:], mode='bilinear', align_corners=False)
        return feat * self.attn_conv(torch.cat([cold_rs, sst_rs], 1))  # modulate features by cold-core attention


# =====================================================================
# SST Encoder — Sea Surface Temperature encoder.
# Introduces a ColdCoreDetector to explicitly capture SST cold wakes
# near TC centers.  SE (Squeeze-and-Excitation) block focuses on the
# most informative channels for thermodynamic feature extraction.
# Output: d_model-dimensional feature vector.
# =====================================================================
class SSTEncoder(nn.Module):
    def __init__(self, in_ch=1, d_model=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch + 4, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.cold = ColdCoreDetector(15)
        self.conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.se = SEBlock(256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, d_model)

    def forward(self, x):
        sst_raw = x  # retain raw SST for cold-core computation
        feat = self.stem(_compute_gradient_channels(x))  # gradient channels + stem conv
        feat = self.conv(self.cold(sst_raw, feat))  # cold-core modulated deep features
        return self.fc(self.pool(self.se(feat)).flatten(1))  # SE attention → global pool → d_model


# Sinusoidal positional encoding for Transformer-based modules
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=2000, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


# =====================================================================
# Text Encoder — Extracts semantic information from the
# textual statistical descriptors of satellite image / GPH / SST data.
# Uses a Transformer encoder to capture dependencies within the descriptors.
# =====================================================================
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, d_model, num_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim, dropout=dropout)
        self.proj = nn.Linear(emb_dim, d_model) if emb_dim != d_model else nn.Identity()
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.drop = nn.Dropout(dropout)

    def forward(self, ids, mask):
        x = self.proj(self.pos(self.emb(ids)))
        return self.drop(self.encoder(x, src_key_padding_mask=~mask))


# =====================================================================
# Symmetric Cross-Attention Module — Image-text alignment.
# Text-to-image attention encourages the model to focus on visual regions
# most relevant to the textual statistical description; bidirectional
# information flow enables multimodal feature alignment and fusion.
# =====================================================================
class CrossAttentionModule(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.n1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_model * 4, d_model), nn.Dropout(dropout))
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, vis, text, text_mask=None):
        q = vis.unsqueeze(1)  # visual feature as query [B, 1, d_model]
        kpm = ~text_mask if text_mask is not None else None
        a, _ = self.attn(q, text, text, key_padding_mask=kpm)  # cross-attend to text
        q = self.n1(q + a)  # residual connection + LayerNorm
        return self.n2(q + self.ffn(q)).squeeze(1)  # FFN + residual → [B, d_model]


# =====================================================================
# Fusion Transformer — Multi-modal feature fusion.
# Stacks the three encoder outputs (image, GPH, SST) as a 3-token
# sequence with learnable positional embeddings, then processes them
# through a Transformer encoder to capture inter-modal interactions.
# Returns: (fused_vec [B, 3*d_model], memory [B, 3, d_model])
#   - fused_vec (768-dim) is exported to prefix_injector for VLM injection.
#   - memory is used by the JSON Decoder for auto-regressive generation.
# =====================================================================
class FusionTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, f_img, f_gph, f_sst):
        tokens = torch.stack([f_img, f_gph, f_sst], dim=1) + self.pos_emb  # [B, 3, d_model]
        out = self.norm(self.encoder(tokens))  # Transformer encoder
        return out.reshape(out.size(0), -1), out  # (fused_vec, memory)


# =====================================================================
# JSON Decoder — Autoregressive Transformer decoder that generates the
# structured JSON output character-by-character.  The decoder attends
# to the fused multi-modal memory from the Fusion Transformer.
# Output format: {"current_tc_count": N, "current_tcs": [{"lat": ..., "lon": ...}]}
# =====================================================================
class JSONDecoder(nn.Module):
    def __init__(self, d_model=256, out_vocab_size=50, max_len=512, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(out_vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True, activation='gelu')
        self.decoder = nn.TransformerDecoder(dec_layer, n_layers)
        self.out_proj = nn.Linear(d_model, out_vocab_size)
        self.scale = math.sqrt(d_model)

    def forward(self, tgt_ids: torch.Tensor, memory: torch.Tensor,
                tgt_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = tgt_ids.shape
        pos = torch.arange(T, device=tgt_ids.device).unsqueeze(0)
        tgt = self.tok_emb(tgt_ids) * self.scale + self.pos_emb(pos)
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=tgt_ids.device)  # causal mask
        out = self.decoder(tgt, memory, tgt_mask=causal, tgt_key_padding_mask=tgt_padding_mask)
        return self.out_proj(out)  # logits over output vocabulary

    @torch.no_grad()
    def generate(self, memory: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        if max_len is None:
            max_len = self.max_len
        B = memory.size(0)
        dev = memory.device
        ids = torch.full((B, 1), BOS_ID, dtype=torch.long, device=dev)
        finished = torch.zeros(B, dtype=torch.bool, device=dev)

        for _ in range(max_len - 1):
            logits = self.forward(ids, memory)
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            finished = finished | (nxt.squeeze(-1) == EOS_ID)
            if finished.all():
                break
        return ids


# =====================================================================
# CycloneFusionModel — Complete multi-modal fusion model.
# Integrates: ImageEncoder + GPHEncoder + SSTEncoder + TextEncoders
#           + CrossAttention (image-text alignment) + FusionTransformer
#           + JSONDecoder.
# During SFT / GRPO training, only the fused_vec (768-dim) is exported
# to the FeaturePrefixEncoder (prefix_injector.py) for physics-aware
# fine-tuning; the JSONDecoder is used in standalone CNN-only ablation.
# =====================================================================
class CycloneFusionModel(nn.Module):
    def __init__(self, cfg: FusionConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.use_gph = cfg.use_gph
        self.use_sst = cfg.use_sst

        self.image_enc = ImageEncoder(in_ch=cfg.image_channels, d_model=cfg.d_model)
        if cfg.use_gph:
            self.gph_enc = GPHEncoder(n_levels=cfg.gph_channels, d_model=cfg.d_model)
        if cfg.use_sst:
            self.sst_enc = SSTEncoder(in_ch=1, d_model=cfg.d_model)

        self.text_enc = TextEncoder(vocab_size, cfg.embedding_dim, cfg.d_model,
                                    num_layers=cfg.transformer_layers,
                                    n_heads=min(cfg.n_heads, 4), dropout=cfg.dropout)

        self.cross_attn_img = CrossAttentionModule(cfg.d_model, cfg.n_heads, cfg.dropout)
        if cfg.use_gph:
            self.cross_attn_gph = CrossAttentionModule(cfg.d_model, cfg.n_heads, cfg.dropout)
        if cfg.use_sst:
            self.cross_attn_sst = CrossAttentionModule(cfg.d_model, cfg.n_heads, cfg.dropout)

        self.fusion_transformer = FusionTransformer(cfg.d_model, cfg.n_heads,
                                                    cfg.fusion_transformer_layers, cfg.dropout)

        self.json_decoder = JSONDecoder(
            d_model=cfg.d_model,
            out_vocab_size=len(OUTPUT_VOCAB),
            max_len=cfg.max_output_len,
            n_heads=cfg.decoder_heads,
            n_layers=cfg.decoder_layers,
            dropout=cfg.dropout,
        )

    def _encode(self, images, text_ids, text_attn_mask,
                gph_images=None, sst_images=None,
                text_ids_gph=None, text_mask_gph=None,
                text_ids_sst=None, text_mask_sst=None):
        """Encode all modalities and fuse them.
        Returns (fused_vec [B, 3*d_model], memory [B, 3, d_model]).
        fused_vec is the physics-aware feature vector injected into VLM."""
        B, dev = images.size(0), images.device

        img_feat = self.image_enc(images)  # Image Encoder → [B, d_model]
        gph_feat = (self.gph_enc(gph_images) if self.use_gph and gph_images is not None
                     and gph_images.abs().sum() > 1e-6 else None)
        sst_feat = (self.sst_enc(sst_images) if self.use_sst and sst_images is not None
                     and sst_images.abs().sum() > 1e-6 else None)

        # Text encoding for each data modality
        txt_img = self.text_enc(text_ids, text_attn_mask)
        txt_gph = self.text_enc(text_ids_gph, text_mask_gph) if text_ids_gph is not None else txt_img
        mk_gph  = text_mask_gph if text_ids_gph is not None else text_attn_mask
        txt_sst = self.text_enc(text_ids_sst, text_mask_sst) if text_ids_sst is not None else txt_img
        mk_sst  = text_mask_sst if text_ids_sst is not None else text_attn_mask

        # Symmetric cross-attention: image-text alignment
        f_img = self.cross_attn_img(img_feat, txt_img, text_attn_mask)
        f_gph = (self.cross_attn_gph(gph_feat, txt_gph, mk_gph) if gph_feat is not None
                 else torch.zeros(B, self.cfg.d_model, device=dev, dtype=img_feat.dtype))
        f_sst = (self.cross_attn_sst(sst_feat, txt_sst, mk_sst) if sst_feat is not None
                 else torch.zeros(B, self.cfg.d_model, device=dev, dtype=img_feat.dtype))

        # Fusion Transformer: fuse image + GPH + SST features
        fused_vec, memory = self.fusion_transformer(f_img, f_gph, f_sst)
        return fused_vec, memory

    def forward(self, images, text_ids, text_attn_mask,
                gph_images=None, sst_images=None,
                text_ids_gph=None, text_mask_gph=None,
                text_ids_sst=None, text_mask_sst=None,
                tgt_ids=None):
        fused_vec, memory = self._encode(
            images, text_ids, text_attn_mask,
            gph_images, sst_images,
            text_ids_gph, text_mask_gph, text_ids_sst, text_mask_sst,
        )
        result: Dict[str, torch.Tensor] = {'fused_vec': fused_vec}

        if tgt_ids is not None:
            dec_input = tgt_ids[:, :-1]
            dec_pad_mask = (dec_input == PAD_ID)
            result['logits'] = self.json_decoder(dec_input, memory, tgt_padding_mask=dec_pad_mask)

        return result

    @torch.no_grad()
    def generate(self, images, text_ids, text_attn_mask,
                 gph_images=None, sst_images=None,
                 text_ids_gph=None, text_mask_gph=None,
                 text_ids_sst=None, text_mask_sst=None,
                 max_len=None):
        fused_vec, memory = self._encode(
            images, text_ids, text_attn_mask,
            gph_images, sst_images,
            text_ids_gph, text_mask_gph, text_ids_sst, text_mask_sst,
        )
        gen_ids = self.json_decoder.generate(memory, max_len)
        return fused_vec, gen_ids

    def generate_json(self, images, text_ids, text_attn_mask, **kw) -> List[dict]:
        _, gen_ids = self.generate(images, text_ids, text_attn_mask, **kw)
        results = []
        for b in range(gen_ids.size(0)):
            text = decode_token_ids(gen_ids[b].tolist())
            try:
                results.append(json.loads(text))
            except Exception:
                results.append({"current_tc_count": 0, "current_tcs": []})
        return results


def _build_input_vocab(cfg: FusionConfig):
    base = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    extra = list(cfg.vocab_extra_chars)
    chars = sorted(set(base + extra))
    vocab = ["<pad>", "<unk>"] + chars
    return vocab, {c: i for i, c in enumerate(vocab)}


# ---------------------------------------------------------------------------
# CycloneFusionDataset — Dataset class for the TCDLD benchmark.
# Each sample contains: satellite image, GPH (6-level), SST, textual
# descriptors, and ground-truth labels (tc_count + tc_positions).
# ---------------------------------------------------------------------------
class CycloneFusionDataset(Dataset):
    def __init__(self, image_files: List[str], cfg: FusionConfig):
        self.files = image_files
        self.cfg = cfg
        self.vocab, self.char2idx = _build_input_vocab(cfg)

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _core(p):
        return os.path.splitext(os.path.basename(p))[0].replace('_image', '')

    def _p(self, folder, core, suffix):
        return os.path.join(folder, f"{core}{suffix}")

    def _load_image(self, path):
        d = np.load(path, allow_pickle=True)
        if isinstance(d, np.ndarray) and d.ndim == 0:
            d = d.item()
        img = d.get('image', list(d.values())[0]) if isinstance(d, dict) else d
        img = img.astype(np.float32)
        r = img.max() - img.min()
        if r > 0:
            img = (img - img.min()) / r
        return img[np.newaxis] if img.ndim == 2 else img

    def _load_gph(self, path):
        if not os.path.exists(path):
            return None
        d = np.load(path, allow_pickle=True)
        if isinstance(d, np.ndarray) and d.ndim == 0:
            d = d.item()
        g = d if isinstance(d, np.ndarray) else np.array(d, dtype=np.float32)
        g = g.astype(np.float32)
        for i in range(g.shape[0]):
            r = g[i].max() - g[i].min()
            if r > 0:
                g[i] = (g[i] - g[i].min()) / r
        return g

    def _load_sst(self, path):
        if not os.path.exists(path):
            return None
        d = np.load(path, allow_pickle=True)
        if isinstance(d, np.ndarray) and d.ndim == 0:
            d = d.item()
        s = d if isinstance(d, np.ndarray) else np.array(d, dtype=np.float32)
        s = s.astype(np.float32)
        r = s.max() - s.min()
        if r > 0:
            s = (s - s.min()) / r
        return s[np.newaxis] if s.ndim == 2 else s

    def _load_label(self, path):
        if not os.path.exists(path):
            return {'tc_count': 0, 'tc_positions': []}
        d = np.load(path, allow_pickle=True)
        if isinstance(d, np.ndarray) and d.ndim == 0:
            d = d.item()
        return d if isinstance(d, dict) else {'tc_count': 0, 'tc_positions': []}

    def _load_doc(self, path):
        if not os.path.exists(path):
            return ""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()[:self.cfg.max_doc_chars]
        except Exception:
            return ""

    def _enc_text(self, txt):
        ids = [self.char2idx.get(c, 1) for c in txt[:self.cfg.max_doc_chars]] or [0]
        t = torch.tensor(ids, dtype=torch.long)
        return t, (t != 0)

    def __getitem__(self, idx):
        path = self.files[idx]
        core = self._core(path)

        img = torch.from_numpy(self._load_image(path))
        gph = self._load_gph(self._p(self.cfg.gph_folder, core, '_gph.npy'))
        sst = self._load_sst(self._p(self.cfg.sst_folder, core, '_sst.npy'))
        label = self._load_label(self._p(self.cfg.label_folder, core, '_label.npy'))

        t_img,  m_img  = self._enc_text(self._load_doc(self._p(self.cfg.docs_folder,     core, '.md')))
        t_gph,  m_gph  = self._enc_text(self._load_doc(self._p(self.cfg.gph_docs_folder, core, '_gph.md')))
        t_sst,  m_sst  = self._enc_text(self._load_doc(self._p(self.cfg.sst_docs_folder, core, '_sst.md')))

        tc_count = int(label.get('tc_count', 0))
        tc_pos   = label.get('tc_positions', [])
        tgt_json = make_target_json(tc_count, tc_pos, self.cfg.max_current)
        tgt_ids  = torch.tensor(encode_json_string(tgt_json), dtype=torch.long)

        if isinstance(tc_pos, np.ndarray):
            tc_pos = tc_pos.tolist()
        max_c = self.cfg.max_current
        raw = torch.zeros(max_c, 2)
        mask = torch.zeros(max_c)
        for i, p in enumerate((tc_pos or [])[:max_c]):
            if isinstance(p, dict):
                raw[i] = torch.tensor([float(p.get('lat', 0)), float(p.get('lon', 0))])
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                raw[i] = torch.tensor([float(p[0]), float(p[1])])
            else:
                continue
            mask[i] = 1.0

        sample = {
            'image': img,
            'text_ids': t_img, 'text_attn_mask': m_img,
            'text_ids_gph': t_gph, 'text_mask_gph': m_gph,
            'text_ids_sst': t_sst, 'text_mask_sst': m_sst,
            'target_ids': tgt_ids,
            'current_tc_count': tc_count,
            'current_tcs_raw': raw,
            'current_tcs_mask': mask,
        }
        if gph is not None:
            sample['gph_image'] = torch.from_numpy(gph)
        if sst is not None:
            sample['sst_image'] = torch.from_numpy(sst)
        return sample


# Collate function: pad variable-length text and target sequences.
def collate_fn(batch: List[dict], pad_token: int = 0) -> dict:
    def _pad(tensors, pv=0):
        ml = max(t.size(0) for t in tensors)
        ids  = torch.full((len(tensors), ml), pv, dtype=tensors[0].dtype)
        mask = torch.zeros(len(tensors), ml, dtype=torch.bool)
        for i, t in enumerate(tensors):
            ids[i, :t.size(0)] = t
            mask[i, :t.size(0)] = True
        return ids, mask

    images = torch.stack([s['image'] for s in batch])
    ti, tm = _pad([s['text_ids'] for s in batch], pad_token)
    tg, mg = _pad([s['text_ids_gph'] for s in batch], pad_token)
    ts, ms = _pad([s['text_ids_sst'] for s in batch], pad_token)
    tgt, _ = _pad([s['target_ids'] for s in batch], pad_token)

    r: Dict[str, torch.Tensor] = {
        'images': images,
        'text_ids': ti, 'text_attn_mask': tm,
        'text_ids_gph': tg, 'text_mask_gph': mg,
        'text_ids_sst': ts, 'text_mask_sst': ms,
        'target_ids': tgt,
        'current_tc_count': torch.tensor([s['current_tc_count'] for s in batch], dtype=torch.long),
        'current_tcs_raw':  torch.stack([s['current_tcs_raw'] for s in batch]),
        'current_tcs_mask': torch.stack([s['current_tcs_mask'] for s in batch]),
    }

    for key, field in [('gph_images', 'gph_image'), ('sst_images', 'sst_image')]:
        if any(field in s for s in batch):
            ref = next((s[field] for s in batch if field in s), None)
            lst = [s.get(field, torch.zeros_like(ref) if ref is not None else torch.zeros(1, 1, 1)) for s in batch]
            r[key] = torch.stack(lst)

    return r


# ---------------------------------------------------------------------------
# Cross-entropy loss for the JSON Decoder with token-level accuracy tracking.
# ---------------------------------------------------------------------------
def loss_fn(pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
            cfg: FusionConfig) -> Dict[str, torch.Tensor]:
    logits = pred['logits']
    tgt = batch['target_ids'][:, 1:]
    min_len = min(logits.size(1), tgt.size(1))
    logits = logits[:, :min_len].contiguous()
    tgt    = tgt[:, :min_len].contiguous()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1),
                           ignore_index=PAD_ID, label_smoothing=cfg.label_smoothing)

    with torch.no_grad():
        mask = (tgt != PAD_ID)
        correct = ((logits.argmax(-1) == tgt) & mask).sum()
        total = mask.sum().clamp(min=1)
        acc = correct.float() / total.float()

    return {'loss': loss, 'token_acc': acc.detach()}


# Haversine formula: compute great-circle distance (km) between two lat/lon points.
# Used to evaluate localization accuracy.
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Evaluation loop: computes val_loss, count accuracy/MAE, Haversine distance,
# and JSON validity rate using greedy auto-regressive decoding.
@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    total_loss, n_batch = 0.0, 0
    count_correct, count_total = 0, 0
    count_mae_sum = 0.0
    hav_sum, hav_cnt = 0.0, 0
    json_valid, json_total = 0, 0
    use_amp = cfg.device.startswith('cuda') and (cfg.fp16 or cfg.bf16)
    amp_dt = torch.float16 if cfg.fp16 else (torch.bfloat16 if cfg.bf16 else torch.float32)

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(cfg.device)
        B = batch['images'].size(0)

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dt):
            out = model(
                batch['images'], batch['text_ids'], batch['text_attn_mask'],
                gph_images=batch.get('gph_images'), sst_images=batch.get('sst_images'),
                text_ids_gph=batch.get('text_ids_gph'), text_mask_gph=batch.get('text_mask_gph'),
                text_ids_sst=batch.get('text_ids_sst'), text_mask_sst=batch.get('text_mask_sst'),
                tgt_ids=batch['target_ids'],
            )
            losses = loss_fn(out, batch, cfg)
            total_loss += losses['loss'].item()
            n_batch += 1

        _, gen_ids = model.generate(
            batch['images'], batch['text_ids'], batch['text_attn_mask'],
            gph_images=batch.get('gph_images'), sst_images=batch.get('sst_images'),
            text_ids_gph=batch.get('text_ids_gph'), text_mask_gph=batch.get('text_mask_gph'),
            text_ids_sst=batch.get('text_ids_sst'), text_mask_sst=batch.get('text_mask_sst'),
        )

        for b in range(B):
            text = decode_token_ids(gen_ids[b].tolist())
            json_total += 1
            try:
                parsed = json.loads(text)
                json_valid += 1
            except Exception:
                parsed = {"current_tc_count": 0, "current_tcs": []}

            pred_count = int(parsed.get('current_tc_count', 0))
            pred_tcs   = parsed.get('current_tcs', [])
            gt_count   = batch['current_tc_count'][b].item()
            gt_raw     = batch['current_tcs_raw'][b].float()
            gt_mask    = batch['current_tcs_mask'][b].float()

            if pred_count == gt_count:
                count_correct += 1
            count_mae_sum += abs(pred_count - gt_count)
            count_total += 1

            gt_list = []
            for i in range(cfg.max_current):
                if gt_mask[i] > 0.5:
                    gt_list.append((gt_raw[i, 0].item(), gt_raw[i, 1].item()))
            for glat, glon in gt_list:
                best = float('inf')
                for tc in (pred_tcs or []):
                    if isinstance(tc, dict):
                        d = _haversine_km(float(tc.get('lat', 0)), float(tc.get('lon', 0)), glat, glon)
                        best = min(best, d)
                if best < float('inf'):
                    hav_sum += best
                    hav_cnt += 1

    return {
        'val_loss':        total_loss / max(1, n_batch),
        'count_acc':       count_correct / max(1, count_total),
        'count_mae':       count_mae_sum / max(1, count_total),
        'hav_current_km':  hav_sum / max(1, hav_cnt),
        'json_valid_rate': json_valid / max(1, json_total),
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description='CNN Encoders + JSON Decoder Multimodal Cyclone Detection Training')
    p.add_argument('--data_folder', type=str, default=None)
    p.add_argument('--docs_folder', type=str, default=None)
    p.add_argument('--gph_folder', type=str, default=None)
    p.add_argument('--gph_docs_folder', type=str, default=None)
    p.add_argument('--sst_folder', type=str, default=None)
    p.add_argument('--sst_docs_folder', type=str, default=None)
    p.add_argument('--label_folder', type=str, default=None)
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--num_epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--learning_rate', type=float, default=None)
    p.add_argument('--warmup_steps', type=int, default=None)
    p.add_argument('--logging_steps', type=int, default=None)
    p.add_argument('--eval_steps', type=int, default=None)
    p.add_argument('--save_steps', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--no_gph', action='store_true')
    p.add_argument('--no_sst', action='store_true')
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--no_bf16', action='store_true')
    p.add_argument('--train_split', type=float, default=None)
    p.add_argument('--enable_early_stopping', action='store_true')
    p.add_argument('--early_stopping_patience', type=int, default=None)
    p.add_argument('--image_channels', type=int, default=None)
    p.add_argument('--decoder_layers', type=int, default=None)
    p.add_argument('--label_smoothing', type=float, default=None)
    args = p.parse_args()

    cfg = FusionConfig()
    for attr in ['data_folder', 'docs_folder', 'gph_folder', 'gph_docs_folder',
                 'sst_folder', 'sst_docs_folder', 'label_folder', 'output_dir']:
        v = getattr(args, attr, None)
        if v is not None:
            setattr(cfg, attr, v)
    for attr, key in [('num_epochs', 'num_epochs'), ('batch_size', 'batch_size'),
                      ('learning_rate', 'lr'), ('warmup_steps', 'warmup_steps'),
                      ('logging_steps', 'logging_steps'), ('eval_steps', 'eval_steps'),
                      ('save_steps', 'save_steps'), ('seed', 'seed'),
                      ('train_split', 'train_split'), ('early_stopping_patience', 'early_stopping_patience'),
                      ('image_channels', 'image_channels'), ('decoder_layers', 'decoder_layers'),
                      ('label_smoothing', 'label_smoothing')]:
        v = getattr(args, attr, None)
        if v is not None:
            setattr(cfg, key, v)
    if args.no_gph:  cfg.use_gph = False
    if args.no_sst:  cfg.use_sst = False
    if args.fp16:    cfg.fp16 = True
    if args.no_bf16: cfg.bf16 = False
    if args.enable_early_stopping: cfg.enable_early_stopping = True

    os.makedirs(cfg.output_dir, exist_ok=True)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        try:
            torch.set_float32_matmul_precision('high')
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    all_files = sorted(glob(os.path.join(cfg.data_folder, '*_image.npy')))
    if not all_files:
        raise RuntimeError(f"No *_image.npy in {cfg.data_folder}")
    split = int(len(all_files) * cfg.train_split)
    train_files, val_files = all_files[:split], all_files[split:]
    print(f"Train={len(train_files)}  Val={len(val_files)}")

    train_ds = CycloneFusionDataset(train_files, cfg)
    val_ds   = CycloneFusionDataset(val_files, cfg)
    vocab_size = len(train_ds.vocab)

    make_loader = lambda ds, shuf: DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuf, collate_fn=collate_fn,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=cfg.dataloader_persistent_workers and cfg.num_workers > 0,
    )
    train_loader = make_loader(train_ds, True)
    val_loader   = make_loader(val_ds, False)

    model = CycloneFusionModel(cfg, vocab_size).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}  |  Output vocab: {len(OUTPUT_VOCAB)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.num_epochs
    def lr_lambda(cur):
        if cur < cfg.warmup_steps:
            return cur / max(1, cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * (cur - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = None
    use_amp = cfg.device.startswith('cuda') and (cfg.fp16 or cfg.bf16)
    if use_amp and cfg.fp16:
        scaler = torch.amp.GradScaler('cuda')

    best_val = math.inf
    best_path = os.path.join(cfg.output_dir, 'best.pt')
    saved_ckpts: List[str] = []
    eval_since_improve = 0
    use_steps = cfg.monitor_during_training and cfg.monitor_strategy == 'steps'
    global_step = 0

    def _save(path, metrics=None):
        nonlocal saved_ckpts
        torch.save({
            'model_state': model.state_dict(),
            'cfg': cfg.__dict__,
            'vocab_size': vocab_size,
            **(({'val_metrics': metrics} if metrics else {})),
        }, path)
        saved_ckpts.append(path)
        if cfg.save_total_limit and len(saved_ckpts) > cfg.save_total_limit:
            for old in saved_ckpts[:-cfg.save_total_limit]:
                try: os.remove(old)
                except: pass
            saved_ckpts = saved_ckpts[-cfg.save_total_limit:]

    amp_dt = torch.float16 if cfg.fp16 else (torch.bfloat16 if cfg.bf16 else torch.float32)

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n{'='*60}\n  Epoch {epoch}/{cfg.num_epochs}\n{'='*60}")
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            for k in batch:
                batch[k] = batch[k].to(cfg.device)

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dt):
                out = model(
                    batch['images'], batch['text_ids'], batch['text_attn_mask'],
                    gph_images=batch.get('gph_images'), sst_images=batch.get('sst_images'),
                    text_ids_gph=batch.get('text_ids_gph'), text_mask_gph=batch.get('text_mask_gph'),
                    text_ids_sst=batch.get('text_ids_sst'), text_mask_sst=batch.get('text_mask_sst'),
                    tgt_ids=batch['target_ids'],
                )
                losses = loss_fn(out, batch, cfg)
                loss = losses['loss'] / max(1, cfg.gradient_accumulation_steps)

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.grad_clip_norm > 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                (scaler.step(optimizer) if scaler else optimizer.step())
                if scaler:
                    scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.logging_steps == 0:
                    print(f"  [train] g={global_step}  loss={losses['loss'].item():.4f}"
                          f"  tok_acc={losses['token_acc'].item():.3f}  lr={scheduler.get_last_lr()[0]:.2e}")

                if use_steps and cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                    metrics = evaluate(model, val_loader, cfg)
                    composite = metrics['val_loss']
                    print(f"  [val] g={global_step}  loss={composite:.4f}  cnt_acc={metrics['count_acc']:.3f}"
                          f"  cnt_mae={metrics['count_mae']:.2f}  hav_km={metrics['hav_current_km']:.1f}"
                          f"  json_ok={metrics['json_valid_rate']:.3f}")
                    if composite + cfg.early_stopping_threshold < best_val:
                        best_val = composite
                        _save(best_path, metrics)
                        print(f"  [save] New best loss={composite:.4f} → {best_path}")
                        eval_since_improve = 0
                    else:
                        eval_since_improve += 1
                        if cfg.enable_early_stopping and eval_since_improve >= cfg.early_stopping_patience:
                            print("  [early-stop]"); break
                    model.train()

                if use_steps and cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    _save(os.path.join(cfg.output_dir, f"ckpt-step-{global_step}.pt"))
        else:
            if not use_steps:
                metrics = evaluate(model, val_loader, cfg)
                composite = metrics['val_loss']
                print(f"  [val] ep={epoch}  loss={composite:.4f}  cnt_acc={metrics['count_acc']:.3f}"
                      f"  cnt_mae={metrics['count_mae']:.2f}  hav_km={metrics['hav_current_km']:.1f}"
                      f"  json_ok={metrics['json_valid_rate']:.3f}")
                if composite + cfg.early_stopping_threshold < best_val:
                    best_val = composite
                    _save(best_path, metrics)
                    print(f"  [save] New best → {best_path}")
                    eval_since_improve = 0
                else:
                    eval_since_improve += 1
                    if cfg.enable_early_stopping and eval_since_improve >= cfg.early_stopping_patience:
                        print("  [early-stop]"); break
            continue
        break

    print(f"\nDone. Best val_loss = {best_val:.6f}  →  {best_path}")

    if val_files:
        model.eval()
        sample = val_ds[0]
        batch = collate_fn([sample])
        for k in batch:
            batch[k] = batch[k].to(cfg.device)
        results = model.generate_json(
            batch['images'], batch['text_ids'], batch['text_attn_mask'],
            gph_images=batch.get('gph_images'), sst_images=batch.get('sst_images'),
            text_ids_gph=batch.get('text_ids_gph'), text_mask_gph=batch.get('text_mask_gph'),
            text_ids_sst=batch.get('text_ids_sst'), text_mask_sst=batch.get('text_mask_sst'),
        )
        print("\nExample inference:\n", json.dumps(results[0], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
