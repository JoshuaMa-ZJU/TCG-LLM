# =============================================================================
# prefix_injector.py — Physics-aware Fine-tuning via KV Prefix Injection
# (TCG-LLM)
#
# This module implements the core mechanism of the physics-aware fine-tuning
# strategy.  It maps the fused physics-aware feature vector z (768-dim,
# from CycloneFusionModel) into prefix Key/Value vectors that are prepended
# to the self-attention KV cache of the VLM (past_key_values).  This allows
# each input token query (Q) to attend to physics-informed representations,
# so the model can "see" external feature vectors and leverage physical
# information to steer the adaptation process during fine-tuning.
#
# Architecture:
#   z [B, z_dim] → MLP → [B, num_kv_heads, prefix_len, head_dim] (K & V)
#   past_key_values = tuple( (K_l, V_l) for each target layer l )
# =============================================================================

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class FeaturePrefixEncoder(nn.Module):
    """Maps an external feature vector z (e.g., the 768-dim fused_vec from the
    CNN encoders) into Transformer self-attention KV prefixes, injected as
    past_key_values so that Q tokens in the VLM can attend to physics-aware
    visual cues.

    Output shape per layer l:
      k: [B, num_kv_heads, prefix_len, head_dim]
      v: [B, num_kv_heads, prefix_len, head_dim]
    past_key_values = tuple( (k_l, v_l) for l in target_layers )

    Args:
        hidden_size  : VLM hidden dimension
        num_heads    : number of attention heads (for head_dim = hidden_size / num_heads)
        num_kv_heads : number of K/V heads (may differ under GQA)
        num_layers   : total VLM layers
        prefix_len   : number of prefix tokens (default 128 in TCG-LLM)
        target_layers: how many layers receive the prefix (default: all)
        share_across_layers: if True, all layers share the same KV prefix
                             (significantly reduces parameters and memory)
        z_dim        : dimensionality of the external feature vector
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        z_dim: int,
        prefix_len: int = 4,
        num_kv_heads: Optional[int] = None,
        target_layers: Optional[int] = None,
        share_across_layers: bool = True,
        mlp_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads) if num_kv_heads is not None else int(num_heads)
        self.num_layers = int(num_layers)
        self.prefix_len = int(prefix_len)
        self.target_layers = int(target_layers) if (target_layers is not None and target_layers > 0) else int(num_layers)
        self.share_across_layers = bool(share_across_layers)
        self.z_dim = int(z_dim)

        head_dim = self.hidden_size // self.num_heads
        self.head_dim = int(head_dim)

        # Total parameters needed per KV projection: num_kv_heads * prefix_len * head_dim
        per_kv = self.num_kv_heads * self.prefix_len * self.head_dim

        hid = mlp_hidden or max(256, min(2048, self.z_dim * 2))  # MLP hidden dimension

        if self.share_across_layers:
            # Single shared K/V projection for all target layers (parameter-efficient)
            self.proj_k = nn.Sequential(
                nn.Linear(self.z_dim, hid), nn.ReLU(), nn.Linear(hid, per_kv)
            )
            self.proj_v = nn.Sequential(
                nn.Linear(self.z_dim, hid), nn.ReLU(), nn.Linear(hid, per_kv)
            )
        else:
            # Separate K/V projections for each target layer
            self.proj_k = nn.ModuleList([
                nn.Sequential(nn.Linear(self.z_dim, hid), nn.ReLU(), nn.Linear(hid, per_kv))
                for _ in range(self.target_layers)
            ])
            self.proj_v = nn.ModuleList([
                nn.Sequential(nn.Linear(self.z_dim, hid), nn.ReLU(), nn.Linear(hid, per_kv))
                for _ in range(self.target_layers)
            ])

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape flat projection [B, per_kv] to [B, num_kv_heads, prefix_len, head_dim]."""
        B = x.size(0)
        return x.view(B, self.num_kv_heads, self.prefix_len, self.head_dim)

    def build_prefix_kv(self, z: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Generate past_key_values from the external physics-aware feature vector z.

        NOTE: no @torch.no_grad() here — when train_prefix_encoder=True during
        cooperative training, gradients must flow through the MLP
        projections so the prefix encoder weights can be updated.

        Args:
            z: [B, z_dim] — fused feature vector from CycloneFusionModel.
        Returns:
            Tuple of (K, V) pairs, one per target layer, ready for VLM injection.
        """
        assert z.dim() == 2 and z.size(1) == self.z_dim, f"z must be [B,{self.z_dim}]"
        B = z.size(0)
        layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
        if self.share_across_layers:
            k = self._reshape(self.proj_k(z))  # [B, H_kv, P, D]
            v = self._reshape(self.proj_v(z))
            for _ in range(self.target_layers):
                layers.append((k, v))
        else:
            for i in range(self.target_layers):
                k = self._reshape(self.proj_k[i](z))
                v = self._reshape(self.proj_v[i](z))
                layers.append((k, v))
        return tuple(layers)


def make_prefix_encoder_from_config(model_config, z_dim: int, prefix_len: int = 4, target_layers: Optional[int] = None, share_across_layers: bool = True) -> FeaturePrefixEncoder:
    """Factory: build FeaturePrefixEncoder from a HuggingFace model config.

    Compatible with nested config structures of multimodal models (e.g.,
    Qwen3-VL-8B): tries top-level first, then falls back to text_config /
    llm_config / language_model sub-configs.

    Required fields (any alias): hidden_size, num_attention_heads,
    num_key_value_heads (optional, defaults to num_attention_heads),
    num_hidden_layers.
    """

    def _resolve_fields(cfg):
        if cfg is None:
            return None
        hs = getattr(cfg, 'hidden_size', None) or getattr(cfg, 'n_embd', None)
        nh = getattr(cfg, 'num_attention_heads', None) or getattr(cfg, 'n_head', None)
        nkv = getattr(cfg, 'num_key_value_heads', None) or getattr(cfg, 'num_attention_heads', None) or getattr(cfg, 'n_head', None)
        nl = getattr(cfg, 'num_hidden_layers', None) or getattr(cfg, 'n_layer', None)
        if hs is None or nh is None or nl is None:
            return None
        return int(hs), int(nh), int(nkv) if nkv is not None else int(nh), int(nl)

    # Candidates: top-level config and common sub-configurations
    cands = [
        model_config,
        getattr(model_config, 'text_config', None),
        getattr(model_config, 'llm_config', None),
        getattr(model_config, 'language_model', None),
        getattr(model_config, 'model_config', None),
    ]
    resolved = None
    for c in cands:
        resolved = _resolve_fields(c)
        if resolved is not None:
            break

    if resolved is None:
        raise ValueError("model_config missing required fields (hidden_size/num_attention_heads/num_hidden_layers)")

    hidden_size, num_heads, num_kv_heads, num_layers = resolved
    enc = FeaturePrefixEncoder(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        z_dim=int(z_dim),
        prefix_len=int(prefix_len),
        target_layers=int(target_layers) if (target_layers is not None and target_layers > 0) else int(num_layers),
        share_across_layers=share_across_layers,
    )
    return enc
