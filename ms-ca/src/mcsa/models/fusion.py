"""Cross-attention fusion between the two encoded streams."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

__all__ = ["CrossAttentionFusion"]


class CrossAttentionFusion(nn.Module):
    """Fuses the two encoded streams via cross-attention.

        Query (Q)       <- Stream B tokens (Socio-Infrastructure)
        Key/Value (K,V) <- Stream A tokens (Environment)

    This direction asks: "given each socio-infrastructure patch, which
    environmental (NO2/SO2/CO) patches are most relevant?" -- the intuitive
    framing for emission attribution.

    The raw per-head attention matrix is returned unmodified for downstream
    XAI visualization (see :func:`msca.models.xai.reshape_attention_to_spatial_grid`).
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(
        self, tokens_b: torch.Tensor, tokens_a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens_b: [B, Lq, D]  Query source (Stream B).
            tokens_a: [B, Lk, D]  Key/Value source (Stream A).

        Returns:
            fused_tokens: [B, Lq, D]
            attn_weights: [B, num_heads, Lq, Lk]  (per-head, un-averaged, for XAI)
        """
        q = self.norm_q(tokens_b)
        kv = self.norm_kv(tokens_a)

        fused, attn_weights = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=True,
            average_attn_weights=False,
        )

        # Residual back onto the query stream so the fused output retains the
        # socio-infrastructure identity.
        fused_tokens = self.norm_out(fused + tokens_b)
        return fused_tokens, attn_weights
