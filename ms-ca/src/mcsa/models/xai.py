"""Explainable-AI helpers for the cross-attention fusion layer."""

from __future__ import annotations

from typing import Tuple

import torch

__all__ = ["reshape_attention_to_spatial_grid"]


def reshape_attention_to_spatial_grid(
    attn_weights: torch.Tensor,
    query_grid: Tuple[int, int],
    key_grid: Tuple[int, int],
    average_heads: bool = True,
) -> torch.Tensor:
    """Convert the flat cross-attention matrix into a spatial tensor.

    Suitable for overlaying on the input raster.

    Args:
        attn_weights: [B, num_heads, Lq, Lk] from ``CrossAttentionFusion``.
        query_grid: (gh, gw) of the query encoder (Stream B), gh*gw == Lq.
        key_grid: (gh, gw) of the key/value encoder (Stream A), gh*gw == Lk.
        average_heads: average across attention heads first.

    Returns:
        ``[B, gh_q, gw_q, gh_k, gw_k]`` if ``average_heads`` else
        ``[B, num_heads, gh_q, gw_q, gh_k, gw_k]``.

    Tip:
        Read the grids straight off the model rather than hardcoding them::

            reshape_attention_to_spatial_grid(
                attn, model.encoder_b.grid_size, model.encoder_a.grid_size
            )
    """
    gh_q, gw_q = query_grid
    gh_k, gw_k = key_grid

    if average_heads:
        attn = attn_weights.mean(dim=1)  # [B, Lq, Lk]
        batch, len_q, len_k = attn.shape
    else:
        attn = attn_weights
        batch, num_heads, len_q, len_k = attn.shape

    if len_q != gh_q * gw_q:
        raise ValueError(
            f"query_grid {query_grid} implies {gh_q * gw_q} tokens, got Lq={len_q}."
        )
    if len_k != gh_k * gw_k:
        raise ValueError(
            f"key_grid {key_grid} implies {gh_k * gw_k} tokens, got Lk={len_k}."
        )

    if average_heads:
        return attn.reshape(batch, gh_q, gw_q, gh_k, gw_k)
    return attn.reshape(batch, num_heads, gh_q, gw_q, gh_k, gw_k)
