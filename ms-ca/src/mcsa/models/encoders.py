"""Stream encoders: the pluggable backbone interface and its ViT implementation.

The ``BaseEncoder`` contract is the extension point of the whole architecture.
Anything that turns ``[B, C, H, W]`` into ``[B, L, D]`` and can report its token
grid shape can be dropped into ``MSCANet`` without touching fusion or the head.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn

__all__ = ["BaseEncoder", "TransformerEncoderBlock", "ViTPatchEncoder"]


class BaseEncoder(nn.Module, ABC):
    """Abstract base class every stream-encoder backbone must implement.

    Contract
    --------
    forward(x)  : ``x`` is a raw image tensor ``[B, C, H, W]`` and MUST return a
                  token sequence ``[B, L, D]``.
    num_patches : the fixed sequence length ``L`` produced by the encoder.
    grid_size   : ``(grid_h, grid_w)`` with ``grid_h * grid_w == L``, so token
                  sequences and attention matrices can be reshaped back into a
                  2D spatial map. This is what keeps the model explainable.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_patches(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def grid_size(self) -> Tuple[int, int]:
        raise NotImplementedError


class TransformerEncoderBlock(nn.Module):
    """A single pre-norm Transformer encoder block (self-attention + MLP)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention sub-block (pre-norm + residual).
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out

        # MLP sub-block (pre-norm + residual).
        x = x + self.mlp(self.norm2(x))
        return x


class ViTPatchEncoder(BaseEncoder):
    """Lightweight Vision Transformer encoder.

    Conv2d patch-embedding -> learned positional embeddings -> N Transformer
    blocks -> final LayerNorm.

    Defaults target this pipeline's 16x16 crops with 4x4 patches:
    ``grid_h = grid_w = 4``, so ``num_patches = 16``.

    No ``[CLS]`` token, deliberately. Keeping the output sequence purely spatial
    means ``L == grid_h * grid_w`` always holds, which is what lets attention
    maps be reshaped into a 2D grid for XAI.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Tuple[int, int] = (16, 16),
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        **_unused_kwargs,
    ) -> None:
        """
        Args:
            in_channels: input channels for this stream (3 = Stream A, 4 = Stream B).
            img_size: (H, W) of the raw input crop.
            patch_size: side length of each square patch.
            embed_dim: token embedding dimension.
            depth: number of stacked Transformer encoder blocks.
            num_heads: self-attention heads per block.
            dropout: dropout probability used throughout the block.
            **_unused_kwargs: absorbs extra keyword arguments so ``MSCANet`` can
                stay backbone-agnostic without every encoder sharing an
                identical constructor signature.
        """
        super().__init__(embed_dim)

        img_h, img_w = img_size
        if img_h % patch_size or img_w % patch_size:
            raise ValueError(
                f"img_size {img_size} must be divisible by patch_size {patch_size}."
            )

        self.patch_size = patch_size
        self.grid_h = img_h // patch_size
        self.grid_w = img_w // patch_size
        self._num_patches = self.grid_h * self.grid_w

        # One strided conv turns each non-overlapping patch into one token.
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self._num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    @property
    def num_patches(self) -> int:
        return self._num_patches

    @property
    def grid_size(self) -> Tuple[int, int]:
        return (self.grid_h, self.grid_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> e.g. [B, 3, 16, 16] or [B, 4, 16, 16]
        x = self.patch_embed(x)           # [B, D, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2)  # [B, L, D]
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)               # [B, L, D]
