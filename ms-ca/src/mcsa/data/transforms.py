"""Synchronized D4 augmentation for the two spatial streams."""

from __future__ import annotations

import random
from typing import Tuple

import torch

__all__ = ["apply_d4_transform", "sample_d4_params"]


def sample_d4_params(rng: random.Random | None = None) -> Tuple[int, bool, bool]:
    """Draw one random element of the D4 symmetry group.

    Args:
        rng: optional ``random.Random`` instance. Passing one lets a caller
            control the augmentation stream explicitly instead of relying on the
            global ``random`` module (which is per-worker in a DataLoader).

    Returns:
        ``(k_rot, flip_rows, flip_cols)``.
    """
    source = rng if rng is not None else random
    return (source.randint(0, 3), source.random() < 0.5, source.random() < 0.5)


def apply_d4_transform(
    stream_a: torch.Tensor,
    stream_b: torch.Tensor,
    k_rot: int,
    flip_rows: bool,
    flip_cols: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply ONE D4 symmetry transform IDENTICALLY to both streams.

    Rotation by ``90 * k_rot`` degrees, then optional row flip, then optional
    column flip. Applying the same transform to both streams preserves the
    spatial alignment the cross-attention fusion layer depends on. The scalar
    regression target is invariant to these transforms, so labels are untouched.

    Args:
        stream_a: [..., H, W] environment proxy.
        stream_b: [..., H, W] socio-infrastructure proxy.
        k_rot: number of 90-degree rotations, 0-3.
        flip_rows: flip along the row axis (dim -2), i.e. a vertical flip.
        flip_cols: flip along the column axis (dim -1), i.e. a horizontal flip.

    Note:
        The original ``_apply_d4_transform`` called these ``flip_h`` / ``flip_v``
        but flipped dims ``-2`` / ``-1`` respectively, which is the opposite of
        what those names suggest. The behaviour is unchanged here -- rotations
        plus both flips generate the same group either way -- only the parameter
        names are corrected.
    """
    if k_rot % 4:
        stream_a = torch.rot90(stream_a, k=k_rot, dims=(-2, -1))
        stream_b = torch.rot90(stream_b, k=k_rot, dims=(-2, -1))
    if flip_rows:
        stream_a = torch.flip(stream_a, dims=(-2,))
        stream_b = torch.flip(stream_b, dims=(-2,))
    if flip_cols:
        stream_a = torch.flip(stream_a, dims=(-1,))
        stream_b = torch.flip(stream_b, dims=(-1,))
    return stream_a, stream_b
