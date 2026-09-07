"""Collate functions.

Both collates drop everything the model does not consume (``mask``, ``date``,
``row``, ``col``) so those never cross the worker->main-process boundary.

``train.py`` and ``evaluation.py`` previously carried two byte-identical copies
of the plain collate under different names. There is now one.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from msca.data.transforms import apply_d4_transform, sample_d4_params

__all__ = ["plain_collate_fn", "d4_augmentation_collate_fn"]


def plain_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Stack the three tensors the model needs. No augmentation."""
    return {
        "stream_a": torch.stack([item["stream_a"] for item in batch], dim=0),
        "stream_b": torch.stack([item["stream_b"] for item in batch], dim=0),
        "label_reg": torch.stack([item["label_reg"] for item in batch], dim=0),
    }


def d4_augmentation_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Per-sample synchronized D4 augmentation, then stack.

    A fresh transform is drawn for each sample (not each batch), so the two
    streams stay aligned with each other while differing across the batch.
    """
    augmented_a, augmented_b, regs = [], [], []
    for item in batch:
        k_rot, flip_rows, flip_cols = sample_d4_params()
        a_aug, b_aug = apply_d4_transform(
            item["stream_a"], item["stream_b"], k_rot, flip_rows, flip_cols
        )
        augmented_a.append(a_aug)
        augmented_b.append(b_aug)
        regs.append(item["label_reg"])

    return {
        "stream_a": torch.stack(augmented_a, dim=0),
        "stream_b": torch.stack(augmented_b, dim=0),
        "label_reg": torch.stack(regs, dim=0),
    }
