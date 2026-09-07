"""DataLoader construction.

Keeps the ~15 lines of DataLoader boilerplate (and its easy-to-get-wrong flags:
``drop_last``, ``persistent_workers``, ``shuffle``) out of the entry points.
"""

from __future__ import annotations

from typing import Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from msca.data.collate import d4_augmentation_collate_fn, plain_collate_fn
from msca.utils.seed import seed_worker

__all__ = ["build_train_loader", "build_eval_loader"]


def build_train_loader(
    subset: Dataset,
    cfg: DictConfig,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """Shuffled, optionally D4-augmented loader. ``drop_last=True``."""
    collate = d4_augmentation_collate_fn if cfg.data.augment else plain_collate_fn
    return DataLoader(
        subset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
        collate_fn=collate,
        persistent_workers=cfg.data.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_eval_loader(
    subset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Deterministic loader for val/test. Never augments, never drops."""
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=plain_collate_fn,
        persistent_workers=num_workers > 0,
    )
