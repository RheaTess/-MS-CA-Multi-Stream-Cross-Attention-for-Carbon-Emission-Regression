"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["set_global_seed", "seed_worker", "make_generator"]


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG the training pipeline touches.

    Args:
        seed: the seed.
        deterministic: also force cuDNN into deterministic mode. Off by default
            because it can cost real throughput; turn it on when you need
            bit-exact reruns rather than statistical reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` that makes DataLoader workers reproducible.

    Why this is needed: ``d4_augmentation_collate_fn`` draws from the global
    ``random`` module, which runs *inside* worker processes. Without this hook,
    each worker inherits a torch-assigned base seed but ``random`` and ``numpy``
    are left in whatever state fork gave them -- so the augmentation stream is
    not reproducible across runs even with ``set_global_seed``. This was a live
    gap in the original ``train.py``.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A seeded generator for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
