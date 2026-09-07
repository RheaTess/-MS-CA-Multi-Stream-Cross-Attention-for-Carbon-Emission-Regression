"""Device selection."""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["resolve_device"]


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Resolve a device string, auto-detecting CUDA when None.

    The same six lines appeared in both ``train.py`` and ``evaluation.py``.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
