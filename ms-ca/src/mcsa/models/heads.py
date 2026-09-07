"""Task heads.

This is a strictly single-task, pure-regression pipeline: ``RegressionHead`` is
the only head, and the 5-class hotspot ``ClassificationHead`` from the old
multi-task version is gone for good.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RegressionHead"]


class RegressionHead(nn.Module):
    """Scalar carbon-emission regression head (the ONLY task head)."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # [B, 1]
