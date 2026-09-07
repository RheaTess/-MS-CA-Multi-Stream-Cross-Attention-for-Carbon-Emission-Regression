"""Experiment logging.

Wraps Weights & Biases behind a tiny interface with a no-op fallback, so the
training loop never needs ``if wandb_run is not None`` guards. The old code had
that check at four call sites; the null-object pattern removes all of them.

W&B stays an optional dependency: if it isn't installed, or logging is disabled
in the config, ``get_logger`` returns a ``NullLogger`` and everything else
proceeds unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from omegaconf import DictConfig

from msca.config import to_plain_dict

try:  # pragma: no cover - depends on the local environment
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WANDB_AVAILABLE = False

__all__ = ["ExperimentLogger", "NullLogger", "WandbLogger", "get_logger"]


class ExperimentLogger(Protocol):
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None: ...
    def watch(self, model: Any) -> None: ...
    def finish(self) -> None: ...


class NullLogger:
    """Console-only fallback. Every method is a no-op."""

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        return None

    def watch(self, model: Any) -> None:
        return None

    def finish(self) -> None:
        return None


class WandbLogger:
    """Thin Weights & Biases adapter."""

    def __init__(self, cfg: DictConfig) -> None:
        self.run = wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            name=cfg.logging.run_name,
            config=to_plain_dict(cfg),
        )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        self.run.log(metrics, step=step)

    def watch(self, model: Any) -> None:
        self.run.watch(model, log="gradients", log_freq=200)

    def finish(self) -> None:
        self.run.finish()


def get_logger(cfg: DictConfig) -> ExperimentLogger:
    """Return a W&B logger if enabled and installed, else a no-op logger."""
    if not cfg.logging.use_wandb:
        return NullLogger()
    if not _WANDB_AVAILABLE:
        print(
            "[Warning] wandb is not installed; continuing console-only. "
            "Run `pip install wandb` to enable it."
        )
        return NullLogger()
    return WandbLogger(cfg)
