"""Configuration schema and loading.

Replaces the ~30 ``argparse`` flags that used to live at the top of ``train.py``
and ``evaluation.py``.

Design
------
Three layers, merged in increasing order of precedence:

    1. The structured dataclass schema below (defaults + type checking).
    2. A YAML file, e.g. ``configs/default.yaml``.
    3. Dotlist overrides from the command line, e.g. ``train.epochs=100``.

The dataclass layer is what makes this better than a bare dict: OmegaConf
validates types against it, so ``train.epochs=abc`` fails loudly at startup
instead of 40 minutes into a run. A typo like ``train.epoch=100`` is likewise
rejected, because the structured schema is closed by default.

Usage
-----
    python scripts/train.py --config configs/default.yaml \\
        data.npz_paths=[data/jan.npz,data/feb.npz] \\
        train.epochs=100 optim.lr=1e-4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from omegaconf import DictConfig, OmegaConf


# ==============================================================================
# SCHEMA
# ==============================================================================
@dataclass
class DataConfig:
    """Dataset construction and splitting.

    Exactly one of ``npz_paths`` / ``tensor_index_csv`` must be set -- the same
    mutually-exclusive contract the old argparse group enforced.
    """

    npz_paths: Optional[List[str]] = None
    tensor_index_csv: Optional[str] = None

    crop_size: int = 16

    # Random 70/15/15 split. See msca.data.splits.build_splits.
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Synchronized D4 augmentation on the TRAIN loader only.
    augment: bool = True

    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Forwarded verbatim to ``MSCANet(**cfg.model)``.

    ``in_channels_a`` / ``in_channels_b`` are exposed but should not be changed:
    the channel layout is fixed by the .npz files (Stream A = 3, Stream B = 4
    with population excluded), and ``SpatialGridDataset`` validates it.
    """

    img_size: List[int] = field(default_factory=lambda: [16, 16])
    patch_size: int = 4
    embed_dim: int = 128
    encoder_depth: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    in_channels_a: int = 3
    in_channels_b: int = 4


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: float = 5.0
    grad_clip: float = 1.0  # <= 0 disables clipping
    cosine_cycles: int = 3


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    use_amp: bool = False
    log_every: int = 50
    output_dir: str = "outputs/checkpoints"
    resume_from: Optional[str] = None


@dataclass
class EvalConfig:
    checkpoint: Optional[str] = None
    batch_size: int = 32
    # Path to the split_indices.json written by training. Strongly preferred
    # over regenerating the split from the seed. See msca.data.splits.
    split_file: Optional[str] = None


@dataclass
class LoggingConfig:
    use_wandb: bool = True
    project: str = "ms-ca-regression"
    entity: Optional[str] = None
    run_name: Optional[str] = None


@dataclass
class Config:
    seed: int = 42
    device: Optional[str] = None  # None -> auto-detect cuda/cpu

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ==============================================================================
# LOADING
# ==============================================================================
def load_config(
    config_path: str | Path | None = None,
    overrides: List[str] | None = None,
) -> DictConfig:
    """Merge schema <- YAML <- CLI dotlist overrides, then validate."""
    cfg = OmegaConf.structured(Config)

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    validate_config(cfg)
    return cfg  # type: ignore[return-value]


def validate_config(cfg: DictConfig) -> None:
    """Fail fast on configurations the pipeline cannot honour."""
    has_paths = bool(cfg.data.npz_paths)
    has_csv = bool(cfg.data.tensor_index_csv)
    if has_paths == has_csv:
        raise ValueError(
            "Set exactly one of 'data.npz_paths' or 'data.tensor_index_csv' "
            f"(got npz_paths={cfg.data.npz_paths!r}, "
            f"tensor_index_csv={cfg.data.tensor_index_csv!r})."
        )

    if not 0.0 < cfg.data.val_ratio < 1.0 or not 0.0 < cfg.data.test_ratio < 1.0:
        raise ValueError("data.val_ratio and data.test_ratio must be in (0, 1).")
    if cfg.data.val_ratio + cfg.data.test_ratio >= 1.0:
        raise ValueError("data.val_ratio + data.test_ratio must be < 1.0.")

    img_h, img_w = cfg.model.img_size
    if img_h % cfg.model.patch_size or img_w % cfg.model.patch_size:
        raise ValueError(
            f"model.img_size {list(cfg.model.img_size)} must be divisible by "
            f"model.patch_size {cfg.model.patch_size}."
        )
    if cfg.model.embed_dim % cfg.model.num_heads:
        raise ValueError(
            f"model.embed_dim ({cfg.model.embed_dim}) must be divisible by "
            f"model.num_heads ({cfg.model.num_heads})."
        )
    if tuple(cfg.model.img_size) != (cfg.data.crop_size, cfg.data.crop_size):
        raise ValueError(
            f"model.img_size {list(cfg.model.img_size)} must match "
            f"data.crop_size {cfg.data.crop_size} on both axes."
        )


def to_plain_dict(cfg: Any) -> dict:
    """Resolve an OmegaConf node into a plain dict (for wandb / checkpoints)."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
