#!/usr/bin/env python
"""Standalone test-set evaluation for the MS-CA regressor.

Loads a trained ``best_model.pth``, runs it over the held-out TEST split, and
reports MAE, RMSE, R^2 and Pearson r.

Reproducing the SAME test split
-------------------------------
Preferred: pass ``eval.split_file=<output_dir>/split_indices.json``, written by
training. The indices are exact and the dataset fingerprint is verified, so a
mismatched dataset fails loudly instead of silently evaluating on training data.

Fallback: omit ``eval.split_file`` and the split is regenerated via
``build_splits``. That only reproduces the original partition if the data files,
their order, ``data.crop_size``, ``seed``, ``val_ratio`` and ``test_ratio`` all
match the training run exactly.

The architecture is rebuilt from the ``model_config`` stored in the checkpoint,
so no architecture settings are needed here.

Usage
-----
    python scripts/evaluate.py --config configs/default.yaml \\
        eval.checkpoint=outputs/checkpoints/best_model.pth \\
        eval.split_file=outputs/checkpoints/split_indices.json
"""

from __future__ import annotations

import argparse
import sys

from msca.config import load_config
from msca.data import (
    SpatialGridDataset,
    build_eval_loader,
    build_splits,
    load_split_indices,
)
from msca.engine import evaluate, format_metrics, load_model_from_checkpoint
from msca.utils import resolve_device, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MS-CA regressor on the held-out test split.",
        epilog="Any config key can be overridden as key.subkey=value.",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. eval.checkpoint=path/to.pth.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    if not cfg.eval.checkpoint:
        raise ValueError(
            "Set eval.checkpoint, e.g. "
            "`python scripts/evaluate.py eval.checkpoint=outputs/checkpoints/best_model.pth`"
        )

    set_global_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"[Setup] Using device: {device}")

    model = load_model_from_checkpoint(cfg.eval.checkpoint, device)

    full_dataset = SpatialGridDataset(
        npz_paths=list(cfg.data.npz_paths) if cfg.data.npz_paths else None,
        tensor_index_csv=cfg.data.tensor_index_csv,
        crop_size=cfg.data.crop_size,
    )

    if cfg.eval.split_file:
        _train, _val, test_subset = load_split_indices(
            cfg.eval.split_file, full_dataset, strict=True
        )
        print(f"[Data] Test split loaded from '{cfg.eval.split_file}' (verified).")
    else:
        _train, _val, test_subset = build_splits(
            full_dataset,
            val_ratio=cfg.data.val_ratio,
            test_ratio=cfg.data.test_ratio,
            seed=cfg.seed,
        )
        print(
            "[Warning] No eval.split_file given; regenerating the split from "
            f"seed={cfg.seed}. This is only valid if the data files, their "
            "order, and the ratios exactly match the training run."
        )

    print(f"[Data] total={len(full_dataset)} | test={len(test_subset)}")

    test_loader = build_eval_loader(
        test_subset,
        batch_size=cfg.eval.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    metrics = evaluate(model, test_loader, device)
    print(format_metrics(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
