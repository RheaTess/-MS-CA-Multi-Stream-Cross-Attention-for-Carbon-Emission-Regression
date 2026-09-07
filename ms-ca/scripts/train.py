#!/usr/bin/env python
"""Training entry point for the MS-CA single-task regression network.

This file is deliberately thin: parse config, wire components, run the loop.
Everything reusable lives in the ``msca`` package.

Usage
-----
    python scripts/train.py --config configs/default.yaml \\
        data.npz_paths=[data/jan.npz,data/feb.npz]

    # Override anything from the CLI:
    python scripts/train.py --config configs/default.yaml \\
        train.epochs=100 optim.lr=1e-4 train.use_amp=true

    # Or point at an index CSV with a 'tensor_path' column:
    python scripts/train.py --config configs/default.yaml \\
        data.tensor_index_csv=data/index.csv

Evaluate afterwards with ``scripts/evaluate.py``, passing the
``split_indices.json`` this script writes next to the checkpoints.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from msca.config import load_config, to_plain_dict
from msca.data import (
    SpatialGridDataset,
    build_eval_loader,
    build_splits,
    build_train_loader,
    save_split_indices,
)
from msca.engine import (
    build_optimizer,
    build_warmup_cosine_scheduler,
    load_checkpoint,
    save_best_model,
    save_latest_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)
from msca.models import MSCANet
from msca.utils import get_logger, make_generator, resolve_device, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the MS-CA single-task regression network.",
        epilog="Any config key can be overridden as key.subkey=value.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. train.epochs=100 optim.lr=1e-4.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    set_global_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"[Setup] Using device: {device}")

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = output_dir / "checkpoint_latest.pth"
    best_ckpt_path = output_dir / "best_model.pth"
    split_path = output_dir / "split_indices.json"

    logger = get_logger(cfg)

    # ------------------------------------------------------------------ #
    # Data -> random 70/15/15 split
    # ------------------------------------------------------------------ #
    full_dataset = SpatialGridDataset(
        npz_paths=list(cfg.data.npz_paths) if cfg.data.npz_paths else None,
        tensor_index_csv=cfg.data.tensor_index_csv,
        crop_size=cfg.data.crop_size,
    )
    train_subset, val_subset, test_subset = build_splits(
        full_dataset,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.seed,
    )

    # Persist the split so evaluation can VERIFY the held-out set rather than
    # re-derive it and hope the inputs were identical.
    save_split_indices(
        split_path,
        full_dataset,
        train_subset,
        val_subset,
        test_subset,
        seed=cfg.seed,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
    )

    generator = make_generator(cfg.seed)
    train_loader = build_train_loader(train_subset, cfg, generator=generator)
    val_loader = build_eval_loader(
        val_subset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    print(
        f"[Data] total={len(full_dataset)} | train={len(train_subset)} "
        f"val={len(val_subset)} test={len(test_subset)} (held out) | "
        f"steps/epoch={len(train_loader)} | augment={cfg.data.augment}"
    )
    print(f"[Data] Split indices written to '{split_path}'.")

    # ------------------------------------------------------------------ #
    # Model (single-task regression)
    # ------------------------------------------------------------------ #
    model_config = to_plain_dict(cfg.model)
    model = MSCANet(**model_config)

    if torch.cuda.device_count() > 1:
        print(f"[Setup] Using {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Standard L1 Loss (MAE) -- the fundamental starting point for this task.
    loss_fn = nn.L1Loss()
    logger.watch(model)

    # ------------------------------------------------------------------ #
    # Optimizer + scheduler + AMP scaler
    # ------------------------------------------------------------------ #
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=int(steps_per_epoch * cfg.optim.warmup_epochs),
        total_steps=steps_per_epoch * cfg.train.epochs,
        cycles=cfg.optim.cosine_cycles,
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=cfg.train.use_amp)

    start_epoch = 0
    best_val_mae = float("inf")
    if cfg.train.resume_from and Path(cfg.train.resume_from).is_file():
        start_epoch, best_val_mae = load_checkpoint(
            cfg.train.resume_from, model, optimizer, scheduler, scaler, device
        )

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    cfg_dict = to_plain_dict(cfg)
    for epoch in range(start_epoch, cfg.train.epochs):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model,
            loss_fn,
            train_loader,
            optimizer,
            scheduler,
            device,
            grad_clip=cfg.optim.grad_clip,
            scaler=scaler,
            use_amp=cfg.train.use_amp,
            epoch=epoch,
            logger=logger,
            log_every=cfg.train.log_every,
        )
        val_metrics = validate_one_epoch(model, loss_fn, val_loader, device, epoch=epoch)

        epoch_time = time.time() - epoch_start
        print(
            f"[Epoch {epoch}] done in {epoch_time:.1f}s | "
            f"train MAE={train_metrics['mae']:.4f} RMSE={train_metrics['rmse']:.4f} | "
            f"val MAE={val_metrics['mae']:.4f} RMSE={val_metrics['rmse']:.4f}"
        )

        logger.log(
            {
                "epoch": epoch,
                "train/mae": train_metrics["mae"],
                "train/rmse": train_metrics["rmse"],
                "val/mae": val_metrics["mae"],
                "val/rmse": val_metrics["rmse"],
                "epoch_time_sec": epoch_time,
            }
        )

        save_latest_checkpoint(
            latest_ckpt_path, model, optimizer, scheduler, scaler,
            model_config, epoch, best_val_mae, cfg_dict,
        )

        # Best model = lowest VALIDATION MAE.
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            save_best_model(
                best_ckpt_path, model, model_config, epoch, best_val_mae, cfg_dict
            )
            print(
                f"[Checkpoint] New best val MAE {best_val_mae:.4f} "
                f"-> saved '{best_ckpt_path}'."
            )

    logger.finish()

    print(f"\n[Done] Training complete. Best val MAE = {best_val_mae:.4f}.")
    print(
        f"[Next] Evaluate the held-out test split with:\n"
        f"       python scripts/evaluate.py --config {args.config} "
        f"eval.checkpoint={best_ckpt_path} eval.split_file={split_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
