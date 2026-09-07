"""Train/val/test splitting and split persistence.

``build_splits`` is behaviourally identical to the original in
``spatial_grid_dataset.py``: PyTorch ``random_split`` with a seeded generator,
70/15/15 by default.

Why the extra persistence layer
-------------------------------
The original contract was "train.py and evaluation.py agree on the test set as
long as they're called with the same seed, ratios, and data". That holds, but it
is an *unverified* contract with a silent failure mode: reorder ``--data_path``,
add a month, or change ``crop_size``, and the sample list changes, so the same
seed yields a different partition -- and evaluation happily reports metrics on
samples the model was trained on. Nothing warns you; the numbers just look good.

So training now writes ``split_indices.json`` next to the checkpoints, recording
the exact indices plus a fingerprint of the dataset that produced them.
Evaluation loads that file and hard-fails on a fingerprint mismatch. The seeded
path remains as a fallback, and the fallback prints a warning.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset, random_split

__all__ = [
    "build_splits",
    "dataset_fingerprint",
    "save_split_indices",
    "load_split_indices",
]


def build_splits(
    dataset: Dataset,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Randomly split a dataset into train/val/test subsets.

    Uses PyTorch's standard ``random_split`` with a seeded generator so the
    partition is reproducible across processes, given identical underlying data
    and identical ``seed`` / ``val_ratio`` / ``test_ratio``.

    Default is a 70 / 15 / 15 train / val / test split.
    """
    if not 0.0 < val_ratio < 1.0 or not 0.0 < test_ratio < 1.0:
        raise ValueError("val_ratio and test_ratio must each be in (0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0 (train needs a share).")

    n_total = len(dataset)  # type: ignore[arg-type]
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"Split leaves no training samples: total={n_total}, "
            f"val={n_val}, test={n_test}."
        )

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )
    return train_subset, val_subset, test_subset


def dataset_fingerprint(dataset: Dataset) -> str:
    """A short, stable hash of the dataset's identity.

    Covers sample count, the ordered ``.npz`` file list, and crop size -- the
    three things that silently change which sample sits at which index.
    """
    parts: list[str] = [str(len(dataset))]  # type: ignore[arg-type]
    parts.append(str(getattr(dataset, "crop_size", "?")))
    for path in getattr(dataset, "npz_paths", []):
        parts.append(Path(path).name)
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def save_split_indices(
    path: str | Path,
    dataset: Dataset,
    train_subset: Subset,
    val_subset: Subset,
    test_subset: Subset,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> None:
    """Persist the exact split indices so evaluation can verify, not guess."""
    payload = {
        "fingerprint": dataset_fingerprint(dataset),
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "n_total": len(dataset),  # type: ignore[arg-type]
        "train": list(train_subset.indices),
        "val": list(val_subset.indices),
        "test": list(test_subset.indices),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def load_split_indices(
    path: str | Path,
    dataset: Dataset,
    strict: bool = True,
) -> tuple[Subset, Subset, Subset]:
    """Rebuild subsets from a saved ``split_indices.json``.

    Args:
        strict: raise if the dataset fingerprint differs from the one recorded
            at training time. Leave this on -- a mismatch means the test set you
            are about to evaluate is not the test set that was held out.
    """
    payload = json.loads(Path(path).read_text())

    current = dataset_fingerprint(dataset)
    recorded = payload.get("fingerprint")
    if current != recorded:
        message = (
            f"Dataset fingerprint mismatch: split file was written for "
            f"'{recorded}' but the current dataset hashes to '{current}'. The "
            f"held-out test set cannot be reproduced -- check that the .npz "
            f"files, their order, and crop_size match the training run."
        )
        if strict:
            raise ValueError(message)
        print(f"[Warning] {message}")

    return (
        Subset(dataset, payload["train"]),
        Subset(dataset, payload["val"]),
        Subset(dataset, payload["test"]),
    )
