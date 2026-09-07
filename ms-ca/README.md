# MS-CA: Multi-Stream Cross-Attention for Carbon-Emission Regression

Predicts continuous carbon-emission values for ~17,000 spatial grids by fusing two
streams of satellite proxy data through a cross-attention Vision Transformer.

This is a **strictly single-task, pure-regression** pipeline. There is no
classification branch, no hotspot grading, and no multi-task loss weighting.

| | |
|---|---|
| **Task** | Scalar regression (continuous carbon emission per grid cell) |
| **Loss** | `nn.L1Loss` (MAE) |
| **Stream A** (Environment) | NO2, SO2, CO — **3 channels** |
| **Stream B** (Socio-Infra) | Nightlight, Urban Fraction, Power Plant, Fossil Capacity — **4 channels** |
| **Fusion** | Cross-attention, Q ← Stream B, K/V ← Stream A |
| **Split** | Random 70 / 15 / 15 via `torch.utils.data.random_split` |
| **Preprocessing** | None — the raw label distribution is preserved (no clipping, capping, or outlier removal) |

---

## Install

```bash
git clone https://github.com/your-org/ms-ca.git
cd ms-ca

python -m venv .venv && source .venv/bin/activate

# Install PyTorch first, matched to your CUDA version:
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -e .            # editable install of the msca package
# or, for development:
pip install -e ".[dev,wandb]"
```

Requires Python ≥ 3.10.

---

## Quickstart

Put your `.npz` tensors in `data/`, then:

```bash
# Train
python scripts/train.py --config configs/default.yaml \
    data.npz_paths=[data/2023_01.npz,data/2023_02.npz]

# Evaluate on the held-out test split
python scripts/evaluate.py --config configs/default.yaml \
    data.npz_paths=[data/2023_01.npz,data/2023_02.npz] \
    eval.checkpoint=outputs/checkpoints/best_model.pth \
    eval.split_file=outputs/checkpoints/split_indices.json
```

Or point at an index CSV with a `tensor_path` column:

```bash
python scripts/train.py --config configs/default.yaml \
    data.tensor_index_csv=data/tensor_index.csv
```

---

## Configuration

All hyperparameters live in [`configs/default.yaml`](configs/default.yaml),
validated against the dataclass schema in `src/msca/config.py`. Override any key
from the command line with dotlist syntax:

```bash
python scripts/train.py --config configs/default.yaml \
    train.epochs=100 \
    optim.lr=1e-4 \
    train.use_amp=true \
    model.embed_dim=256 \
    logging.use_wandb=false
```

Precedence: **schema defaults → YAML file → CLI overrides**.

Because the schema is a closed structured config, mistakes fail at startup
rather than 40 minutes into a run:

```
$ python scripts/train.py train.epoch=100
omegaconf.errors.ConfigKeyError: Key 'epoch' not in 'TrainConfig'

$ python scripts/train.py train.epochs=abc
omegaconf.errors.ValidationError: Value 'abc' of type 'str' could not be converted to Integer
```

---

## Reproducing the test split

Training writes `split_indices.json` alongside the checkpoints, recording the
exact train/val/test indices plus a fingerprint of the dataset that produced
them. Pass it to `evaluate.py` via `eval.split_file=`.

**Prefer this over regenerating from the seed.** Seed-based reproduction is
still supported as a fallback, but it only works if the `.npz` files, *their
order*, `data.crop_size`, `seed`, `val_ratio` and `test_ratio` all match the
training run exactly. Add a month or reorder the paths and the same seed yields
a *different* partition — you would evaluate on training samples and never know.
The fingerprint check turns that silent corruption into a loud error:

```
ValueError: Dataset fingerprint mismatch: split file was written for
'fb52f85ebffce24a' but the current dataset hashes to 'd853fe0434f92539'.
```

---

## Project layout

```
ms-ca/
├── configs/                  # YAML configs (default + experiment overrides)
├── data/                     # .npz tensors & index CSVs (gitignored)
├── scripts/
│   ├── train.py              # entry point: parse config, wire, loop
│   └── evaluate.py           # entry point: load checkpoint, report metrics
├── src/msca/
│   ├── config.py             # OmegaConf schema, loading, validation
│   ├── data/
│   │   ├── dataset.py        # SpatialGridDataset (patch extraction)
│   │   ├── splits.py         # build_splits + index persistence/verification
│   │   ├── transforms.py     # synchronized D4 augmentation
│   │   ├── collate.py        # plain & augmenting collate functions
│   │   └── loaders.py        # DataLoader assembly
│   ├── models/
│   │   ├── encoders.py       # BaseEncoder interface + ViTPatchEncoder
│   │   ├── fusion.py         # CrossAttentionFusion
│   │   ├── heads.py          # RegressionHead
│   │   ├── msca_net.py       # MSCANet (composition root)
│   │   └── xai.py            # attention → spatial grid reshaping
│   ├── engine/
│   │   ├── trainer.py        # train_one_epoch / validate_one_epoch
│   │   ├── evaluator.py      # prediction collection + metric reporting
│   │   ├── optim.py          # AdamW param groups + warmup-cosine schedule
│   │   └── checkpoint.py     # save/load best & latest
│   └── utils/
│       ├── metrics.py        # RegressionMeter + compute_metrics
│       ├── seed.py           # global + per-worker seeding
│       ├── device.py
│       └── logging.py        # W&B wrapper with no-op fallback
└── tests/
```

---

## Data format

Each `.npz` must contain:

| Array | Shape | Notes |
|---|---|---|
| `stream_a` | `[3, H, W]` | NO2, SO2, CO |
| `stream_b` | `[4, H, W]` | Nightlight, Urban Fraction, Power Plant, Fossil Capacity |
| `label_reg` | `[H, W]` | continuous target |
| `mask` | `[H, W]` | 1 = valid cell, becomes a sample |
| `date` | scalar (optional) | falls back to the filename stem |

`label_cls` may still be present; it is **never read**. Every cell where
`mask == 1` becomes one sample, centre-cropped to `crop_size` with zero padding
at raster edges.

---

## Swapping the backbone

`MSCANet` is backbone-agnostic. Subclass `BaseEncoder`, honour the contract
(`forward` returns `[B, L, D]`; `grid_size` returns `(gh, gw)` with
`gh * gw == L`), and inject it:

```python
from msca.models import BaseEncoder, MSCANet

class MyCNNEncoder(BaseEncoder):
    ...

model = MSCANet(encoder_a_cls=MyCNNEncoder, encoder_b_cls=MyCNNEncoder)
```

Nothing in the fusion layer or the regression head changes.

---

## Explainability

The fusion layer returns raw per-head attention. Reshape it onto the spatial
grid for overlays:

```python
from msca.models import reshape_attention_to_spatial_grid

reg_out, attn = model(stream_a, stream_b, return_attention=True)
spatial = reshape_attention_to_spatial_grid(
    attn, model.encoder_b.grid_size, model.encoder_a.grid_size
)  # [B, gh_q, gw_q, gh_k, gw_k]
```

Read: *for each socio-infrastructure patch, which environmental patches did the
model attend to?*

---

## Tests

```bash
pytest
```

Covers the invariants that matter: single scalar output, 3/4 channel layout,
`label_cls` never surfacing, labels never clipped, splits disjoint and
reproducible, fingerprint mismatch detection, D4 stream synchronization, and
agreement between `RegressionMeter` and `compute_metrics`.

---

## License

MIT — see [LICENSE](LICENSE).
