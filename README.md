# AMFE

Conservative, research-oriented implementation of the **AMFE-Backbone + AMF-Neck + Ultralytics Detect head** detector described in `AMFE_AMF_Codex_Implementation_Spec.md`.

## Phase E status

The repository now supports two validated paths:

- Phase D synthetic smoke validation for model/loss/backward wiring.
- Phase E local real-data integration for the existing VisDrone YOLO dataset under `true_datasets/visdrone_yolo`.

Phase E does **not** claim that the full formal experiment is complete. It only establishes that the local dataset is recognized correctly, the training pipeline can run end to end on real data, validation can run on the `val` split, and logs/checkpoints are produced.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Why `opencv-python-headless`?

Some headless environments cannot import the regular OpenCV wheel because `libGL.so.1` is missing. This repository therefore depends on `opencv-python-headless` and also ships a tiny fallback compatibility shim so module-level smoke tests can still import Ultralytics in dependency-limited environments.

## Model instantiation

```python
from amfe.models import build_amfe_detector, build_model_from_yaml

model = build_amfe_detector(num_classes=3)
# or
model = build_model_from_yaml("configs/model/amfe_amf_yolo.yaml")
```

## Feature hierarchy

For an input tensor `[B, 3, 640, 640]` the detector follows:

- Backbone outputs
  - `F3`: `[B, 512, 80, 80]`
  - `F4`: `[B, 1024, 40, 40]`
  - `F5`: `[B, 2048, 20, 20]`
- Neck outputs
  - `N3`: `[B, 256, 80, 80]`
  - `N4`: `[B, 256, 40, 40]`
  - `N5`: `[B, 256, 20, 20]`
- Detect head input
  - `N3 / N4 / N5`

## Synthetic smoke path

Run a full build/forward/loss/backward/step smoke test with synthetic tensors:

```bash
.venv\Scripts\python.exe tools/train.py --config configs/train/train_default.yaml --synthetic-smoke
```

This remains the fastest conservative check for model construction, stride initialization, Ultralytics loss reuse, backward propagation, and a single optimizer step.

## Local real-data path

### Expected dataset

Phase E assumes the local VisDrone YOLO dataset already exists at:

- Absolute path: `D:\code\AMFE\true_datasets\visdrone_yolo`
- Repo-relative path: `true_datasets/visdrone_yolo`

Expected layout:

```text
true_datasets/visdrone_yolo/
├─ dataset.yaml
├─ images/
│  ├─ train/
│  └─ val/
└─ labels/
   ├─ train/
   └─ val/
```

The tracked project-local config for this dataset is [`configs/data/visdrone_local.yaml`](/D:/code/AMFE/configs/data/visdrone_local.yaml), and the matching 10-class model config is [`configs/model/amfe_amf_yolo_visdrone.yaml`](/D:/code/AMFE/configs/model/amfe_amf_yolo_visdrone.yaml).

### Validate the dataset

Before training, validate the local dataset config and label structure:

```bash
.venv\Scripts\python.exe tools/validate_dataset.py --data configs/data/visdrone_local.yaml --model-config configs/model/amfe_amf_yolo_visdrone.yaml
```

This checks:

- dataset root and split directories exist
- `train` and `val` paths resolve correctly
- image and label stems match
- class ids are contiguous and start at `0`
- every label row is `class x_center y_center width height`
- coordinates are normalized
- the AMFE model config `num_classes` and `in_channels` match the dataset config

### Run the real-data smoke training job

The committed Phase E smoke-training config is [`configs/train/train_visdrone_smoke.yaml`](/D:/code/AMFE/configs/train/train_visdrone_smoke.yaml). It uses:

- the full local dataset as-is
- `epochs: 1`
- `imgsz: 320`
- `batch: 2`
- `workers: 0`
- `amp: false` to keep the default smoke path free of Ultralytics AMP self-check downloads
- validation enabled

Run it with:

```bash
.venv\Scripts\python.exe tools/train.py --config configs/train/train_visdrone_smoke.yaml
```

Artifacts are written under `runs/phase_e/visdrone_smoke/` and include:

- `args.yaml`
- `results.csv`
- `weights/last.pt`
- `weights/best.pt`
- Ultralytics training logs for the smoke run

## Switching to a larger dataset later

Phase E keeps the real-data path explicit rather than hardcoding conversion logic into training:

1. Prepare a YOLO detect dataset root with `images/{train,val}` and `labels/{train,val}`.
2. Create or update a project-local data config like [`configs/data/visdrone_local.yaml`](/D:/code/AMFE/configs/data/visdrone_local.yaml).
3. Make the AMFE model config `num_classes` match the dataset `nc`.
4. Duplicate and tune [`configs/train/train_visdrone_smoke.yaml`](/D:/code/AMFE/configs/train/train_visdrone_smoke.yaml) for longer runs.

## Testing

Run the test suite with:

```bash
.venv\Scripts\python.exe -m pytest
```

The core module tests remain synthetic. Phase E also adds dataset-config and local real-dataset integration checks that skip cleanly if the local VisDrone dataset is not present in the current workspace.
