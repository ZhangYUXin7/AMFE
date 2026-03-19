# AMFE

Conservative, research-oriented implementation of the **AMFE-Backbone + AMF-Neck + Ultralytics Detect head** detector described in `AMFE_AMF_Codex_Implementation_Spec.md`.

## Phase D status

Phase D integrates the implemented backbone and neck into a runnable detector path that:

- builds `AMFEBackbone -> AMFNeck -> Ultralytics Detect` end to end,
- computes Ultralytics `v8DetectionLoss` without redesigning the head or loss,
- supports a synthetic forward/loss/backward/optimizer-step smoke test,
- provides reproducible YAML configuration and a minimal launcher script.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Why `opencv-python-headless`?

Some headless environments cannot import the regular OpenCV wheel because `libGL.so.1` is missing. This repository therefore depends on `opencv-python-headless` and also ships a tiny fallback compatibility shim so module-level smoke tests can still import Ultralytics in dependency-limited CI environments.

## Model instantiation

### Python API

```python
from amfe.models import build_amfe_detector, build_model_from_yaml

model = build_amfe_detector(num_classes=3)
# or
model = build_model_from_yaml("configs/model/amfe_amf_yolo.yaml")
```

### Feature hierarchy

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

## Smoke test

Run a full build/forward/loss/backward/step smoke test with synthetic tensors:

```bash
python tools/train.py --config configs/train/train_default.yaml --synthetic-smoke
```

This command is the recommended Phase D validation path because it exercises:

1. model construction,
2. stride initialization for the Ultralytics Detect head,
3. forward pass through backbone + neck + head,
4. Ultralytics `v8DetectionLoss`,
5. backward propagation,
6. one optimizer step.

## Minimal training path

Phase D does **not** add dataset download logic or large experiment orchestration. Instead it provides the minimal reproducible entry point required to start training later:

1. Update `configs/data/dataset_example.yaml` with your dataset root and splits.
2. Adjust `configs/model/amfe_amf_yolo.yaml` if `num_classes` changes.
3. Optionally tune defaults in `configs/train/train_default.yaml`.
4. Start with the synthetic smoke test above.
5. When a prepared Ultralytics-format dataset is available, reuse `tools/train.py` as the canonical launcher entry and replace the placeholder dataset YAML.

## Testing

```bash
pytest
```

The test suite uses synthetic tensors only and does not download datasets.
