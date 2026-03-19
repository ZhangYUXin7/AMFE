"""Minimal training launcher for the integrated AMFE detector.

Phase D intentionally focuses on a conservative, reproducible path that proves
model build, forward, loss, backward, and optimizer-step compatibility. The
launcher therefore supports two modes:

1. ``--synthetic-smoke``: one optimizer step on synthetic data.
2. Config validation for future real-data runs, without hardcoding dataset paths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amfe.models import build_model_from_yaml, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or smoke-test the AMFE detector.")
    parser.add_argument(
        "--config",
        default="configs/train/train_default.yaml",
        help="Path to the training configuration YAML.",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run one synthetic optimizer step instead of expecting a prepared dataset.",
    )
    return parser.parse_args()


def synthetic_batch(batch_size: int, image_size: int, num_classes: int, device: torch.device) -> dict[str, Tensor]:
    """Create a lightweight synthetic detection batch in Ultralytics loss format."""

    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    batch_idx = torch.arange(batch_size, device=device, dtype=torch.long)
    cls = torch.arange(batch_size, device=device, dtype=torch.long).remainder(num_classes)
    bboxes = torch.tensor(
        [[0.50, 0.50, 0.25, 0.25], [0.35, 0.35, 0.20, 0.20]],
        dtype=images.dtype,
        device=device,
    )
    if batch_size != 2:
        repeats = (batch_size + 1) // 2
        bboxes = bboxes.repeat(repeats, 1)[:batch_size]
    return {
        "img": images,
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
    }


def run_synthetic_smoke(config: dict) -> None:
    """Build the model and execute one forward/loss/backward/step cycle."""

    training_cfg = config.get("training", {})
    model = build_model_from_yaml(config["model_config"])
    device = torch.device(training_cfg.get("device", "cpu"))
    model.to(device)
    model.train()

    batch_size = int(training_cfg.get("synthetic_batch_size", 2))
    image_size = int(training_cfg.get("synthetic_image_size", 128))
    num_classes = int(training_cfg.get("synthetic_num_classes", model.config.num_classes))
    batch = synthetic_batch(batch_size=batch_size, image_size=image_size, num_classes=num_classes, device=device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(training_cfg.get("lr0", 0.01)),
        momentum=0.9,
        weight_decay=float(training_cfg.get("weight_decay", 5e-4)),
    )
    optimizer.zero_grad(set_to_none=True)
    total_loss, loss_items = model.loss(batch)
    total_loss.backward()
    optimizer.step()

    print(
        "Synthetic smoke step completed:",
        {
            "total_loss": float(total_loss.detach().cpu().item()),
            "loss_items": [float(x) for x in loss_items.detach().cpu().tolist()],
            "stride": [float(x) for x in model.stride.detach().cpu().tolist()],
        },
    )


def validate_real_run_inputs(config: dict) -> None:
    """Validate config paths for later real-data training without starting dataset-specific logic."""

    data_path = Path(config["data"])
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Training data config was not found: {data_path}. Provide a valid Ultralytics-style dataset YAML."
        )

    data_cfg = load_yaml_config(data_path)
    dataset_root = Path(str(data_cfg.get("path", "")))
    if not dataset_root.exists():
        raise FileNotFoundError(
            "Dataset root in the data config does not exist. Update configs/data/dataset_example.yaml "
            "or pass a prepared dataset YAML before attempting a real training run."
        )

    raise RuntimeError(
        "Real dataset-driven training is intentionally not launched in Phase D. "
        "Use --synthetic-smoke to verify the training path today, then point this script at a prepared "
        "Ultralytics-format dataset in a later phase."
    )


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if "model_config" not in config:
        raise KeyError("Training config must define 'model_config'.")
    if args.synthetic_smoke or config.get("training", {}).get("synthetic_smoke", False):
        run_synthetic_smoke(config)
        return
    validate_real_run_inputs(config)


if __name__ == "__main__":
    main()
