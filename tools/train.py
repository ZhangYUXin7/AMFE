"""Training and smoke-test launcher for the AMFE detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amfe.data import inspect_yolo_data_config
from amfe.models import build_model_from_yaml, load_yaml_config
from amfe.training import AMFEDetectionTrainer


def parse_args() -> argparse.Namespace:
    """Parse the launcher arguments."""

    parser = argparse.ArgumentParser(description="Train or smoke-test the AMFE detector.")
    parser.add_argument(
        "--config",
        default="configs/train/train_default.yaml",
        help="Path to the training configuration YAML.",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run one synthetic optimizer step instead of a real dataset-driven training job.",
    )
    parser.add_argument(
        "--check-data-only",
        action="store_true",
        help="Validate and summarize the configured real dataset without launching training.",
    )
    return parser.parse_args()


def synthetic_batch(batch_size: int, image_size: int, num_classes: int, device: torch.device) -> dict[str, Tensor]:
    """Build a lightweight synthetic detection batch compatible with Ultralytics loss."""

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


def run_synthetic_smoke(config: dict[str, Any]) -> None:
    """Run one forward/loss/backward/optimizer-step cycle with synthetic tensors."""

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


def inspect_real_run_inputs(config: dict[str, Any]) -> Any:
    """Validate the real dataset config and print a concise summary."""

    summary = inspect_yolo_data_config(config["data"], model_config=config["model_config"])
    print(
        "Real dataset summary:",
        {
            "data_yaml": str(summary.data_yaml),
            "dataset_root": str(summary.dataset_root),
            "dataset_yaml_in_root": str(summary.dataset_yaml_in_root) if summary.dataset_yaml_in_root else None,
            "train_images_dir": str(summary.train_images_dir),
            "val_images_dir": str(summary.val_images_dir),
            "train_labels_dir": str(summary.train_labels_dir),
            "val_labels_dir": str(summary.val_labels_dir),
            "nc": summary.nc,
            "names": list(summary.names),
            "split_image_counts": summary.split_image_counts,
            "split_label_counts": summary.split_label_counts,
            "annotation_rows": summary.annotation_rows,
            "empty_label_files": summary.empty_label_files,
        },
    )
    return summary


def build_trainer_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Translate the project training YAML into Ultralytics trainer overrides."""

    training_cfg = config.get("training", {})
    model_payload = load_yaml_config(config["model_config"])
    model_cfg = model_payload.get("model", model_payload)
    loss_cfg = model_cfg.get("loss_hyperparameters", {})

    overrides: dict[str, Any] = {
        "model": str(Path(config["model_config"]).resolve()),
        "data": str(Path(config["data"]).resolve()),
        "imgsz": int(training_cfg.get("imgsz", 640)),
        "epochs": int(training_cfg.get("epochs", 300)),
        "batch": int(training_cfg.get("batch", 8)),
        "workers": int(training_cfg.get("workers", 8)),
        "optimizer": str(training_cfg.get("optimizer", "SGD")),
        "lr0": float(training_cfg.get("lr0", 0.01)),
        "weight_decay": float(training_cfg.get("weight_decay", 5e-4)),
        "device": training_cfg.get("device", "cpu"),
        "amp": bool(training_cfg.get("amp", False)),
        "save": bool(training_cfg.get("save", True)),
        "val": bool(training_cfg.get("val", True)),
        "plots": bool(training_cfg.get("plots", False)),
        "cache": bool(training_cfg.get("cache", False)),
        "project": str(Path(training_cfg.get("project", "runs")).resolve()),
        "name": str(training_cfg.get("name", "train")),
        "exist_ok": bool(training_cfg.get("exist_ok", False)),
        "patience": int(training_cfg.get("patience", 100)),
        "seed": int(training_cfg.get("seed", 0)),
        "deterministic": bool(training_cfg.get("deterministic", True)),
        "verbose": bool(training_cfg.get("verbose", True)),
        "pretrained": training_cfg.get("pretrained", False),
        "box": float(loss_cfg.get("box", 7.5)),
        "cls": float(loss_cfg.get("cls", 0.5)),
        "dfl": float(loss_cfg.get("dfl", 1.5)),
    }

    optional_keys = (
        "resume",
        "close_mosaic",
        "save_period",
        "warmup_epochs",
        "momentum",
        "single_cls",
        "cos_lr",
        "degrees",
        "translate",
        "scale",
        "fliplr",
        "flipud",
        "mosaic",
        "mixup",
        "copy_paste",
    )
    for key in optional_keys:
        if key in training_cfg:
            overrides[key] = training_cfg[key]
    return overrides


def run_real_training(config: dict[str, Any]) -> None:
    """Launch a real-data training run through the Ultralytics detection trainer."""

    inspect_real_run_inputs(config)
    training_cfg = config.get("training", {})
    trainer = AMFEDetectionTrainer(
        overrides=build_trainer_overrides(config),
        fps_benchmark=training_cfg.get("fps_benchmark"),
    )
    trainer.train()

    summary = {
        "save_dir": str(trainer.save_dir),
        "best": str(trainer.best),
        "last": str(trainer.last),
        "results_csv": str(trainer.csv),
    }
    if trainer.fps_benchmark_result is not None:
        summary["fps_benchmark"] = trainer.fps_benchmark_result

    print("Real-data training completed:", summary)


def main() -> None:
    """Load the selected config and execute the requested Phase D/E path."""

    args = parse_args()
    config = load_yaml_config(args.config)
    if "model_config" not in config:
        raise KeyError("Training config must define 'model_config'.")
    if "data" not in config:
        raise KeyError("Training config must define 'data'.")

    if args.synthetic_smoke or config.get("training", {}).get("synthetic_smoke", False):
        run_synthetic_smoke(config)
        return
    if args.check_data_only:
        inspect_real_run_inputs(config)
        return
    run_real_training(config)


if __name__ == "__main__":
    main()
