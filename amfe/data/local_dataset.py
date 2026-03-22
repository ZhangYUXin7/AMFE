"""Local YOLO detection dataset inspection and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics.data.utils import check_det_dataset

from amfe.models import load_yaml_config

from .visdrone_conversion import validate_yolo_dataset


@dataclass(frozen=True)
class YOLODataConfigSummary:
    """Resolved dataset information used by the local real-data Phase E path."""

    data_yaml: Path
    dataset_root: Path
    dataset_yaml_in_root: Path | None
    train_images_dir: Path
    val_images_dir: Path
    train_labels_dir: Path
    val_labels_dir: Path
    nc: int
    names: tuple[str, ...]
    image_count: int
    label_count: int
    empty_label_files: int
    annotation_rows: int
    split_image_counts: dict[str, int]
    split_label_counts: dict[str, int]


def inspect_yolo_data_config(
    data_yaml: str | Path,
    *,
    model_config: str | Path | None = None,
) -> YOLODataConfigSummary:
    """Resolve a YOLO dataset config, validate the dataset, and optionally cross-check the model config."""

    data_yaml_path = Path(data_yaml).resolve()
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"Dataset config was not found: {data_yaml_path}")

    resolved = check_det_dataset(str(data_yaml_path), autodownload=False)
    dataset_root = Path(resolved["path"]).resolve()
    train_images_dir = Path(str(resolved["train"])).resolve()
    val_images_dir = Path(str(resolved["val"])).resolve()
    train_labels_dir = dataset_root / "labels" / "train"
    val_labels_dir = dataset_root / "labels" / "val"

    for directory in (train_images_dir, val_images_dir, train_labels_dir, val_labels_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Expected dataset directory was not found: {directory}")

    names = _normalize_names(resolved["names"])
    dataset_yaml_in_root = dataset_root / "dataset.yaml"
    if dataset_yaml_in_root.is_file():
        root_names = _normalize_names(load_yaml_config(dataset_yaml_in_root).get("names", {}))
        if root_names != names:
            raise ValueError(
                f"Class names in {data_yaml_path} do not match {dataset_yaml_in_root}. "
                "Update the project-local config so the training path stays explicit and reproducible."
            )

    if model_config is not None:
        _validate_model_config_against_dataset(model_config, num_classes=len(names), channels=int(resolved["channels"]))

    stats = validate_yolo_dataset(dataset_root)
    split_label_counts = {
        "train": len(list(train_labels_dir.glob("*.txt"))),
        "val": len(list(val_labels_dir.glob("*.txt"))),
    }
    return YOLODataConfigSummary(
        data_yaml=data_yaml_path,
        dataset_root=dataset_root,
        dataset_yaml_in_root=dataset_yaml_in_root if dataset_yaml_in_root.is_file() else None,
        train_images_dir=train_images_dir,
        val_images_dir=val_images_dir,
        train_labels_dir=train_labels_dir,
        val_labels_dir=val_labels_dir,
        nc=len(names),
        names=names,
        image_count=stats.image_count,
        label_count=stats.label_count,
        empty_label_files=stats.empty_label_files,
        annotation_rows=stats.annotation_rows,
        split_image_counts=stats.split_image_counts,
        split_label_counts=split_label_counts,
    )


def _normalize_names(names: Any) -> tuple[str, ...]:
    """Normalize list-or-mapping class names into an ordered tuple."""

    if isinstance(names, dict):
        normalized = {int(key): str(value) for key, value in names.items()}
        expected = list(range(len(normalized)))
        if sorted(normalized) != expected:
            raise ValueError(f"Class names must use contiguous ids starting at 0, found: {sorted(normalized)}")
        return tuple(normalized[index] for index in expected)
    if isinstance(names, list):
        return tuple(str(value) for value in names)
    raise TypeError("Dataset names must be defined as a list or id->name mapping.")


def _validate_model_config_against_dataset(model_config: str | Path, *, num_classes: int, channels: int) -> None:
    """Ensure the AMFE model config agrees with the target dataset before training starts."""

    payload = load_yaml_config(model_config)
    model_cfg = payload.get("model", payload)
    if not isinstance(model_cfg, dict):
        raise TypeError("Model config must deserialize to a mapping.")

    configured_classes = int(model_cfg.get("num_classes", num_classes))
    if configured_classes != num_classes:
        raise ValueError(
            f"Model config num_classes={configured_classes} does not match dataset nc={num_classes}: {model_config}"
        )

    configured_channels = int(model_cfg.get("in_channels", channels))
    if configured_channels != channels:
        raise ValueError(
            f"Model config in_channels={configured_channels} does not match dataset channels={channels}: {model_config}"
        )


__all__ = ["YOLODataConfigSummary", "inspect_yolo_data_config"]
