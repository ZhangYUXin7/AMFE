"""Validate and summarize a local YOLO detection dataset config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amfe.data import inspect_yolo_data_config


def parse_args() -> argparse.Namespace:
    """Parse dataset validation arguments."""

    parser = argparse.ArgumentParser(description="Validate a local YOLO detection dataset config.")
    parser.add_argument(
        "--data",
        default="configs/data/visdrone_local.yaml",
        help="Path to the project-local dataset YAML.",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Optional AMFE model config to cross-check against dataset nc/channels.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate the configured dataset and print a summary."""

    args = parse_args()
    summary = inspect_yolo_data_config(args.data, model_config=args.model_config)
    print(
        "Dataset validation passed:",
        {
            "data_yaml": str(summary.data_yaml),
            "dataset_root": str(summary.dataset_root),
            "train_images_dir": str(summary.train_images_dir),
            "val_images_dir": str(summary.val_images_dir),
            "train_labels_dir": str(summary.train_labels_dir),
            "val_labels_dir": str(summary.val_labels_dir),
            "nc": summary.nc,
            "names": list(summary.names),
            "image_count": summary.image_count,
            "label_count": summary.label_count,
            "annotation_rows": summary.annotation_rows,
            "empty_label_files": summary.empty_label_files,
            "split_image_counts": summary.split_image_counts,
            "split_label_counts": summary.split_label_counts,
        },
    )


if __name__ == "__main__":
    main()
