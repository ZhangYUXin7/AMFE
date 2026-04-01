from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from amfe.data import inspect_yolo_data_config
from amfe.training import AMFEDetectionTrainer


def _write_stub_dataset(dataset_root: Path) -> None:
    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "images" / split / f"{split}_sample.jpg").write_bytes(b"stub")
        (dataset_root / "labels" / split / f"{split}_sample.txt").write_text("0 0.5 0.5 0.25 0.25", encoding="utf-8")


def _write_data_yaml(path: Path, dataset_root: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "path": dataset_root.resolve().as_posix(),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "object"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_model_yaml(path: Path, *, num_classes: int = 1) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "num_classes": num_classes,
                    "in_channels": 3,
                    "neck_channels": 256,
                    "msb_variant": "yolov8_s",
                    "stride_init_image_size": 256,
                    "loss_hyperparameters": {"box": 7.5, "cls": 0.5, "dfl": 1.5},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_inspect_yolo_data_config_validates_project_local_yaml(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_stub_dataset(dataset_root)
    _write_data_yaml(dataset_root / "dataset.yaml", dataset_root)

    project_data_yaml = tmp_path / "configs" / "data" / "local.yaml"
    project_data_yaml.parent.mkdir(parents=True, exist_ok=True)
    _write_data_yaml(project_data_yaml, dataset_root)

    model_yaml = tmp_path / "configs" / "model" / "amfe.yaml"
    model_yaml.parent.mkdir(parents=True, exist_ok=True)
    _write_model_yaml(model_yaml)

    summary = inspect_yolo_data_config(project_data_yaml, model_config=model_yaml)

    assert summary.dataset_root == dataset_root.resolve()
    assert summary.nc == 1
    assert summary.names == ("object",)
    assert summary.image_count == 2
    assert summary.label_count == 2
    assert summary.annotation_rows == 2
    assert summary.split_image_counts == {"train": 1, "val": 1}
    assert summary.split_label_counts == {"train": 1, "val": 1}


def test_inspect_yolo_data_config_rejects_model_class_mismatch(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_stub_dataset(dataset_root)
    _write_data_yaml(dataset_root / "dataset.yaml", dataset_root)

    project_data_yaml = tmp_path / "configs" / "data" / "local.yaml"
    project_data_yaml.parent.mkdir(parents=True, exist_ok=True)
    _write_data_yaml(project_data_yaml, dataset_root)

    model_yaml = tmp_path / "configs" / "model" / "amfe.yaml"
    model_yaml.parent.mkdir(parents=True, exist_ok=True)
    _write_model_yaml(model_yaml, num_classes=2)

    with pytest.raises(ValueError, match="num_classes=2 does not match dataset nc=1"):
        inspect_yolo_data_config(project_data_yaml, model_config=model_yaml)


def test_local_visdrone_real_dataset_path_and_dataloader_when_available(tmp_path: Path) -> None:
    data_yaml = Path("configs/data/visdrone_local.yaml")
    model_yaml = Path("configs/model/amfe_amf_yolo_visdrone.yaml")
    if not data_yaml.is_file() or not model_yaml.is_file():
        pytest.skip("Local Phase E configs are not present.")

    summary = inspect_yolo_data_config(data_yaml, model_config=model_yaml)
    if not summary.dataset_root.exists():
        pytest.skip("Local VisDrone dataset is not available in this workspace.")

    assert summary.split_image_counts["train"] > 0
    assert summary.split_image_counts["val"] > 0
    assert summary.split_image_counts == summary.split_label_counts
    assert summary.nc == 10

    trainer = AMFEDetectionTrainer(
        overrides={
            "model": str(model_yaml.resolve()),
            "data": str(data_yaml.resolve()),
            "imgsz": 320,
            "epochs": 1,
            "batch": 2,
            "workers": 0,
            "device": "cpu",
            "amp": False,
            "save": False,
            "val": False,
            "plots": False,
            "project": str(tmp_path.resolve()),
            "name": "local_real_data_loader",
            "exist_ok": True,
        }
    )
    trainer.model = trainer.get_model(cfg=str(model_yaml.resolve()), verbose=False)
    batch = next(iter(trainer.get_dataloader(trainer.data["train"], batch_size=2, rank=-1, mode="train")))

    assert {"img", "cls", "bboxes", "batch_idx", "im_file"}.issubset(batch)
    assert batch["img"].shape[0] == 2
