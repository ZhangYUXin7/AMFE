from __future__ import annotations

from pathlib import Path

from amfe.evaluation import (
    ValidationResult,
    build_validation_overrides,
    load_evaluation_context,
    validate_trained_detector,
    validation_result_to_dict,
)


def test_load_evaluation_context_from_training_yaml() -> None:
    context = load_evaluation_context("configs/train/train_default.yaml")

    assert context.data_yaml == Path("configs/data/dataset_example.yaml").resolve()
    assert context.model_config == Path("configs/model/amfe_amf_yolo.yaml").resolve()
    assert context.training["imgsz"] == 640
    assert context.training["batch"] == 8


def test_build_validation_overrides_uses_training_defaults() -> None:
    overrides = build_validation_overrides(
        weights="runs/train/weights/best.pt",
        data_yaml="configs/data/dataset_example.yaml",
        training_cfg={
            "imgsz": 320,
            "batch": 4,
            "workers": 2,
            "device": "cpu",
            "amp": True,
        },
    )

    assert overrides["task"] == "detect"
    assert overrides["mode"] == "val"
    assert overrides["imgsz"] == 320
    assert overrides["batch"] == 4
    assert overrides["workers"] == 2
    assert overrides["device"] == "cpu"
    assert overrides["half"] is True
    assert overrides["save_json"] is False


def test_build_validation_overrides_accepts_explicit_overrides() -> None:
    overrides = build_validation_overrides(
        weights="runs/train/weights/best.pt",
        data_yaml="configs/data/dataset_example.yaml",
        training_cfg={"imgsz": 320, "batch": 4, "workers": 2, "device": "cpu", "amp": True},
        split="test",
        imgsz=512,
        batch=1,
        workers=0,
        device="0",
        half=False,
        plots=True,
        save_json=True,
        conf=0.2,
        iou=0.6,
        max_det=100,
    )

    assert overrides["split"] == "test"
    assert overrides["imgsz"] == 512
    assert overrides["batch"] == 1
    assert overrides["workers"] == 0
    assert overrides["device"] == "0"
    assert overrides["half"] is False
    assert overrides["plots"] is True
    assert overrides["save_json"] is True
    assert overrides["conf"] == 0.2
    assert overrides["iou"] == 0.6
    assert overrides["max_det"] == 100


def test_validate_trained_detector_uses_standard_validator(monkeypatch, tmp_path: Path) -> None:
    class DummyValidator:
        def __init__(self, *, args, **_: object) -> None:
            self.args = args
            self.save_dir = tmp_path / "runs" / "eval"
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.speed = {"preprocess": 1.0, "inference": 2.0, "loss": 0.0, "postprocess": 0.5}

        def __call__(self, *, model: object) -> dict[str, float]:
            assert model is not None
            return {"metrics/mAP50(B)": 0.5}

    monkeypatch.setattr("amfe.evaluation.inspect_yolo_data_config", lambda *args, **kwargs: object())
    monkeypatch.setattr("amfe.evaluation.select_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr("amfe.evaluation.load_checkpoint", lambda *args, **kwargs: (object(), {"train_metrics": None}))
    monkeypatch.setattr("amfe.evaluation._compute_model_complexity", lambda *args, **kwargs: {"params_m": 1.23, "flops_g": 4.56})
    monkeypatch.setattr("amfe.evaluation.DetectionValidator", DummyValidator)

    result = validate_trained_detector(
        weights=tmp_path / "best.pt",
        data_yaml=tmp_path / "dataset.yaml",
        validator_overrides={
            "device": "cpu",
            "split": "test",
            "imgsz": 640,
            "half": False,
        },
    )

    assert result.metrics["metrics/mAP50(B)"] == 0.5
    assert result.speed_ms["inference"] == 2.0
    assert result.model_complexity == {"params_m": 1.23, "flops_g": 4.56}


def test_validation_result_to_dict_preserves_metrics_and_speed() -> None:
    result = ValidationResult(
        weights=Path("runs/train/weights/best.pt").resolve(),
        data_yaml=Path("configs/data/dataset_example.yaml").resolve(),
        split="val",
        save_dir=Path("runs/eval/eval").resolve(),
        metrics={"metrics/mAP50(B)": 0.5, "metrics/recall(B)": 0.75},
        speed_ms={"preprocess": 1.0, "inference": 2.0, "loss": 0.0, "postprocess": 0.5},
        checkpoint_train_metrics={"fitness": 0.44},
        model_complexity={"params_m": 12.3, "flops_g": 45.6},
        fps_benchmark={"fps": 88.0},
    )

    summary = validation_result_to_dict(result)

    assert summary["weights"].endswith("best.pt")
    assert summary["split"] == "val"
    assert summary["metrics"]["metrics/mAP50(B)"] == 0.5
    assert summary["speed_ms_per_image"]["inference"] == 2.0
    assert summary["checkpoint_train_metrics"]["fitness"] == 0.44
    assert summary["model_complexity"]["params_m"] == 12.3
    assert summary["model_complexity"]["flops_g"] == 45.6
    assert summary["fps_benchmark"]["fps"] == 88.0
