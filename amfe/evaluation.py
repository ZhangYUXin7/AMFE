"""Evaluation helpers for trained AMFE detection checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import select_device

from amfe.data import inspect_yolo_data_config
from amfe.models import load_yaml_config
from amfe.training import (
    FPSBenchmarkConfig,
    _compute_model_complexity,
    _format_model_complexity_summary,
    benchmark_model_fps,
)


@dataclass(frozen=True)
class EvaluationContext:
    """Resolved config inputs required to validate a trained checkpoint."""

    data_yaml: Path
    model_config: Path | None
    training: dict[str, Any]


@dataclass(frozen=True)
class ValidationResult:
    """Structured validation output for terminal and scripting use."""

    weights: Path
    data_yaml: Path
    split: str
    save_dir: Path
    metrics: dict[str, float]
    speed_ms: dict[str, float]
    checkpoint_train_metrics: dict[str, Any] | None
    model_complexity: dict[str, float | None] | None = None
    fps_benchmark: dict[str, Any] | None = None


def load_evaluation_context(training_config: str | Path) -> EvaluationContext:
    """Load evaluation defaults from a training config YAML."""

    payload = load_yaml_config(training_config)
    if "data" not in payload:
        raise KeyError("Training config must define 'data' for standalone evaluation.")

    model_config = payload.get("model_config")
    return EvaluationContext(
        data_yaml=Path(payload["data"]).resolve(),
        model_config=Path(model_config).resolve() if model_config is not None else None,
        training=dict(payload.get("training", {})),
    )


def build_validation_overrides(
    *,
    weights: str | Path,
    data_yaml: str | Path,
    training_cfg: dict[str, Any] | None = None,
    split: str = "val",
    imgsz: int | None = None,
    batch: int | None = None,
    workers: int | None = None,
    device: Any | None = None,
    half: bool | None = None,
    plots: bool = False,
    save_json: bool = False,
    project: str | Path = "runs/eval",
    name: str = "eval",
    exist_ok: bool = False,
    conf: float | None = None,
    iou: float | None = None,
    max_det: int = 300,
) -> dict[str, Any]:
    """Translate repo config and CLI overrides into Ultralytics validator args."""

    training_cfg = {} if training_cfg is None else dict(training_cfg)
    overrides: dict[str, Any] = {
        "task": "detect",
        "mode": "val",
        "model": str(Path(weights).resolve()),
        "data": str(Path(data_yaml).resolve()),
        "split": split,
        "imgsz": int(training_cfg.get("imgsz", 640) if imgsz is None else imgsz),
        "batch": int(training_cfg.get("batch", 1) if batch is None else batch),
        "workers": int(training_cfg.get("workers", 8) if workers is None else workers),
        "device": training_cfg.get("device", "cpu") if device is None else device,
        "half": bool(training_cfg.get("amp", False) if half is None else half),
        "plots": bool(plots),
        "save_json": bool(save_json),
        "project": str(Path(project).resolve()),
        "name": str(name),
        "exist_ok": bool(exist_ok),
        "max_det": int(max_det),
    }
    if conf is not None:
        overrides["conf"] = float(conf)
    if iou is not None:
        overrides["iou"] = float(iou)
    return overrides


def validate_trained_detector(
    *,
    weights: str | Path,
    data_yaml: str | Path,
    model_config: str | Path | None = None,
    validator_overrides: dict[str, Any],
    fps_benchmark: bool | dict[str, Any] | None = None,
) -> ValidationResult:
    """Run standalone validation for a trained AMFE checkpoint."""

    weights_path = Path(weights).resolve()
    data_yaml_path = Path(data_yaml).resolve()
    model_config_path = Path(model_config).resolve() if model_config is not None else None

    inspect_yolo_data_config(data_yaml_path, model_config=model_config_path)

    selected_device = select_device(str(validator_overrides["device"]), verbose=False)
    model, checkpoint = load_checkpoint(weights_path, device=selected_device)
    model_complexity = _compute_model_complexity(model, validator_overrides["imgsz"])

    validator = DetectionValidator(args=validator_overrides)
    metrics = validator(model=model)
    speed_ms = {name: float(value) for name, value in validator.speed.items()}

    fps_result = None
    if fps_benchmark is not None:
        fps_config = FPSBenchmarkConfig.from_mapping(
            fps_benchmark,
            default_imgsz=validator_overrides["imgsz"],
            default_use_amp=bool(validator_overrides["half"]),
        )
        if fps_config.enabled:
            fps_result = benchmark_model_fps(model, fps_config)

    return ValidationResult(
        weights=weights_path,
        data_yaml=data_yaml_path,
        split=str(validator_overrides["split"]),
        save_dir=validator.save_dir,
        metrics={key: float(value) for key, value in metrics.items()},
        speed_ms=speed_ms,
        checkpoint_train_metrics=checkpoint.get("train_metrics"),
        model_complexity=model_complexity,
        fps_benchmark=fps_result,
    )


def validation_result_to_dict(result: ValidationResult) -> dict[str, Any]:
    """Convert a validation result into a terminal-friendly summary dictionary."""

    summary: dict[str, Any] = {
        "weights": str(result.weights),
        "data_yaml": str(result.data_yaml),
        "split": result.split,
        "save_dir": str(result.save_dir),
        "metrics": dict(result.metrics),
        "speed_ms_per_image": dict(result.speed_ms),
    }
    if result.checkpoint_train_metrics is not None:
        summary["checkpoint_train_metrics"] = dict(result.checkpoint_train_metrics)
    if result.model_complexity is not None:
        summary["model_complexity"] = dict(result.model_complexity)
    if result.fps_benchmark is not None:
        summary["fps_benchmark"] = dict(result.fps_benchmark)
    return summary


__all__ = [
    "EvaluationContext",
    "ValidationResult",
    "build_validation_overrides",
    "load_evaluation_context",
    "_format_model_complexity_summary",
    "validate_trained_detector",
    "validation_result_to_dict",
]
