"""Ultralytics training bridge for the migrated 3-scale AMFE detector."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.checks import check_file
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import get_flops, get_num_params

from amfe.models import AMFEYOLODetectionModel, build_model_from_config, load_yaml_config


def _require_positive_int(name: str, value: Any) -> int:
    """Normalize a positive integer config value or fail explicitly."""

    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer, received {value!r}.")
    return normalized


def _require_non_negative_int(name: str, value: Any) -> int:
    """Normalize a non-negative integer config value or fail explicitly."""

    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer, received {value!r}.")
    return normalized


def _normalize_image_size(imgsz: int | list[int] | tuple[int, int]) -> tuple[int, int]:
    """Normalize an Ultralytics-style image size value into ``(height, width)``."""

    if isinstance(imgsz, int):
        size = _require_positive_int("fps_benchmark.imgsz", imgsz)
        return (size, size)
    if isinstance(imgsz, list):
        imgsz = tuple(imgsz)
    if isinstance(imgsz, tuple):
        if len(imgsz) == 1:
            size = _require_positive_int("fps_benchmark.imgsz[0]", imgsz[0])
            return (size, size)
        if len(imgsz) == 2:
            height = _require_positive_int("fps_benchmark.imgsz[0]", imgsz[0])
            width = _require_positive_int("fps_benchmark.imgsz[1]", imgsz[1])
            return (height, width)
    raise TypeError("fps_benchmark.imgsz must be an integer or a one/two-item list/tuple of integers.")


@dataclass(frozen=True)
class FPSBenchmarkConfig:
    """Configuration for the post-training inference throughput benchmark."""

    enabled: bool = True
    batch_size: int = 1
    warmup_iters: int = 10
    timed_iters: int = 30
    imgsz: tuple[int, int] = (640, 640)
    use_amp: bool = False

    @classmethod
    def from_mapping(
        cls,
        values: bool | Mapping[str, Any] | None,
        *,
        default_imgsz: int | list[int] | tuple[int, int],
        default_use_amp: bool,
    ) -> "FPSBenchmarkConfig":
        """Build benchmark config from YAML values while enforcing explicit validation."""

        if isinstance(values, bool):
            return cls(
                enabled=values,
                imgsz=_normalize_image_size(default_imgsz),
                use_amp=bool(default_use_amp),
            )
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise TypeError("training.fps_benchmark must be a boolean or mapping when provided.")

        return cls(
            enabled=bool(values.get("enabled", True)),
            batch_size=_require_positive_int("fps_benchmark.batch_size", values.get("batch_size", 1)),
            warmup_iters=_require_non_negative_int("fps_benchmark.warmup_iters", values.get("warmup_iters", 10)),
            timed_iters=_require_positive_int("fps_benchmark.timed_iters", values.get("timed_iters", 30)),
            imgsz=_normalize_image_size(values.get("imgsz", default_imgsz)),
            use_amp=bool(values.get("use_amp", default_use_amp)),
        )


def _format_epoch_summary(
    *,
    epoch_index: int,
    total_epochs: int,
    elapsed_seconds: float,
    losses: Mapping[str, float] | None,
) -> str:
    """Build a compact epoch-level training summary for terminal output."""

    parts = [
        f"Epoch {epoch_index}/{total_epochs}",
        f"time={elapsed_seconds:.2f}s",
    ]
    if not losses:
        parts.append("loss=n/a")
        return " | ".join(parts)

    total_loss = sum(losses.values())
    parts.append(f"loss={total_loss:.5f}")
    parts.extend(f"{name}={value:.5f}" for name, value in losses.items())
    return " | ".join(parts)


def _compute_model_complexity(model: AMFEYOLODetectionModel, imgsz: int | list[int] | tuple[int, int]) -> dict[str, float | None]:
    """Return model parameter count and FLOPs in training-friendly units."""

    return {
        "params_m": get_num_params(model) / 1e6,
        "flops_g": float(get_flops(model, imgsz)),
    }


def _format_model_complexity_summary(*, params_m: float, flops_g: float | None) -> str:
    """Build a compact one-line model complexity summary for terminal output."""

    flops_text = f"{flops_g:.3f}" if flops_g is not None and flops_g > 0.0 else "n/a"
    return f"Model complexity | params/M={params_m:.3f} | FLOPs/G={flops_text}"


def _resolve_model_device(model: nn.Module) -> torch.device:
    """Return the device that should be used for benchmarking the model."""

    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def _resolve_model_dtype(model: nn.Module) -> torch.dtype:
    """Return the dtype of the model tensors used for inference."""

    try:
        return next(model.parameters()).dtype
    except StopIteration:
        try:
            return next(model.buffers()).dtype
        except StopIteration:
            return torch.float32


def _resolve_model_in_channels(model: nn.Module) -> int:
    """Return the declared input channel count required to build a dummy batch."""

    config = getattr(model, "config", None)
    in_channels = getattr(config, "in_channels", None)
    if in_channels is None:
        raise AttributeError("Benchmark model must expose config.in_channels.")
    return int(in_channels)


def _synchronize_device(device: torch.device) -> None:
    """Synchronize asynchronous accelerators so wall time reflects actual inference latency."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _benchmark_precision_label(*, model_dtype: torch.dtype, use_amp: bool) -> str:
    """Describe the effective inference precision used by the synthetic benchmark."""

    if use_amp:
        return "amp_fp16"
    if model_dtype == torch.float16:
        return "fp16"
    if model_dtype == torch.bfloat16:
        return "bf16"
    return "fp32"


def benchmark_model_fps(model: nn.Module, config: FPSBenchmarkConfig) -> dict[str, Any]:
    """Measure inference latency and FPS using synthetic inputs on the model's device."""

    device = _resolve_model_device(model)
    model_dtype = _resolve_model_dtype(model)
    if config.use_amp and device.type != "cuda":
        raise ValueError("fps_benchmark.use_amp=true requires the benchmark model to run on CUDA.")

    in_channels = _resolve_model_in_channels(model)
    height, width = config.imgsz
    batch = torch.zeros(config.batch_size, in_channels, height, width, device=device, dtype=model_dtype)
    autocast_dtype = torch.float16 if config.use_amp else None
    was_training = model.training

    try:
        model.eval()
        with torch.inference_mode():
            for _ in range(config.warmup_iters):
                with (
                    torch.autocast(device_type="cuda", dtype=autocast_dtype)
                    if config.use_amp
                    else nullcontext()
                ):
                    model(batch)
                _synchronize_device(device)

            start = time.perf_counter()
            for _ in range(config.timed_iters):
                with (
                    torch.autocast(device_type="cuda", dtype=autocast_dtype)
                    if config.use_amp
                    else nullcontext()
                ):
                    model(batch)
            _synchronize_device(device)
            elapsed_seconds = max(time.perf_counter() - start, 1e-12)
    finally:
        if was_training:
            model.train()

    latency_ms = elapsed_seconds / config.timed_iters * 1000.0
    fps = (config.batch_size * config.timed_iters) / elapsed_seconds
    return {
        "device": str(device),
        "precision": _benchmark_precision_label(model_dtype=model_dtype, use_amp=config.use_amp),
        "batch_size": config.batch_size,
        "imgsz": config.imgsz,
        "warmup_iters": config.warmup_iters,
        "timed_iters": config.timed_iters,
        "latency_ms": latency_ms,
        "latency_ms_per_image": latency_ms / config.batch_size,
        "fps": fps,
    }


def _format_fps_benchmark_summary(result: Mapping[str, Any]) -> str:
    """Build a compact one-line summary for the post-training FPS benchmark."""

    height, width = result["imgsz"]
    return (
        "Inference benchmark"
        f" | device={result['device']}"
        f" | precision={result['precision']}"
        f" | imgsz={height}x{width}"
        f" | batch={result['batch_size']}"
        f" | warmup={result['warmup_iters']}"
        f" | iters={result['timed_iters']}"
        f" | latency/batch={result['latency_ms']:.2f}ms"
        f" | latency/img={result['latency_ms_per_image']:.2f}ms"
        f" | FPS={result['fps']:.2f}"
    )


class AMFEDetectionTrainer(DetectionTrainer):
    """Detection trainer that builds the AMFE detector instead of a stock YOLO backbone."""

    _RESUME_OVERRIDE_KEYS = (
        "imgsz",
        "batch",
        "device",
        "close_mosaic",
        "augmentations",
        "save_period",
        "workers",
        "cache",
        "patience",
        "time",
        "freeze",
        "val",
        "plots",
        "project",
        "name",
        "exist_ok",
    )

    def __init__(
        self,
        cfg: Any = None,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
        fps_benchmark: bool | Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the trainer and register epoch-level terminal logging callbacks."""

        trainer_cfg = DEFAULT_CFG if cfg is None else cfg
        super().__init__(cfg=trainer_cfg, overrides=overrides, _callbacks=_callbacks)
        self._epoch_start_time: float | None = None
        self.model_complexity: dict[str, float | None] | None = None
        self.fps_benchmark_config = FPSBenchmarkConfig.from_mapping(
            fps_benchmark,
            default_imgsz=self.args.imgsz,
            default_use_amp=bool(self.args.amp),
        )
        self.fps_benchmark_result: dict[str, Any] | None = None
        self.add_callback("on_train_start", self._on_train_start)
        self.add_callback("on_train_epoch_start", self._on_train_epoch_start)
        self.add_callback("on_train_epoch_end", self._on_train_epoch_end)
        self.add_callback("on_train_end", self._on_train_end)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: Any | None = None,
        verbose: bool = True,
    ) -> AMFEYOLODetectionModel:
        """Build the configured AMFE detector and optionally load checkpoint weights."""

        del verbose  # The AMFE builder does not currently expose a verbosity toggle.

        if cfg is None:
            raise ValueError("AMFEDetectionTrainer requires an AMFE model configuration.")
        config = self._load_model_config(cfg)
        model_cfg = config.get("model", config)
        if not isinstance(model_cfg, dict):
            raise TypeError("AMFE model configuration must deserialize to a mapping.")

        dataset_nc = int(self.data["nc"])
        model_nc = int(model_cfg.get("num_classes", dataset_nc))
        if model_nc != dataset_nc:
            raise ValueError(
                f"Model config num_classes={model_nc} does not match dataset nc={dataset_nc}. "
                "Use a matching model config for the target dataset."
            )

        dataset_channels = int(self.data.get("channels", 3))
        model_channels = int(model_cfg.get("in_channels", dataset_channels))
        if model_channels != dataset_channels:
            raise ValueError(
                f"Model config in_channels={model_channels} does not match dataset channels={dataset_channels}."
            )

        model = build_model_from_config(config)
        if weights is not None:
            self._load_model_weights(model, weights)
        return model

    def check_resume(self, overrides: dict[str, Any] | None) -> None:
        """Allow workspace-specific save paths to override checkpoint args when resuming training."""

        overrides = {} if overrides is None else dict(overrides)
        resume = self.args.resume
        if not resume:
            self.resume = resume
            return

        try:
            exists = isinstance(resume, (str, Path)) and Path(resume).exists()
            last = Path(check_file(resume) if exists else get_latest_run())

            ckpt_args = dict(load_checkpoint(last)[0].args)
            if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                ckpt_args["data"] = self.args.data

            if any(key in overrides for key in ("project", "name", "exist_ok")):
                ckpt_args["save_dir"] = None

            for key in self._RESUME_OVERRIDE_KEYS:
                if key in overrides:
                    ckpt_args[key] = overrides[key]

            resume = True
            self.args = get_cfg(ckpt_args)
            self.args.model = self.args.resume = str(last)

            if ckpt_args.get("augmentations") is not None:
                LOGGER.warning(
                    "Custom Albumentations transforms were used in the original training run but are not "
                    "being restored. To preserve custom augmentations when resuming, you need to pass the "
                    "'augmentations' parameter again to get expected results. Example: \n"
                    f"model.train(resume=True, augmentations={ckpt_args['augmentations']})"
                )
        except Exception as exc:
            raise FileNotFoundError(
                "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                "i.e. 'yolo train resume model=path/to/last.pt'"
            ) from exc

        self.resume = resume

    @staticmethod
    def _load_model_config(cfg: str | Path | dict[str, Any]) -> dict[str, Any]:
        """Load the AMFE model config from disk or normalize an in-memory mapping."""

        if isinstance(cfg, (str, Path)):
            return load_yaml_config(cfg)
        if isinstance(cfg, dict):
            return deepcopy(cfg)
        raise TypeError(f"Unsupported AMFE model config type: {type(cfg)!r}")

    @staticmethod
    def _load_model_weights(model: AMFEYOLODetectionModel, weights: Any) -> None:
        """Load checkpoint weights into the AMFE detector with an explicit failure mode."""

        source_state = weights.state_dict() if hasattr(weights, "state_dict") else weights
        if not isinstance(source_state, dict):
            raise TypeError("Checkpoint weights must expose a state_dict-compatible mapping.")
        model.load_state_dict(source_state)

    def _on_train_epoch_start(self, trainer: DetectionTrainer) -> None:
        """Capture the wall-clock start time for the current epoch."""

        del trainer  # The callback is instance-scoped; the bound trainer is sufficient.
        self._epoch_start_time = time.perf_counter()

    def _on_train_start(self, trainer: DetectionTrainer) -> None:
        """Log parameter count and FLOPs once after the trainer has fully configured the model."""

        del trainer  # The callback is instance-scoped; the bound trainer is sufficient.
        if RANK not in {-1, 0}:
            return

        try:
            if self.model_complexity is None:
                self.model_complexity = _compute_model_complexity(self.model, self.args.imgsz)
        except Exception as exc:
            LOGGER.warning(f"Unable to compute model complexity summary: {exc}")
            return

        LOGGER.info(
            _format_model_complexity_summary(
                params_m=float(self.model_complexity["params_m"]),
                flops_g=self.model_complexity["flops_g"],
            )
        )

    def _on_train_epoch_end(self, trainer: DetectionTrainer) -> None:
        """Emit a concise terminal summary for the completed epoch on the main process only."""

        del trainer  # The callback is instance-scoped; the bound trainer is sufficient.
        if RANK not in {-1, 0}:
            return

        LOGGER.info(
            _format_epoch_summary(
                epoch_index=self.epoch + 1,
                total_epochs=self.epochs,
                elapsed_seconds=self._epoch_elapsed_seconds(),
                losses=self._epoch_loss_summary(),
            )
        )

    def _on_train_end(self, trainer: DetectionTrainer) -> None:
        """Run an optional post-training inference benchmark and log the result."""

        del trainer  # The callback is instance-scoped; the bound trainer is sufficient.
        if RANK not in {-1, 0} or not self.fps_benchmark_config.enabled:
            return

        benchmark_model = self._benchmark_model_source()
        try:
            self.fps_benchmark_result = benchmark_model_fps(benchmark_model, self.fps_benchmark_config)
        except Exception as exc:
            LOGGER.warning(f"Unable to compute post-training FPS benchmark: {exc}")
            return

        LOGGER.info(_format_fps_benchmark_summary(self.fps_benchmark_result))

    def _epoch_elapsed_seconds(self) -> float:
        """Return elapsed epoch wall time, defaulting to zero if the start hook was skipped."""

        if self._epoch_start_time is None:
            return 0.0
        return max(time.perf_counter() - self._epoch_start_time, 0.0)

    def _epoch_loss_summary(self) -> dict[str, float] | None:
        """Return the current epoch mean loss items as a plain dictionary."""

        if self.tloss is None:
            return None

        labeled_losses = self.label_loss_items(loss_items=self.tloss, prefix="train")
        return {
            name.split("/", 1)[-1]: float(value)
            for name, value in labeled_losses.items()
        }

    def _benchmark_model_source(self) -> nn.Module:
        """Prefer the EMA weights for inference benchmarking when they are available."""

        ema = getattr(self, "ema", None)
        ema_model = getattr(ema, "ema", None)
        return ema_model if ema_model is not None else self.model

__all__ = [
    "AMFEDetectionTrainer",
    "FPSBenchmarkConfig",
    "_compute_model_complexity",
    "_format_epoch_summary",
    "_format_fps_benchmark_summary",
    "_format_model_complexity_summary",
    "benchmark_model_fps",
]
