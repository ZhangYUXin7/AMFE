"""Ultralytics training bridge for the AMFE detector."""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.checks import check_file
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import get_flops, get_num_params

from amfe.models import AMFEYOLODetectionModel, build_model_from_config, load_yaml_config


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

    def __init__(self, cfg: Any = None, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None) -> None:
        """Initialize the trainer and register epoch-level terminal logging callbacks."""

        trainer_cfg = DEFAULT_CFG if cfg is None else cfg
        super().__init__(cfg=trainer_cfg, overrides=overrides, _callbacks=_callbacks)
        self._epoch_start_time: float | None = None
        self.model_complexity: dict[str, float | None] | None = None
        self.add_callback("on_train_start", self._on_train_start)
        self.add_callback("on_train_epoch_start", self._on_train_epoch_start)
        self.add_callback("on_train_epoch_end", self._on_train_epoch_end)

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

__all__ = [
    "AMFEDetectionTrainer",
    "_compute_model_complexity",
    "_format_epoch_summary",
    "_format_model_complexity_summary",
]
