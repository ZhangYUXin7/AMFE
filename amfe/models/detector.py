"""Detector wiring for Phase D.

This module connects the implemented AMFE-Backbone and AMF-Neck to the native
Ultralytics ``Detect`` head, and exposes a minimal training-style path that
reuses Ultralytics' detection loss without redesigning the head or objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn

from amfe.ultralytics_compat import Detect, v8DetectionLoss

from .backbone import AMFEBackbone
from .neck import AMFNeck


@dataclass(frozen=True)
class LossHyperparameters:
    """Minimal subset of Ultralytics detection gains required by ``v8DetectionLoss``."""

    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5


@dataclass(frozen=True)
class AMFEModelConfig:
    """Configuration for the integrated AMFE detector."""

    num_classes: int = 80
    in_channels: int = 3
    neck_channels: int = 256
    stride_init_image_size: int = 256
    loss_hyperparameters: LossHyperparameters = LossHyperparameters()

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "AMFEModelConfig":
        """Create a config from a plain mapping.

        The accepted keys mirror the YAML files committed in ``configs/model``.
        """

        loss_values = values.get("loss_hyperparameters", {})
        if not isinstance(loss_values, dict):
            raise TypeError("loss_hyperparameters must be a mapping when provided.")
        return cls(
            num_classes=int(values.get("num_classes", cls.num_classes)),
            in_channels=int(values.get("in_channels", cls.in_channels)),
            neck_channels=int(values.get("neck_channels", cls.neck_channels)),
            stride_init_image_size=int(values.get("stride_init_image_size", cls.stride_init_image_size)),
            loss_hyperparameters=LossHyperparameters(
                box=float(loss_values.get("box", LossHyperparameters.box)),
                cls=float(loss_values.get("cls", LossHyperparameters.cls)),
                dfl=float(loss_values.get("dfl", LossHyperparameters.dfl)),
            ),
        )


class AMFEYOLODetectionModel(nn.Module):
    """AMFE detector with Ultralytics Detect head and loss-compatible model metadata.

    The model keeps the architecture separation explicit:

    Input -> AMFEBackbone -> AMFNeck -> Ultralytics Detect

    ``self.model`` is a ``ModuleList`` solely to match the small contract expected
    by Ultralytics' ``v8DetectionLoss`` helper, which looks up the final Detect
    module as ``model.model[-1]``.
    """

    def __init__(self, config: AMFEModelConfig) -> None:
        super().__init__()
        self.config = config
        self.nc = config.num_classes
        self.backbone = AMFEBackbone(in_channels=config.in_channels)
        self.neck = AMFNeck(
            in_channels=(
                self.backbone.output_channels.f3,
                self.backbone.output_channels.f4,
                self.backbone.output_channels.f5,
            ),
            out_channels=config.neck_channels,
        )
        self.detect = Detect(nc=config.num_classes, ch=(config.neck_channels,) * 3)
        self.model = nn.ModuleList([self.backbone, self.neck, self.detect])
        self.args = SimpleNamespace(
            box=config.loss_hyperparameters.box,
            cls=config.loss_hyperparameters.cls,
            dfl=config.loss_hyperparameters.dfl,
        )
        self.end2end = False
        self.task = "detect"
        self.yaml = {
            "num_classes": config.num_classes,
            "in_channels": config.in_channels,
            "neck_channels": config.neck_channels,
            "stride_init_image_size": config.stride_init_image_size,
            "loss_hyperparameters": {
                "box": config.loss_hyperparameters.box,
                "cls": config.loss_hyperparameters.cls,
                "dfl": config.loss_hyperparameters.dfl,
            },
            "channels": config.in_channels,
        }
        self.names = {index: str(index) for index in range(config.num_classes)}
        self.stride = self._initialize_detect_head()
        self.criterion: v8DetectionLoss | None = None

    def _initialize_detect_head(self) -> Tensor:
        """Infer Detect strides from a dummy feature pass and initialize head biases."""

        image_size = self.config.stride_init_image_size
        if image_size % 32 != 0:
            raise ValueError(
                "stride_init_image_size must be divisible by 32 so the backbone stride contract remains valid."
            )

        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, self.config.in_channels, image_size, image_size)
            features = self.forward_features(dummy)
        stride = torch.tensor(
            [image_size / feature.shape[-2] for feature in features],
            dtype=features[0].dtype,
            device=features[0].device,
        )
        self.detect.stride = stride
        self.detect.bias_init()
        if was_training:
            self.train()
        return stride

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return N3/N4/N5 features before the Detect head."""

        return self.neck(self.backbone(x))

    def forward(self, x: Tensor | dict[str, Tensor], *args: Any, **kwargs: Any) -> Any:
        """Match the Ultralytics ``BaseModel`` forward contract for trainer compatibility."""

        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def init_criterion(self) -> v8DetectionLoss:
        """Create the Ultralytics detection loss used by the minimal training path."""

        return v8DetectionLoss(self)

    def loss(self, batch: dict[str, Tensor], preds: Any | None = None) -> tuple[Tensor, Tensor]:
        """Compute Ultralytics-compatible detection loss for a training batch."""

        self._validate_batch(batch)
        if self.criterion is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])
        loss_vector, loss_items = self.criterion(preds, batch)
        return loss_vector.sum(), loss_items

    def predict(
        self,
        x: Tensor,
        profile: bool = False,
        visualize: bool = False,
        augment: bool = False,
        embed: list[int] | None = None,
    ) -> Any:
        """Run inference while accepting the keyword arguments Ultralytics validators pass through."""

        if profile:
            raise NotImplementedError("Per-layer profiling is not implemented for AMFEYOLODetectionModel.")
        if visualize:
            raise NotImplementedError("Feature-map visualization hooks are not implemented for AMFEYOLODetectionModel.")
        if augment:
            raise NotImplementedError("Augmented inference is not implemented for AMFEYOLODetectionModel.")
        if embed is not None:
            raise NotImplementedError("Embedding extraction is not implemented for AMFEYOLODetectionModel.")

        features = self.forward_features(x)
        return self.detect(list(features))

    def _validate_batch(self, batch: dict[str, Tensor]) -> None:
        """Validate the minimal batch contract required by Ultralytics loss."""

        required = {"img", "batch_idx", "cls", "bboxes"}
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(f"Training batch is missing required keys: {missing}")

        images = batch["img"]
        if images.ndim != 4:
            raise ValueError(f"batch['img'] must be [B, C, H, W], received {tuple(images.shape)}.")
        if images.shape[1] != self.config.in_channels:
            raise ValueError(
                f"Model expects {self.config.in_channels} input channels, received {images.shape[1]}."
            )
        batch_idx = batch["batch_idx"]
        cls = batch["cls"]
        bboxes = batch["bboxes"]
        if batch_idx.ndim != 1:
            raise ValueError("batch['batch_idx'] must be a 1D tensor of image indices.")
        if cls.ndim not in {1, 2}:
            raise ValueError("batch['cls'] must be a 1D or 2D tensor.")
        if bboxes.ndim != 2 or bboxes.shape[1] != 4:
            raise ValueError("batch['bboxes'] must be shaped [N, 4] in normalized xywh format.")
        num_targets = batch_idx.shape[0]
        if cls.shape[0] != num_targets or bboxes.shape[0] != num_targets:
            raise ValueError("batch target tensors must agree on the number of annotations.")


AMFEDetector = AMFEYOLODetectionModel


def build_amfe_detector(
    num_classes: int = 80,
    in_channels: int = 3,
    *,
    neck_channels: int = 256,
) -> AMFEYOLODetectionModel:
    """Build an integrated AMFE detector with explicit configuration wiring."""

    return AMFEYOLODetectionModel(
        AMFEModelConfig(num_classes=num_classes, in_channels=in_channels, neck_channels=neck_channels)
    )
