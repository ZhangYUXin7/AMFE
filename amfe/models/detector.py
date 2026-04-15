"""Detector wiring for the migrated 3-scale AMFE model."""

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
    """Configuration for the integrated 3-scale AMFE detector."""

    num_classes: int = 80
    in_channels: int = 3
    neck_channels: int = 256
    msb_variant: str = "yolov8_s"
    use_lem: bool = False
    lem_channels: int = 32
    fusion_gate_reduction: int = 8
    tdsf_spg_reduction: int = 8
    tdsf_dpg_kernels: tuple[int, int, int] = (3, 5, 7)
    rfb_channels: int = 512
    rfb_expand_ratio: float = 1.0
    rfb_dilations: tuple[int, int] = (3, 5)
    detect_feature_strides: tuple[int, int, int] = (4, 8, 16)
    stride_init_image_size: int = 256
    loss_hyperparameters: LossHyperparameters = LossHyperparameters()

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "AMFEModelConfig":
        """Create a config from a plain mapping."""

        loss_values = values.get("loss_hyperparameters", {})
        if not isinstance(loss_values, dict):
            raise TypeError("loss_hyperparameters must be a mapping when provided.")

        dpg_kernels = values.get("tdsf_dpg_kernels", cls.tdsf_dpg_kernels)
        if isinstance(dpg_kernels, list):
            dpg_kernels = tuple(int(kernel) for kernel in dpg_kernels)
        elif isinstance(dpg_kernels, tuple):
            dpg_kernels = tuple(int(kernel) for kernel in dpg_kernels)
        if not isinstance(dpg_kernels, tuple) or len(dpg_kernels) != 3:
            raise TypeError("tdsf_dpg_kernels must be a three-item list or tuple of integers.")

        rfb_dilations = values.get("rfb_dilations", cls.rfb_dilations)
        if isinstance(rfb_dilations, list):
            rfb_dilations = tuple(int(dilation) for dilation in rfb_dilations)
        elif isinstance(rfb_dilations, tuple):
            rfb_dilations = tuple(int(dilation) for dilation in rfb_dilations)
        if not isinstance(rfb_dilations, tuple) or len(rfb_dilations) != 2:
            raise TypeError("rfb_dilations must be a two-item list or tuple of integers.")

        detect_feature_strides = values.get("detect_feature_strides", cls.detect_feature_strides)
        if isinstance(detect_feature_strides, list):
            detect_feature_strides = tuple(int(stride) for stride in detect_feature_strides)
        elif isinstance(detect_feature_strides, tuple):
            detect_feature_strides = tuple(int(stride) for stride in detect_feature_strides)
        if not isinstance(detect_feature_strides, tuple) or len(detect_feature_strides) != 3:
            raise TypeError("detect_feature_strides must be a three-item list or tuple of integers.")

        fusion_gate_reduction = int(
            values.get(
                "fusion_gate_reduction",
                values.get("mbfm_gate_reduction", cls.fusion_gate_reduction),
            )
        )

        return cls(
            num_classes=int(values.get("num_classes", cls.num_classes)),
            in_channels=int(values.get("in_channels", cls.in_channels)),
            neck_channels=int(values.get("neck_channels", cls.neck_channels)),
            msb_variant=str(values.get("msb_variant", cls.msb_variant)),
            use_lem=bool(values.get("use_lem", cls.use_lem)),
            lem_channels=int(values.get("lem_channels", cls.lem_channels)),
            fusion_gate_reduction=fusion_gate_reduction,
            tdsf_spg_reduction=int(values.get("tdsf_spg_reduction", cls.tdsf_spg_reduction)),
            tdsf_dpg_kernels=dpg_kernels,
            rfb_channels=int(values.get("rfb_channels", cls.rfb_channels)),
            rfb_expand_ratio=float(values.get("rfb_expand_ratio", cls.rfb_expand_ratio)),
            rfb_dilations=rfb_dilations,
            detect_feature_strides=detect_feature_strides,
            stride_init_image_size=int(values.get("stride_init_image_size", cls.stride_init_image_size)),
            loss_hyperparameters=LossHyperparameters(
                box=float(loss_values.get("box", LossHyperparameters.box)),
                cls=float(loss_values.get("cls", LossHyperparameters.cls)),
                dfl=float(loss_values.get("dfl", LossHyperparameters.dfl)),
            ),
        )


class AMFEYOLODetectionModel(nn.Module):
    """AMFE detector with a 3-scale Ultralytics Detect head."""

    def __init__(self, config: AMFEModelConfig) -> None:
        super().__init__()
        self.config = config
        self.nc = config.num_classes
        self.backbone = AMFEBackbone(
            in_channels=config.in_channels,
            msb_variant=config.msb_variant,
            use_lem=config.use_lem,
            lem_channels=config.lem_channels,
            fusion_gate_reduction=config.fusion_gate_reduction,
            rfb_channels=config.rfb_channels,
            rfb_expand_ratio=config.rfb_expand_ratio,
            rfb_dilations=config.rfb_dilations,
        )
        self.neck = AMFNeck(
            in_channels=(
                self.backbone.output_channels.f2,
                self.backbone.output_channels.f3,
                self.backbone.output_channels.f4e,
            ),
            out_channels=config.neck_channels,
            spg_reduction=config.tdsf_spg_reduction,
            dpg_kernels=config.tdsf_dpg_kernels,
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
            "msb_variant": config.msb_variant,
            "use_lem": config.use_lem,
            "lem_channels": config.lem_channels,
            "fusion_gate_reduction": config.fusion_gate_reduction,
            "tdsf_spg_reduction": config.tdsf_spg_reduction,
            "tdsf_dpg_kernels": list(config.tdsf_dpg_kernels),
            "rfb_channels": config.rfb_channels,
            "rfb_expand_ratio": config.rfb_expand_ratio,
            "rfb_dilations": list(config.rfb_dilations),
            "detect_feature_strides": list(config.detect_feature_strides),
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
        max_stride = max(self.config.detect_feature_strides)
        if image_size % max_stride != 0:
            raise ValueError(
                "stride_init_image_size must be divisible by the deepest detection stride so the backbone "
                "stride contract remains valid."
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
        expected_stride = torch.tensor(
            self.config.detect_feature_strides,
            dtype=features[0].dtype,
            device=features[0].device,
        )
        if not torch.equal(stride, expected_stride):
            raise AssertionError(
                f"Detect stride mismatch: expected {tuple(expected_stride.tolist())}, "
                f"received {tuple(stride.tolist())}."
            )
        self.detect.stride = stride
        self.detect.bias_init()
        if was_training:
            self.train()
        return stride

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return N2/N3/N4 features before the Detect head."""

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
    msb_variant: str = "yolov8_s",
    use_lem: bool = False,
    lem_channels: int = 32,
    fusion_gate_reduction: int = 8,
    tdsf_spg_reduction: int = 8,
    tdsf_dpg_kernels: tuple[int, int, int] = (3, 5, 7),
    rfb_channels: int = 512,
    rfb_expand_ratio: float = 1.0,
    rfb_dilations: tuple[int, int] = (3, 5),
    detect_feature_strides: tuple[int, int, int] = (4, 8, 16),
) -> AMFEYOLODetectionModel:
    """Build an integrated AMFE detector with explicit configuration wiring."""

    return AMFEYOLODetectionModel(
        AMFEModelConfig(
            num_classes=num_classes,
            in_channels=in_channels,
            neck_channels=neck_channels,
            msb_variant=msb_variant,
            use_lem=use_lem,
            lem_channels=lem_channels,
            fusion_gate_reduction=fusion_gate_reduction,
            tdsf_spg_reduction=tdsf_spg_reduction,
            tdsf_dpg_kernels=tdsf_dpg_kernels,
            rfb_channels=rfb_channels,
            rfb_expand_ratio=rfb_expand_ratio,
            rfb_dilations=rfb_dilations,
            detect_feature_strides=detect_feature_strides,
        )
    )
