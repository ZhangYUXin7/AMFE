"""Model wrapper that connects AMFE-Backbone, AMF-Neck, and Ultralytics Detect."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from .backbone import AMFEBackbone
from .neck import AMFNeck

try:
    from ultralytics.nn.modules import Detect
except ImportError:  # pragma: no cover - exercised only in dependency-limited environments.
    Detect = None  # type: ignore[assignment]


@dataclass(frozen=True)
class AMFEModelConfig:
    """Minimal Phase A wiring config.

    This keeps only the parameters needed to scaffold the backbone, neck, and
    Detect head interface while avoiding early over-building.
    """

    num_classes: int = 80
    in_channels: int = 3
    neck_channels: int = 256


class AMFEDetector(nn.Module):
    """Wrapper module that reuses the Ultralytics Detect head without redesigning it."""

    def __init__(self, config: AMFEModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = AMFEBackbone(in_channels=config.in_channels)
        self.neck = AMFNeck(
            in_channels=(
                self.backbone.output_channels.f3,
                self.backbone.output_channels.f4,
                self.backbone.output_channels.f5,
            ),
            out_channels=config.neck_channels,
        )
        if Detect is None:
            self.detect = None
        else:
            self.detect = Detect(nc=config.num_classes, ch=(config.neck_channels,) * 3)

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return N3/N4/N5 features before the Detect head."""

        return self.neck(self.backbone(x))

    def forward(self, x: Tensor) -> Any:
        features = self.forward_features(x)
        if self.detect is None:
            raise ImportError(
                "Ultralytics is not installed, so the Detect head scaffold cannot run. "
                "Install the project dependencies from pyproject.toml to enable full forward passes."
            )
        return self.detect(list(features))


def build_amfe_detector(num_classes: int = 80, in_channels: int = 3) -> AMFEDetector:
    """Build a Phase A AMFE detector with minimal configuration wiring."""

    return AMFEDetector(AMFEModelConfig(num_classes=num_classes, in_channels=in_channels))
