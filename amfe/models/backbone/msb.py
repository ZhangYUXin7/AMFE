"""Main Semantic Branch (MSB) for AMFE-Backbone."""

from __future__ import annotations

import importlib.util
from collections import OrderedDict
from typing import Any

from torch import Tensor, nn

from ..common import ConvBNAct


class TorchvisionCompatibleBottleneck(nn.Module):
    """Fallback Bottleneck matching the torchvision ResNet-50 block contract."""

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


if importlib.util.find_spec("torchvision") is not None:
    from torchvision.models import ResNet50_Weights, resnet50
    from torchvision.models.resnet import Bottleneck
else:  # pragma: no cover - exercised only in environments without torchvision.
    ResNet50_Weights = None
    Bottleneck = TorchvisionCompatibleBottleneck
    resnet50 = None


class MSB(nn.Module):
    """Main Semantic Branch based on the ResNet-50 stage layout.

    The branch starts from the shared shallow feature S2 instead of the original
    torchvision stem and returns (C2, C3, C4, C5).
    """

    def __init__(self, in_channels: int = 64, block: type[nn.Module] = Bottleneck) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.block = block
        self._planes = 64
        self.pre_c2 = ConvBNAct(in_channels, self._planes, kernel_size=1)
        self.layer1 = self._make_layer(block, planes=64, blocks=3, stride=1)
        self.layer2 = self._make_layer(block, planes=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(block, planes=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(block, planes=512, blocks=3, stride=2)

    def _make_layer(self, block: type[nn.Module], planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample: nn.Module | None = None
        out_channels = planes * block.expansion
        if stride != 1 or self._planes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self._planes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers: list[nn.Module] = [block(self._planes, planes, stride=stride, downsample=downsample)]
        self._planes = out_channels
        for _ in range(1, blocks):
            layers.append(block(self._planes, planes))
        return nn.Sequential(*layers)

    def forward(self, s2: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if s2.ndim != 4:
            raise ValueError(f"MSB expects a 4D tensor [B, C, H, W], received {tuple(s2.shape)}.")
        if s2.shape[1] != self.in_channels:
            raise ValueError(
                f"MSB expected {self.in_channels} channels from S2, received {s2.shape[1]}."
            )

        x = self.pre_c2(s2)
        c2 = self.layer1(x)  # [B, 256, H, W] relative to S2 => stride 4 from input.
        c3 = self.layer2(c2)  # [B, 512, H/2, W/2] => stride 8 from input.
        c4 = self.layer3(c3)  # [B, 1024, H/4, W/4] => stride 16 from input.
        c5 = self.layer4(c4)  # [B, 2048, H/8, W/8] => stride 32 from input.

        expected_channels = (256, 512, 1024, 2048)
        for feature, channels, name in zip(
            (c2, c3, c4, c5),
            expected_channels,
            ("C2", "C3", "C4", "C5"),
            strict=True,
        ):
            if feature.shape[1] != channels:
                raise AssertionError(
                    f"{name} channel contract mismatch: expected {channels}, got {feature.shape[1]}."
                )
        return c2, c3, c4, c5

    def load_torchvision_resnet50_weights(self, weights: Any | None = None) -> tuple[list[str], list[str]]:
        """Load torchvision ResNet-50 weights into layer1-layer4 when available."""

        if resnet50 is None:
            raise ImportError(
                "torchvision is not installed, so ResNet-50 pretrained weights cannot be loaded."
            )

        resolved_weights = weights
        if resolved_weights is None and ResNet50_Weights is not None:
            resolved_weights = ResNet50_Weights.DEFAULT
        reference_model = resnet50(weights=resolved_weights)

        mapping = OrderedDict(
            [
                ("layer1", self.layer1),
                ("layer2", self.layer2),
                ("layer3", self.layer3),
                ("layer4", self.layer4),
            ]
        )
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
        for layer_name, target_layer in mapping.items():
            source_state = reference_model.get_submodule(layer_name).state_dict()
            result = target_layer.load_state_dict(source_state, strict=False)
            missing_keys.extend(f"{layer_name}.{key}" for key in result.missing_keys)
            unexpected_keys.extend(f"{layer_name}.{key}" for key in result.unexpected_keys)
        return missing_keys, unexpected_keys
