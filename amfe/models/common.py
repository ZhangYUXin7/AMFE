"""Shared building blocks used by the Phase A AMFE model scaffolding."""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

_DEFAULT_EPS: Final[float] = 1e-5
_DEFAULT_MOMENTUM: Final[float] = 0.1


class ConvBNAct(nn.Sequential):
    """Convolution followed by batch normalization and SiLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        activation: bool = True,
    ) -> None:
        padding = ((kernel_size - 1) // 2) * dilation
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, eps=_DEFAULT_EPS, momentum=_DEFAULT_MOMENTUM),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class ResidualProjection(nn.Module):
    """Projects a residual branch when spatial stride or channels do not match."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.proj = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=stride, activation=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution used for lightweight placeholder blocks."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = ConvBNAct(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels)
        self.pointwise = ConvBNAct(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


def ensure_feature_channels(feature: Tensor, *, expected_channels: int, name: str) -> None:
    """Raise a clear error when a feature tensor has an unexpected channel count."""

    if feature.ndim != 4:
        raise ValueError(f"{name} must be a 4D tensor, but received shape {tuple(feature.shape)}.")
    if feature.shape[1] != expected_channels:
        raise ValueError(
            f"{name} expected {expected_channels} channels, but received {feature.shape[1]}."
        )
