"""Phase A AMF-Neck scaffolding.

The neck keeps channel alignment, top-down selective fusion, and bottom-up
refinement as separate modules so later phases can increase fidelity without
rewiring the model interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import ConvBNAct, ensure_feature_channels


@dataclass(frozen=True)
class NeckOutputChannels:
    """Named channel dimensions for the AMF neck outputs."""

    n3: int = 256
    n4: int = 256
    n5: int = 256


class CAF(nn.Module):
    """Channel Alignment Fusion block that normalizes F3/F4/F5 to 256 channels."""

    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.align = ConvBNAct(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.align(x)


class TDSF(nn.Module):
    """Top-Down Selective Fusion.

    The Phase A version uses lightweight gating to keep top-down fusion distinct
    from the bottom-up refinement path.
    """

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=1), nn.Sigmoid())
        self.refine = ConvBNAct(channels, channels, kernel_size=3)

    def forward(self, lateral: Tensor, topdown: Tensor) -> Tensor:
        if lateral.shape[-2:] != topdown.shape[-2:]:
            raise ValueError("TDSF expects spatially aligned lateral and top-down tensors.")
        weights = self.gate(torch.cat([lateral, topdown], dim=1))
        return self.refine(lateral + weights * topdown)


class BURF(nn.Module):
    """Bottom-Up Refinement Fusion."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.downsample = ConvBNAct(channels, channels, kernel_size=3, stride=2)
        self.refine = ConvBNAct(channels, channels, kernel_size=3)

    def forward(self, lower: Tensor, higher: Tensor) -> Tensor:
        lower_down = self.downsample(lower)
        if lower_down.shape[-2:] != higher.shape[-2:]:
            raise ValueError("BURF expects aligned refinement features after downsampling.")
        return self.refine(lower_down + higher)


class AMFNeck(nn.Module):
    """Conservative Phase A scaffold for AMF-Neck.

    Input feature order is explicit and fixed: (F3, F4, F5).
    Output feature order is explicit and fixed: (N3, N4, N5).
    """

    output_channels = NeckOutputChannels()

    def __init__(self, in_channels: tuple[int, int, int] = (512, 1024, 2048), out_channels: int = 256) -> None:
        super().__init__()
        self.expected_in_channels = in_channels
        self.caf3 = CAF(in_channels[0], out_channels=out_channels)
        self.caf4 = CAF(in_channels[1], out_channels=out_channels)
        self.caf5 = CAF(in_channels[2], out_channels=out_channels)
        self.tdsf4 = TDSF(channels=out_channels)
        self.tdsf3 = TDSF(channels=out_channels)
        self.burf4 = BURF(channels=out_channels)
        self.burf5 = BURF(channels=out_channels)

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        if len(features) != 3:
            raise ValueError(f"AMFNeck expects exactly three backbone features, received {len(features)}.")
        f3, f4, f5 = features
        ensure_feature_channels(f3, expected_channels=self.expected_in_channels[0], name="F3")
        ensure_feature_channels(f4, expected_channels=self.expected_in_channels[1], name="F4")
        ensure_feature_channels(f5, expected_channels=self.expected_in_channels[2], name="F5")

        l3 = self.caf3(f3)
        l4 = self.caf4(f4)
        l5 = self.caf5(f5)

        td4 = self.tdsf4(l4, F.interpolate(l5, size=l4.shape[-2:], mode="nearest"))
        n3 = self.tdsf3(l3, F.interpolate(td4, size=l3.shape[-2:], mode="nearest"))
        n4 = self.burf4(n3, td4)
        n5 = self.burf5(n4, l5)

        ensure_feature_channels(n3, expected_channels=self.output_channels.n3, name="N3")
        ensure_feature_channels(n4, expected_channels=self.output_channels.n4, name="N4")
        ensure_feature_channels(n5, expected_channels=self.output_channels.n5, name="N5")
        return n3, n4, n5
