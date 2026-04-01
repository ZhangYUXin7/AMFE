"""Lightweight Global Context Block (LGCB) for AMFE-Backbone."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..common import ConvBNAct


class LGCB(nn.Module):
    """Lightweight Global Context Block.

    Fixed structure:
    - C5 [B, 512, 20, 20]
    - 1x1 reduction to 256 channels
    - Depthwise 1x7 then depthwise 7x1
    - GAP + sigmoid gate
    - G5 [B, 256, 20, 20]
    - Nearest-neighbor upsample to G4 [B, 256, 40, 40] and G3 [B, 256, 80, 80]
    """

    def __init__(self, in_channels: int = 512, context_channels: int = 256) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.context_channels = context_channels
        self.reduce = ConvBNAct(in_channels, context_channels, kernel_size=1)
        self.dwconv_1x7 = ConvBNAct(
            context_channels,
            context_channels,
            kernel_size=(1, 7),
            groups=context_channels,
        )
        self.dwconv_7x1 = ConvBNAct(
            context_channels,
            context_channels,
            kernel_size=(7, 1),
            groups=context_channels,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, c3: Tensor, c4: Tensor, c5: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        expected = (("C3", c3, 256), ("C4", c4, 512), ("C5", c5, self.in_channels))
        for name, feature, expected_channels in expected:
            if feature.ndim != 4:
                raise ValueError(f"{name} must be a 4D tensor, received {tuple(feature.shape)}.")
            if feature.shape[1] != expected_channels:
                raise ValueError(f"{name} expected {expected_channels} channels, received {feature.shape[1]}.")

        g5 = self.reduce(c5)
        g5 = self.dwconv_1x7(g5)
        g5 = self.dwconv_7x1(g5)
        g5 = g5 * torch.sigmoid(self.pool(g5))

        g4 = F.interpolate(g5, size=c4.shape[-2:], mode="nearest")
        g3 = F.interpolate(g5, size=c3.shape[-2:], mode="nearest")

        if g5.shape[1] != self.context_channels:
            raise AssertionError("LGCB G5 channels do not match the configured context width.")
        if g4.shape != (c4.shape[0], self.context_channels, *c4.shape[-2:]):
            raise AssertionError("LGCB G4 must align with C4 spatial size and context width.")
        if g3.shape != (c3.shape[0], self.context_channels, *c3.shape[-2:]):
            raise AssertionError("LGCB G3 must align with C3 spatial size and context width.")
        return g3, g4, g5
