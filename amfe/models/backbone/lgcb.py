"""Lightweight Global Context Block (LGCB) for AMFE-Backbone."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from ..common import ConvBNAct


class LGCB(nn.Module):
    """Lightweight Global Context Block.

    Fixed structure:
    C5
    → Conv1×1(2048→512)
    → DWConv1×7
    → DWConv7×1
    → GlobalAvgPool
    → Conv1×1
    → Sigmoid Gate
    → Channel Reweight
    → Conv1×1
    → G5

    The module returns (G3, G4, G5) using explicit interpolation from the deep
    semantic feature C5 to the spatial sizes of C3 and C4.
    """

    def __init__(self, in_channels: int = 2048, context_channels: int = 512) -> None:
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
        self.gate = nn.Sequential(
            nn.Conv2d(context_channels, context_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.project = ConvBNAct(context_channels, context_channels, kernel_size=1)

    def forward(self, c3: Tensor, c4: Tensor, c5: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        for name, feature, expected_channels in (("C3", c3, 512), ("C4", c4, 1024), ("C5", c5, self.in_channels)):
            if feature.ndim != 4:
                raise ValueError(f"{name} must be a 4D tensor, received {tuple(feature.shape)}.")
            if feature.shape[1] != expected_channels:
                raise ValueError(f"{name} expected {expected_channels} channels, received {feature.shape[1]}.")

        g5 = self.reduce(c5)
        g5 = self.dwconv_1x7(g5)
        g5 = self.dwconv_7x1(g5)
        gate = self.gate(self.pool(g5))
        g5 = self.project(g5 * gate)
        g4 = F.interpolate(g5, size=c4.shape[-2:], mode="nearest")
        g3 = F.interpolate(g5, size=c3.shape[-2:], mode="nearest")

        if g5.shape[1] != self.context_channels:
            raise AssertionError("LGCB G5 channels do not match the configured context width.")
        if g4.shape[-2:] != c4.shape[-2:] or g3.shape[-2:] != c3.shape[-2:]:
            raise AssertionError("LGCB interpolation must align G4/G3 with C4/C3 spatial sizes.")
        return g3, g4, g5
