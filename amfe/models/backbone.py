"""Phase A AMFE-Backbone scaffolding.

The implementation in this phase keeps the architecture wiring explicit without
claiming full paper-level fidelity yet. It preserves the intended module names,
output feature hierarchy, and shape contracts so later phases can replace the
internal placeholder blocks safely.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import ConvBNAct, ResidualProjection, ensure_feature_channels


@dataclass(frozen=True)
class BackboneOutputChannels:
    """Named channel dimensions for the AMFE backbone outputs."""

    f3: int = 512
    f4: int = 1024
    f5: int = 2048


class LEM(nn.Module):
    """Lightweight Enhancement Module.

    Shape contract:
    - input: [B, 3, H, W]
    - output: [B, 32, H, W]
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 32) -> None:
        super().__init__()
        self.conv_in = ConvBNAct(in_channels, out_channels, kernel_size=3)
        self.dwconv = ConvBNAct(out_channels, out_channels, kernel_size=3, groups=out_channels)
        self.pwconv = ConvBNAct(out_channels, out_channels, kernel_size=1, activation=False)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else ResidualProjection(in_channels, out_channels, stride=1)
        )
        self.conv_out = ConvBNAct(out_channels, out_channels, kernel_size=3)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        x = self.conv_in(x)
        x = self.pwconv(self.dwconv(x))
        x = self.activation(x + residual)
        return self.conv_out(x)


class DPSStem(nn.Module):
    """Detail-Preserving Stem.

    Shape contract:
    - input: [B, 32, H, W]
    - output S2: [B, 64, H/4, W/4]
    """

    def __init__(self, in_channels: int = 32, out_channels: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNAct(in_channels, out_channels, kernel_size=3, stride=2),
            ConvBNAct(out_channels, out_channels, kernel_size=3, stride=1),
            ConvBNAct(out_channels, out_channels, kernel_size=3, stride=1),
            ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DEB(nn.Module):
    """Detail Enhancement Block used by the lightweight ADB scaffold."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.dw1 = ConvBNAct(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels)
        self.pw1 = ConvBNAct(in_channels, out_channels, kernel_size=1)
        self.dw2 = ConvBNAct(out_channels, out_channels, kernel_size=3, dilation=2, groups=out_channels)
        self.pw2 = ConvBNAct(out_channels, out_channels, kernel_size=1, activation=False)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else ResidualProjection(in_channels, out_channels, stride=stride)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        x = self.pw1(self.dw1(x))
        x = self.pw2(self.dw2(x))
        return self.activation(x + residual)


class LGCB(nn.Module):
    """Lightweight Global Context Block.

    Produces G5 directly from C5, then later phases can reuse the returned G5 to
    derive G4 and G3 by interpolation.
    """

    def __init__(self, in_channels: int = 2048, context_channels: int = 512) -> None:
        super().__init__()
        self.reduce = ConvBNAct(in_channels, context_channels, kernel_size=1)
        self.dwconv_h = ConvBNAct(
            context_channels,
            context_channels,
            kernel_size=7,
            groups=context_channels,
        )
        self.dwconv_w = ConvBNAct(
            context_channels,
            context_channels,
            kernel_size=7,
            groups=context_channels,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(context_channels, context_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.project = ConvBNAct(context_channels, context_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.reduce(x)
        x = self.dwconv_w(self.dwconv_h(x))
        gate = self.gate(self.pool(x))
        return self.project(x * gate)


class CDG(nn.Module):
    """Context-Detail Gate used by the MBFM fusion placeholder."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, anchor: Tensor, prior: Tensor) -> Tensor:
        gate = self.gate(torch.cat([anchor, prior], dim=1))
        return anchor + gate * prior


class MBFM(nn.Module):
    """Multi-Branch Fusion Module.

    This scaffold keeps the main branch as the anchor and applies lightweight
    context-detail gating before a final projection.
    """

    def __init__(self, semantic_channels: int, detail_channels: int, context_channels: int, out_channels: int) -> None:
        super().__init__()
        self.detail_adapter = None if detail_channels == 0 else ConvBNAct(detail_channels, semantic_channels, kernel_size=1)
        self.context_adapter = ConvBNAct(context_channels, semantic_channels, kernel_size=1)
        self.detail_gate = CDG(semantic_channels)
        self.context_gate = CDG(semantic_channels)
        self.project = ConvBNAct(semantic_channels, out_channels, kernel_size=3)

    def forward(self, semantic: Tensor, detail: Tensor | None, context: Tensor) -> Tensor:
        fused = semantic
        if detail is not None:
            if self.detail_adapter is None:
                raise ValueError("MBFM received detail features but no detail adapter is configured.")
            detail = self.detail_adapter(detail)
            fused = self.detail_gate(fused, detail)
        context = self.context_adapter(context)
        fused = self.context_gate(fused, context)
        return self.project(fused)


class AMFEBackbone(nn.Module):
    """Conservative Phase A scaffold for AMFE-Backbone.

    Output features follow the fixed contract:
    - F3: stride 8, channels 512
    - F4: stride 16, channels 1024
    - F5: stride 32, channels 2048
    """

    output_channels = BackboneOutputChannels()

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.lem = LEM(in_channels=in_channels, out_channels=32)
        self.stem = DPSStem(in_channels=32, out_channels=64)

        # Placeholder semantic branch that preserves the target feature hierarchy.
        self.c2 = ConvBNAct(64, 256, kernel_size=3, stride=1)
        self.c3 = nn.Sequential(ConvBNAct(256, 512, kernel_size=3, stride=2), ConvBNAct(512, 512, kernel_size=3))
        self.c4 = nn.Sequential(ConvBNAct(512, 1024, kernel_size=3, stride=2), ConvBNAct(1024, 1024, kernel_size=3))
        self.c5 = nn.Sequential(ConvBNAct(1024, 2048, kernel_size=3, stride=2), ConvBNAct(2048, 2048, kernel_size=3))

        self.deb3 = DEB(64, 256, stride=2)
        self.deb4 = DEB(256, 512, stride=2)
        self.lgcb = LGCB(in_channels=2048, context_channels=512)
        self.mbfm3 = MBFM(semantic_channels=512, detail_channels=256, context_channels=512, out_channels=512)
        self.mbfm4 = MBFM(semantic_channels=1024, detail_channels=512, context_channels=512, out_channels=1024)
        self.mbfm5 = MBFM(semantic_channels=2048, detail_channels=0, context_channels=512, out_channels=2048)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 4:
            raise ValueError(f"AMFEBackbone expects [B, C, H, W], received {tuple(x.shape)}.")
        if x.shape[-1] % 32 != 0 or x.shape[-2] % 32 != 0:
            raise ValueError("AMFEBackbone expects height and width divisible by 32.")

        x = self.lem(x)  # [B, 32, H, W]
        s2 = self.stem(x)  # [B, 64, H/4, W/4]

        c2 = self.c2(s2)  # [B, 256, H/4, W/4]
        c3 = self.c3(c2)  # [B, 512, H/8, W/8]
        c4 = self.c4(c3)  # [B, 1024, H/16, W/16]
        c5 = self.c5(c4)  # [B, 2048, H/32, W/32]

        d3 = self.deb3(s2)  # [B, 256, H/8, W/8]
        d4 = self.deb4(d3)  # [B, 512, H/16, W/16]

        g5 = self.lgcb(c5)  # [B, 512, H/32, W/32]
        g4 = F.interpolate(g5, size=c4.shape[-2:], mode="nearest")
        g3 = F.interpolate(g5, size=c3.shape[-2:], mode="nearest")

        f3 = self.mbfm3(c3, d3, g3)
        f4 = self.mbfm4(c4, d4, g4)
        f5 = self.mbfm5(c5, None, g5)

        ensure_feature_channels(f3, expected_channels=self.output_channels.f3, name="F3")
        ensure_feature_channels(f4, expected_channels=self.output_channels.f4, name="F4")
        ensure_feature_channels(f5, expected_channels=self.output_channels.f5, name="F5")
        return f3, f4, f5
