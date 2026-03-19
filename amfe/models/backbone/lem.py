"""Lightweight Enhancement Module (LEM) for AMFE-Backbone."""

from __future__ import annotations

from torch import Tensor, nn

from ..common import ConvBNAct, ResidualProjection


class LEM(nn.Module):
    """Lightweight Enhancement Module.

    The module follows the fixed Phase B structure:

    Input
    → Conv3×3
    → DWConv3×3
    → PWConv1×1
    → Residual Add
    → Conv3×3
    → Output

    Shape contract:
    - input: [B, in_channels, H, W]
    - output: [B, out_channels, H, W]
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_in = ConvBNAct(in_channels, out_channels, kernel_size=3)
        self.dwconv = ConvBNAct(out_channels, out_channels, kernel_size=3, groups=out_channels)
        self.pwconv = ConvBNAct(out_channels, out_channels, kernel_size=1, activation=False)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else ResidualProjection(in_channels, out_channels, stride=1)
        )
        self.activation = nn.SiLU(inplace=True)
        self.conv_out = ConvBNAct(out_channels, out_channels, kernel_size=3)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"LEM expects a 4D tensor [B, C, H, W], received {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"LEM expected {self.in_channels} input channels, received {x.shape[1]}."
            )

        residual = self.shortcut(x)
        enhanced = self.conv_in(x)
        enhanced = self.dwconv(enhanced)
        enhanced = self.pwconv(enhanced)
        enhanced = self.activation(enhanced + residual)
        enhanced = self.conv_out(enhanced)

        if enhanced.shape[1] != self.out_channels:
            raise AssertionError("LEM output channel mismatch after enhancement.")
        return enhanced
