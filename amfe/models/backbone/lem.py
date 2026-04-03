"""Lightweight Enhancement Module (LEM) for AMFE-Backbone."""

from __future__ import annotations

from torch import Tensor, nn

from ..common import ConvBNAct


class LEM(nn.Module):
    """Lightweight low-level enhancement before the DPS stem.

    The module preserves the input resolution while expanding RGB inputs to a
    stable 32-channel shallow representation:

    Input
    -> 3x3 Conv
    -> 3x3 DWConv
    -> 1x1 PWConv
    -> Residual Add
    -> 3x3 Conv
    -> Output
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pre = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=1)
        self.dwconv = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            groups=out_channels,
        )
        self.pwconv = ConvBNAct(out_channels, out_channels, kernel_size=1, activation=False)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else ConvBNAct(in_channels, out_channels, kernel_size=1, activation=False)
        )
        self.residual_activation = nn.SiLU(inplace=True)
        self.post = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"LEM expects a 4D tensor [B, C, H, W], received {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"LEM expected {self.in_channels} input channels, received {x.shape[1]}.")

        residual = self.shortcut(x)
        enhanced = self.pre(x)
        enhanced = self.pwconv(self.dwconv(enhanced))
        enhanced = self.residual_activation(enhanced + residual)
        enhanced = self.post(enhanced)

        if enhanced.shape != (x.shape[0], self.out_channels, *x.shape[-2:]):
            raise AssertionError("LEM must preserve the input spatial size and configured channel width.")
        return enhanced


__all__ = ["LEM"]
