"""Detail-Preserving Stem (DPSStem) for AMFE-Backbone."""

from __future__ import annotations

from torch import Tensor, nn

from ..common import ConvBNAct


class DPSStem(nn.Module):
    """Detail-Preserving Stem.

    This module conservatively replaces the original ResNet 7×7 + max-pool stem.

    Shape contract:
    - input: [B, in_channels, H, W]
    - output S2: [B, out_channels, H/4, W/4]
    """

    def __init__(self, in_channels: int = 32, out_channels: int = 64) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            ConvBNAct(in_channels, out_channels, kernel_size=3, stride=2),
            ConvBNAct(out_channels, out_channels, kernel_size=3),
            ConvBNAct(out_channels, out_channels, kernel_size=3),
            ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"DPSStem expects a 4D tensor [B, C, H, W], received {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"DPSStem expected {self.in_channels} input channels, received {x.shape[1]}."
            )
        if x.shape[-2] % 4 != 0 or x.shape[-1] % 4 != 0:
            raise ValueError("DPSStem expects spatial dimensions divisible by 4.")

        s2 = self.layers(x)
        expected_hw = (x.shape[-2] // 4, x.shape[-1] // 4)
        if s2.shape[1] != self.out_channels or s2.shape[-2:] != expected_hw:
            raise AssertionError("DPSStem must preserve the configured stride-4 output contract.")
        return s2
