"""WIP Detail-Preserving Stem (DPSStem) for AMFE-Backbone."""

from __future__ import annotations

from torch import Tensor, nn

from ..common import ConvBNAct


class DPSStem(nn.Module):
    """WIP-DPS Stem.

    Fixed 640-input shape trace:
    - Input: [B, Cin, 640, 640]
    - Conv 3x3, s=2, c=32 -> [B, 32, 320, 320]
    - DWConv 3x3, s=1, c=32 -> [B, 32, 320, 320]
    - Path-A Conv 3x3, s=2, c=64 -> [B, 64, 160, 160]
    - Path-B AvgPool 2x2, s=2 + Conv 1x1, c=64 -> [B, 64, 160, 160]
    - Add -> S2 [B, 64, 160, 160]

    The stem keeps early downsampling mild so weak texture, weak edges, and
    small-object cues are less likely to be destroyed before the semantic trunk.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64, stem_channels: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stem_channels = stem_channels

        self.conv1 = ConvBNAct(in_channels, stem_channels, kernel_size=3, stride=2)
        self.dwconv = ConvBNAct(
            stem_channels,
            stem_channels,
            kernel_size=3,
            groups=stem_channels,
        )
        self.path_a = ConvBNAct(stem_channels, out_channels, kernel_size=3, stride=2, activation=False)
        self.path_b_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.path_b_proj = ConvBNAct(stem_channels, out_channels, kernel_size=1, activation=False)
        self.out_activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"DPSStem expects a 4D tensor [B, C, H, W], received {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"DPSStem expected {self.in_channels} input channels, received {x.shape[1]}."
            )
        if x.shape[-2] % 4 != 0 or x.shape[-1] % 4 != 0:
            raise ValueError("DPSStem expects spatial dimensions divisible by 4.")

        x = self.conv1(x)
        x = self.dwconv(x)

        path_a = self.path_a(x)
        path_b = self.path_b_proj(self.path_b_pool(x))
        s2 = self.out_activation(path_a + path_b)

        expected_hw = (x.shape[-2] // 2, x.shape[-1] // 2)
        if s2.shape[1] != self.out_channels or s2.shape[-2:] != expected_hw:
            raise AssertionError(
                "DPSStem must preserve the configured stride-4 output contract for S2."
            )
        return s2
