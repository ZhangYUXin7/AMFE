"""Auxiliary detail branch modules for AMFE-Backbone."""

from __future__ import annotations

from torch import Tensor, nn

from ..common import ConvBNAct, ResidualProjection


class DEB(nn.Module):
    """Detail Enhancement Block used inside the lightweight ADB.

    Fixed structure:
    - DWConv 3x3, stride=s
    - PWConv 1x1 to the target width
    - DWConv 3x3, stride=1, dilation=2
    - PWConv 1x1 to the target width
    - Residual projection when shape changes
    - SiLU activation after residual addition
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"DEB supports stride 1 or 2, received {stride}.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dwconv1 = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )
        self.pwconv1 = ConvBNAct(in_channels, out_channels, kernel_size=1)
        self.dwconv2 = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            groups=out_channels,
            dilation=2,
        )
        self.pwconv2 = ConvBNAct(out_channels, out_channels, kernel_size=1, activation=False)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else ResidualProjection(in_channels, out_channels, stride=stride)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"DEB expects a 4D tensor [B, C, H, W], received {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"DEB expected {self.in_channels} input channels, received {x.shape[1]}."
            )

        detail = self.pwconv1(self.dwconv1(x))
        detail = self.pwconv2(self.dwconv2(detail))
        shortcut = self.shortcut(x)
        if detail.shape != shortcut.shape:
            raise AssertionError(
                "DEB residual branches must be shape-aligned before addition, "
                f"received main={tuple(detail.shape)} and shortcut={tuple(shortcut.shape)}."
            )
        detail = self.activation(detail + shortcut)

        expected_hw = (x.shape[-2] // self.stride, x.shape[-1] // self.stride)
        if detail.shape[1] != self.out_channels or detail.shape[-2:] != expected_hw:
            raise AssertionError("DEB output shape does not match the configured stride/channel contract.")
        return detail


class ADB(nn.Module):
    """Auxiliary Detail Branch.

    The branch starts from the shared shallow feature S2 and only extends to
    D3/D4 so it remains a lightweight local-detail compensator instead of a
    second semantic backbone.
    """

    def __init__(self, in_channels: int = 64, d3_channels: int = 128, d4_channels: int = 256) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.deb3 = DEB(in_channels, d3_channels, stride=2)
        self.deb4 = DEB(d3_channels, d4_channels, stride=2)

    def forward(self, s2: Tensor) -> tuple[Tensor, Tensor]:
        if s2.ndim != 4:
            raise ValueError(f"ADB expects a 4D tensor [B, C, H, W], received {tuple(s2.shape)}.")
        if s2.shape[1] != self.in_channels:
            raise ValueError(
                f"ADB expected {self.in_channels} input channels from S2, received {s2.shape[1]}."
            )

        # D3 [B, 128, H/2, W/2] relative to S2 -> stride 8 from the image.
        d3 = self.deb3(s2)
        # D4 [B, 256, H/4, W/4] relative to S2 -> stride 16 from the image.
        d4 = self.deb4(d3)
        return d3, d4
