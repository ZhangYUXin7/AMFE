"""Phase A AMF-Neck scaffolding.

The neck keeps channel alignment, top-down selective fusion, and bottom-up
refinement as separate modules so later phases can increase fidelity without
rewiring the model interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn import functional as F

from .common import ConvBNAct, ensure_feature_channels


@dataclass(frozen=True)
class NeckOutputChannels:
    """Named channel dimensions for the AMF neck outputs."""

    n2: int = 256
    n3: int = 256
    n4: int = 256
    n5: int = 256


class CAF(nn.Module):
    """Channel Alignment Fusion block that normalizes F2/F3/F4/F5 to 256 channels."""

    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.align = ConvBNAct(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.align(x)


class SPG(nn.Module):
    """Semantic Prior Gate producing a lightweight channel prior."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        if reduction < 1:
            raise ValueError(f"SPG reduction must be positive, received {reduction}.")
        hidden_channels = max(channels // reduction, 1)
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True)
        self.activation = nn.SiLU(inplace=True)
        self.expand = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"SPG expects a 4D tensor [B, C, H, W], received {tuple(x.shape)}.")
        if x.shape[1] != self.channels:
            raise ValueError(f"SPG expected {self.channels} channels, received {x.shape[1]}.")
        return self.output_activation(self.expand(self.activation(self.reduce(self.pool(x)))))


class DPG(nn.Module):
    """Detail Prior Gate producing a lightweight spatial prior."""

    def __init__(self, channels: int, kernels: tuple[int, int, int] = (3, 5, 7)) -> None:
        super().__init__()
        if len(kernels) != 3:
            raise ValueError(f"DPG expects exactly three kernel sizes, received {kernels}.")
        if any(kernel % 2 == 0 for kernel in kernels):
            raise ValueError(f"DPG only supports odd kernel sizes, received {kernels}.")
        self.channels = channels
        self.branches = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
                groups=channels,
                bias=False,
            )
            for kernel in kernels
        )
        self.proj = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"DPG expects a 4D tensor [B, C, H, W], received {tuple(x.shape)}.")
        if x.shape[1] != self.channels:
            raise ValueError(f"DPG expected {self.channels} channels, received {x.shape[1]}.")
        fused = sum(branch(x) for branch in self.branches)
        return self.output_activation(self.proj(fused))


class TDSF(nn.Module):
    """Top-Down Selective Fusion with lightweight channel and spatial priors."""

    def __init__(
        self,
        channels: int = 256,
        *,
        spg_reduction: int = 8,
        dpg_kernels: tuple[int, int, int] = (3, 5, 7),
    ) -> None:
        super().__init__()
        self.channels = channels
        self.spg = SPG(channels, reduction=spg_reduction)
        self.dpg = DPG(channels, kernels=dpg_kernels)
        self.refine = ConvBNAct(channels, channels, kernel_size=3)

    def forward(self, lateral: Tensor, topdown: Tensor) -> Tensor:
        if lateral.ndim != 4 or topdown.ndim != 4:
            raise ValueError("TDSF expects 4D lateral and top-down tensors.")
        if lateral.shape[1] != self.channels or topdown.shape[1] != self.channels:
            raise ValueError(
                f"TDSF expected both inputs to have {self.channels} channels, "
                f"received {lateral.shape[1]} and {topdown.shape[1]}."
            )
        if lateral.shape[-2:] != topdown.shape[-2:]:
            raise ValueError("TDSF expects spatially aligned lateral and top-down tensors.")

        fusion_context = lateral + topdown
        channel_prior = self.spg(topdown)
        spatial_prior = self.dpg(fusion_context)
        selective_topdown = topdown * channel_prior * spatial_prior
        return self.refine(lateral + selective_topdown)


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
    """Conservative AMF-Neck scaffold.

    Input feature order is explicit and fixed: (F2, F3, F4, F5).
    Output feature order is explicit and fixed: (N2, N3, N4, N5).

    Current backbone contract before the neck:
    - F2: [B, 256, H/4, W/4]
    - F3: [B, 256, H/8, W/8]
    - F4: [B, 512, H/16, W/16]
    - F5: [B, 512, H/32, W/32]
    """

    output_channels = NeckOutputChannels()

    def __init__(
        self,
        in_channels: tuple[int, int, int, int] = (256, 256, 512, 512),
        out_channels: int = 256,
        *,
        spg_reduction: int = 8,
        dpg_kernels: tuple[int, int, int] = (3, 5, 7),
    ) -> None:
        super().__init__()
        self.expected_in_channels = in_channels
        self.caf2 = CAF(in_channels[0], out_channels=out_channels)
        self.caf3 = CAF(in_channels[1], out_channels=out_channels)
        self.caf4 = CAF(in_channels[2], out_channels=out_channels)
        self.caf5 = CAF(in_channels[3], out_channels=out_channels)
        self.tdsf4 = TDSF(
            channels=out_channels,
            spg_reduction=spg_reduction,
            dpg_kernels=dpg_kernels,
        )
        self.tdsf3 = TDSF(
            channels=out_channels,
            spg_reduction=spg_reduction,
            dpg_kernels=dpg_kernels,
        )
        self.tdsf2 = TDSF(
            channels=out_channels,
            spg_reduction=spg_reduction,
            dpg_kernels=dpg_kernels,
        )
        self.burf3 = BURF(channels=out_channels)
        self.burf4 = BURF(channels=out_channels)
        self.burf5 = BURF(channels=out_channels)

    def forward(
        self, features: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if len(features) != 4:
            raise ValueError(f"AMFNeck expects exactly four backbone features, received {len(features)}.")
        f2, f3, f4, f5 = features
        ensure_feature_channels(f2, expected_channels=self.expected_in_channels[0], name="F2")
        ensure_feature_channels(f3, expected_channels=self.expected_in_channels[1], name="F3")
        ensure_feature_channels(f4, expected_channels=self.expected_in_channels[2], name="F4")
        ensure_feature_channels(f5, expected_channels=self.expected_in_channels[3], name="F5")

        l2 = self.caf2(f2)
        l3 = self.caf3(f3)
        l4 = self.caf4(f4)
        l5 = self.caf5(f5)

        td4 = self.tdsf4(l4, F.interpolate(l5, size=l4.shape[-2:], mode="nearest"))
        td3 = self.tdsf3(l3, F.interpolate(td4, size=l3.shape[-2:], mode="nearest"))
        n2 = self.tdsf2(l2, F.interpolate(td3, size=l2.shape[-2:], mode="nearest"))
        n3 = self.burf3(n2, td3)
        n4 = self.burf4(n3, td4)
        n5 = self.burf5(n4, l5)

        ensure_feature_channels(n2, expected_channels=self.output_channels.n2, name="N2")
        ensure_feature_channels(n3, expected_channels=self.output_channels.n3, name="N3")
        ensure_feature_channels(n4, expected_channels=self.output_channels.n4, name="N4")
        ensure_feature_channels(n5, expected_channels=self.output_channels.n5, name="N5")
        return n2, n3, n4, n5
