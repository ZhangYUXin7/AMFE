"""Main Semantic Branch (MSB) for AMFE-Backbone."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..common import ConvBNAct


class YOLOStyleBottleneck(nn.Module):
    """Lightweight bottleneck used inside the project-local C2f stages."""

    def __init__(self, channels: int, *, hidden_ratio: float = 0.5) -> None:
        super().__init__()
        hidden_channels = max(int(channels * hidden_ratio), 1)
        self.channels = channels
        self.cv1 = ConvBNAct(channels, hidden_channels, kernel_size=3)
        self.cv2 = ConvBNAct(hidden_channels, channels, kernel_size=3, activation=False)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"YOLOStyleBottleneck expects a 4D tensor [B, C, H, W], received {tuple(x.shape)}."
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"YOLOStyleBottleneck expected {self.channels} channels, received {x.shape[1]}."
            )
        return self.activation(x + self.cv2(self.cv1(x)))


class C2fBlock(nn.Module):
    """Project-local C2f block following the YOLOv8-native stage pattern."""

    def __init__(self, in_channels: int, out_channels: int, num_bottlenecks: int) -> None:
        super().__init__()
        if num_bottlenecks < 1:
            raise ValueError(f"C2fBlock requires at least one bottleneck, received {num_bottlenecks}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bottlenecks = num_bottlenecks
        hidden_channels = max(out_channels // 2, 1)

        self.reduce = ConvBNAct(in_channels, hidden_channels * 2, kernel_size=1)
        self.blocks = nn.ModuleList(
            YOLOStyleBottleneck(hidden_channels) for _ in range(num_bottlenecks)
        )
        self.fuse = ConvBNAct(hidden_channels * (2 + num_bottlenecks), out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"C2fBlock expects a 4D tensor [B, C, H, W], received {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"C2fBlock expected {self.in_channels} channels, received {x.shape[1]}.")

        parts = list(self.reduce(x).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        fused = self.fuse(torch.cat(parts, dim=1))
        if fused.shape[1] != self.out_channels or fused.shape[-2:] != x.shape[-2:]:
            raise AssertionError("C2fBlock output does not match the configured channel/spatial contract.")
        return fused


class SPPFLite(nn.Module):
    """Lightweight SPPF tail for the deepest MSB stage."""

    def __init__(self, in_channels: int, out_channels: int, pool_kernel_size: int = 5) -> None:
        super().__init__()
        if pool_kernel_size % 2 == 0:
            raise ValueError("SPPFLite requires an odd pool kernel size for same-shape pooling.")

        hidden_channels = max(in_channels // 2, 1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cv1 = ConvBNAct(in_channels, hidden_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=pool_kernel_size // 2)
        self.cv2 = ConvBNAct(hidden_channels * 4, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"SPPFLite expects a 4D tensor [B, C, H, W], received {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"SPPFLite expected {self.in_channels} channels, received {x.shape[1]}.")

        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        pooled = self.cv2(torch.cat([x, y1, y2, y3], dim=1))
        if pooled.shape[1] != self.out_channels or pooled.shape[-2:] != x.shape[-2:]:
            raise AssertionError("SPPFLite output does not match the configured contract.")
        return pooled


class MSB(nn.Module):
    """YOLOv8-native style semantic backbone used as the AMFE MSB.

    Fixed shape contract from S2:
    - S2: [B, 64, 160, 160]
    - Stage-3: Downsample Conv -> C2f -> C3 [B, 256, 80, 80]
    - Stage-4: Downsample Conv -> C2f -> C4 [B, 512, 40, 40]
    - Stage-5: Downsample Conv -> C2f -> SPPF-Lite -> C5 [B, 512, 20, 20]
    """

    SUPPORTED_VARIANT = "yolov8_s"
    SUPPORTED_VARIANTS = {"yolov8_s", "yolov8_native"}
    OUTPUT_CHANNELS = (256, 512, 512)
    STAGE_DEPTHS = (2, 2, 1)

    def __init__(self, in_channels: int = 64, *, variant: str = SUPPORTED_VARIANT) -> None:
        super().__init__()
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                "MSB only supports the normalized YOLOv8-native style variants "
                f"{sorted(self.SUPPORTED_VARIANTS)}, received '{variant}'."
            )

        self.in_channels = in_channels
        self.variant = variant

        c3_channels, c4_channels, c5_channels = self.OUTPUT_CHANNELS
        d3, d4, d5 = self.STAGE_DEPTHS

        self.downsample3 = ConvBNAct(in_channels, c3_channels, kernel_size=3, stride=2)
        self.stage3 = C2fBlock(c3_channels, c3_channels, num_bottlenecks=d3)

        self.downsample4 = ConvBNAct(c3_channels, c4_channels, kernel_size=3, stride=2)
        self.stage4 = C2fBlock(c4_channels, c4_channels, num_bottlenecks=d4)

        self.downsample5 = ConvBNAct(c4_channels, c5_channels, kernel_size=3, stride=2)
        self.stage5 = C2fBlock(c5_channels, c5_channels, num_bottlenecks=d5)
        self.sppf = SPPFLite(c5_channels, c5_channels)

    def forward(self, s2: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if s2.ndim != 4:
            raise ValueError(f"MSB expects a 4D tensor [B, C, H, W], received {tuple(s2.shape)}.")
        if s2.shape[1] != self.in_channels:
            raise ValueError(
                f"MSB expected {self.in_channels} channels from S2, received {s2.shape[1]}."
            )
        if s2.shape[-2] % 8 != 0 or s2.shape[-1] % 8 != 0:
            raise ValueError("MSB expects S2 height and width divisible by 8.")

        # Stage-3 semantic extraction: S2 [B, 64, H/4, W/4] -> C3 [B, 256, H/8, W/8]
        c3 = self.stage3(self.downsample3(s2))
        # Stage-4 semantic extraction: C3 [B, 256, H/8, W/8] -> C4 [B, 512, H/16, W/16]
        c4 = self.stage4(self.downsample4(c3))
        # Stage-5 semantic extraction: C4 [B, 512, H/16, W/16] -> C5 [B, 512, H/32, W/32]
        c5 = self.sppf(self.stage5(self.downsample5(c4)))

        expected_shapes = (
            (self.OUTPUT_CHANNELS[0], s2.shape[-2] // 2, s2.shape[-1] // 2),
            (self.OUTPUT_CHANNELS[1], s2.shape[-2] // 4, s2.shape[-1] // 4),
            (self.OUTPUT_CHANNELS[2], s2.shape[-2] // 8, s2.shape[-1] // 8),
        )
        for feature, expected, name in zip((c3, c4, c5), expected_shapes, ("C3", "C4", "C5"), strict=True):
            expected_channels, expected_h, expected_w = expected
            if feature.shape[1] != expected_channels or feature.shape[-2:] != (expected_h, expected_w):
                raise AssertionError(
                    f"{name} contract mismatch: expected [B, {expected_channels}, {expected_h}, {expected_w}], "
                    f"received {tuple(feature.shape)}."
                )
        return c3, c4, c5
