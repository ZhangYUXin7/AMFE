"""Receptive field enhancement blocks for the deepest semantic feature."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..common import ConvBNAct


class RFBLite(nn.Module):
    """Lightweight receptive field block used as the deep semantic enhancer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        expand_ratio: float = 1.0,
        dilations: tuple[int, int] = (3, 5),
        use_refine: bool = True,
    ) -> None:
        super().__init__()
        if expand_ratio <= 0:
            raise ValueError(f"RFBLite expand_ratio must be positive, received {expand_ratio}.")
        if len(dilations) != 2:
            raise ValueError(f"RFBLite expects exactly two dilation values, received {dilations}.")
        if any(dilation < 1 for dilation in dilations):
            raise ValueError(f"RFBLite dilations must be positive, received {dilations}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.dilations = dilations

        branch_channels = max(int(out_channels * expand_ratio / 4), 1)
        self.branch1 = ConvBNAct(in_channels, branch_channels, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, kernel_size=1),
            ConvBNAct(branch_channels, branch_channels, kernel_size=3),
        )
        self.branch3 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, kernel_size=1),
            ConvBNAct(branch_channels, branch_channels, kernel_size=3, dilation=dilations[0]),
        )
        self.branch4 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, kernel_size=1),
            ConvBNAct(branch_channels, branch_channels, kernel_size=3, dilation=dilations[1]),
        )
        self.fuse = ConvBNAct(branch_channels * 4, out_channels, kernel_size=1, activation=False)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else ConvBNAct(in_channels, out_channels, kernel_size=1, activation=False)
        )
        self.activation = nn.SiLU(inplace=True)
        self.refine = (
            ConvBNAct(out_channels, out_channels, kernel_size=3, groups=out_channels)
            if use_refine
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"RFBLite expects a 4D tensor [B, C, H, W], received {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"RFBLite expected {self.in_channels} channels, received {x.shape[1]}.")

        fused = self.fuse(
            torch.cat(
                [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
                dim=1,
            )
        )
        output = self.refine(self.activation(fused + self.shortcut(x)))
        if output.shape != (x.shape[0], self.out_channels, *x.shape[-2:]):
            raise AssertionError("RFBLite output does not match the configured contract.")
        return output


DeepSemanticEnhancer = RFBLite


__all__ = ["DeepSemanticEnhancer", "RFBLite"]
