"""Fusion modules for AMFE-Backbone."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..common import ConvBNAct


class CDG(nn.Module):
    """Context-Detail Gate.

    Input order is explicit and fixed: semantic anchor, aligned detail prior,
    aligned context prior.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, semantic: Tensor, detail: Tensor, context: Tensor) -> Tensor:
        if semantic.shape != detail.shape or semantic.shape != context.shape:
            raise ValueError(
                "CDG expects semantic, detail, and context tensors with identical shapes."
            )
        return self.gate(torch.cat([semantic, detail, context], dim=1))


class MBFM(nn.Module):
    """Multi-Branch Fusion Module.

    The semantic branch remains the anchor. Detail and context features are first
    aligned, then fused with a lightweight gate that preserves the semantic
    residual path.
    """

    def __init__(
        self,
        semantic_channels: int,
        detail_channels: int | None,
        context_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.semantic_channels = semantic_channels
        self.detail_channels = detail_channels
        self.context_channels = context_channels
        self.out_channels = out_channels
        self.detail_align = (
            ConvBNAct(detail_channels, semantic_channels, kernel_size=1)
            if detail_channels is not None
            else None
        )
        self.context_align = ConvBNAct(context_channels, semantic_channels, kernel_size=1)
        self.cdg = CDG(semantic_channels) if detail_channels is not None else None
        fusion_inputs = semantic_channels * 3 if detail_channels is not None else semantic_channels * 2
        self.fuse = ConvBNAct(fusion_inputs, out_channels, kernel_size=3, activation=False)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, semantic: Tensor, detail: Tensor | None, context: Tensor) -> Tensor:
        if semantic.ndim != 4 or context.ndim != 4:
            raise ValueError("MBFM expects 4D semantic and context tensors.")
        if semantic.shape[1] != self.semantic_channels:
            raise ValueError(
                f"MBFM expected semantic channels={self.semantic_channels}, received {semantic.shape[1]}."
            )
        if context.shape[1] != self.context_channels:
            raise ValueError(
                f"MBFM expected context channels={self.context_channels}, received {context.shape[1]}."
            )

        aligned_context = self.context_align(context)
        if aligned_context.shape[-2:] != semantic.shape[-2:]:
            raise ValueError("MBFM context feature must be spatially aligned with the semantic anchor.")

        if self.detail_align is None:
            if detail is not None:
                raise ValueError("This MBFM stage does not accept a detail branch.")
            fused = self.fuse(torch.cat([semantic, aligned_context], dim=1))
            output = self.activation(fused + semantic)
            if output.shape[1] != self.out_channels:
                raise AssertionError("MBFM F5 channel contract was violated.")
            return output

        if detail is None:
            raise ValueError("This MBFM stage requires a detail feature.")
        if detail.ndim != 4:
            raise ValueError("MBFM expects a 4D detail tensor when detail fusion is enabled.")
        if self.detail_channels is None or detail.shape[1] != self.detail_channels:
            raise ValueError(
                f"MBFM expected detail channels={self.detail_channels}, received {detail.shape[1]}."
            )

        aligned_detail = self.detail_align(detail)
        if aligned_detail.shape[-2:] != semantic.shape[-2:]:
            raise ValueError("MBFM detail feature must be spatially aligned with the semantic anchor.")
        gate = self.cdg(semantic, aligned_detail, aligned_context)
        fused = self.fuse(
            torch.cat(
                [semantic, gate * aligned_detail, (1.0 - gate) * aligned_context],
                dim=1,
            )
        )
        output = self.activation(fused + semantic)
        if output.shape[1] != self.out_channels or output.shape[-2:] != semantic.shape[-2:]:
            raise AssertionError("MBFM fusion output does not match the semantic anchor contract.")
        return output
