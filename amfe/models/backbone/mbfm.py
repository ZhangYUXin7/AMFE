"""Lightweight channel-gated fusion modules for AMFE-Backbone."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..common import ConvBNAct


class CDG(nn.Module):
    """Legacy Context-Detail Gate retained for compatibility with earlier imports."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, semantic: Tensor, detail: Tensor, context: Tensor) -> Tensor:
        if semantic.shape != detail.shape or semantic.shape != context.shape:
            raise ValueError("CDG expects semantic, detail, and context tensors with identical shapes.")
        return self.gate(torch.cat([semantic, detail, context], dim=1))


class ChannelGate(nn.Module):
    """Lightweight squeeze-style channel gate used by the upgraded MBFM."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        if reduction < 1:
            raise ValueError(f"ChannelGate reduction must be positive, received {reduction}.")
        hidden_channels = max(channels // reduction, 1)
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True)
        self.activation = nn.SiLU(inplace=True)
        self.expand = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"ChannelGate expects a 4D tensor [B, C, H, W], received {tuple(x.shape)}.")
        if x.shape[1] != self.channels:
            raise ValueError(f"ChannelGate expected {self.channels} channels, received {x.shape[1]}.")
        return self.output_activation(self.expand(self.activation(self.reduce(self.pool(x)))))


class SRAFMBFM(nn.Module):
    """Semantic-anchor residual additive fusion with lightweight channel gating.

    Updated fusion equations:
    - F3 = DWConv(C3 + gate_d(Align(D3)) * Align(D3) + gate_g(Align(G3)) * Align(G3))
    - F4 = DWConv(C4 + gate_d(Align(D4)) * Align(D4) + gate_g(Align(G4)) * Align(G4))
    - F5 = DWConv(C5 + gate_g(Align(G5)) * Align(G5))

    The semantic branch remains the anchor. Detail and context branches only act
    as lightweight residual compensations, avoiding concat-heavy fusion and
    large post-fusion reconstruction stacks.
    """

    def __init__(
        self,
        semantic_channels: int,
        detail_channels: int | None,
        context_channels: int,
        out_channels: int,
        *,
        gate_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.semantic_channels = semantic_channels
        self.detail_channels = detail_channels
        self.context_channels = context_channels
        self.out_channels = out_channels

        self.semantic_anchor = (
            nn.Identity()
            if semantic_channels == out_channels
            else ConvBNAct(semantic_channels, out_channels, kernel_size=1, activation=False)
        )
        self.detail_align = (
            ConvBNAct(detail_channels, out_channels, kernel_size=1, activation=False)
            if detail_channels is not None
            else None
        )
        self.context_align = ConvBNAct(context_channels, out_channels, kernel_size=1, activation=False)
        self.detail_gate = (
            ChannelGate(out_channels, reduction=gate_reduction)
            if detail_channels is not None
            else None
        )
        self.context_gate = ChannelGate(out_channels, reduction=gate_reduction)
        self.reconstruct = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=3,
            groups=out_channels,
        )

    def forward(self, semantic: Tensor, detail: Tensor | None, context: Tensor) -> Tensor:
        if semantic.ndim != 4 or context.ndim != 4:
            raise ValueError("SRAFMBFM expects 4D semantic and context tensors.")
        if semantic.shape[1] != self.semantic_channels:
            raise ValueError(
                f"SRAFMBFM expected semantic channels={self.semantic_channels}, received {semantic.shape[1]}."
            )
        if context.shape[1] != self.context_channels:
            raise ValueError(
                f"SRAFMBFM expected context channels={self.context_channels}, received {context.shape[1]}."
            )
        if context.shape[-2:] != semantic.shape[-2:]:
            raise ValueError("SRAFMBFM context feature must align with the semantic spatial size.")

        fused = self.semantic_anchor(semantic)
        if self.detail_align is None:
            if detail is not None:
                raise ValueError("This SRAFMBFM stage does not accept a detail feature.")
        else:
            if detail is None:
                raise ValueError("This SRAFMBFM stage requires a detail feature.")
            if detail.ndim != 4:
                raise ValueError("SRAFMBFM expects a 4D detail tensor when detail fusion is enabled.")
            if self.detail_channels is None or detail.shape[1] != self.detail_channels:
                raise ValueError(
                    f"SRAFMBFM expected detail channels={self.detail_channels}, received {detail.shape[1]}."
                )
            if detail.shape[-2:] != semantic.shape[-2:]:
                raise ValueError("SRAFMBFM detail feature must align with the semantic spatial size.")
            aligned_detail = self.detail_align(detail)
            if self.detail_gate is None:
                raise AssertionError("detail_gate must exist when detail fusion is enabled.")
            fused = fused + self.detail_gate(aligned_detail) * aligned_detail

        aligned_context = self.context_align(context)
        fused = fused + self.context_gate(aligned_context) * aligned_context
        fused = self.reconstruct(fused)
        if fused.shape != (semantic.shape[0], self.out_channels, *semantic.shape[-2:]):
            raise AssertionError("SRAFMBFM output does not match the configured semantic anchor contract.")
        return fused


SimpleResidualAdditiveFusion = SRAFMBFM
MBFM = SRAFMBFM


__all__ = ["CDG", "MBFM", "SRAFMBFM", "SimpleResidualAdditiveFusion"]
