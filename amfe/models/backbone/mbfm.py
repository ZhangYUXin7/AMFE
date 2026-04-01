"""Simple residual additive fusion modules for AMFE-Backbone."""

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


class SRAFMBFM(nn.Module):
    """Simple Residual Additive Fusion MBFM.

    Fixed first-pass fusion equations:
    - F3 = C3 + alpha3 * Align(D3) + beta3 * Align(G3)
    - F4 = C4 + alpha4 * Align(D4) + beta4 * Align(G4)
    - F5 = C5 + beta5 * Align(G5)

    The residual weights are per-scale learnable scalars constrained with a
    sigmoid. The fusion intentionally avoids concat-heavy fusion and extra 3x3
    reconstruction convolutions.
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

        self.alpha = nn.Parameter(torch.zeros(1)) if detail_channels is not None else None
        self.beta = nn.Parameter(torch.zeros(1))

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
            fused = fused + torch.sigmoid(self.alpha) * self.detail_align(detail)

        fused = fused + torch.sigmoid(self.beta) * self.context_align(context)
        if fused.shape != (semantic.shape[0], self.out_channels, *semantic.shape[-2:]):
            raise AssertionError("SRAFMBFM output does not match the configured semantic anchor contract.")
        return fused


SimpleResidualAdditiveFusion = SRAFMBFM
MBFM = SRAFMBFM


__all__ = ["CDG", "MBFM", "SRAFMBFM", "SimpleResidualAdditiveFusion"]
