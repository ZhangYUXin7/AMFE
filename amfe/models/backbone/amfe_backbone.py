"""AMFE-Backbone Phase B implementation."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .adb import ADB
from .dps_stem import DPSStem
from .lem import LEM
from .lgcb import LGCB
from .mbfm import MBFM
from .msb import MSB
from ..common import ensure_feature_channels


@dataclass(frozen=True)
class BackboneOutputChannels:
    """Named channel dimensions for the AMFE backbone outputs."""

    f3: int = 512
    f4: int = 1024
    f5: int = 2048


class AMFEBackbone(nn.Module):
    """Asymmetric Multi-branch Feature Enhancement Backbone.

    Fixed flow:
    Input
    → LEM
    → DPS Stem
    → Shared Shallow Feature S2
    → MSB + ADB + LGCB
    → MBFM
    → F3 / F4 / F5

    Output contract:
    - F3: stride 8,  channels 512
    - F4: stride 16, channels 1024
    - F5: stride 32, channels 2048
    """

    output_channels = BackboneOutputChannels()

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.lem = LEM(in_channels=in_channels, out_channels=32)
        self.stem = DPSStem(in_channels=32, out_channels=64)
        self.msb = MSB(in_channels=64)
        self.adb = ADB(in_channels=64, d3_channels=256, d4_channels=512)
        self.lgcb = LGCB(in_channels=2048, context_channels=512)
        self.mbfm3 = MBFM(semantic_channels=512, detail_channels=256, context_channels=512, out_channels=512)
        self.mbfm4 = MBFM(semantic_channels=1024, detail_channels=512, context_channels=512, out_channels=1024)
        self.mbfm5 = MBFM(semantic_channels=2048, detail_channels=None, context_channels=512, out_channels=2048)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 4:
            raise ValueError(f"AMFEBackbone expects [B, C, H, W], received {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"AMFEBackbone expected {self.in_channels} input channels, received {x.shape[1]}."
            )
        if x.shape[-2] % 32 != 0 or x.shape[-1] % 32 != 0:
            raise ValueError("AMFEBackbone expects height and width divisible by 32.")

        input_hw = x.shape[-2:]
        enhanced = self.lem(x)  # [B, 32, H, W]
        s2 = self.stem(enhanced)  # [B, 64, H/4, W/4]
        c2, c3, c4, c5 = self.msb(s2)
        d3, d4 = self.adb(s2)
        g3, g4, g5 = self.lgcb(c3, c4, c5)

        f3 = self.mbfm3(c3, d3, g3)
        f4 = self.mbfm4(c4, d4, g4)
        f5 = self.mbfm5(c5, None, g5)

        ensure_feature_channels(f3, expected_channels=self.output_channels.f3, name="F3")
        ensure_feature_channels(f4, expected_channels=self.output_channels.f4, name="F4")
        ensure_feature_channels(f5, expected_channels=self.output_channels.f5, name="F5")

        expected_shapes = {
            "F3": (input_hw[0] // 8, input_hw[1] // 8),
            "F4": (input_hw[0] // 16, input_hw[1] // 16),
            "F5": (input_hw[0] // 32, input_hw[1] // 32),
        }
        for name, feature in (("F3", f3), ("F4", f4), ("F5", f5)):
            if feature.shape[-2:] != expected_shapes[name]:
                raise AssertionError(
                    f"{name} stride contract mismatch: expected spatial size {expected_shapes[name]}, "
                    f"received {feature.shape[-2:]}."
                )

        _ = c2  # Explicitly retained for readability and future debugging hooks.
        return f3, f4, f5
