"""AMFE-Backbone with a restrained pre-neck redesign."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .adb import ADB
from .dps_stem import DPSStem
from .lem import LEM
from .lgcb import LGCB
from .mbfm import SRAFMBFM
from .msb import MSB
from ..common import ConvBNAct, ensure_feature_channels


@dataclass(frozen=True)
class BackboneOutputChannels:
    """Named channel dimensions for the AMFE backbone outputs."""

    f2: int = 256
    f3: int = 256
    f4: int = 512
    f5: int = 512


class AMFEBackbone(nn.Module):
    """Asymmetric Multi-branch Feature Enhancement Backbone.

    Fixed flow after the first revision:
    Input
    -> LEM
    -> WIP-DPS Stem
    -> S2
    -> independent F2 branch from S2
    -> MSB + ADB + LGCB + SRAF-MBFM
    -> F2 / F3 / F4 / F5

    Output contract:
    - F2: stride 4, channels 256
    - F3: stride 8, channels 256
    - F4: stride 16, channels 512
    - F5: stride 32, channels 512
    """

    output_channels = BackboneOutputChannels()

    def __init__(
        self,
        in_channels: int = 3,
        *,
        msb_variant: str = MSB.SUPPORTED_VARIANT,
        lem_channels: int = 32,
        mbfm_gate_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.msb_variant = msb_variant

        self.lem = LEM(in_channels=in_channels, out_channels=lem_channels)
        self.stem = DPSStem(in_channels=lem_channels, out_channels=64, stem_channels=lem_channels)
        self.f2_branch = nn.Sequential(
            ConvBNAct(64, self.output_channels.f2, kernel_size=1),
            ConvBNAct(
                self.output_channels.f2,
                self.output_channels.f2,
                kernel_size=3,
                groups=self.output_channels.f2,
            ),
        )
        self.msb = MSB(in_channels=64, variant=msb_variant)
        self.adb = ADB(in_channels=64, d3_channels=128, d4_channels=256)
        self.lgcb = LGCB(in_channels=512, context_channels=256)

        self.mbfm3 = SRAFMBFM(
            semantic_channels=256,
            detail_channels=128,
            context_channels=256,
            out_channels=256,
            gate_reduction=mbfm_gate_reduction,
        )
        self.mbfm4 = SRAFMBFM(
            semantic_channels=512,
            detail_channels=256,
            context_channels=256,
            out_channels=512,
            gate_reduction=mbfm_gate_reduction,
        )
        self.mbfm5 = SRAFMBFM(
            semantic_channels=512,
            detail_channels=None,
            context_channels=256,
            out_channels=512,
            gate_reduction=mbfm_gate_reduction,
        )
        self.last_forward_shapes: dict[str, tuple[int, ...]] = {}

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.ndim != 4:
            raise ValueError(f"AMFEBackbone expects [B, C, H, W], received {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"AMFEBackbone expected {self.in_channels} input channels, received {x.shape[1]}."
            )
        if x.shape[-2] % 32 != 0 or x.shape[-1] % 32 != 0:
            raise ValueError("AMFEBackbone expects height and width divisible by 32.")

        input_hw = x.shape[-2:]

        lem = self.lem(x)  # LEM [B, 32, H, W]
        s2 = self.stem(lem)  # S2 [B, 64, H/4, W/4]
        f2 = self.f2_branch(s2)  # F2 [B, 256, H/4, W/4]
        c3, c4, c5 = self.msb(s2)  # C3/C4/C5 = 256/512/512 channels
        d3, d4 = self.adb(s2)  # D3/D4 = 128/256 channels
        g3, g4, g5 = self.lgcb(c3, c4, c5)  # G3/G4/G5 = 256 channels

        f3 = self.mbfm3(c3, d3, g3)  # F3 [B, 256, H/8, W/8]
        f4 = self.mbfm4(c4, d4, g4)  # F4 [B, 512, H/16, W/16]
        f5 = self.mbfm5(c5, None, g5)  # F5 [B, 512, H/32, W/32]

        expected = {
            "LEM": (self.lem.out_channels, input_hw[0], input_hw[1]),
            "S2": (64, input_hw[0] // 4, input_hw[1] // 4),
            "F2": (self.output_channels.f2, input_hw[0] // 4, input_hw[1] // 4),
            "C3": (256, input_hw[0] // 8, input_hw[1] // 8),
            "C4": (512, input_hw[0] // 16, input_hw[1] // 16),
            "C5": (512, input_hw[0] // 32, input_hw[1] // 32),
            "D3": (128, input_hw[0] // 8, input_hw[1] // 8),
            "D4": (256, input_hw[0] // 16, input_hw[1] // 16),
            "G3": (256, input_hw[0] // 8, input_hw[1] // 8),
            "G4": (256, input_hw[0] // 16, input_hw[1] // 16),
            "G5": (256, input_hw[0] // 32, input_hw[1] // 32),
            "F3": (256, input_hw[0] // 8, input_hw[1] // 8),
            "F4": (512, input_hw[0] // 16, input_hw[1] // 16),
            "F5": (512, input_hw[0] // 32, input_hw[1] // 32),
        }
        features = {
            "LEM": lem,
            "S2": s2,
            "F2": f2,
            "C3": c3,
            "C4": c4,
            "C5": c5,
            "D3": d3,
            "D4": d4,
            "G3": g3,
            "G4": g4,
            "G5": g5,
            "F3": f3,
            "F4": f4,
            "F5": f5,
        }
        for name, feature in features.items():
            expected_channels, expected_h, expected_w = expected[name]
            if feature.shape[1] != expected_channels or feature.shape[-2:] != (expected_h, expected_w):
                raise AssertionError(
                    f"{name} contract mismatch: expected [B, {expected_channels}, {expected_h}, {expected_w}], "
                    f"received {tuple(feature.shape)}."
                )

        ensure_feature_channels(f2, expected_channels=self.output_channels.f2, name="F2")
        ensure_feature_channels(f3, expected_channels=self.output_channels.f3, name="F3")
        ensure_feature_channels(f4, expected_channels=self.output_channels.f4, name="F4")
        ensure_feature_channels(f5, expected_channels=self.output_channels.f5, name="F5")

        self.last_forward_shapes = {name: tuple(feature.shape) for name, feature in features.items()}
        return f2, f3, f4, f5
