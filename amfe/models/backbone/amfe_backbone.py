"""AMFE backbone migrated to the new 3-scale semantic/detail/RFB design."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .adb import ADB
from .dps_stem import DPSStem
from .lem import LEM
from .mbfm import SemanticDetailFusion
from .msb import MSB
from .rfb import RFBLite
from ..common import ensure_feature_channels


@dataclass(frozen=True)
class BackboneOutputChannels:
    """Named channel dimensions for the AMFE backbone outputs."""

    f2: int = 256
    f3: int = 256
    f4e: int = 512

    @property
    def f4(self) -> int:
        """Legacy alias kept for compatibility with older call sites."""

        return self.f4e


class AMFEBackbone(nn.Module):
    """3-scale AMFE backbone.

    Main flow:
    Input
    -> optional LEM (default disabled via Identity)
    -> DPSStem
    -> shared shallow feature S2
    -> MSB produces C2/C3/C4
    -> ADB produces D2/D3
    -> SemanticDetailFusion produces F2/F3
    -> RFBLite enhances C4 into F4e
    """

    output_channels = BackboneOutputChannels()

    def __init__(
        self,
        in_channels: int = 3,
        *,
        msb_variant: str = MSB.SUPPORTED_VARIANT,
        use_lem: bool = False,
        lem_channels: int = 32,
        adb_detail_channels: int = 128,
        fusion_gate_reduction: int = 8,
        rfb_channels: int = 512,
        rfb_expand_ratio: float = 1.0,
        rfb_dilations: tuple[int, int] = (3, 5),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.msb_variant = msb_variant
        self.use_lem = use_lem
        self.lem_out_channels = lem_channels if use_lem else in_channels

        self.lem = (
            LEM(in_channels=in_channels, out_channels=lem_channels)
            if use_lem
            else nn.Identity()
        )
        if not use_lem:
            self.lem.in_channels = in_channels
            self.lem.out_channels = in_channels
        self.stem = DPSStem(
            in_channels=self.lem_out_channels,
            out_channels=64,
            stem_channels=lem_channels,
        )
        self.msb = MSB(in_channels=64, variant=msb_variant)
        self.adb = ADB(in_channels=64, d2_channels=adb_detail_channels, d3_channels=adb_detail_channels)
        self.fusion2 = SemanticDetailFusion(
            semantic_channels=MSB.OUTPUT_CHANNELS[0],
            detail_channels=adb_detail_channels,
            out_channels=self.output_channels.f2,
            gate_reduction=fusion_gate_reduction,
        )
        self.fusion3 = SemanticDetailFusion(
            semantic_channels=MSB.OUTPUT_CHANNELS[1],
            detail_channels=adb_detail_channels,
            out_channels=self.output_channels.f3,
            gate_reduction=fusion_gate_reduction,
        )
        self.deep_semantic_enhancer = RFBLite(
            in_channels=MSB.OUTPUT_CHANNELS[2],
            out_channels=rfb_channels,
            expand_ratio=rfb_expand_ratio,
            dilations=rfb_dilations,
        )
        self.output_channels = BackboneOutputChannels(f4e=rfb_channels)
        self.last_forward_shapes: dict[str, tuple[int, ...]] = {}

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 4:
            raise ValueError(f"AMFEBackbone expects [B, C, H, W], received {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"AMFEBackbone expected {self.in_channels} input channels, received {x.shape[1]}."
            )
        if x.shape[-2] % 16 != 0 or x.shape[-1] % 16 != 0:
            raise ValueError("AMFEBackbone expects height and width divisible by 16.")

        input_hw = x.shape[-2:]

        lem = self.lem(x)
        s2 = self.stem(lem)
        c2, c3, c4 = self.msb(s2)
        d2, d3 = self.adb(s2)

        f2 = self.fusion2(c2, d2)
        f3 = self.fusion3(c3, d3)
        f4e = self.deep_semantic_enhancer(c4)

        expected = {
            "LEM": (self.lem_out_channels, input_hw[0], input_hw[1]),
            "S2": (64, input_hw[0] // 4, input_hw[1] // 4),
            "C2": (MSB.OUTPUT_CHANNELS[0], input_hw[0] // 4, input_hw[1] // 4),
            "C3": (MSB.OUTPUT_CHANNELS[1], input_hw[0] // 8, input_hw[1] // 8),
            "C4": (MSB.OUTPUT_CHANNELS[2], input_hw[0] // 16, input_hw[1] // 16),
            "D2": (self.adb.d2_channels, input_hw[0] // 4, input_hw[1] // 4),
            "D3": (self.adb.d3_channels, input_hw[0] // 8, input_hw[1] // 8),
            "F2": (self.output_channels.f2, input_hw[0] // 4, input_hw[1] // 4),
            "F3": (self.output_channels.f3, input_hw[0] // 8, input_hw[1] // 8),
            "F4e": (self.output_channels.f4e, input_hw[0] // 16, input_hw[1] // 16),
        }
        features = {
            "LEM": lem,
            "S2": s2,
            "C2": c2,
            "C3": c3,
            "C4": c4,
            "D2": d2,
            "D3": d3,
            "F2": f2,
            "F3": f3,
            "F4e": f4e,
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
        ensure_feature_channels(f4e, expected_channels=self.output_channels.f4e, name="F4e")

        self.last_forward_shapes = {name: tuple(feature.shape) for name, feature in features.items()}
        return f2, f3, f4e
