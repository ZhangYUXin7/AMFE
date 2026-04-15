from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from amfe.models.backbone import (
    ADB,
    AMFEBackbone,
    DEB,
    DPSStem,
    LEM,
    MSB,
    RFBLite,
    SemanticDetailFusion,
)


@torch.no_grad()
def test_lem_output_shape() -> None:
    module = LEM(in_channels=3, out_channels=32).eval()
    outputs = module(torch.randn(2, 3, 640, 640))
    assert outputs.shape == (2, 32, 640, 640)


@torch.no_grad()
def test_dps_stem_output_shape_without_lem() -> None:
    module = DPSStem(in_channels=3, out_channels=64, stem_channels=32).eval()
    outputs = module(torch.randn(2, 3, 640, 640))
    assert outputs.shape == (2, 64, 160, 160)


@torch.no_grad()
def test_deb_output_shape() -> None:
    module = DEB(in_channels=64, out_channels=128, stride=2).eval()
    outputs = module(torch.randn(2, 64, 160, 160))
    assert outputs.shape == (2, 128, 80, 80)


@torch.no_grad()
def test_deb_stride1_output_shape() -> None:
    module = DEB(in_channels=128, out_channels=128, stride=1).eval()
    outputs = module(torch.randn(2, 128, 80, 80))
    assert outputs.shape == (2, 128, 80, 80)


@torch.no_grad()
def test_adb_output_shapes() -> None:
    module = ADB(in_channels=64, d2_channels=128, d3_channels=128).eval()
    d2, d3 = module(torch.randn(2, 64, 160, 160))

    assert d2.shape == (2, 128, 160, 160)
    assert d3.shape == (2, 128, 80, 80)


@torch.no_grad()
def test_msb_output_shapes() -> None:
    module = MSB(in_channels=64).eval()
    c2, c3, c4 = module(torch.randn(2, 64, 160, 160))

    assert c2.shape == (2, 256, 160, 160)
    assert c3.shape == (2, 256, 80, 80)
    assert c4.shape == (2, 512, 40, 40)


@torch.no_grad()
def test_semantic_detail_fusion_output_shape() -> None:
    module = SemanticDetailFusion(
        semantic_channels=256,
        detail_channels=128,
        out_channels=256,
    ).eval()
    semantic = torch.randn(2, 256, 80, 80)
    detail = torch.randn(2, 128, 80, 80)

    fused = module(semantic, detail)

    assert fused.shape == (2, 256, 80, 80)


@torch.no_grad()
def test_semantic_detail_fusion_channel_mismatch_raises() -> None:
    module = SemanticDetailFusion(
        semantic_channels=256,
        detail_channels=128,
        out_channels=256,
    ).eval()

    with pytest.raises(ValueError, match="detail channels"):
        module(torch.randn(2, 256, 80, 80), torch.randn(2, 64, 80, 80))


@torch.no_grad()
def test_semantic_detail_fusion_spatial_mismatch_raises() -> None:
    module = SemanticDetailFusion(
        semantic_channels=256,
        detail_channels=128,
        out_channels=256,
    ).eval()

    with pytest.raises(ValueError, match="spatial size"):
        module(torch.randn(2, 256, 80, 80), torch.randn(2, 128, 40, 40))


@torch.no_grad()
def test_rfb_lite_output_shape() -> None:
    module = RFBLite(in_channels=512, out_channels=512).eval()
    outputs = module(torch.randn(2, 512, 40, 40))
    assert outputs.shape == (2, 512, 40, 40)


@torch.no_grad()
def test_rfb_lite_invalid_input_raises() -> None:
    module = RFBLite(in_channels=512, out_channels=512).eval()

    with pytest.raises(ValueError, match="expected 512 channels"):
        module(torch.randn(2, 256, 40, 40))


@torch.no_grad()
def test_amfe_backbone_output_shapes_and_shape_trace() -> None:
    model = AMFEBackbone(use_lem=False).eval()
    inputs = torch.randn(1, 3, 640, 640)

    outputs = model(inputs)
    assert len(outputs) == 3
    f2, f3, f4e = outputs

    assert isinstance(model.lem, torch.nn.Identity)
    assert f2.shape == (1, 256, 160, 160)
    assert f3.shape == (1, 256, 80, 80)
    assert f4e.shape == (1, 512, 40, 40)
    assert model.last_forward_shapes == {
        "LEM": (1, 3, 640, 640),
        "S2": (1, 64, 160, 160),
        "C2": (1, 256, 160, 160),
        "C3": (1, 256, 80, 80),
        "C4": (1, 512, 40, 40),
        "D2": (1, 128, 160, 160),
        "D3": (1, 128, 80, 80),
        "F2": (1, 256, 160, 160),
        "F3": (1, 256, 80, 80),
        "F4e": (1, 512, 40, 40),
    }
