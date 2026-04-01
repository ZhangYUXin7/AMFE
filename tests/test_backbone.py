from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from amfe.models.backbone import ADB, AMFEBackbone, DEB, DPSStem, LGCB, MBFM, MSB, SRAFMBFM


@torch.no_grad()
def test_dps_stem_output_shape() -> None:
    module = DPSStem(in_channels=3, out_channels=64).eval()
    outputs = module(torch.randn(2, 3, 640, 640))
    assert outputs.shape == (2, 64, 160, 160)


@torch.no_grad()
def test_deb_output_shape() -> None:
    module = DEB(in_channels=64, out_channels=128, stride=2).eval()
    outputs = module(torch.randn(2, 64, 160, 160))
    assert outputs.shape == (2, 128, 80, 80)


@torch.no_grad()
def test_adb_output_shapes() -> None:
    module = ADB(in_channels=64, d3_channels=128, d4_channels=256).eval()
    d3, d4 = module(torch.randn(2, 64, 160, 160))

    assert d3.shape == (2, 128, 80, 80)
    assert d4.shape == (2, 256, 40, 40)


@torch.no_grad()
def test_msb_output_shapes() -> None:
    module = MSB(in_channels=64).eval()
    c3, c4, c5 = module(torch.randn(2, 64, 160, 160))

    assert c3.shape == (2, 256, 80, 80)
    assert c4.shape == (2, 512, 40, 40)
    assert c5.shape == (2, 512, 20, 20)


@torch.no_grad()
def test_lgcb_output_shapes() -> None:
    module = LGCB(in_channels=512, context_channels=256).eval()
    c3 = torch.randn(2, 256, 80, 80)
    c4 = torch.randn(2, 512, 40, 40)
    c5 = torch.randn(2, 512, 20, 20)

    g3, g4, g5 = module(c3, c4, c5)

    assert g3.shape == (2, 256, 80, 80)
    assert g4.shape == (2, 256, 40, 40)
    assert g5.shape == (2, 256, 20, 20)


@torch.no_grad()
def test_sraf_mbfm_output_shapes() -> None:
    assert MBFM is SRAFMBFM

    module = MBFM(
        semantic_channels=256,
        detail_channels=128,
        context_channels=256,
        out_channels=256,
    ).eval()
    semantic = torch.randn(2, 256, 80, 80)
    detail = torch.randn(2, 128, 80, 80)
    context = torch.randn(2, 256, 80, 80)

    fused = module(semantic, detail, context)

    assert fused.shape == (2, 256, 80, 80)


@torch.no_grad()
def test_sraf_mbfm_f5_output_shape() -> None:
    module = SRAFMBFM(
        semantic_channels=512,
        detail_channels=None,
        context_channels=256,
        out_channels=512,
    ).eval()
    semantic = torch.randn(2, 512, 20, 20)
    context = torch.randn(2, 256, 20, 20)

    fused = module(semantic, None, context)

    assert fused.shape == (2, 512, 20, 20)


@torch.no_grad()
def test_amfe_backbone_output_shapes_and_shape_trace() -> None:
    model = AMFEBackbone().eval()
    inputs = torch.randn(1, 3, 640, 640)

    f3, f4, f5 = model(inputs)

    assert f3.shape == (1, 256, 80, 80)
    assert f4.shape == (1, 512, 40, 40)
    assert f5.shape == (1, 512, 20, 20)
    assert model.last_forward_shapes == {
        "S2": (1, 64, 160, 160),
        "C3": (1, 256, 80, 80),
        "C4": (1, 512, 40, 40),
        "C5": (1, 512, 20, 20),
        "D3": (1, 128, 80, 80),
        "D4": (1, 256, 40, 40),
        "G3": (1, 256, 80, 80),
        "G4": (1, 256, 40, 40),
        "G5": (1, 256, 20, 20),
        "F3": (1, 256, 80, 80),
        "F4": (1, 512, 40, 40),
        "F5": (1, 512, 20, 20),
    }
