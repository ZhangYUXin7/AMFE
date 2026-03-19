from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from amfe.models.backbone import AMFEBackbone, DEB, DPSStem, LEM, LGCB, MBFM


@torch.no_grad()
def test_lem_output_shape() -> None:
    module = LEM(in_channels=3, out_channels=32).eval()
    outputs = module(torch.randn(2, 3, 128, 128))
    assert outputs.shape == (2, 32, 128, 128)


@torch.no_grad()
def test_dps_stem_output_shape() -> None:
    module = DPSStem(in_channels=32, out_channels=64).eval()
    outputs = module(torch.randn(2, 32, 128, 128))
    assert outputs.shape == (2, 64, 32, 32)


@torch.no_grad()
def test_deb_output_shape() -> None:
    module = DEB(in_channels=64, out_channels=256, stride=2).eval()
    outputs = module(torch.randn(2, 64, 160, 160))
    assert outputs.shape == (2, 256, 80, 80)


@torch.no_grad()
def test_lgcb_output_shapes() -> None:
    module = LGCB(in_channels=2048, context_channels=512).eval()
    c3 = torch.randn(2, 512, 80, 80)
    c4 = torch.randn(2, 1024, 40, 40)
    c5 = torch.randn(2, 2048, 20, 20)

    g3, g4, g5 = module(c3, c4, c5)

    assert g3.shape == (2, 512, 80, 80)
    assert g4.shape == (2, 512, 40, 40)
    assert g5.shape == (2, 512, 20, 20)


@torch.no_grad()
def test_mbfm_output_shapes() -> None:
    module = MBFM(semantic_channels=512, detail_channels=256, context_channels=512, out_channels=512).eval()
    semantic = torch.randn(2, 512, 80, 80)
    detail = torch.randn(2, 256, 80, 80)
    context = torch.randn(2, 512, 80, 80)

    fused = module(semantic, detail, context)

    assert fused.shape == (2, 512, 80, 80)


@torch.no_grad()
def test_amfe_backbone_output_shapes() -> None:
    model = AMFEBackbone().eval()
    inputs = torch.randn(1, 3, 640, 640)

    f3, f4, f5 = model(inputs)

    assert f3.shape == (1, 512, 80, 80)
    assert f4.shape == (1, 1024, 40, 40)
    assert f5.shape == (1, 2048, 20, 20)
