from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from amfe.models.neck import AMFNeck, BURF, DPG, SPG, TDSF


@torch.no_grad()
def test_spg_output_shape() -> None:
    module = SPG(channels=256).eval()
    outputs = module(torch.randn(2, 256, 32, 32))
    assert outputs.shape == (2, 256, 1, 1)


@torch.no_grad()
def test_dpg_output_shape() -> None:
    module = DPG(channels=256).eval()
    outputs = module(torch.randn(2, 256, 32, 32))
    assert outputs.shape == (2, 1, 32, 32)


@torch.no_grad()
def test_tdsf_output_shape() -> None:
    module = TDSF(channels=256).eval()
    lateral = torch.randn(2, 256, 32, 32)
    topdown = torch.randn(2, 256, 32, 32)

    fused = module(lateral, topdown)

    assert fused.shape == (2, 256, 32, 32)


@torch.no_grad()
def test_burf_output_shape() -> None:
    module = BURF(channels=256).eval()
    lower = torch.randn(2, 256, 32, 32)
    higher = torch.randn(2, 256, 16, 16)

    fused = module(lower, higher)

    assert fused.shape == (2, 256, 16, 16)


@torch.no_grad()
def test_burf_rejects_shape_mismatch_after_downsampling() -> None:
    module = BURF(channels=256).eval()
    lower = torch.randn(2, 256, 32, 32)
    higher = torch.randn(2, 256, 15, 15)

    with pytest.raises(ValueError, match="aligned refinement features"):
        module(lower, higher)


@torch.no_grad()
def test_amf_neck_output_shapes() -> None:
    model = AMFNeck()
    features = (
        torch.randn(2, 256, 160, 160),
        torch.randn(2, 256, 80, 80),
        torch.randn(2, 512, 40, 40),
    )

    outputs = model(features)
    assert len(outputs) == 3
    n2, n3, n4 = outputs

    assert n2.shape == (2, 256, 160, 160)
    assert n3.shape == (2, 256, 80, 80)
    assert n4.shape == (2, 256, 40, 40)


@torch.no_grad()
def test_amf_neck_rejects_four_input_features() -> None:
    model = AMFNeck()
    with pytest.raises(ValueError, match="exactly three"):
        model(
            (
                torch.randn(1, 256, 160, 160),
                torch.randn(1, 256, 80, 80),
                torch.randn(1, 512, 40, 40),
                torch.randn(1, 512, 20, 20),
            )
        )
