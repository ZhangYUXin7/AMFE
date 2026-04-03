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
def test_amf_neck_output_shapes() -> None:
    model = AMFNeck()
    features = (
        torch.randn(2, 256, 64, 64),
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 512, 16, 16),
        torch.randn(2, 512, 8, 8),
    )

    n2, n3, n4, n5 = model(features)

    assert n2.shape == (2, 256, 64, 64)
    assert n3.shape == (2, 256, 32, 32)
    assert n4.shape == (2, 256, 16, 16)
    assert n5.shape == (2, 256, 8, 8)
