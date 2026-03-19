from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from amfe.models.neck import AMFNeck


def test_amf_neck_output_shapes() -> None:
    model = AMFNeck()
    features = (
        torch.randn(2, 512, 32, 32),
        torch.randn(2, 1024, 16, 16),
        torch.randn(2, 2048, 8, 8),
    )

    n3, n4, n5 = model(features)

    assert n3.shape == (2, 256, 32, 32)
    assert n4.shape == (2, 256, 16, 16)
    assert n5.shape == (2, 256, 8, 8)
