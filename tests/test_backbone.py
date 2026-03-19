from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from amfe.models.backbone import AMFEBackbone


def test_amfe_backbone_output_shapes() -> None:
    model = AMFEBackbone()
    inputs = torch.randn(2, 3, 256, 256)

    f3, f4, f5 = model(inputs)

    assert f3.shape == (2, 512, 32, 32)
    assert f4.shape == (2, 1024, 16, 16)
    assert f5.shape == (2, 2048, 8, 8)
