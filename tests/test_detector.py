from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from amfe.models.detector import build_amfe_detector


def test_amfe_detector_forward_smoke() -> None:
    model = build_amfe_detector(num_classes=5)
    model.train()
    outputs = model(torch.randn(1, 3, 256, 256))

    assert outputs is not None
