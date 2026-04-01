from __future__ import annotations

from pathlib import Path

import pytest
import torch

from amfe.models import AMFEModelConfig, AMFEYOLODetectionModel, build_amfe_detector, build_model_from_yaml


def _synthetic_batch(batch_size: int = 2, image_size: int = 128) -> dict[str, torch.Tensor]:
    images = torch.randn(batch_size, 3, image_size, image_size)
    return {
        "img": images,
        "batch_idx": torch.tensor([0, 1], dtype=torch.long),
        "cls": torch.tensor([0, 1], dtype=torch.long),
        "bboxes": torch.tensor([[0.50, 0.50, 0.25, 0.25], [0.35, 0.35, 0.20, 0.20]], dtype=images.dtype),
    }


@torch.no_grad()
def test_amfe_detector_forward_smoke() -> None:
    model = build_amfe_detector(num_classes=5)
    model.train()
    outputs = model(torch.randn(1, 3, 128, 128))

    assert isinstance(outputs, dict)
    assert set(outputs.keys()) == {"boxes", "scores", "feats"}
    assert outputs["scores"].shape[1] == 5
    assert len(outputs["feats"]) == 3


@torch.no_grad()
def test_amfe_detector_feature_shapes_and_backbone_trace() -> None:
    model = AMFEYOLODetectionModel(AMFEModelConfig(num_classes=3))
    n3, n4, n5 = model.forward_features(torch.randn(1, 3, 128, 128))

    assert n3.shape == (1, 256, 16, 16)
    assert n4.shape == (1, 256, 8, 8)
    assert n5.shape == (1, 256, 4, 4)
    assert tuple(model.stride.tolist()) == (8.0, 16.0, 32.0)
    assert model.backbone.last_forward_shapes == {
        "S2": (1, 64, 32, 32),
        "C3": (1, 256, 16, 16),
        "C4": (1, 512, 8, 8),
        "C5": (1, 512, 4, 4),
        "D3": (1, 128, 16, 16),
        "D4": (1, 256, 8, 8),
        "G3": (1, 256, 16, 16),
        "G4": (1, 256, 8, 8),
        "G5": (1, 256, 4, 4),
        "F3": (1, 256, 16, 16),
        "F4": (1, 512, 8, 8),
        "F5": (1, 512, 4, 4),
    }


def test_amfe_detector_loss_backward_and_optimizer_step() -> None:
    model = build_amfe_detector(num_classes=3)
    model.train()
    batch = _synthetic_batch()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    tracked_parameter = next(model.detect.parameters())
    before = tracked_parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    total_loss, loss_items = model.loss(batch)
    total_loss.backward()
    optimizer.step()
    after = tracked_parameter.detach()

    assert total_loss.requires_grad
    assert total_loss.detach().item() > 0
    assert loss_items.shape == (3,)
    assert tracked_parameter.grad is not None
    assert not torch.equal(before, after)


@torch.no_grad()
def test_build_model_from_yaml() -> None:
    model = build_model_from_yaml("configs/model/amfe_amf_yolo.yaml")
    outputs = model(torch.randn(1, 3, 128, 128))

    assert isinstance(outputs, dict)
    assert model.config.msb_variant == "yolov8_s"
    assert model.config.neck_channels == 256
    assert model.backbone.output_channels.f3 == 256
    assert model.backbone.output_channels.f4 == 512
    assert model.backbone.output_channels.f5 == 512


@torch.no_grad()
def test_build_visdrone_model_from_yaml() -> None:
    path = Path("configs/model/amfe_amf_yolo_visdrone.yaml")
    model = build_model_from_yaml(path)
    outputs = model(torch.randn(1, 3, 128, 128))

    assert isinstance(outputs, dict)
    assert model.config.num_classes == 10
    assert model.config.neck_channels == 256
