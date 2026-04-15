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

    assert outputs is not None
    if isinstance(outputs, list):
        assert len(outputs) == 3
    elif isinstance(outputs, tuple):
        assert len(outputs) >= 1
    elif isinstance(outputs, dict):
        assert outputs
    else:
        assert torch.is_tensor(outputs)


@torch.no_grad()
def test_amfe_detector_feature_shapes_and_backbone_trace() -> None:
    model = AMFEYOLODetectionModel(AMFEModelConfig(num_classes=3))
    outputs = model.forward_features(torch.randn(1, 3, 128, 128))

    assert len(outputs) == 3
    n2, n3, n4 = outputs
    assert n2.shape == (1, 256, 32, 32)
    assert n3.shape == (1, 256, 16, 16)
    assert n4.shape == (1, 256, 8, 8)
    assert tuple(model.stride.tolist()) == (4.0, 8.0, 16.0)
    assert model.backbone.last_forward_shapes == {
        "LEM": (1, 3, 128, 128),
        "S2": (1, 64, 32, 32),
        "C2": (1, 256, 32, 32),
        "C3": (1, 256, 16, 16),
        "C4": (1, 512, 8, 8),
        "D2": (1, 128, 32, 32),
        "D3": (1, 128, 16, 16),
        "F2": (1, 256, 32, 32),
        "F3": (1, 256, 16, 16),
        "F4e": (1, 512, 8, 8),
    }


@torch.no_grad()
def test_forward_features_returns_three_scales_and_detect_uses_three_inputs() -> None:
    model = build_amfe_detector(num_classes=3)
    features = model.forward_features(torch.randn(1, 3, 128, 128))

    assert len(features) == 3
    assert model.detect.nl == 3
    assert tuple(model.detect.stride.tolist()) == (4.0, 8.0, 16.0)


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

    assert outputs is not None
    assert model.config.msb_variant == "yolov8_s"
    assert model.config.use_lem is False
    assert model.config.lem_channels == 32
    assert model.config.fusion_gate_reduction == 8
    assert model.config.tdsf_spg_reduction == 8
    assert model.config.tdsf_dpg_kernels == (3, 5, 7)
    assert model.config.rfb_channels == 512
    assert model.config.rfb_expand_ratio == 1.0
    assert model.config.rfb_dilations == (3, 5)
    assert model.config.detect_feature_strides == (4, 8, 16)
    assert model.config.neck_channels == 256
    assert model.backbone.output_channels.f2 == 256
    assert model.backbone.output_channels.f3 == 256
    assert model.backbone.output_channels.f4e == 512


@torch.no_grad()
def test_build_visdrone_model_from_yaml() -> None:
    path = Path("configs/model/amfe_amf_yolo_visdrone.yaml")
    model = build_model_from_yaml(path)
    outputs = model(torch.randn(1, 3, 128, 128))

    assert outputs is not None
    assert model.config.num_classes == 10
    assert model.config.use_lem is False
    assert model.config.lem_channels == 32
    assert model.config.neck_channels == 256
    assert model.config.detect_feature_strides == (4, 8, 16)
