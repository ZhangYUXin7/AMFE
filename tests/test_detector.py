from __future__ import annotations

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
def test_amfe_detector_feature_shapes() -> None:
    model = AMFEYOLODetectionModel(AMFEModelConfig(num_classes=3))
    n3, n4, n5 = model.forward_features(torch.randn(1, 3, 128, 128))

    assert n3.shape == (1, 256, 16, 16)
    assert n4.shape == (1, 256, 8, 8)
    assert n5.shape == (1, 256, 4, 4)
    assert tuple(model.stride.tolist()) == (8.0, 16.0, 32.0)


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
