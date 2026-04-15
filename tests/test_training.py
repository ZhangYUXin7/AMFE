from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from amfe.training import FPSBenchmarkConfig, _format_fps_benchmark_summary, benchmark_model_fps


class _DummyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(in_channels=3)
        self.conv = nn.Conv2d(3, 8, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _DtypeTrackingDetector(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = SimpleNamespace(in_channels=3)
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=dtype))
        self.last_input_dtype: torch.dtype | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input_dtype = x.dtype
        return x


def test_fps_benchmark_config_uses_training_defaults() -> None:
    config = FPSBenchmarkConfig.from_mapping(None, default_imgsz=320, default_use_amp=True)

    assert config.enabled is True
    assert config.batch_size == 1
    assert config.warmup_iters == 10
    assert config.timed_iters == 30
    assert config.imgsz == (320, 320)
    assert config.use_amp is True


def test_fps_benchmark_config_can_be_disabled_with_boolean() -> None:
    config = FPSBenchmarkConfig.from_mapping(False, default_imgsz=(256, 384), default_use_amp=False)

    assert config.enabled is False
    assert config.imgsz == (256, 384)
    assert config.use_amp is False


def test_benchmark_model_fps_returns_positive_metrics_and_restores_train_mode() -> None:
    model = _DummyDetector()
    model.train()

    result = benchmark_model_fps(
        model,
        FPSBenchmarkConfig(
            enabled=True,
            batch_size=2,
            warmup_iters=1,
            timed_iters=3,
            imgsz=(16, 16),
            use_amp=False,
        ),
    )

    assert model.training is True
    assert result["device"] == "cpu"
    assert result["precision"] == "fp32"
    assert result["batch_size"] == 2
    assert result["imgsz"] == (16, 16)
    assert result["warmup_iters"] == 1
    assert result["timed_iters"] == 3
    assert result["latency_ms"] > 0.0
    assert result["latency_ms_per_image"] > 0.0
    assert result["fps"] > 0.0


def test_benchmark_model_fps_rejects_amp_without_cuda() -> None:
    model = _DummyDetector()

    with pytest.raises(ValueError, match="requires the benchmark model to run on CUDA"):
        benchmark_model_fps(
            model,
            FPSBenchmarkConfig(
                enabled=True,
                batch_size=1,
                warmup_iters=0,
                timed_iters=1,
                imgsz=(16, 16),
                use_amp=True,
            ),
        )


def test_benchmark_model_fps_uses_model_dtype_for_inputs() -> None:
    model = _DtypeTrackingDetector(dtype=torch.float16)

    result = benchmark_model_fps(
        model,
        FPSBenchmarkConfig(
            enabled=True,
            batch_size=1,
            warmup_iters=1,
            timed_iters=2,
            imgsz=(8, 8),
            use_amp=False,
        ),
    )

    assert model.last_input_dtype == torch.float16
    assert result["precision"] == "fp16"


def test_format_fps_benchmark_summary() -> None:
    summary = _format_fps_benchmark_summary(
        {
            "device": "cpu",
            "precision": "fp32",
            "batch_size": 1,
            "imgsz": (640, 640),
            "warmup_iters": 10,
            "timed_iters": 30,
            "latency_ms": 12.345,
            "latency_ms_per_image": 12.345,
            "fps": 81.0,
        }
    )

    assert "Inference benchmark" in summary
    assert "device=cpu" in summary
    assert "imgsz=640x640" in summary
    assert "latency/batch=12.35ms" in summary
    assert "FPS=81.00" in summary
