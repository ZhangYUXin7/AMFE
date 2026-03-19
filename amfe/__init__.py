"""AMFE project package.

This package provides a conservative AMFE-Backbone + AMF-Neck implementation
and a minimal Ultralytics-compatible detector/training path for Phase D.
"""

from .models import (
    AMFEBackbone,
    AMFEDetector,
    AMFEModelConfig,
    AMFEYOLODetectionModel,
    AMFNeck,
    build_amfe_detector,
    build_model_from_yaml,
)

__all__ = [
    "AMFEBackbone",
    "AMFEDetector",
    "AMFEModelConfig",
    "AMFEYOLODetectionModel",
    "AMFNeck",
    "build_amfe_detector",
    "build_model_from_yaml",
]
