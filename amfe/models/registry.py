"""Configuration helpers for AMFE detector construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .detector import AMFEModelConfig, AMFEYOLODetectionModel


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file with a clear error message."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Configuration at {config_path} must deserialize to a mapping.")
    return data


def build_model_from_config(config: dict[str, Any]) -> AMFEYOLODetectionModel:
    """Build an ``AMFEYOLODetectionModel`` from a config mapping."""

    model_cfg = config.get("model", config)
    if not isinstance(model_cfg, dict):
        raise TypeError("Model configuration must be a mapping.")
    return AMFEYOLODetectionModel(AMFEModelConfig.from_mapping(model_cfg))


def build_model_from_yaml(path: str | Path) -> AMFEYOLODetectionModel:
    """Build an ``AMFEYOLODetectionModel`` from a YAML file."""

    return build_model_from_config(load_yaml_config(path))


__all__ = ["build_model_from_config", "build_model_from_yaml", "load_yaml_config"]
