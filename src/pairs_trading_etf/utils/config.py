"""Configuration loading helpers for YAML-based project settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(RuntimeError):
    """Raised when configuration files cannot be loaded or parsed."""


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a dictionary.

    Args:
        path: File system path to the YAML document.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        ConfigError: If the file is missing or cannot be parsed as YAML.
    """

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found at {config_path.resolve()}")

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - exercised via tests
        raise ConfigError(f"Failed to parse YAML config {config_path}: {exc}") from exc

    if not isinstance(data, dict):  # pragma: no cover - guard for malformed files
        raise ConfigError(f"Config {config_path} must define a mapping at the top level")

    return data
