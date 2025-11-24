"""Tests for configuration loading helpers."""

from pathlib import Path

import pytest

from pairs_trading_etf.utils.config import ConfigError, load_yaml_config


def test_load_yaml_config_returns_expected_sections() -> None:
    config_path = Path("configs/data.yaml")
    if not config_path.exists():
        pytest.skip("configs/data.yaml missing; skipping config test")

    config = load_yaml_config(config_path)

    assert "universe" in config
    assert "data" in config
    assert isinstance(config["universe"].get("etfs", []), list)


def test_load_yaml_config_missing_file() -> None:
    with pytest.raises(ConfigError):
        load_yaml_config("configs/missing.yaml")
