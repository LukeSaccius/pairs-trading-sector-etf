"""Tests for ETF universe utilities."""

from pathlib import Path

import pytest

from pairs_trading_etf.data.universe import (
    ETFUniverse,
    load_configured_universe,
    load_etf_metadata,
    resolve_universe,
)
from pairs_trading_etf.utils.config import load_yaml_config


@pytest.mark.skipif(not Path("configs/etf_metadata.yaml").exists(), reason="metadata file missing")
def test_load_etf_metadata_contains_expected_fields() -> None:
    catalog = load_etf_metadata(Path("configs/etf_metadata.yaml"))

    assert "XLK" in catalog
    tech_meta = catalog["XLK"]
    assert tech_meta.sector == "Technology"
    assert tech_meta.issuer is not None


@pytest.mark.skipif(not Path("configs/data.yaml").exists(), reason="config file missing")
def test_resolve_universe_uses_default_list() -> None:
    config = load_yaml_config(Path("configs/data.yaml"))

    metadata_path = config.get("metadata", {}).get("etf_info_path")
    metadata = load_etf_metadata(metadata_path) if metadata_path else None

    universe = resolve_universe(config, metadata=metadata)

    assert isinstance(universe, ETFUniverse)
    assert universe.name == config["universe"].get("default_list", "legacy")
    assert "XLK" in universe.tickers
    assert not universe.missing_metadata()


@pytest.mark.skipif(
    (not Path("configs/data.yaml").exists()) or (not Path("configs/etf_metadata.yaml").exists()),
    reason="config or metadata missing",
)
def test_load_configured_universe_round_trip() -> None:
    universe = load_configured_universe(Path("configs/data.yaml"))

    assert universe.tickers
    assert universe.metadata
    assert "XLF" in universe.tickers
