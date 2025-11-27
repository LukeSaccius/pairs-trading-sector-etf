"""Smoke tests for data download utilities."""

from datetime import UTC, datetime, timedelta
import os

import pytest

from pairs_trading_etf.data.ingestion import download_etf_data
from pairs_trading_etf.utils.config import load_yaml_config

NETWORK_UNAVAILABLE = os.environ.get("SKIP_NETWORK_TESTS") == "1"


@pytest.mark.skipif(NETWORK_UNAVAILABLE, reason="Network-dependent test disabled via env flag")
def test_download_single_ticker_smoke() -> None:
    config = load_yaml_config("configs/data.yaml")
    ticker = config["universe"]["etfs"][0]

    end_date = datetime.now(UTC).date()
    start_date = (end_date - timedelta(days=365)).isoformat()

    df = download_etf_data([ticker], start=start_date, end=end_date.isoformat())

    assert df.shape[0] > 200
    assert list(df.columns) == [ticker]
