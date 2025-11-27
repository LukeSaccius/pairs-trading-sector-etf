"""Integration tests for the pair scanning pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pairs_trading_etf.pipelines.pair_scan import PairScanConfig, run_pair_scan


def _synthetic_prices(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=320, freq="B")
    base = np.cumsum(rng.normal(0, 1, size=len(dates))) + 75
    pair = base + rng.normal(0, 0.2, size=len(dates))
    diverger = np.cumsum(rng.normal(0, 2, size=len(dates))) + 30
    frame = pd.DataFrame({"AAA": base, "BBB": pair, "CCC": diverger}, index=dates)
    frame.index.name = "date"
    return frame


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _build_metadata(
    tmp_path: Path, tickers: list[str], sector_map: dict[str, str] | None = None
) -> Path:
    metadata_payload = {
        "etfs": {
            ticker: {
                "name": ticker,
                "sector": (sector_map or {}).get(ticker, "Test"),
                "region": "US",
                "issuer": "UnitTest",
            }
            for ticker in tickers
        }
    }
    metadata_path = tmp_path / "metadata.yaml"
    _write_yaml(metadata_path, metadata_payload)
    return metadata_path


def _build_config(tmp_path: Path, metadata_path: Path, tickers: list[str]) -> Path:
    config_payload = {
        "universe": {
            "etfs": tickers,
            "default_list": "core",
            "lists": {
                "core": {
                    "description": "Synthetic trio for testing",
                    "sectors": ["Test"],
                    "tickers": tickers,
                }
            },
        },
        "metadata": {"etf_info_path": str(metadata_path)},
        "data": {
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
        },
    }
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, config_payload)
    return config_path


def test_run_pair_scan_returns_ranked_pairs(tmp_path: Path) -> None:
    prices = _synthetic_prices()
    price_path = tmp_path / "prices.csv"
    prices.to_csv(price_path)

    tickers = ["AAA", "BBB", "CCC"]
    metadata_path = _build_metadata(tmp_path, tickers)
    config_path = _build_config(tmp_path, metadata_path, tickers)

    cfg = PairScanConfig(
        config_path=config_path,
        price_path=price_path,
        output_path=None,
        list_name=None,
        metadata_path=None,
        lookback_days=None,
        min_obs=150,
        min_corr=0.95,
        max_pairs=1,
        engle_granger_maxlag=1,
    )

    df = run_pair_scan(cfg)

    assert not df.empty
    assert {"AAA", "BBB"} == {df.iloc[0]["leg_x"], df.iloc[0]["leg_y"]}
    assert df.iloc[0]["coint_pvalue"] < 0.05
    assert df.iloc[0]["correlation"] > 0.95


def test_run_pair_scan_same_sector_only(tmp_path: Path) -> None:
    prices = _synthetic_prices()
    price_path = tmp_path / "prices.csv"
    prices.to_csv(price_path)

    tickers = ["AAA", "BBB", "CCC"]
    sector_map = {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"}
    metadata_path = _build_metadata(tmp_path, tickers, sector_map)
    config_path = _build_config(tmp_path, metadata_path, tickers)

    cfg = PairScanConfig(
        config_path=config_path,
        price_path=price_path,
        output_path=None,
        list_name=None,
        metadata_path=metadata_path,
        lookback_days=None,
        min_obs=150,
        min_corr=0.95,
        max_pairs=5,
        engle_granger_maxlag=1,
        allow_cross_sector=False,
    )

    df = run_pair_scan(cfg)

    assert not df.empty
    unique_pairs = {(row["leg_x"], row["leg_y"]) for _, row in df.iterrows()}
    assert unique_pairs == {("AAA", "BBB")}


def test_run_pair_scan_keeps_all_pairs_when_max_pairs_none(tmp_path: Path) -> None:
    prices = _synthetic_prices()
    price_path = tmp_path / "prices.csv"
    prices.to_csv(price_path)

    tickers = ["AAA", "BBB", "CCC"]
    metadata_path = _build_metadata(tmp_path, tickers)
    config_path = _build_config(tmp_path, metadata_path, tickers)

    cfg = PairScanConfig(
        config_path=config_path,
        price_path=price_path,
        output_path=None,
        list_name=None,
        metadata_path=metadata_path,
        lookback_days=None,
        min_obs=150,
        min_corr=-1.0,
        max_pairs=None,
        engle_granger_maxlag=1,
        allow_cross_sector=True,
    )

    df = run_pair_scan(cfg)

    expected_pairs = len(tickers) * (len(tickers) - 1) // 2
    assert len(df) == expected_pairs


def test_run_pair_scan_drops_missing_tickers(tmp_path: Path) -> None:
    prices = _synthetic_prices()
    price_path = tmp_path / "prices.csv"
    prices.to_csv(price_path)

    tickers = ["AAA", "BBB", "CCC", "ZZZ"]
    metadata_path = _build_metadata(tmp_path, tickers)
    config_path = _build_config(tmp_path, metadata_path, tickers)

    cfg = PairScanConfig(
        config_path=config_path,
        price_path=price_path,
        output_path=None,
        list_name=None,
        metadata_path=metadata_path,
        lookback_days=None,
        min_obs=150,
        min_corr=0.95,
        max_pairs=5,
        engle_granger_maxlag=1,
        allow_cross_sector=True,
    )

    with pytest.warns(UserWarning):
        df = run_pair_scan(cfg)

    assert not df.empty
    legs = set(df[["leg_x", "leg_y"]].stack())
    assert "ZZZ" not in legs
