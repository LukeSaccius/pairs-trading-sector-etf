"""Integration tests for the pair scanning pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pairs_trading_etf.features.pair_generation import PairScore
from pairs_trading_etf.pipelines.pair_scan import (
    PairScanConfig,
    _filter_cointegration_metrics,
    _filter_high_correlation_pairs,
    _filter_same_index_pairs,
    run_pair_scan,
)
from pairs_trading_etf.data.universe import ETFUniverse, ETFMetadata


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
        # Relax cointegration filters for synthetic data
        pvalue_threshold=1.0,  # Accept all p-values
        min_half_life=0.0,
        max_half_life=float('inf'),
        min_spread_range_pct=0.0,
    )

    df = run_pair_scan(cfg)

    assert not df.empty
    assert {"AAA", "BBB"} == {df.iloc[0]["leg_x"], df.iloc[0]["leg_y"]}
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
        # Relax cointegration filters for synthetic data
        pvalue_threshold=1.0,
        min_half_life=0.0,
        max_half_life=float('inf'),
        min_spread_range_pct=0.0,
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
        # Relax cointegration filters for synthetic data
        pvalue_threshold=1.0,
        min_half_life=0.0,
        max_half_life=float('inf'),
        min_spread_range_pct=0.0,
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
        # Relax cointegration filters for synthetic data
        pvalue_threshold=1.0,
        min_half_life=0.0,
        max_half_life=float('inf'),
        min_spread_range_pct=0.0,
    )

    with pytest.warns(UserWarning):
        df = run_pair_scan(cfg)

    assert not df.empty
    legs = set(df[["leg_x", "leg_y"]].stack())
    assert "ZZZ" not in legs


# ============================================================================
# Filter Function Tests
# ============================================================================


def _make_pair_score(
    leg_x: str = "AAA",
    leg_y: str = "BBB",
    correlation: float = 0.90,
    coint_pvalue: float | None = 0.05,
    half_life: float | None = 30.0,
    spread_range_pct: float | None = 15.0,
) -> PairScore:
    """Helper to create PairScore objects for testing."""
    return PairScore(
        leg_x=leg_x,
        leg_y=leg_y,
        correlation=correlation,
        n_obs=300,
        spread_mean=0.0,
        spread_std=0.02,
        hedge_ratio=1.0,
        coint_statistic=-3.5,
        coint_pvalue=coint_pvalue,
        half_life=half_life,
        spread_range_pct=spread_range_pct,
    )


class TestFilterHighCorrelationPairs:
    """Tests for _filter_high_correlation_pairs function."""

    def test_keeps_pairs_below_threshold(self) -> None:
        scores = [
            _make_pair_score(leg_x="A", leg_y="B", correlation=0.85),
            _make_pair_score(leg_x="C", leg_y="D", correlation=0.90),
        ]
        
        kept, excluded = _filter_high_correlation_pairs(scores, max_corr=0.99)
        
        assert len(kept) == 2
        assert len(excluded) == 0

    def test_excludes_pairs_above_threshold(self) -> None:
        scores = [
            _make_pair_score(leg_x="A", leg_y="B", correlation=0.995),
            _make_pair_score(leg_x="C", leg_y="D", correlation=0.85),
        ]
        
        kept, excluded = _filter_high_correlation_pairs(scores, max_corr=0.99)
        
        assert len(kept) == 1
        assert kept[0].leg_x == "C"
        assert len(excluded) == 1
        assert "high_corr" in excluded[0][1]

    def test_excludes_all_duplicates(self) -> None:
        scores = [
            _make_pair_score(leg_x="SPY", leg_y="IVV", correlation=0.998),
            _make_pair_score(leg_x="VOO", leg_y="IVV", correlation=0.997),
        ]
        
        kept, excluded = _filter_high_correlation_pairs(scores, max_corr=0.99)
        
        assert len(kept) == 0
        assert len(excluded) == 2


class TestFilterCointegrationMetrics:
    """Tests for _filter_cointegration_metrics function."""

    def test_keeps_valid_pairs(self) -> None:
        scores = [
            _make_pair_score(coint_pvalue=0.05, half_life=50.0, spread_range_pct=15.0),
        ]
        
        kept, excluded = _filter_cointegration_metrics(
            scores,
            pvalue_threshold=0.10,
            min_half_life=15.0,
            max_half_life=180.0,
            min_spread_range_pct=8.0,
        )
        
        assert len(kept) == 1
        assert len(excluded) == 0

    def test_excludes_high_pvalue(self) -> None:
        scores = [
            _make_pair_score(coint_pvalue=0.15),  # > 0.10
        ]
        
        kept, excluded = _filter_cointegration_metrics(
            scores,
            pvalue_threshold=0.10,
            min_half_life=15.0,
            max_half_life=180.0,
            min_spread_range_pct=8.0,
        )
        
        assert len(kept) == 0
        assert len(excluded) == 1
        assert "pvalue" in excluded[0][1]

    def test_excludes_none_pvalue(self) -> None:
        scores = [
            _make_pair_score(coint_pvalue=None),
        ]
        
        kept, excluded = _filter_cointegration_metrics(
            scores,
            pvalue_threshold=0.10,
            min_half_life=15.0,
            max_half_life=180.0,
            min_spread_range_pct=8.0,
        )
        
        assert len(kept) == 0
        assert "pvalue:None" in excluded[0][1]

    def test_excludes_half_life_too_low(self) -> None:
        scores = [
            _make_pair_score(half_life=10.0),  # < 15
        ]
        
        kept, excluded = _filter_cointegration_metrics(
            scores,
            pvalue_threshold=0.10,
            min_half_life=15.0,
            max_half_life=180.0,
            min_spread_range_pct=8.0,
        )
        
        assert len(kept) == 0
        assert "half_life_low" in excluded[0][1]

    def test_excludes_half_life_too_high(self) -> None:
        scores = [
            _make_pair_score(half_life=500.0),  # > 180
        ]
        
        kept, excluded = _filter_cointegration_metrics(
            scores,
            pvalue_threshold=0.10,
            min_half_life=15.0,
            max_half_life=180.0,
            min_spread_range_pct=8.0,
        )
        
        assert len(kept) == 0
        assert "half_life_high" in excluded[0][1]

    def test_excludes_none_half_life(self) -> None:
        scores = [
            _make_pair_score(half_life=None),
        ]
        
        kept, excluded = _filter_cointegration_metrics(
            scores,
            pvalue_threshold=0.10,
            min_half_life=15.0,
            max_half_life=180.0,
            min_spread_range_pct=8.0,
        )
        
        assert len(kept) == 0
        assert "half_life:None" in excluded[0][1]

    def test_excludes_low_spread_range(self) -> None:
        scores = [
            _make_pair_score(spread_range_pct=5.0),  # < 8.0
        ]
        
        kept, excluded = _filter_cointegration_metrics(
            scores,
            pvalue_threshold=0.10,
            min_half_life=15.0,
            max_half_life=180.0,
            min_spread_range_pct=8.0,
        )
        
        assert len(kept) == 0
        assert "spread_range_low" in excluded[0][1]

    def test_allows_none_spread_range(self) -> None:
        """None spread_range should not cause exclusion."""
        scores = [
            _make_pair_score(spread_range_pct=None),
        ]
        
        kept, excluded = _filter_cointegration_metrics(
            scores,
            pvalue_threshold=0.10,
            min_half_life=15.0,
            max_half_life=180.0,
            min_spread_range_pct=8.0,
        )
        
        assert len(kept) == 1
        assert len(excluded) == 0

    def test_multiple_filters_applied_in_order(self) -> None:
        """Test that multiple scores are filtered correctly."""
        scores = [
            _make_pair_score(leg_x="A", leg_y="B", coint_pvalue=0.05, half_life=50.0),  # Keep
            _make_pair_score(leg_x="C", leg_y="D", coint_pvalue=0.20, half_life=50.0),  # Exclude: pvalue
            _make_pair_score(leg_x="E", leg_y="F", coint_pvalue=0.05, half_life=500.0), # Exclude: HL high
            _make_pair_score(leg_x="G", leg_y="H", coint_pvalue=0.05, half_life=5.0),   # Exclude: HL low
        ]
        
        kept, excluded = _filter_cointegration_metrics(
            scores,
            pvalue_threshold=0.10,
            min_half_life=15.0,
            max_half_life=180.0,
            min_spread_range_pct=8.0,
        )
        
        assert len(kept) == 1
        assert kept[0].leg_x == "A"
        assert len(excluded) == 3


class TestFilterSameIndexPairs:
    """Tests for _filter_same_index_pairs function."""

    def test_excludes_same_index_pairs(self) -> None:
        scores = [
            _make_pair_score(leg_x="SPY", leg_y="IVV"),
            _make_pair_score(leg_x="XLK", leg_y="IYW"),
        ]
        
        # Create universe with metadata
        metadata = {
            "SPY": ETFMetadata(ticker="SPY", name="SPY", sector="Broad", tracks_index="sp500"),
            "IVV": ETFMetadata(ticker="IVV", name="IVV", sector="Broad", tracks_index="sp500"),
            "XLK": ETFMetadata(ticker="XLK", name="XLK", sector="Tech", tracks_index="tech_select"),
            "IYW": ETFMetadata(ticker="IYW", name="IYW", sector="Tech", tracks_index="dow_tech"),
        }
        universe = ETFUniverse(
            name="test",
            tickers=["SPY", "IVV", "XLK", "IYW"],
            metadata=metadata,
        )
        
        kept, excluded = _filter_same_index_pairs(scores, universe)
        
        assert len(kept) == 1
        assert kept[0].leg_x == "XLK"
        assert len(excluded) == 1
        assert "same_index:sp500" in excluded[0][1]

    def test_keeps_different_index_pairs(self) -> None:
        scores = [
            _make_pair_score(leg_x="XLK", leg_y="XLF"),
        ]
        
        metadata = {
            "XLK": ETFMetadata(ticker="XLK", name="XLK", sector="Tech", tracks_index="tech_select"),
            "XLF": ETFMetadata(ticker="XLF", name="XLF", sector="Financials", tracks_index="fin_select"),
        }
        universe = ETFUniverse(
            name="test",
            tickers=["XLK", "XLF"],
            metadata=metadata,
        )
        
        kept, excluded = _filter_same_index_pairs(scores, universe)
        
        assert len(kept) == 1
        assert len(excluded) == 0

    def test_keeps_pairs_without_index_info(self) -> None:
        """Pairs without tracks_index should not be excluded."""
        scores = [
            _make_pair_score(leg_x="AAA", leg_y="BBB"),
        ]
        
        metadata = {
            "AAA": ETFMetadata(ticker="AAA", name="AAA", sector="Test", tracks_index=None),
            "BBB": ETFMetadata(ticker="BBB", name="BBB", sector="Test", tracks_index=None),
        }
        universe = ETFUniverse(
            name="test",
            tickers=["AAA", "BBB"],
            metadata=metadata,
        )
        
        kept, excluded = _filter_same_index_pairs(scores, universe)
        
        assert len(kept) == 1
        assert len(excluded) == 0

    def test_no_metadata_keeps_all(self) -> None:
        """Without metadata, all pairs should be kept."""
        scores = [
            _make_pair_score(leg_x="AAA", leg_y="BBB"),
        ]
        
        universe = ETFUniverse(
            name="test",
            tickers=["AAA", "BBB"],
            metadata=None,
        )
        
        kept, excluded = _filter_same_index_pairs(scores, universe)
        
        assert len(kept) == 1
        assert len(excluded) == 0
