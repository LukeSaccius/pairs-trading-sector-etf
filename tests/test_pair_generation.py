"""Unit tests for pair generation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs_trading_etf.features.pair_generation import enumerate_pairs, score_pairs


def _synthetic_prices(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    base = np.cumsum(rng.normal(0, 1, size=len(dates))) + 100
    pair = base + rng.normal(0, 0.2, size=len(dates))
    diverger = np.cumsum(rng.normal(0, 2, size=len(dates))) + 50
    return pd.DataFrame({"AAA": base, "BBB": pair, "CCC": diverger}, index=dates)


def test_enumerate_pairs_generates_unique_combinations() -> None:
    pairs = enumerate_pairs(["AAA", "bbb", "CCC"])
    assert ("AAA", "BBB") in pairs
    assert len(pairs) == 3  # 3 choose 2


def test_score_pairs_surfaces_cointegrated_pair() -> None:
    prices = _synthetic_prices()
    scores = score_pairs(
        prices,
        min_obs=200,
        min_corr=0.95,
        lookback=None,
        max_pairs=1,
    )

    assert scores, "Expected at least one qualifying pair"
    top = scores[0]
    assert {top.leg_x, top.leg_y} == {"AAA", "BBB"}
    assert top.coint_pvalue is not None and top.coint_pvalue < 0.05
    assert top.correlation > 0.95
