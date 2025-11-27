"""Pair enumeration and scoring helpers used by scanning pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from pairs_trading_etf.cointegration.engle_granger import EngleGrangerResult, run_engle_granger

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairScore:
    """Summary statistics for a candidate ETF pair."""

    leg_x: str
    leg_y: str
    correlation: float
    n_obs: int
    spread_mean: float | None = None
    spread_std: float | None = None
    hedge_ratio: float | None = None
    coint_statistic: float | None = None
    coint_pvalue: float | None = None
    half_life: float | None = None

    def as_dict(self) -> Mapping[str, float | int | None]:
        return {
            "leg_x": self.leg_x,
            "leg_y": self.leg_y,
            "correlation": self.correlation,
            "n_obs": self.n_obs,
            "spread_mean": self.spread_mean,
            "spread_std": self.spread_std,
            "hedge_ratio": self.hedge_ratio,
            "coint_statistic": self.coint_statistic,
            "coint_pvalue": self.coint_pvalue,
            "half_life": self.half_life,
        }


def enumerate_pairs(tickers: Sequence[str]) -> list[tuple[str, str]]:
    """Enumerate unique unordered ticker combinations in uppercase form."""

    cleaned = [str(t).strip().upper() for t in tickers]
    pairs: list[tuple[str, str]] = []
    for idx in range(len(cleaned)):
        for jdx in range(idx + 1, len(cleaned)):
            pairs.append((cleaned[idx], cleaned[jdx]))
    return pairs


def _engle_granger_fields(result: EngleGrangerResult | None) -> dict[str, float | None]:
    """Extract spread/cointegration metrics from an Engle–Granger result object."""
    if result is None:
        return {
            "spread_mean": None,
            "spread_std": None,
            "hedge_ratio": None,
            "coint_statistic": None,
            "coint_pvalue": None,
            "half_life": None,
        }

    return {
        "spread_mean": result.spread_mean,
        "spread_std": result.spread_std,
        "hedge_ratio": result.hedge_ratio,
        "coint_statistic": result.test_statistic,
        "coint_pvalue": result.pvalue,
        "half_life": result.half_life,
    }


def score_pairs(
    prices: pd.DataFrame,
    min_obs: int = 252,
    min_corr: float = 0.80,
    lookback: int | None = None,
    max_pairs: int | None = None,
    run_cointegration: bool = True,
    engle_granger_kwargs: Mapping[str, object] | None = None,
) -> list[PairScore]:
    """Rank ETF pairs by correlation strength and Engle–Granger diagnostics."""

    if prices.empty:
        return []

    prices = prices.copy()
    prices.columns = [str(col).upper() for col in prices.columns]
    if lookback is not None and lookback > 0:
        prices = prices.tail(lookback)

    returns = prices.pct_change().dropna()
    if returns.empty:
        return []

    candidates = enumerate_pairs(returns.columns)
    if not candidates:
        return []

    granger_kwargs = dict(engle_granger_kwargs or {})

    # Diagnostic counters
    min_obs_fails = 0
    min_corr_fails = 0
    eg_fails = 0
    pvalues: list[float] = []

    scored: list[PairScore] = []
    for leg_x, leg_y in candidates:
        pair_returns = returns[[leg_x, leg_y]].dropna()
        n_obs = pair_returns.shape[0]
        if n_obs < min_obs:
            min_obs_fails += 1
            continue

        corr = pair_returns[leg_x].corr(pair_returns[leg_y])
        if pd.isna(corr) or corr < min_corr:
            min_corr_fails += 1
            continue

        pair_prices = prices[[leg_x, leg_y]].loc[pair_returns.index].dropna()
        eg_result: EngleGrangerResult | None = None
        if run_cointegration:
            try:
                eg_result = run_engle_granger(
                    pair_prices[leg_x], pair_prices[leg_y], **granger_kwargs
                )
                if eg_result is not None:
                    pvalues.append(eg_result.pvalue)
            except ValueError as exc:
                logger.debug("Engle-Granger failed for %s-%s: %s", leg_x, leg_y, exc)
                eg_fails += 1
                continue

        fields = _engle_granger_fields(eg_result)
        scored.append(
            PairScore(
                leg_x=leg_x,
                leg_y=leg_y,
                correlation=float(corr),
                n_obs=int(n_obs),
                **fields,
            )
        )

    # Log diagnostic summary
    logger.info(
        "score_pairs: %d candidates | min_obs_fails=%d | min_corr_fails=%d | eg_fails=%d | scored=%d",
        len(candidates),
        min_obs_fails,
        min_corr_fails,
        eg_fails,
        len(scored),
    )
    if pvalues:
        import numpy as np
        pv_arr = np.array(pvalues)
        logger.info(
            "p-value distribution: min=%.4f, median=%.4f, mean=%.4f, max=%.4f, <=0.05: %d/%d",
            pv_arr.min(),
            np.median(pv_arr),
            pv_arr.mean(),
            pv_arr.max(),
            int((pv_arr <= 0.05).sum()),
            len(pv_arr),
        )

    def _sort_key(item: PairScore) -> tuple[float, float]:
        pvalue = item.coint_pvalue if item.coint_pvalue is not None else 1.0
        return (pvalue, -item.correlation)

    scored.sort(key=_sort_key)

    if max_pairs is not None and max_pairs > 0:
        scored = scored[:max_pairs]

    return scored
