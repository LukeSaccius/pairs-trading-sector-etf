"""Lightweight Engle-Granger cointegration helper utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngleGrangerResult:
    """Container for Engle-Granger outputs used by pipelines."""

    test_statistic: float
    pvalue: float
    crit_values: tuple[float, float, float]
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    half_life: float | None

    def as_dict(self) -> Mapping[str, float | tuple[float, float, float] | None]:
        return {
            "test_statistic": self.test_statistic,
            "pvalue": self.pvalue,
            "crit_values": self.crit_values,
            "hedge_ratio": self.hedge_ratio,
            "spread_mean": self.spread_mean,
            "spread_std": self.spread_std,
            "half_life": self.half_life,
        }


def _align_series(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    frame = pd.concat([x, y], axis=1, join="inner").dropna()
    if frame.shape[0] < 30:
        raise ValueError("Need at least 30 overlapping observations for Engle-Granger")
    return frame.iloc[:, 0], frame.iloc[:, 1]


def _estimate_half_life(spread: pd.Series) -> float | None:
    spread = spread.dropna()
    if spread.shape[0] < 30:
        return None

    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    aligned = pd.concat([lagged, delta], axis=1, join="inner").dropna()
    if aligned.empty:
        return None

    # Simple OLS slope estimate without statsmodels dependency for speed.
    x = aligned.iloc[:, 0].values
    y = aligned.iloc[:, 1].values
    beta = np.linalg.lstsq(x.reshape(-1, 1), y, rcond=None)[0][0]
    if beta >= 0:
        return None

    return float(-np.log(2) / beta)


def run_engle_granger(
    series_x: pd.Series,
    series_y: pd.Series,
    maxlag: int = 1,
    trend: str = "c",
    autolag: str = "aic",
    use_log: bool = True,
) -> EngleGrangerResult:
    """Execute the Engle-Granger two-step test for a pair of price series.

    Parameters
    ----------
    series_x, series_y : pd.Series
        Price series for the two legs of the pair.
    maxlag : int
        Maximum lag for ADF test inside statsmodels coint.
    trend : str
        Trend specification passed to coint ('c', 'ct', 'ctt', 'n').
    autolag : str
        Lag selection method ('aic', 'bic', 't-stat', None).
    use_log : bool
        If True, apply natural log to prices before testing. Recommended for
        scale invariance and alignment with standard stat-arb practice.
    """

    aligned_x, aligned_y = _align_series(series_x, series_y)

    if use_log:
        aligned_x = np.log(aligned_x.where(aligned_x > 0)).replace([np.inf, -np.inf], np.nan).dropna()
        aligned_y = np.log(aligned_y.where(aligned_y > 0)).replace([np.inf, -np.inf], np.nan).dropna()
        # Re-align after log transform in case any rows were dropped
        aligned_x, aligned_y = _align_series(aligned_x, aligned_y)

    test_stat, pvalue, crit_values = coint(aligned_x, aligned_y, trend=trend, maxlag=maxlag, autolag=autolag)
    logger.debug(
        "Engle-Granger test: t-stat=%.3f, p=%.4f, use_log=%s",
        test_stat,
        pvalue,
        use_log,
    )

    hedge_ratio = float(np.polyfit(aligned_y, aligned_x, 1)[0])
    spread = aligned_x - hedge_ratio * aligned_y
    half_life = _estimate_half_life(spread)

    return EngleGrangerResult(
        test_statistic=float(test_stat),
        pvalue=float(pvalue),
        crit_values=tuple(float(v) for v in crit_values),
        hedge_ratio=hedge_ratio,
        spread_mean=float(spread.mean()),
        spread_std=float(spread.std(ddof=1)),
        half_life=half_life,
    )
