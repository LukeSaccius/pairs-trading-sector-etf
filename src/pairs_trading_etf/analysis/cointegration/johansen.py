"""Johansen cointegration test implementation with basket discovery."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen, JohansenTestResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JohansenResult:
    """Container for Johansen cointegration test results."""

    tickers: tuple[str, ...]
    basket_size: int
    cointegration_rank: int  # Number of cointegrating relationships at 95% level
    trace_stats: tuple[float, ...]
    crit_90: tuple[float, ...]
    crit_95: tuple[float, ...]
    crit_99: tuple[float, ...]
    eigenvalues: tuple[float, ...]
    eigenvectors: tuple[tuple[float, ...], ...]  # Each row is a cointegrating vector
    score: float  # Composite score for ranking

    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "tickers": list(self.tickers),
            "basket_size": self.basket_size,
            "cointegration_rank": self.cointegration_rank,
            "trace_stats": list(self.trace_stats),
            "crit_90": list(self.crit_90),
            "crit_95": list(self.crit_95),
            "crit_99": list(self.crit_99),
            "eigenvalues": list(self.eigenvalues),
            "eigenvectors": [list(v) for v in self.eigenvectors],
            "score": self.score,
            "hedge_ratios": list(self.eigenvectors[0]) if self.eigenvectors else [],
        }


def _compute_johansen_score(trace_stats: np.ndarray, crit_95: np.ndarray, coint_rank: int) -> float:
    """
    Compute a composite score for ranking Johansen results.
    
    Score = (trace_stat[0] - crit_95[0]) / crit_95[0] * (1 + coint_rank)
    Higher score = stronger cointegration evidence.
    """
    if coint_rank == 0:
        return 0.0
    margin = (trace_stats[0] - crit_95[0]) / crit_95[0]
    return float(margin * (1 + coint_rank))


def run_johansen_test(
    prices: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> JohansenResult:
    """
    Run the Johansen cointegration test and return full results including eigenvectors.

    Args:
        prices: DataFrame of asset prices (columns are tickers).
        det_order: Order of deterministic trend (-1 to 1).
        k_ar_diff: Number of lagged differences in the VECM.

    Returns:
        JohansenResult with trace stats, cointegration rank, and hedge ratios.
    """
    if prices.empty:
        raise ValueError("Price frame is empty; cannot run Johansen test")

    tickers = tuple(str(c).upper() for c in prices.columns)
    
    # Log-transform prices for the test
    log_prices = np.log(prices.where(prices > 0)).dropna()

    if log_prices.shape[0] < 30:
        raise ValueError(f"Need at least 30 observations, got {log_prices.shape[0]}")

    result: JohansenTestResult = coint_johansen(
        log_prices, det_order=det_order, k_ar_diff=k_ar_diff
    )

    trace_stats = result.lr1
    crit_vals = result.cvt  # Critical values (90%, 95%, 99%)
    eigenvalues = result.eig
    eigenvectors = result.evec  # Each column is a cointegrating vector

    # Count cointegration rank at 95% level
    coint_rank = int(np.sum(trace_stats > crit_vals[:, 1]))

    # Compute score for ranking
    score = _compute_johansen_score(trace_stats, crit_vals[:, 1], coint_rank)

    # Normalize eigenvectors: first element = 1 for interpretability
    normalized_evecs = []
    for i in range(eigenvectors.shape[1]):
        evec = eigenvectors[:, i]
        if abs(evec[0]) > 1e-10:
            evec = evec / evec[0]
        normalized_evecs.append(tuple(float(v) for v in evec))

    return JohansenResult(
        tickers=tickers,
        basket_size=len(tickers),
        cointegration_rank=coint_rank,
        trace_stats=tuple(float(t) for t in trace_stats),
        crit_90=tuple(float(c) for c in crit_vals[:, 0]),
        crit_95=tuple(float(c) for c in crit_vals[:, 1]),
        crit_99=tuple(float(c) for c in crit_vals[:, 2]),
        eigenvalues=tuple(float(e) for e in eigenvalues),
        eigenvectors=tuple(normalized_evecs),
        score=score,
    )


def scan_johansen_baskets(
    prices: pd.DataFrame,
    min_basket_size: int = 3,
    max_basket_size: int = 5,
    det_order: int = 0,
    k_ar_diff: int = 1,
    min_corr_prefilter: float | None = 0.70,
    max_baskets: int | None = 50,
) -> list[JohansenResult]:
    """
    Scan all ticker combinations for Johansen cointegration.

    Args:
        prices: DataFrame of asset prices (columns are tickers).
        min_basket_size: Minimum number of assets in a basket (default 3).
        max_basket_size: Maximum number of assets in a basket (default 5).
        det_order: Order of deterministic trend for Johansen test.
        k_ar_diff: Number of lagged differences in the VECM.
        min_corr_prefilter: If set, skip baskets where avg pairwise correlation < threshold.
        max_baskets: Maximum number of baskets to return (ranked by score).

    Returns:
        List of JohansenResult objects, sorted by score descending.
    """
    if prices.empty:
        return []

    prices = prices.copy()
    prices.columns = [str(c).upper() for c in prices.columns]
    tickers = list(prices.columns)
    
    # Pre-compute correlation matrix for filtering
    returns = prices.pct_change().dropna()
    corr_matrix = returns.corr()

    results: list[JohansenResult] = []
    tested = 0
    skipped_corr = 0
    skipped_error = 0

    for size in range(min_basket_size, max_basket_size + 1):
        for combo in combinations(tickers, size):
            # Pre-filter by average pairwise correlation
            if min_corr_prefilter is not None:
                combo_corrs = []
                for i, t1 in enumerate(combo):
                    for t2 in combo[i + 1:]:
                        combo_corrs.append(corr_matrix.loc[t1, t2])
                avg_corr = np.mean(combo_corrs)
                if avg_corr < min_corr_prefilter:
                    skipped_corr += 1
                    continue

            try:
                basket_prices = prices[list(combo)].dropna()
                result = run_johansen_test(basket_prices, det_order=det_order, k_ar_diff=k_ar_diff)
                if result.cointegration_rank > 0:
                    results.append(result)
                tested += 1
            except ValueError as e:
                logger.debug("Johansen test failed for %s: %s", combo, e)
                skipped_error += 1
                continue

    logger.info(
        "scan_johansen_baskets: tested=%d, cointegrated=%d, skipped_corr=%d, skipped_error=%d",
        tested, len(results), skipped_corr, skipped_error
    )

    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)

    if max_baskets is not None and len(results) > max_baskets:
        results = results[:max_baskets]

    return results
