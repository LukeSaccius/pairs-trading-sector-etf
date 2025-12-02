"""Pair enumeration and scoring helpers used by scanning pipelines.

Optimized version with:
- Vectorized correlation matrix computation
- Parallel Engle-Granger tests using ThreadPoolExecutor
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from pairs_trading_etf.cointegration.engle_granger import EngleGrangerResult, run_engle_granger

logger = logging.getLogger(__name__)

# Number of parallel workers (default: CPU count - 1, min 1)
N_JOBS = max(1, (os.cpu_count() or 4) - 1)


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
    spread_range_pct: float | None = None

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
            "spread_range_pct": self.spread_range_pct,
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


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute full correlation matrix using vectorized operations.
    
    This is O(n²) but highly optimized by numpy/pandas.
    For 200 ETFs: ~0.1 seconds vs ~10 seconds iterative.
    """
    return returns.corr()


def filter_pairs_by_correlation(
    tickers: Sequence[str],
    corr_matrix: pd.DataFrame,
    min_corr: float = 0.60,
    max_corr: float = 0.99,
) -> list[tuple[str, str, float]]:
    """Filter pairs by correlation threshold using the pre-computed matrix.
    
    Returns list of (leg_x, leg_y, correlation) tuples.
    """
    filtered = []
    tickers = list(tickers)
    
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            leg_x, leg_y = tickers[i], tickers[j]
            if leg_x not in corr_matrix.columns or leg_y not in corr_matrix.columns:
                continue
            corr = corr_matrix.loc[leg_x, leg_y]
            if pd.isna(corr):
                continue
            if min_corr <= corr <= max_corr:
                filtered.append((leg_x, leg_y, float(corr)))
    
    return filtered


def _process_single_pair(
    leg_x: str,
    leg_y: str,
    correlation: float,
    prices_x: np.ndarray,
    prices_y: np.ndarray,
    index: pd.DatetimeIndex,
    n_obs: int,
    granger_kwargs: dict,
) -> PairScore | None:
    """Process a single pair - designed for parallel execution."""
    try:
        # Convert back to Series for engle_granger
        px = pd.Series(prices_x, index=index, name=leg_x)
        py = pd.Series(prices_y, index=index, name=leg_y)
        
        eg_result = run_engle_granger(px, py, **granger_kwargs)
        fields = _engle_granger_fields(eg_result)
        
        # Calculate spread range % on last 252 days
        spread_range_pct = None
        try:
            prices_1y_x = prices_x[-252:] if len(prices_x) >= 252 else prices_x
            prices_1y_y = prices_y[-252:] if len(prices_y) >= 252 else prices_y
            if len(prices_1y_x) >= 126:
                px_norm = prices_1y_x / prices_1y_x[0] * 100
                py_norm = prices_1y_y / prices_1y_y[0] * 100
                spread_pct = px_norm - py_norm
                spread_range_pct = float(np.max(spread_pct) - np.min(spread_pct))
        except Exception:
            pass
        
        return PairScore(
            leg_x=leg_x,
            leg_y=leg_y,
            correlation=correlation,
            n_obs=n_obs,
            spread_range_pct=spread_range_pct,
            **fields,
        )
    except Exception as e:
        logger.debug("EG failed for %s-%s: %s", leg_x, leg_y, e)
        return None


def score_pairs(
    prices: pd.DataFrame,
    min_obs: int = 252,
    min_corr: float = 0.60,
    max_corr: float = 0.99,
    lookback: int | None = None,
    max_pairs: int | None = None,
    run_cointegration: bool = True,
    engle_granger_kwargs: Mapping[str, object] | None = None,
    n_jobs: int | None = None,
) -> list[PairScore]:
    """Rank ETF pairs by correlation strength and Engle–Granger diagnostics.
    
    Optimized pipeline:
    1. Compute correlation matrix (vectorized, ~0.1s for 200 ETFs)
    2. Filter pairs by correlation threshold
    3. Run Engle-Granger tests in parallel
    
    Args:
        prices: DataFrame with ETF prices (columns = tickers)
        min_obs: Minimum overlapping observations required
        min_corr: Minimum correlation threshold
        max_corr: Maximum correlation threshold (filter duplicates)
        lookback: Optional lookback window (None = full history)
        max_pairs: Maximum pairs to return (None = all)
        run_cointegration: Whether to run Engle-Granger tests
        engle_granger_kwargs: Additional kwargs for EG test
        n_jobs: Number of parallel workers (None = auto)
    
    Returns:
        List of PairScore objects sorted by p-value (ascending)
    """
    if prices.empty:
        return []

    prices = prices.copy()
    prices.columns = [str(col).upper() for col in prices.columns]
    if lookback is not None and lookback > 0:
        prices = prices.tail(lookback)

    # Step 1: Compute returns and correlation matrix (FAST - vectorized)
    returns = prices.pct_change(fill_method=None).dropna()
    if returns.empty:
        return []

    logger.info("Computing correlation matrix for %d ETFs...", len(returns.columns))
    corr_matrix = compute_correlation_matrix(returns)
    
    # Step 2: Filter pairs by correlation (FAST)
    tickers = list(returns.columns)
    filtered_pairs = filter_pairs_by_correlation(tickers, corr_matrix, min_corr, max_corr)
    
    total_pairs = len(tickers) * (len(tickers) - 1) // 2
    logger.info(
        "Correlation filter: %d/%d pairs passed (%.1f%%)",
        len(filtered_pairs), total_pairs, 100 * len(filtered_pairs) / max(total_pairs, 1)
    )
    
    if not filtered_pairs:
        return []
    
    if not run_cointegration:
        # Return pairs without EG test
        return [
            PairScore(leg_x=x, leg_y=y, correlation=c, n_obs=len(returns))
            for x, y, c in filtered_pairs
        ]
    
    # Step 3: Prepare data for parallel processing
    granger_kwargs = dict(engle_granger_kwargs or {})
    workers = n_jobs if n_jobs is not None else N_JOBS
    
    # Pre-extract price data to numpy arrays (faster serialization)
    tasks = []
    for leg_x, leg_y, corr in filtered_pairs:
        pair_prices = prices[[leg_x, leg_y]].dropna()
        if len(pair_prices) < min_obs:
            continue
        tasks.append({
            "leg_x": leg_x,
            "leg_y": leg_y,
            "correlation": corr,
            "prices_x": pair_prices[leg_x].values,
            "prices_y": pair_prices[leg_y].values,
            "index": pair_prices.index,
            "n_obs": len(pair_prices),
            "granger_kwargs": granger_kwargs,
        })
    
    logger.info("Running Engle-Granger tests on %d pairs (workers=%d)...", len(tasks), workers)
    
    # Step 4: Run EG tests (parallel for large workloads)
    scored: list[PairScore] = []
    
    if workers > 1 and len(tasks) > 50:
        # Parallel execution using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_process_single_pair, **task)
                for task in tasks
            ]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    scored.append(result)
    else:
        # Sequential for small workloads
        for task in tasks:
            result = _process_single_pair(**task)
            if result is not None:
                scored.append(result)
    
    logger.info("Scored %d pairs successfully", len(scored))
    
    # Sort by p-value (best cointegration first)
    def _sort_key(item: PairScore) -> tuple[float, float]:
        pvalue = item.coint_pvalue if item.coint_pvalue is not None else 1.0
        return (pvalue, -item.correlation)
    
    scored.sort(key=_sort_key)
    
    if max_pairs is not None and max_pairs > 0:
        scored = scored[:max_pairs]
    
    return scored
