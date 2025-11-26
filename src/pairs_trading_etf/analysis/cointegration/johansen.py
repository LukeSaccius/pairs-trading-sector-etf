"""Johansen cointegration test implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen, JohansenTestResult


def johansen_cointegration_test(
    prices: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> pd.DataFrame:
    """
    Run the Johansen cointegration test on a DataFrame of asset prices.

    Args:
        prices: DataFrame of asset prices (columns are tickers).
        det_order: Order of deterministic trend (-1 to 1).
                   -1: No constant, no trend.
                    0: Constant, no trend.
                    1: Constant and trend.
        k_ar_diff: Number of lagged differences in the VECM.

    Returns:
        DataFrame containing trace statistics and critical values.
    """
    if prices.empty:
        raise ValueError("Price frame is empty; cannot run Johansen test")

    # Log-transform prices for the test
    log_prices = np.log(prices.where(prices > 0)).dropna()

    if log_prices.empty:
        raise ValueError("Log-prices are empty after dropping NaNs/zeros")

    result: JohansenTestResult = coint_johansen(
        log_prices, det_order=det_order, k_ar_diff=k_ar_diff
    )

    trace_stats = result.lr1
    crit_vals = result.cvt  # Critical values (90%, 95%, 99%)

    # Create a summary DataFrame
    summary_df = pd.DataFrame(
        {
            "trace_stat": trace_stats,
            "crit_90": crit_vals[:, 0],
            "crit_95": crit_vals[:, 1],
            "crit_99": crit_vals[:, 2],
        },
        index=[f"r <= {i}" for i in range(len(trace_stats))],
    )

    return summary_df
