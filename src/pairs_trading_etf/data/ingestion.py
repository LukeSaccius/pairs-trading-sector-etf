"""Data download, persistence, and validation helpers for ETF prices."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf


def download_etf_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for ETFs from Yahoo Finance.

    Args:
        tickers: List of ticker symbols to download.
        start: ISO date string (YYYY-MM-DD) marking the inclusive start date.
        end: ISO date string marking the inclusive end date.

    Returns:
        Wide DataFrame with a DatetimeIndex and one column per ticker.
    """

    if not tickers:
        raise ValueError("At least one ticker is required for download_etf_data")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if "Adj Close" in raw:
        prices = raw["Adj Close"]
    else:
        prices = raw

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    return prices


def save_raw_data(df: pd.DataFrame, path: Path) -> None:
    """Persist raw price data to CSV, creating parent folders if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def validate_price_data(df: pd.DataFrame) -> Dict[str, object]:
    """Run lightweight validation checks on price data.

    Args:
        df: Wide DataFrame of prices indexed by date.

    Returns:
        Dictionary summarizing row/column counts, missing percentages, and
        counts of extreme daily moves (>15% absolute returns).
    """

    if df.empty:
        return {
            "n_rows": 0,
            "n_cols": 0,
            "missing_pct": {},
            "extreme_moves": {},
        }

    missing_pct = df.isna().mean().round(4).to_dict()
    returns = df.pct_change().abs()
    extreme = (returns > 0.15).sum().to_dict()

    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_pct": missing_pct,
        "extreme_moves": extreme,
        "all_nan_columns": [col for col, pct in missing_pct.items() if pct == 1.0],
    }

    return summary
