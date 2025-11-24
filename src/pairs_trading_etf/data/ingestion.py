"""Data download, persistence, and validation helpers for ETF prices."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf


def download_etf_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Download daily adjusted close prices for ETFs from Yahoo Finance.

    Args:
        tickers:
            List of ticker symbols to download, e.g. ["XLK", "XLF"].
        start:
            ISO date string "YYYY-MM-DD" marking the inclusive start date.
        end:
            ISO date string marking the inclusive end date.
            Note: yfinance treats `end` as *exclusive*, so we add +1 day
            to really include the end date you pass in.

    Returns:
        Wide DataFrame with a DatetimeIndex and one column per ticker.
    """
    if not tickers:
        raise ValueError("At least one ticker is required for download_etf_data")

    # yfinance end date is exclusive → cộng 1 ngày để lấy đủ
    end_plus_one = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end_plus_one,
        interval="1d",
        auto_adjust=True,   # dùng giá đã điều chỉnh
        progress=False,
        group_by="column",
    )

    if raw.empty:
        raise RuntimeError(
            f"yfinance returned empty data for tickers={tickers} "
            f"from {start} to {end}"
        )

    # yfinance thường trả MultiIndex với level 0 = field ("Adj Close")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"]
        else:
            # fallback: lấy level đầu tiên bất kỳ (thường là 'Close')
            first_level = raw.columns.levels[0][0]
            prices = raw[first_level]
    else:
        # Đã là wide DataFrame rồi (1 field) → dùng luôn
        prices = raw

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]

    # Khi chỉ có một ticker, yfinance đôi khi trả Series
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    # Đảm bảo cột là đúng thứ tự tickers (nếu có sẵn)
    cols_in_data = [t for t in tickers if t in prices.columns]
    prices = prices[cols_in_data]

    return prices


def save_raw_data(df: pd.DataFrame, path: Path | str) -> None:
    """Persist raw price data to CSV, creating parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def validate_price_data(df: pd.DataFrame) -> Dict[str, object]:
    """
    Run lightweight validation checks on price data.

    Args:
        df: Wide DataFrame of prices indexed by date.

    Returns:
        Dictionary summarizing:
            - n_rows, n_cols
            - missing_pct per ticker
            - extreme_moves: count of days |return| > 15%
            - all_nan_columns: columns which are entirely NaN
    """
    if df.empty:
        return {
            "n_rows": 0,
            "n_cols": 0,
            "missing_pct": {},
            "extreme_moves": {},
            "all_nan_columns": [],
        }

    missing_pct = df.isna().mean().round(4).to_dict()
    returns = df.pct_change().abs()
    extreme = (returns > 0.15).sum().to_dict()

    summary: Dict[str, object] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_pct": missing_pct,
        "extreme_moves": extreme,
        "all_nan_columns": [col for col, pct in missing_pct.items() if pct == 1.0],
    }

    return summary
