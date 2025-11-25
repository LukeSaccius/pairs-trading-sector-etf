"""Helpers for loading and preparing stored ETF price data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import warnings

import numpy as np
import pandas as pd


class PriceLoaderError(RuntimeError):
    """Raised when persisted price data cannot be loaded or prepared."""


@dataclass(slots=True, frozen=True)
class PriceFrame:
    """Convenience wrapper bundling aligned prices and returns."""

    prices: pd.DataFrame
    returns: pd.DataFrame

    def slice_last(self, periods: int) -> "PriceFrame":
        """Return a new PriceFrame limited to the most recent `periods` rows."""

        if periods <= 0:
            return self
        prices_tail = self.prices.tail(periods)
        idx = prices_tail.index
        returns_tail = self.returns.loc[idx.intersection(self.returns.index)]
        return PriceFrame(prices=prices_tail, returns=returns_tail)


def _sanitize_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_price_history(path: str | Path) -> pd.DataFrame:
    """Load a wide price CSV that was produced by ingestion.save_raw_data."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise PriceLoaderError(f"Price file not found at {csv_path}")

    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    except Exception as exc:  # pragma: no cover - pandas raises rich errors
        raise PriceLoaderError(f"Failed to read price CSV {csv_path}: {exc}") from exc

    if df.empty:
        raise PriceLoaderError(f"Price file {csv_path} is empty")

    return _sanitize_frame(df)


def select_tickers(
    df: pd.DataFrame,
    tickers: Sequence[str] | None = None,
    *,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Return a view containing only the requested tickers (if provided)."""

    if tickers is None:
        return df

    normalized = [str(t).strip().upper() for t in tickers]
    missing = [t for t in normalized if t not in df.columns]
    if missing and not allow_missing:
        raise PriceLoaderError(
            f"Requested tickers missing from price frame: {', '.join(missing)}"
        )

    if missing and allow_missing:
        warnings.warn(
            "Dropping tickers without price history: " + ", ".join(missing),
            UserWarning,
            stacklevel=2,
        )
        normalized = [t for t in normalized if t in df.columns]

    if not normalized:
        raise PriceLoaderError("No usable tickers remain after filtering price history")

    return df[normalized]


def drop_sparse_columns(df: pd.DataFrame, min_non_na: int) -> pd.DataFrame:
    """Remove tickers with insufficient observations."""

    if min_non_na <= 0:
        return df

    valid_cols = [col for col in df.columns if df[col].count() >= min_non_na]
    if not valid_cols:
        raise PriceLoaderError(
            "Dropping sparse columns removed every ticker; relax min_non_na"
        )
    return df[valid_cols]


def compute_returns(df: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute daily returns from price levels."""

    if method not in {"log", "simple"}:
        raise ValueError("method must be 'log' or 'simple'")

    if method == "log":
        safe_prices = df.where(df > 0)
        returns = np.log(safe_prices).diff()
    else:
        returns = df.pct_change()

    returns = returns.replace([np.inf, -np.inf], pd.NA)
    return returns.dropna(how="all")


def build_price_frame(
    price_path: str | Path,
    tickers: Sequence[str] | None = None,
    min_non_na: int = 252,
    return_method: str = "log",
    *,
    allow_missing: bool = False,
) -> PriceFrame:
    """Load, filter, and compute returns for downstream analytics."""

    prices = load_price_history(price_path)
    prices = select_tickers(prices, tickers, allow_missing=allow_missing)
    prices = drop_sparse_columns(prices, min_non_na=min_non_na)
    returns = compute_returns(prices, method=return_method)
    return PriceFrame(prices=prices, returns=returns)
