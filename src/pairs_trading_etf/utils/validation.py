"""Train/test split and walk-forward validation utilities.

Provides rolling window validation to prevent overfitting in pairs trading research.
The formation-trading period approach follows Gatev et al. (2006).

References:
- Gatev, E., Goetzmann, W.N., Rouwenhorst, K.G. (2006). 
  "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """A single train/test window in walk-forward validation."""
    
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days
    
    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days
    
    def __repr__(self) -> str:
        return (
            f"Window {self.window_id}: "
            f"Train [{self.train_start.date()} - {self.train_end.date()}] "
            f"→ Test [{self.test_start.date()} - {self.test_end.date()}]"
        )


def walk_forward_split(
    df: pd.DataFrame,
    train_months: int = 12,
    test_months: int = 6,
    step_months: int | None = None,
    date_column: str | None = None,
) -> Iterator[tuple[WalkForwardWindow, pd.DataFrame, pd.DataFrame]]:
    """Generate walk-forward train/test splits from a time-series DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index or a date column.
    train_months : int
        Length of formation/training period in months.
    test_months : int
        Length of trading/test period in months.
    step_months : int, optional
        Step size between windows. Defaults to test_months (non-overlapping test).
    date_column : str, optional
        If provided, use this column for dates instead of the index.
    
    Yields
    ------
    tuple[WalkForwardWindow, pd.DataFrame, pd.DataFrame]
        (window_info, train_df, test_df) for each rolling window.
        
    Notes
    -----
    The Gatev et al. (2006) approach uses:
    - 12-month formation period (identify pairs, estimate parameters)
    - 6-month trading period (evaluate performance out-of-sample)
    - Roll forward by 6 months (non-overlapping test periods)
    
    Example
    -------
    >>> for window, train_df, test_df in walk_forward_split(prices, 12, 6):
    ...     # Estimate parameters on train_df
    ...     # Backtest on test_df
    ...     print(window)
    """
    if step_months is None:
        step_months = test_months
    
    # Get date index
    if date_column is not None:
        dates = pd.to_datetime(df[date_column])
        df = df.set_index(dates)
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex or specify date_column")
    
    df = df.sort_index()
    
    start_date = df.index.min()
    end_date = df.index.max()
    
    window_id = 0
    current_train_start = start_date
    
    while True:
        # Calculate window boundaries
        train_end = current_train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        
        # Check if we have enough data
        if test_end > end_date:
            logger.debug(
                "Walk-forward split: stopping at window %d (test_end %s > data end %s)",
                window_id, test_end.date(), end_date.date()
            )
            break
        
        # Extract train and test sets
        train_df = df[(df.index >= current_train_start) & (df.index < train_end)]
        test_df = df[(df.index >= test_start) & (df.index < test_end)]
        
        if train_df.empty or test_df.empty:
            logger.warning("Empty train or test set at window %d, skipping", window_id)
            current_train_start += pd.DateOffset(months=step_months)
            continue
        
        window = WalkForwardWindow(
            window_id=window_id,
            train_start=current_train_start.to_pydatetime(),
            train_end=train_end.to_pydatetime(),
            test_start=test_start.to_pydatetime(),
            test_end=test_end.to_pydatetime(),
        )
        
        yield window, train_df, test_df
        
        window_id += 1
        current_train_start += pd.DateOffset(months=step_months)


def simple_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    date_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple chronological train/test split (no rolling).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index or a date column.
    train_ratio : float
        Fraction of data to use for training (0-1).
    date_column : str, optional
        If provided, use this column for dates instead of the index.
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    if date_column is not None:
        df = df.copy()
        df = df.set_index(pd.to_datetime(df[date_column]))
    
    df = df.sort_index()
    
    n = len(df)
    split_idx = int(n * train_ratio)
    
    return df.iloc[:split_idx], df.iloc[split_idx:]


def count_walk_forward_windows(
    start_date: datetime | str,
    end_date: datetime | str,
    train_months: int = 12,
    test_months: int = 6,
    step_months: int | None = None,
) -> int:
    """Count how many walk-forward windows can be generated.
    
    Useful for estimating computational cost before running full validation.
    """
    if step_months is None:
        step_months = test_months
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    count = 0
    current = start
    
    while True:
        train_end = current + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        
        if test_end > end:
            break
        
        count += 1
        current += pd.DateOffset(months=step_months)
    
    return count
