"""Z-score based trading signal generation for pairs trading.

Implements the standard mean-reversion entry/exit logic:
- Entry: |z-score| >= entry_threshold (typically 2.0)
- Exit: z-score crosses 0 OR |z-score| <= exit_threshold OR time > max_holding

References:
- Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis"
- Gatev, E., Goetzmann, W.N., Rouwenhorst, K.G. (2006). "Pairs Trading"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Position(Enum):
    """Trading position states."""
    FLAT = 0
    LONG_SPREAD = 1   # Long leg_x, Short leg_y (spread expected to increase)
    SHORT_SPREAD = -1  # Short leg_x, Long leg_y (spread expected to decrease)


@dataclass
class TradeSignal:
    """A single trade signal."""
    date: pd.Timestamp
    position: Position
    z_score: float
    spread_value: float
    reason: str  # 'entry_long', 'entry_short', 'exit_mean', 'exit_threshold', 'exit_timeout'


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    entry_threshold: float = 2.0      # Entry when |z| >= this
    exit_threshold: float = 0.5       # Exit when |z| <= this (profit taking)
    exit_on_mean_cross: bool = True   # Exit when z crosses 0
    max_holding_periods: int | None = None  # Max bars to hold (e.g., 2 * half_life)
    lookback_window: int = 20         # Window for rolling z-score calculation


def calculate_z_score(
    spread: pd.Series,
    lookback: int = 20,
    use_expanding: bool = False,
) -> pd.Series:
    """Calculate rolling z-score of the spread.
    
    Parameters
    ----------
    spread : pd.Series
        The spread time series (log_x - hedge_ratio * log_y).
    lookback : int
        Rolling window size for mean/std calculation.
    use_expanding : bool
        If True, use expanding window instead of rolling.
    
    Returns
    -------
    pd.Series
        Z-score series: (spread - mean) / std
    """
    if use_expanding:
        mean = spread.expanding(min_periods=lookback).mean()
        std = spread.expanding(min_periods=lookback).std()
    else:
        mean = spread.rolling(window=lookback, min_periods=lookback).mean()
        std = spread.rolling(window=lookback, min_periods=lookback).std()
    
    z_score = (spread - mean) / std
    return z_score.replace([np.inf, -np.inf], np.nan)


def generate_signals(
    spread: pd.Series,
    config: SignalConfig | None = None,
    half_life: float | None = None,
) -> tuple[pd.DataFrame, list[TradeSignal]]:
    """Generate trading signals from spread z-scores.
    
    Parameters
    ----------
    spread : pd.Series
        The spread time series with datetime index.
    config : SignalConfig, optional
        Signal generation parameters. Uses defaults if not provided.
    half_life : float, optional
        If provided and config.max_holding_periods is None, 
        sets max holding to 2 * half_life.
    
    Returns
    -------
    tuple[pd.DataFrame, list[TradeSignal]]
        (signal_df with columns [z_score, position, signal], list of TradeSignal objects)
    """
    if config is None:
        config = SignalConfig()
    
    # Set max holding based on half-life if not specified
    max_holding = config.max_holding_periods
    if max_holding is None and half_life is not None:
        max_holding = int(2 * half_life)
    
    # Calculate z-scores
    z_score = calculate_z_score(spread, lookback=config.lookback_window)
    
    # Initialize output
    n = len(spread)
    positions = np.zeros(n, dtype=int)
    signals = []
    
    current_position = Position.FLAT
    entry_bar = None
    
    for i in range(config.lookback_window, n):
        date = spread.index[i]
        z = z_score.iloc[i]
        spread_val = spread.iloc[i]
        
        if np.isnan(z):
            positions[i] = current_position.value
            continue
        
        # Check for exit conditions first
        if current_position != Position.FLAT:
            bars_held = i - entry_bar if entry_bar is not None else 0
            should_exit = False
            exit_reason = None
            
            # Exit on mean cross
            if config.exit_on_mean_cross:
                if current_position == Position.LONG_SPREAD and z <= 0:
                    should_exit = True
                    exit_reason = "exit_mean_cross"
                elif current_position == Position.SHORT_SPREAD and z >= 0:
                    should_exit = True
                    exit_reason = "exit_mean_cross"
            
            # Exit on threshold (profit taking / convergence)
            if not should_exit and abs(z) <= config.exit_threshold:
                should_exit = True
                exit_reason = "exit_threshold"
            
            # Exit on timeout
            if not should_exit and max_holding is not None and bars_held >= max_holding:
                should_exit = True
                exit_reason = "exit_timeout"
            
            if should_exit:
                signals.append(TradeSignal(
                    date=date,
                    position=Position.FLAT,
                    z_score=z,
                    spread_value=spread_val,
                    reason=exit_reason,
                ))
                current_position = Position.FLAT
                entry_bar = None
        
        # Check for entry conditions
        if current_position == Position.FLAT:
            if z <= -config.entry_threshold:
                # Spread is low -> expect it to increase -> Long spread
                current_position = Position.LONG_SPREAD
                entry_bar = i
                signals.append(TradeSignal(
                    date=date,
                    position=Position.LONG_SPREAD,
                    z_score=z,
                    spread_value=spread_val,
                    reason="entry_long",
                ))
            elif z >= config.entry_threshold:
                # Spread is high -> expect it to decrease -> Short spread
                current_position = Position.SHORT_SPREAD
                entry_bar = i
                signals.append(TradeSignal(
                    date=date,
                    position=Position.SHORT_SPREAD,
                    z_score=z,
                    spread_value=spread_val,
                    reason="entry_short",
                ))
        
        positions[i] = current_position.value
    
    # Build output DataFrame
    signal_df = pd.DataFrame({
        'spread': spread,
        'z_score': z_score,
        'position': positions,
    }, index=spread.index)
    
    # Add signal column (1 for entry, -1 for exit, 0 for hold)
    signal_df['signal'] = 0
    for sig in signals:
        idx = signal_df.index.get_loc(sig.date)
        if sig.position == Position.FLAT:
            signal_df.iloc[idx, signal_df.columns.get_loc('signal')] = -1  # Exit
        else:
            signal_df.iloc[idx, signal_df.columns.get_loc('signal')] = 1  # Entry
    
    return signal_df, signals


def signals_to_dataframe(signals: Sequence[TradeSignal]) -> pd.DataFrame:
    """Convert list of TradeSignal objects to a DataFrame."""
    records = []
    for sig in signals:
        records.append({
            'date': sig.date,
            'position': sig.position.name,
            'position_value': sig.position.value,
            'z_score': sig.z_score,
            'spread_value': sig.spread_value,
            'reason': sig.reason,
        })
    return pd.DataFrame(records)


def summarize_signals(signals: Sequence[TradeSignal]) -> dict:
    """Generate summary statistics for a signal sequence."""
    if not signals:
        return {'total_signals': 0}
    
    entries = [s for s in signals if 'entry' in s.reason]
    exits = [s for s in signals if 'exit' in s.reason]
    
    exit_reasons = {}
    for s in exits:
        exit_reasons[s.reason] = exit_reasons.get(s.reason, 0) + 1
    
    return {
        'total_signals': len(signals),
        'total_entries': len(entries),
        'total_exits': len(exits),
        'long_entries': sum(1 for s in entries if s.position == Position.LONG_SPREAD),
        'short_entries': sum(1 for s in entries if s.position == Position.SHORT_SPREAD),
        'exit_reasons': exit_reasons,
    }
