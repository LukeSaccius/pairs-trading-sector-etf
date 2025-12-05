"""
Validation utilities for pairs trading backtests.

This module provides:
- Pair stability validation (train/val split within formation period)
- Rolling consistency checks
- Adaptive half-life estimation
- Market regime detection

References:
- Bailey et al. (2015) "The Probability of Backtest Overfitting"
- Vidyamurthy (2004) "Pairs Trading: Quantitative Methods and Analysis"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd
from scipy import stats

from .cpcv_correct import WalkForwardCPCV

if TYPE_CHECKING:
    from .config import BacktestConfig

logger = logging.getLogger(__name__)


# =============================================================================
# PAIR STABILITY VALIDATION
# =============================================================================

def validate_pair_stability(
    prices: pd.DataFrame,
    pair: Tuple[str, str],
    use_log: bool = True,
    pvalue_threshold: float = 0.10,
    min_half_life: float = 5.0,
    max_half_life: float = 30.0,
    train_pct: float = 0.75,
) -> Dict[str, Any]:
    """
    Validate pair selection stability across train/validation split.
    
    Splits formation period into:
    - Train: First 75% (e.g., Jan-Sep)
    - Validation: Last 25% (e.g., Oct-Dec)
    
    Pair passes if:
    1. Cointegrated in BOTH periods
    2. Half-life is similar (within 100% deviation)
    3. Hedge ratio is similar (within 40% deviation)
    
    Parameters
    ----------
    prices : pd.DataFrame
        Formation period price data
    pair : tuple
        (ticker_x, ticker_y) pair
    use_log : bool
        Use log prices for spread calculation
    pvalue_threshold : float
        Maximum p-value for cointegration test
    min_half_life : float
        Minimum half-life in days
    max_half_life : float
        Maximum half-life in days
    train_pct : float
        Fraction of data for training (default 0.75)
        
    Returns
    -------
    dict
        Contains: stable, reason, train_hl, val_hl, stability_score
    """
    from .engine import run_engle_granger_test
    
    n = len(prices)
    split_idx = int(n * train_pct)
    
    if split_idx < 60 or (n - split_idx) < 30:
        return {'stable': False, 'reason': 'insufficient_data'}
    
    train_prices = prices.iloc[:split_idx]
    val_prices = prices.iloc[split_idx:]
    
    leg_x, leg_y = pair
    
    # Check if both tickers exist
    if leg_x not in prices.columns or leg_y not in prices.columns:
        return {'stable': False, 'reason': 'missing_tickers'}
    
    # Test on train period
    train_result = run_engle_granger_test(
        train_prices[leg_x], 
        train_prices[leg_y],
        use_log=use_log,
        pvalue_threshold=pvalue_threshold,
        min_half_life=min_half_life,
        max_half_life=max_half_life,
    )
    
    if train_result is None:
        return {'stable': False, 'reason': 'train_failed'}
    
    # Test on validation period with relaxed thresholds
    val_result = run_engle_granger_test(
        val_prices[leg_x], 
        val_prices[leg_y],
        use_log=use_log,
        pvalue_threshold=pvalue_threshold * 1.5,  # Slightly relaxed
        min_half_life=min_half_life * 0.5,
        max_half_life=max_half_life * 2.0,
    )
    
    if val_result is None:
        return {'stable': False, 'reason': 'validation_failed'}
    
    # Check stability metrics
    # Guard against division by zero and near-zero values (epsilon = 1e-8)
    EPSILON = 1e-8
    if abs(train_result['half_life']) < EPSILON or abs(train_result['hedge_ratio']) < EPSILON:
        return {'stable': False, 'reason': 'zero_train_values'}

    hl_ratio = val_result['half_life'] / max(abs(train_result['half_life']), EPSILON)
    hr_ratio = abs(val_result['hedge_ratio']) / max(abs(train_result['hedge_ratio']), EPSILON)
    
    # Stability thresholds
    hl_stable = 0.5 <= hl_ratio <= 2.0
    hr_stable = 0.6 <= hr_ratio <= 1.67
    
    if not hl_stable:
        return {
            'stable': False, 
            'reason': f'hl_unstable: ratio={hl_ratio:.2f}',
            'train_hl': train_result['half_life'],
            'val_hl': val_result['half_life'],
        }
    
    if not hr_stable:
        return {
            'stable': False, 
            'reason': f'hr_unstable: ratio={hr_ratio:.2f}',
            'train_hr': train_result['hedge_ratio'],
            'val_hr': val_result['hedge_ratio'],
        }
    
    # Calculate stability score (0-1, higher is better)
    hl_deviation = abs(hl_ratio - 1.0)
    hr_deviation = abs(hr_ratio - 1.0)
    stability_score = 1.0 - (hl_deviation + hr_deviation) / 2
    stability_score = max(0, stability_score)
    
    return {
        'stable': True,
        'reason': 'passed',
        'train_hl': train_result['half_life'],
        'val_hl': val_result['half_life'],
        'train_hr': train_result['hedge_ratio'],
        'val_hr': val_result['hedge_ratio'],
        'hl_ratio': hl_ratio,
        'hr_ratio': hr_ratio,
        'stability_score': stability_score,
        'train_pvalue': train_result['pvalue'],
        'val_pvalue': val_result['pvalue'],
    }


# =============================================================================
# ROLLING CONSISTENCY CHECK
# =============================================================================

def check_rolling_consistency(
    prices: pd.DataFrame,
    pair: Tuple[str, str],
    use_log: bool = True,
    pvalue_threshold: float = 0.10,
    min_half_life: float = 5.0,
    max_half_life: float = 30.0,
    n_windows: int = 4,
    min_passing: int = 2,
) -> Dict[str, Any]:
    """
    Check if cointegration is consistent across rolling windows.
    
    Splits formation period into n_windows overlapping windows,
    tests each for cointegration.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Formation period price data
    pair : tuple
        (ticker_x, ticker_y) pair
    n_windows : int
        Number of sub-windows to test
    min_passing : int
        Minimum windows that must pass cointegration test
        
    Returns
    -------
    dict
        Contains: passes, passing_windows, score, hl_values, hr_values
    """
    from .engine import run_engle_granger_test
    
    leg_x, leg_y = pair
    n = len(prices)
    
    if leg_x not in prices.columns or leg_y not in prices.columns:
        return {'passes': False, 'reason': 'missing_tickers'}
    
    # Calculate window parameters
    # Use overlapping windows for better coverage
    window_size = int(n * 0.6)  # 60% of data per window
    step = (n - window_size) // max(1, n_windows - 1)
    
    if window_size < 60:
        return {'passes': False, 'reason': 'insufficient_data'}
    
    passing_windows = 0
    hl_values = []
    hr_values = []
    pvalues = []
    
    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + window_size
        
        if end_idx > n:
            end_idx = n
            start_idx = max(0, end_idx - window_size)
        
        window_prices = prices.iloc[start_idx:end_idx]
        
        if len(window_prices) < 60:
            continue
        
        result = run_engle_granger_test(
            window_prices[leg_x], 
            window_prices[leg_y],
            use_log=use_log,
            pvalue_threshold=pvalue_threshold * 1.2,  # Slightly relaxed
            min_half_life=min_half_life * 0.7,
            max_half_life=max_half_life * 1.5,
        )
        
        if result is not None:
            passing_windows += 1
            hl_values.append(result['half_life'])
            hr_values.append(result['hedge_ratio'])
            pvalues.append(result['pvalue'])
    
    passes = passing_windows >= min_passing
    
    # Stability score based on consistency of HL and HR across windows
    if len(hl_values) >= 2:
        hl_mean = np.mean(hl_values)
        hr_mean = np.mean(hr_values)
        # Guard against division by zero and near-zero values
        EPSILON = 1e-8
        hl_cv = np.std(hl_values) / max(abs(hl_mean), EPSILON) if abs(hl_mean) > EPSILON else 1.0
        hr_cv = np.std(hr_values) / max(abs(hr_mean), EPSILON) if abs(hr_mean) > EPSILON else 1.0
        score = max(0, 1.0 - (hl_cv + hr_cv) / 2)
    else:
        score = 0.0 if not passes else 0.5
    
    return {
        'passes': passes,
        'passing_windows': passing_windows,
        'total_windows': n_windows,
        'min_required': min_passing,
        'score': score,
        'hl_values': hl_values,
        'hr_values': hr_values,
        'pvalues': pvalues,
        'hl_mean': np.mean(hl_values) if hl_values else None,
        'hl_std': np.std(hl_values) if len(hl_values) >= 2 else None,
        'hr_mean': np.mean(hr_values) if hr_values else None,
    }


# =============================================================================
# ADAPTIVE HALF-LIFE ESTIMATION
# =============================================================================

def estimate_current_half_life(
    spread_history: pd.Series,
    min_samples: int = 20,
) -> Optional[float]:
    """
    Re-estimate half-life from recent spread history.
    
    Uses AR(1) model on spread changes to estimate current half-life.
    
    Parameters
    ----------
    spread_history : pd.Series
        Recent spread values
    min_samples : int
        Minimum samples required for estimation
        
    Returns
    -------
    float or None
        Current half-life estimate, or None if invalid
    """
    if len(spread_history) < min_samples:
        return None
    
    spread = spread_history.iloc[-min_samples:]
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    
    common_idx = spread_lag.index.intersection(spread_diff.index)
    if len(common_idx) < 10:
        return None
    
    spread_lag = spread_lag.loc[common_idx].values
    spread_diff = spread_diff.loc[common_idx].values
    
    try:
        slope, _, _, _, _ = stats.linregress(spread_lag, spread_diff)
    except Exception:
        return None
    
    if slope >= 0:
        return None  # Not mean-reverting
    
    phi = 1 + slope
    if phi <= 0 or phi >= 1:
        return None
    
    half_life = -np.log(2) / np.log(phi)
    
    # Sanity check
    if not (1 <= half_life <= 200):
        return None
    
    return half_life


def get_adaptive_max_holding(
    trade_spread_history: pd.Series,
    formation_half_life: float,
    max_holding_multiplier: float = 2.5,
    max_holding_days: int = 45,
    blend_weight: float = 0.5,
) -> int:
    """
    Get adaptive max holding based on current half-life estimate.
    
    Blends formation period HL with current estimate for robustness.
    
    Parameters
    ----------
    trade_spread_history : pd.Series
        Spread history during trade
    formation_half_life : float
        Half-life estimated during formation period
    max_holding_multiplier : float
        Multiplier for half-life to get max holding
    max_holding_days : int
        Absolute maximum holding period
    blend_weight : float
        Weight for current HL (0-1), rest goes to formation HL
        
    Returns
    -------
    int
        Adaptive max holding period in days
    """
    current_hl = estimate_current_half_life(trade_spread_history)
    
    if current_hl is not None and 3 <= current_hl <= 100:
        # Blend formation and current estimates
        effective_hl = blend_weight * current_hl + (1 - blend_weight) * formation_half_life
    else:
        effective_hl = formation_half_life
    
    max_hold = int(max_holding_multiplier * effective_hl)
    max_hold = min(max_hold, max_holding_days)
    max_hold = max(max_hold, 5)  # Minimum 5 days
    
    return max_hold


# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================

@dataclass
class RegimeInfo:
    """Market regime information."""
    regime: str  # 'HIGH_VOL', 'LOW_VOL', 'NORMAL'
    vix_level: float
    vix_percentile: float
    market_trend: float  # 60-day market return
    vol_regime_score: float  # 0-1, higher = more volatile
    
    def __repr__(self) -> str:
        return f"RegimeInfo({self.regime}, VIX={self.vix_level:.1f}, trend={self.market_trend:.1%})"


def detect_market_regime(
    prices: pd.DataFrame,
    current_date: pd.Timestamp,
    lookback_days: int = 60,
) -> RegimeInfo:
    """
    Detect current market regime for parameter adjustment.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data including VIX if available
    current_date : pd.Timestamp
        Current trading date
    lookback_days : int
        Days to look back for trend calculation
        
    Returns
    -------
    RegimeInfo
        Current market regime information
    """
    # Get VIX if available
    vix_col = None
    for col in ['VIX', '^VIX', 'VIXY']:
        if col in prices.columns:
            vix_col = col
            break
    
    if vix_col:
        vix_data = prices[vix_col]
        available = vix_data[vix_data.index <= current_date].dropna()
        current_vix = available.iloc[-1] if len(available) > 0 else 20.0
        
        # VIX percentile over last 252 days
        vix_history = available.iloc[-252:] if len(available) >= 252 else available
        vix_percentile = (vix_history < current_vix).mean() if len(vix_history) > 0 else 0.5
    else:
        current_vix = 20.0
        vix_percentile = 0.5
    
    # Market trend (SPY or first column)
    spy_col = None
    for col in ['SPY', 'IVV', 'VOO']:
        if col in prices.columns:
            spy_col = col
            break
    
    if spy_col is None:
        spy_col = prices.columns[0]
    
    spy_data = prices[spy_col]
    available = spy_data[spy_data.index <= current_date].dropna()
    
    if len(available) >= lookback_days:
        market_trend = available.iloc[-1] / available.iloc[-lookback_days] - 1
    else:
        market_trend = 0.0
    
    # Classify regime
    if current_vix > 30:
        regime = 'HIGH_VOL'
        vol_score = min(current_vix / 50, 1.0)
    elif current_vix < 15:
        regime = 'LOW_VOL'
        vol_score = 0.2
    else:
        regime = 'NORMAL'
        vol_score = (current_vix - 15) / 15  # 0.0 to 1.0
    
    return RegimeInfo(
        regime=regime,
        vix_level=float(current_vix),
        vix_percentile=float(vix_percentile),
        market_trend=float(market_trend),
        vol_regime_score=float(vol_score),
    )


def adjust_config_for_regime(
    cfg: 'BacktestConfig',
    regime: RegimeInfo,
) -> 'BacktestConfig':
    """
    Adjust config parameters based on market regime.
    
    Returns a modified copy of the config.
    
    Parameters
    ----------
    cfg : BacktestConfig
        Original configuration
    regime : RegimeInfo
        Current market regime
        
    Returns
    -------
    BacktestConfig
        Adjusted configuration
    """
    from .config import BacktestConfig
    
    # Create copy
    adjusted = BacktestConfig(**cfg.to_dict())
    
    if regime.regime == 'HIGH_VOL':
        # In high vol: be more selective, smaller positions
        adjusted.entry_zscore = cfg.entry_zscore * 0.9  # Enter earlier (more opportunities)
        adjusted.max_positions = max(3, cfg.max_positions - 2)  # Fewer positions
        adjusted.vol_size_max = 1.5  # Cap position sizes
        adjusted.stop_loss_zscore = cfg.stop_loss_zscore * 0.9  # Tighter stops
        
    elif regime.regime == 'LOW_VOL':
        # In low vol: be more patient, larger positions OK
        adjusted.entry_zscore = cfg.entry_zscore * 1.1  # More selective
        adjusted.max_positions = cfg.max_positions + 2  # More positions OK
        adjusted.vol_size_max = 2.5  # Allow larger positions
        
    return adjusted


# =============================================================================
# MINIMUM VOLATILITY THRESHOLD
# =============================================================================

def calculate_safe_vol_sizing(
    base_capital: float,
    spread_volatility: float,
    target_volatility: float = 0.02,
    min_scale: float = 0.25,
    max_scale: float = 2.0,
    min_vol_threshold: float = 0.003,  # 0.3% minimum
) -> Tuple[float, float]:
    """
    Calculate volatility-adjusted position size with safety bounds.
    
    Adds minimum volatility threshold to prevent extreme scaling.
    
    Parameters
    ----------
    base_capital : float
        Base capital allocation
    spread_volatility : float
        Daily volatility of spread
    target_volatility : float
        Target daily vol (default 2%)
    min_scale : float
        Minimum position scale (25%)
    max_scale : float
        Maximum position scale (200%)
    min_vol_threshold : float
        Minimum spread vol to prevent division issues
        
    Returns
    -------
    tuple
        (adjusted_capital, scale_factor)
    """
    # Apply minimum volatility threshold
    effective_vol = max(spread_volatility, min_vol_threshold)
    
    # Handle edge cases
    if effective_vol <= 0 or np.isnan(effective_vol):
        return base_capital, 1.0
    
    # Calculate scale factor
    scale = target_volatility / effective_vol
    scale = max(min_scale, min(max_scale, scale))
    
    return base_capital * scale, scale


# =============================================================================
# PURGED WALK-FORWARD VALIDATOR
# =============================================================================


@dataclass
class WalkForwardSplitResult:
    """Per-split walk-forward metrics."""

    train_year: int
    test_year: int
    train_return: float
    test_return: float
    train_days: int
    test_days: int
    positive: bool


@dataclass
class WalkForwardValidationResult:
    """Aggregate walk-forward validation output."""

    splits: List[WalkForwardSplitResult]
    avg_is_return: float
    avg_oos_return: float
    positive_ratio: float
    purge_days: int
    embargo_days: int
    min_positive_ratio: float
    min_avg_oos_return: float
    passed: bool
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "PURGED WALK-FORWARD VALIDATION",
            f"  Splits analyzed: {len(self.splits)}",
            f"  Purge / Embargo: {self.purge_days}d / {self.embargo_days}d",
            f"  Avg IS return: {self.avg_is_return:.4%}",
            f"  Avg OOS return: {self.avg_oos_return:.4%}",
            f"  Positive OOS splits: {self.positive_ratio:.1%}",
            f"  Minimum positive ratio: {self.min_positive_ratio:.1%}",
            f"  Minimum avg OOS return: {self.min_avg_oos_return:.4%}",
            f"  Passed: {'YES' if self.passed else 'NO'}",
        ]

        if self.warnings:
            lines.append("  Warnings:")
            for warn in self.warnings:
                lines.append(f"    - {warn}")

        return "\n".join(lines)


class PurgedWalkForwardValidator:
    """Optional validator that reports purged walk-forward health."""

    def __init__(
        self,
        train_years: int = 1,
        test_years: int = 1,
        min_positive_ratio: float = 0.55,
        min_avg_oos_return: float = 0.0,
        default_purge_days: int = 21,
        default_embargo_days: int = 5,
    ) -> None:
        self.train_years = train_years
        self.test_years = test_years
        self.min_positive_ratio = min_positive_ratio
        self.min_avg_oos_return = min_avg_oos_return
        self.default_purge_days = default_purge_days
        self.default_embargo_days = default_embargo_days

    def evaluate(
        self,
        returns_series: np.ndarray,
        dates: pd.DatetimeIndex,
        purge_days: Optional[int] = None,
        embargo_days: Optional[int] = None,
    ) -> WalkForwardValidationResult:
        """Run purged walk-forward validation on daily returns."""
        if len(dates) == 0 or len(returns_series) == 0:
            raise ValueError("No data available for walk-forward validation")

        if len(returns_series) != len(dates):
            raise ValueError("Returns series and dates must have the same length")

        purge = int(purge_days) if purge_days else self.default_purge_days
        embargo = int(embargo_days) if embargo_days else self.default_embargo_days

        wf = WalkForwardCPCV(
            train_years=self.train_years,
            test_years=self.test_years,
            purge_days=purge,
            embargo_days=embargo,
        )

        splits = wf.generate_splits(dates)
        if not splits:
            raise ValueError("Not enough history for walk-forward validation")

        returns = np.asarray(returns_series, dtype=float)
        split_results: List[WalkForwardSplitResult] = []
        positive_count = 0

        for train_mask, test_mask, train_year, test_year in splits:
            train_ret = float(returns[train_mask].sum())
            test_ret = float(returns[test_mask].sum())
            positive = test_ret > 0
            if positive:
                positive_count += 1

            split_results.append(
                WalkForwardSplitResult(
                    train_year=train_year,
                    test_year=test_year,
                    train_return=train_ret,
                    test_return=test_ret,
                    train_days=int(train_mask.sum()),
                    test_days=int(test_mask.sum()),
                    positive=positive,
                )
            )

        avg_is = float(np.mean([s.train_return for s in split_results]))
        avg_oos = float(np.mean([s.test_return for s in split_results]))
        positive_ratio = positive_count / len(split_results)

        warnings: List[str] = []
        if positive_ratio < self.min_positive_ratio:
            warnings.append(
                f"OOS positive ratio {positive_ratio:.1%} < {self.min_positive_ratio:.1%}"
            )
        if avg_oos < self.min_avg_oos_return:
            warnings.append(
                f"Average OOS return {avg_oos:.4%} < {self.min_avg_oos_return:.4%}"
            )

        passed = positive_ratio >= self.min_positive_ratio and avg_oos >= self.min_avg_oos_return

        return WalkForwardValidationResult(
            splits=split_results,
            avg_is_return=avg_is,
            avg_oos_return=avg_oos,
            positive_ratio=positive_ratio,
            purge_days=purge,
            embargo_days=embargo,
            min_positive_ratio=self.min_positive_ratio,
            min_avg_oos_return=self.min_avg_oos_return,
            passed=passed,
            warnings=warnings,
        )
