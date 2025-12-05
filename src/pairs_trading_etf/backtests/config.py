"""
Backtest configuration management.

This module provides a unified configuration system for pairs trading backtests,
supporting both programmatic configuration via dataclasses and YAML file loading.
"""

from __future__ import annotations

import yaml
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from ..utils.sectors import DEFAULT_EXCLUDED_SECTORS


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class BacktestConfig:
    """
    Configuration for pairs trading backtest.
    
    This unified config replaces the scattered configurations in various scripts.
    Can be loaded from YAML or created programmatically.
    """
    
    # ==========================================================================
    # Experiment Metadata
    # ==========================================================================
    experiment_name: str = "default"
    description: str = ""
    
    # ==========================================================================
    # Time Windows
    # ==========================================================================
    formation_days: int = 252        # 1 year for pair selection
    trading_days: int = 252          # 1 year trading period
    hedge_update_days: int = 63      # Quarterly hedge ratio update
    
    # ==========================================================================
    # Cointegration Testing
    # ==========================================================================
    # WARNING: p-value must be 0.01 or 0.05 ONLY. Never increase above 0.05!
    # Higher p-values lead to false positives and poor out-of-sample performance.
    # Per research findings: "trong những dự án sau tuyệt đối không được chỉnh lên"
    pvalue_threshold: float = 0.05   # E-G cointegration p-value (LOCKED: 0.01 or 0.05)
    min_half_life: float = 2.0       # Ch.7: Too fast = noise (< 2 days)
    max_half_life: float = 50.0      # Ch.7: Too slow = capital inefficient (> 50 days)
    use_log_prices: bool = True      # Use log prices for spread
    
    # ==========================================================================
    # Correlation Filter
    # ==========================================================================
    min_correlation: float = 0.75    # Minimum correlation
    max_correlation: float = 0.95    # Maximum (avoid identical ETFs)
    
    # ==========================================================================
    # Rolling Consistency (optional)
    # ==========================================================================
    rolling_consistency: bool = False  # Check consistency across windows
    n_rolling_windows: int = 4        # Number of sub-windows
    min_passing_windows: int = 2      # Windows that must pass
    
    # ==========================================================================
    # Pair Selection
    # ==========================================================================
    top_pairs: int = 20              # Max pairs to select per year
    max_pairs_per_sector: int = 5    # Diversification limit per sector
    max_pairs_per_etf: int = 2       # Diversification limit per ETF
    min_spread_range_pct: float = 0.02  # Min expected spread movement
    
    # ==========================================================================
    # Sector Focus
    # ==========================================================================
    sector_focus: bool = True        # Only same-sector pairs
    exclude_sectors: Tuple[str, ...] = DEFAULT_EXCLUDED_SECTORS
    
    # ==========================================================================
    # Trading Signals (Vidyamurthy Ch.8: Optimal Threshold Design)
    # ==========================================================================
    # Per Ch.8: Optimal threshold Δ* ≈ 0.75σ maximizes profit function f(Δ) = Δ × [1 - N(Δ)]
    # This balances profit-per-trade (higher Δ) vs trade frequency (lower Δ)
    # Traditional z-score (2.0-2.5) is statistically motivated, NOT economically optimal
    # Use compute_optimal_threshold() to verify: solves d/dΔ[Δ(1-N(Δ))] = 0 → Δ* ≈ 0.7477
    entry_threshold_sigma: float = 0.75   # Ch.8 optimal Δ* (NOT traditional 2.0!)
    exit_threshold_sigma: float = 0.0     # Exit at mean (spread = equilibrium)
    exit_tolerance_sigma: float = 0.1     # Ch.8: Exit if |z - exit_threshold| <= tolerance
    stop_loss_sigma: float = 4.0          # Z-score stop (backup if time_based_stops=False)
    zscore_lookback: int = 60             # Default lookback (overridden if use_adaptive_lookback=True)
    
    # ==========================================================================
    # Adaptive Z-Score Lookback (QMA: lookback = f(half_life))
    # ==========================================================================
    # Per QMA: Lookback should scale with pair's half-life for consistent signal quality
    # Formula: lookback = clamp(4 * half_life, min_lookback, max_lookback)
    # Faster mean-reversion (small HL) → shorter lookback; slower → longer lookback
    use_adaptive_lookback: bool = True  # RECOMMENDED: True for QMA compliance
    adaptive_lookback_multiplier: float = 4.0  # lookback = multiplier * half_life
    adaptive_lookback_min: int = 30     # Minimum lookback (avoid noisy estimates)
    adaptive_lookback_max: int = 120    # Maximum lookback (avoid stale estimates)
    
    # ==========================================================================
    # QMA Level 2: Fixed Exit Parameters
    # ==========================================================================
    # Per Quantitative Methods for Algorithmic Trading (QMA):
    # Exit z-score should use FIXED μ_entry, σ_entry captured at entry time.
    # This prevents "Rolling Beta Trap" where exit z uses different distribution.
    # When enabled, exit z = (spread - μ_entry) / σ_entry
    use_fixed_exit_params: bool = True  # RECOMMENDED: True for QMA compliance
    
    # ==========================================================================
    # Hedge Ratio Filter (NEW - improves win rate)
    # ==========================================================================
    min_hedge_ratio: float = 0.5     # Min |HR| - avoids directional bets
    max_hedge_ratio: float = 2.0     # Max |HR| - avoids over-hedging
    
    # ==========================================================================
    # Position Management
    # ==========================================================================
    max_holding_days: int = 60       # Ch.8: ~3 × typical HL (fallback exit)
    max_positions: int = 10          # Max concurrent positions (0 = unlimited)
    dynamic_hedge: bool = True       # Update hedge ratio during trading
    dynamic_max_holding: bool = True # Scale max holding by half-life
    max_holding_multiplier: float = 3.0  # max_hold = min(multiplier * half_life, max_holding_days)
    
    # ==========================================================================
    # Capital Allocation (see engine.py for logic)
    # ==========================================================================
    # When compounding=True:
    #   position_capital = (current_capital * leverage) / max_positions
    #   capital_per_pair is IGNORED
    # When compounding=False:
    #   position_capital = capital_per_pair * leverage
    capital_per_pair: float = 10000.0  # Fixed notional per pair (ONLY used when compounding=False)
    
    # ==========================================================================
    # Vidyamurthy Framework - SNR & Tradability Filters (Ch.7)
    # ==========================================================================
    min_snr: float = 0.0             # Minimum Signal-to-Noise Ratio (0 = disabled, recommend 1.5+)
    min_zero_crossing_rate: float = 0.0  # Min zero crossings per year (0 = disabled, recommend 5+)
    time_based_stops: bool = True    # Ch.8: RECOMMENDED - time-based stop tightening
    stop_tightening_rate: float = 0.15 # Ch.8: Tighten 15% per HL elapsed
    
    # [Vidyamurthy:Ch.7:p114-115] Bootstrap procedure for holding period estimation
    use_bootstrap_holding_period: bool = True  # Use bootstrap for HL estimation
    bootstrap_n_samples: int = 1000  # Number of bootstrap resamples
    
    # ==========================================================================
    # Kalman Filter Dynamic Hedge Ratio
    # ==========================================================================
    # Based on Palomar (2025) Chapter 15.6 "Kalman Filtering for Pairs Trading"
    use_kalman_hedge: bool = False   # Use Kalman filter for dynamic hedge ratio
    kalman_delta: float = 0.00001    # Process noise (smaller = more stable, typical: 1e-5 to 1e-6)
    kalman_vw: float = 0.001         # Initial observation noise (will be adapted online)
    kalman_use_momentum: bool = True # Use momentum model (Eq. 15.4) for smoother hedge ratios
    kalman_zscore_regime: bool = True  # Use z-score based regime break (vs spread sign change)
    kalman_regime_zscore: float = 3.0  # Z-score threshold for regime break when using Kalman
    
    # ==========================================================================
    # VIX Regime Filter
    # ==========================================================================
    use_vix_filter: bool = False     # Enable VIX-based regime filter
    vix_threshold: float = 30.0      # Halt new entries when VIX > threshold
    vix_lookback_days: int = 5       # Days to average VIX over
    
    # ==========================================================================
    # Volatility-Adjusted Position Sizing
    # ==========================================================================
    use_vol_sizing: bool = False     # Enable volatility-adjusted position sizing
    target_daily_vol: float = 0.02   # Target daily volatility (2%)
    vol_size_min: float = 0.25       # Minimum position size (25% of base)
    vol_size_max: float = 2.0        # Maximum position size (200% of base)
    
    # ==========================================================================
    # Dynamic Z-Score Exit
    # ==========================================================================
    use_dynamic_z_exit: bool = False         # Enable dynamic exit based on Z-score divergence
    dynamic_z_exit_hl_ratio: float = 1.5     # Check after this many half-lives
    dynamic_z_exit_threshold: float = 0.0    # Exit if |current_z| >= |entry_z| + threshold
    
    # ==========================================================================
    # Slow Convergence Exit
    # ==========================================================================
    use_slow_convergence_exit: bool = False  # Exit if Z hasn't converged enough
    slow_conv_hl_ratio: float = 1.5          # Check after this many half-lives
    slow_conv_z_pct: float = 0.50            # Exit if Z remaining > this % of entry Z
    
    # ==========================================================================
    # Compounding & Leverage
    # ==========================================================================
    initial_capital: float = 50000.0   # Starting capital
    leverage: float = 1.0              # Leverage multiplier (2.0 = 2x leverage)
    compounding: bool = False          # Compound returns after each trade
    unlimited_pairs: bool = False      # Allow unlimited pairs (no top_pairs limit)
    max_capital_per_trade: float = 0.0 # Max capital per trade (0 = no limit)
    min_pairs_for_trading: int = 3     # Minimum pairs required to trade (risk diversification)
    
    # ==========================================================================
    # Costs and Risk
    # ==========================================================================
    transaction_cost_bps: float = 10.0  # Round-trip cost in basis points
    
    # ==========================================================================
    # Blacklist
    # ==========================================================================
    blacklist_stoploss_rate: float = 0.30  # Blacklist if >30% stop-loss
    blacklist_min_trades: int = 3          # Min trades before blacklisting
    
    # ==========================================================================
    # Output
    # ==========================================================================
    output_dir: str = "results"
    save_trades: bool = True
    save_summary: bool = True
    save_config_snapshot: bool = True
    timestamped_output: bool = True  # Create timestamped subfolder
    
    # ==========================================================================
    # Data Paths
    # ==========================================================================
    price_data_path: str = "data/raw/etf_prices_fresh.csv"
    etf_metadata_path: str = "configs/etf_metadata.yaml"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert tuple from list if loaded from YAML
        if isinstance(self.exclude_sectors, list):
            self.exclude_sectors = tuple(self.exclude_sectors)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        d = asdict(self)
        # Convert tuple to list for YAML serialization
        d['exclude_sectors'] = list(d['exclude_sectors'])
        return d
    
    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def get_output_path(self) -> Path:
        """Get output path, creating timestamped folder if needed."""
        base_path = Path(self.output_dir)
        
        if self.timestamped_output:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            folder_name = f"{timestamp}_{self.experiment_name}"
            output_path = base_path / folder_name
        else:
            output_path = base_path
        
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


# =============================================================================
# YAML LOADING
# =============================================================================

def load_config(path: str) -> BacktestConfig:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    path : str
        Path to YAML configuration file
        
    Returns
    -------
    BacktestConfig
        Configuration object
        
    Example
    -------
    >>> cfg = load_config('configs/experiments/conservative.yaml')
    >>> print(cfg.pvalue_threshold)
    0.01
    """
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Handle nested structure if present
    if 'backtest' in data:
        data = data['backtest']
    
    return BacktestConfig(**data)


def merge_configs(base: BacktestConfig, overrides: Dict[str, Any]) -> BacktestConfig:
    """
    Merge override values into base config.
    
    Parameters
    ----------
    base : BacktestConfig
        Base configuration
    overrides : dict
        Dictionary of values to override
        
    Returns
    -------
    BacktestConfig
        New configuration with overrides applied
    """
    base_dict = base.to_dict()
    base_dict.update(overrides)
    return BacktestConfig(**base_dict)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def compute_zscore_lookback(half_life: float) -> int:
    """
    Compute optimal zscore lookback window based on half-life.
    
    Per QMA research: lookback should scale with half-life to capture
    the mean-reverting behavior correctly.
    
    Formula: max(30, min(120, 4 * half_life))
    - At least 30 days for statistical significance
    - At most 120 days to avoid too much smoothing
    - 4x half_life as the base scaling factor
    
    Parameters
    ----------
    half_life : float
        The mean-reversion half-life in days
        
    Returns
    -------
    int
        Optimal lookback window for z-score calculation
    """
    return max(30, min(120, int(4 * half_life)))


def compute_optimal_threshold(slippage_bps: float = 0.0) -> float:
    """
    Compute optimal entry threshold per QMA Chapter 8.
    
    For white noise spread, the optimal threshold Δ* maximizes:
        f(Δ) = Δ × [1 - N(Δ)]
    
    where N(Δ) is the CDF of standard normal distribution.
    
    Solving the first-order condition:
        d/dΔ [Δ(1 - N(Δ))] = 0
        [1 - N(Δ)] - Δ × n(Δ) = 0
    
    gives Δ* ≈ 0.7477 (approximately 0.75σ).
    
    Interpretation:
    - Δ too small → many trades, but small profit per trade
    - Δ too large → big profit per trade, but few trades
    - Δ* = 0.75σ is the economically optimal balance
    
    Parameters
    ----------
    slippage_bps : float
        Transaction cost in basis points. If > 0, adjusts threshold
        to ensure profit > slippage.
        
    Returns
    -------
    float
        Optimal threshold in units of standard deviation
        
    Example
    -------
    >>> compute_optimal_threshold()
    0.7477  # approximately 0.75
    
    >>> compute_optimal_threshold(slippage_bps=10)  # With 10 bps slippage
    0.78  # Slightly higher to cover costs
    
    Notes
    -----
    This assumes white noise spread. For ARMA spreads, use Rice's formula
    to compute level-crossing rates (not implemented here).
    """
    # Profit function: f(delta) = delta * (1 - N(delta))
    # We want to MAXIMIZE this, so minimize the negative
    def neg_profit(delta: float) -> float:
        if delta <= 0:
            return 0.0
        return -delta * (1 - norm.cdf(delta))
    
    # Find optimal delta
    result = minimize_scalar(neg_profit, bounds=(0.1, 3.0), method='bounded')
    optimal_delta = result.x
    
    # Adjust for slippage if needed
    # Slippage in sigma units (rough approximation: 10 bps ≈ 0.01 sigma for typical spread)
    if slippage_bps > 0:
        slippage_sigma = slippage_bps / 1000  # Rough conversion
        # Ensure profit per trade (2*delta) > slippage
        min_delta = slippage_sigma / 2
        optimal_delta = max(optimal_delta, min_delta)
    
    return round(optimal_delta, 4)


def compute_nonparametric_threshold(
    spread_series: np.ndarray,
    slippage_bps: float = 10.0,
    n_levels: int = 30
) -> float:
    """
    Compute optimal threshold using nonparametric approach from QMA Chapter 8.
    
    Instead of assuming white noise, this method:
    1. Counts actual level crossings at various thresholds
    2. Computes profit = threshold × crossings for each level
    3. Returns threshold that maximizes profit
    
    This handles ARMA-like spreads that deviate from white noise assumption.
    
    Parameters
    ----------
    spread_series : np.ndarray
        Historical spread values (should be standardized: mean=0, std=1)
    slippage_bps : float
        Transaction cost in basis points
    n_levels : int
        Number of threshold levels to evaluate
        
    Returns
    -------
    float
        Optimal threshold based on historical data
    """
    # Standardize spread
    spread = np.asarray(spread_series)
    spread_std = (spread - np.mean(spread)) / np.std(spread)
    
    # Candidate thresholds
    deltas = np.linspace(0.3, 2.5, n_levels)
    profits = []
    
    for delta in deltas:
        # Count level crossings (transitions across ±delta)
        above_upper = spread_std >= delta
        below_lower = spread_std <= -delta
        
        # Entry signals: crossing into extreme region
        long_entries = ((~below_lower[:-1]) & below_lower[1:]).sum()
        short_entries = ((~above_upper[:-1]) & above_upper[1:]).sum()
        
        total_crossings = long_entries + short_entries
        
        # Profit per trade = 2 * delta (buy at -delta, sell at +delta)
        # Minus slippage (converted to sigma units)
        slippage_sigma = slippage_bps / 1000
        profit_per_trade = 2 * delta - slippage_sigma
        
        total_profit = profit_per_trade * total_crossings
        profits.append(total_profit)
    
    # Find optimal
    optimal_idx = np.argmax(profits)
    return round(deltas[optimal_idx], 2)


def bootstrap_holding_period(
    spread_series: np.ndarray,
    n_bootstrap: int = 1000,
    percentiles: Tuple[float, ...] = (5, 25, 50, 75, 95)
) -> Dict[str, float]:
    """
    Bootstrap estimate of holding period distribution per QMA Chapter 7.
    
    The time between zero crossings indicates expected holding period.
    This is used for time-based stop design.
    
    Parameters
    ----------
    spread_series : np.ndarray
        Historical spread values
    n_bootstrap : int
        Number of bootstrap samples
    percentiles : tuple
        Percentiles to report
        
    Returns
    -------
    dict
        Holding period statistics including median and percentiles
        
    Example
    -------
    >>> stats = bootstrap_holding_period(spread)
    >>> print(f"Median holding: {stats['median']:.1f} days")
    >>> print(f"95th percentile: {stats['p95']:.1f} days")
    """
    spread = np.asarray(spread_series)
    mean = np.mean(spread)
    
    # Find zero crossings (transitions across mean)
    above_mean = spread > mean
    crossings = np.where(above_mean[:-1] != above_mean[1:])[0]
    
    if len(crossings) < 2:
        # Not enough crossings
        return {'median': np.nan, 'mean': np.nan, 'p5': np.nan, 'p95': np.nan}
    
    # Time between crossings
    holding_times = np.diff(crossings)
    
    if len(holding_times) < 3:
        return {
            'median': np.median(holding_times),
            'mean': np.mean(holding_times),
            'p5': np.min(holding_times),
            'p95': np.max(holding_times)
        }
    
    # Bootstrap resampling
    bootstrap_medians = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(holding_times, size=len(holding_times), replace=True)
        bootstrap_medians.append(np.median(sample))
    
    result = {
        'median': np.median(holding_times),
        'mean': np.mean(holding_times),
        'std': np.std(holding_times),
        'min': np.min(holding_times),
        'max': np.max(holding_times),
    }
    
    for p in percentiles:
        result[f'p{int(p)}'] = np.percentile(holding_times, p)
    
    # Bootstrap confidence interval for median
    result['median_ci_low'] = np.percentile(bootstrap_medians, 2.5)
    result['median_ci_high'] = np.percentile(bootstrap_medians, 97.5)
    
    return result


def get_conservative_config() -> BacktestConfig:
    """Get conservative (low risk) configuration."""
    return BacktestConfig(
        experiment_name="conservative",
        description="Conservative settings: strict p-value, EUROPE focus",
        pvalue_threshold=0.01,
        min_half_life=5.0,
        max_half_life=15.0,
        min_correlation=0.80,
        max_correlation=0.95,
        sector_focus=True,
        exclude_sectors=('EMERGING', 'BONDS_GOV', 'US_GROWTH', 
                        'INDUSTRIALS', 'HEALTHCARE', 'COMMODITIES'),
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_loss_zscore=3.5,
        max_holding_days=30,
        top_pairs=10,
    )


def get_aggressive_config() -> BacktestConfig:
    """Get aggressive (higher risk) configuration."""
    return BacktestConfig(
        experiment_name="aggressive",
        description="Aggressive settings: relaxed filters, more sectors",
        pvalue_threshold=0.05,
        min_half_life=3.0,
        max_half_life=45.0,
        min_correlation=0.70,
        max_correlation=0.95,
        sector_focus=True,
        exclude_sectors=('EMERGING',),  # Only exclude emerging
        entry_zscore=1.5,
        exit_zscore=0.3,
        stop_loss_zscore=4.0,
        max_holding_days=60,
        top_pairs=25,
    )


def get_europe_only_config() -> BacktestConfig:
    """Get EUROPE-focused configuration (best performing sector)."""
    return BacktestConfig(
        experiment_name="europe_only",
        description="Focus only on EUROPE sector pairs",
        pvalue_threshold=0.05,
        sector_focus=True,
        exclude_sectors=tuple(s for s in DEFAULT_EXCLUDED_SECTORS) + 
                       ('US_BROAD', 'US_VALUE', 'US_SMALL', 'US_MID', 
                        'TECH', 'FINANCIALS', 'CONSUMER_DISC', 'CONSUMER_STAPLES',
                        'ENERGY', 'MATERIALS', 'UTILITIES', 'REITS',
                        'ASIA_DEV', 'BONDS_CORP', 'COMMODITIES'),
        max_pairs_per_sector=10,
        top_pairs=15,
    )
