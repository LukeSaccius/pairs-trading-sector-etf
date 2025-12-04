"""
Backtest configuration management.

This module provides a unified configuration system for pairs trading backtests,
supporting both programmatic configuration via dataclasses and YAML file loading.
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

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
    pvalue_threshold: float = 0.05   # E-G cointegration p-value
    min_half_life: float = 5.0       # Minimum half-life (days)
    max_half_life: float = 30.0      # Maximum half-life (days)
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
    # Trading Signals
    # ==========================================================================
    entry_zscore: float = 2.0        # Z-score for entry (recommend 2.5 for quality)
    exit_zscore: float = 0.5         # Z-score for exit (mean reversion)
    stop_loss_zscore: float = 4.0    # Z-score for stop-loss
    zscore_lookback: int = 60        # Rolling window for z-score
    
    # ==========================================================================
    # Hedge Ratio Filter (NEW - improves win rate)
    # ==========================================================================
    min_hedge_ratio: float = 0.5     # Min |HR| - avoids directional bets
    max_hedge_ratio: float = 2.0     # Max |HR| - avoids over-hedging
    
    # ==========================================================================
    # Position Management
    # ==========================================================================
    max_holding_days: int = 45       # Maximum days to hold position (fallback)
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
    # Vidyamurthy Framework - SNR & Tradability Filters
    # ==========================================================================
    min_snr: float = 0.0             # Minimum Signal-to-Noise Ratio (0 = disabled, recommend 1.5+)
    min_zero_crossing_rate: float = 0.0  # Min zero crossings per year (0 = disabled, recommend 5+)
    time_based_stops: bool = False   # Enable time-based stop tightening
    stop_tightening_rate: float = 0.1  # Rate of stop tightening per half-life elapsed
    
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
