"""
CSCV-Integrated Backtest Framework.

This module implements a proper 3-phase backtesting framework:
1. TRAIN: Parameter optimization (multiple configurations)
2. VALIDATION: CSCV analysis to detect overfitting
3. TEST: Final unbiased evaluation

The key insight from Bailey & López de Prado (2014):
- PBO (Probability of Backtest Overfitting) measures likelihood that 
  the best in-sample strategy will underperform out-of-sample
- If PBO > 0.50, the strategy selection is likely overfit

References:
- Bailey et al. (2014) "The Probability of Backtest Overfitting"
- Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .engine import run_walkforward_backtest
from .cross_validation import (
    CSCVResult,
    run_cscv_analysis,
    calculate_deflated_sharpe,
    print_cscv_report,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA SPLIT CONFIGURATION
# =============================================================================

@dataclass
class CSCVBacktestSplit:
    """
    Three-phase split for CSCV-integrated backtesting.
    
    Default periods:
    - TRAIN: 2009-2016 (8 years) - Test multiple configurations
    - VALIDATION: 2017-2020 (4 years) - CSCV analysis
    - TEST: 2021-2024 (4 years) - Final unbiased evaluation
    
    CRITICAL: TEST period must NEVER be used for any optimization!
    """
    train_start: int = 2009
    train_end: int = 2016
    val_start: int = 2017
    val_end: int = 2020
    test_start: int = 2021
    test_end: int = 2024
    
    def __post_init__(self) -> None:
        assert self.train_end < self.val_start, "Train/Val overlap!"
        assert self.val_end < self.test_start, "Val/Test overlap!"
    
    @property
    def train_years(self) -> range:
        return range(self.train_start, self.train_end + 1)
    
    @property
    def val_years(self) -> range:
        return range(self.val_start, self.val_end + 1)
    
    @property
    def test_years(self) -> range:
        return range(self.test_start, self.test_end + 1)


# =============================================================================
# PARAMETER GRID
# =============================================================================

@dataclass
class ParameterGrid:
    """Define parameter ranges for grid search."""
    
    entry_zscore: list[float] = field(default_factory=lambda: [2.5, 2.8, 3.0])
    exit_zscore: list[float] = field(default_factory=lambda: [0.3, 0.5])
    max_positions: list[int] = field(default_factory=lambda: [5, 8, 10])
    max_half_life: list[float] = field(default_factory=lambda: [20, 25, 30])
    vol_size_min: list[float] = field(default_factory=lambda: [0.3, 0.5])
    
    def generate_configs(
        self, base_config: BacktestConfig
    ) -> list[tuple[str, BacktestConfig]]:
        """Generate all config combinations."""
        configs = []
        
        for entry_z, exit_z, max_pos, max_hl, vol_min in product(
            self.entry_zscore,
            self.exit_zscore,
            self.max_positions,
            self.max_half_life,
            self.vol_size_min,
        ):
            # Create modified config
            cfg_dict = base_config.to_dict()
            cfg_dict['entry_zscore'] = entry_z
            cfg_dict['exit_zscore'] = exit_z
            cfg_dict['max_positions'] = max_pos
            cfg_dict['max_half_life'] = max_hl
            cfg_dict['vol_size_min'] = vol_min
            
            name = f"ez{entry_z}_xz{exit_z}_mp{max_pos}_hl{max_hl}_vs{vol_min}"
            configs.append((name, BacktestConfig(**cfg_dict)))
        
        return configs
    
    @property
    def n_configs(self) -> int:
        return (len(self.entry_zscore) * len(self.exit_zscore) * 
                len(self.max_positions) * len(self.max_half_life) * 
                len(self.vol_size_min))


# =============================================================================
# CSCV BACKTEST RESULT
# =============================================================================

@dataclass
class CSCVBacktestResult:
    """Complete result from CSCV-integrated backtest."""
    
    # Split info
    split: CSCVBacktestSplit
    
    # Training phase results
    n_configs_tested: int = 0
    train_results: dict[str, dict] = field(default_factory=dict)
    
    # Validation phase results
    cscv_result: CSCVResult | None = None
    deflated_sharpe: float = 0.0
    dsr_p_value: float = 1.0
    
    # Best config from validation
    best_config_name: str = ""
    best_config: BacktestConfig | None = None
    
    # Final test results (ONLY filled after test phase)
    test_pnl: float | None = None
    test_sharpe: float | None = None
    test_trades: int | None = None
    test_win_rate: float | None = None
    
    @property
    def is_overfit(self) -> bool:
        """Check if strategy is overfit based on CSCV."""
        if self.cscv_result is None:
            return True  # No CSCV = assume overfit
        return self.cscv_result.pbo > 0.50 or self.deflated_sharpe < 0
    
    @property
    def recommendation(self) -> str:
        """Get recommendation based on CSCV results."""
        if self.cscv_result is None:
            return "NO CSCV - Cannot assess overfitting"
        
        pbo = self.cscv_result.pbo
        if pbo < 0.25 and self.deflated_sharpe > 0:
            return "✅ LOW overfitting risk - Proceed to test"
        elif pbo < 0.50:
            return "⚠️ MODERATE overfitting risk - Use caution"
        else:
            return "❌ HIGH overfitting risk - DO NOT deploy"
    
    def to_dict(self) -> dict:
        return {
            'n_configs_tested': self.n_configs_tested,
            'best_config': self.best_config_name,
            'pbo': self.cscv_result.pbo if self.cscv_result else None,
            'deflated_sharpe': self.deflated_sharpe,
            'is_overfit': self.is_overfit,
            'recommendation': self.recommendation,
            'test_pnl': self.test_pnl,
            'test_sharpe': self.test_sharpe,
        }


# =============================================================================
# CSCV BACKTEST RUNNER
# =============================================================================

def run_cscv_backtest(
    prices: pd.DataFrame,
    base_config: BacktestConfig,
    split: CSCVBacktestSplit | None = None,
    param_grid: ParameterGrid | None = None,
    n_cscv_partitions: int = 8,
    verbose: bool = True,
) -> CSCVBacktestResult:
    """
    Run complete CSCV-integrated backtest.
    
    Three phases:
    1. TRAIN: Run backtest for all parameter configurations
    2. VALIDATION: Run CSCV analysis to detect overfitting
    3. TEST: If not overfit, evaluate on held-out test period
    
    Parameters
    ----------
    prices : pd.DataFrame
        Full price data covering all periods
    base_config : BacktestConfig
        Base configuration to modify
    split : CSCVBacktestSplit, optional
        Period definitions (default: 2009-2016/2017-2020/2021-2024)
    param_grid : ParameterGrid, optional
        Parameter ranges for grid search
    n_cscv_partitions : int
        Number of partitions for CSCV (default 8)
    verbose : bool
        Print progress and results
        
    Returns
    -------
    CSCVBacktestResult
        Complete results including CSCV analysis
    """
    if split is None:
        split = CSCVBacktestSplit()
    
    if param_grid is None:
        param_grid = ParameterGrid()
    
    result = CSCVBacktestResult(split=split)
    
    # Generate all configurations
    configs = param_grid.generate_configs(base_config)
    result.n_configs_tested = len(configs)
    
    if verbose:
        print("=" * 70)
        print("CSCV-INTEGRATED BACKTEST")
        print("=" * 70)
        print(f"\nPhase 1: TRAINING ({split.train_start}-{split.train_end})")
        print(f"Testing {len(configs)} configurations...")
    
    # ==========================================================================
    # PHASE 1: TRAINING - Run all configurations on train period
    # ==========================================================================
    
    train_prices = prices[
        (prices.index >= f'{split.train_start}-01-01') &
        (prices.index <= f'{split.train_end}-12-31')
    ]
    
    train_returns: dict[str, pd.Series] = {}  # config_name -> daily returns series
    train_metrics: dict[str, dict] = {}  # config_name -> {pnl, sharpe, trades, ...}
    
    for i, (name, cfg) in enumerate(configs):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Config {i+1}/{len(configs)}: {name}")
        
        try:
            trades, _ = run_walkforward_backtest(
                train_prices, cfg,
                start_year=split.train_start + 1,  # +1 because first year is formation
                end_year=split.train_end,
            )
            
            total_pnl = sum(t['pnl'] for t in trades)
            n_trades = len(trades)
            win_rate = sum(1 for t in trades if t['pnl'] > 0) / n_trades if n_trades > 0 else 0
            
            # Build daily returns
            daily_pnl = _trades_to_daily_returns(trades, train_prices.index)
            train_returns[name] = daily_pnl
            
            train_metrics[name] = {
                'pnl': total_pnl,
                'trades': n_trades,
                'win_rate': win_rate,
                'config': cfg,
            }
            
        except Exception:
            logger.warning(f"Config {name} failed")
            continue
    
    result.train_results = train_metrics
    
    if verbose:
        print(f"\n  Completed {len(train_metrics)} configurations")
        
        # Show top 5 by PnL
        sorted_configs = sorted(train_metrics.items(), key=lambda x: x[1]['pnl'], reverse=True)
        print("\n  Top 5 by Train PnL:")
        for name, m in sorted_configs[:5]:
            print(f"    {name}: ${m['pnl']:,.0f} ({m['trades']} trades, {m['win_rate']*100:.0f}% WR)")
    
    # ==========================================================================
    # PHASE 2: VALIDATION - CSCV Analysis
    # ==========================================================================
    
    if verbose:
        print(f"\nPhase 2: VALIDATION ({split.val_start}-{split.val_end})")
        print("Running CSCV analysis...")
    
    val_prices = prices[
        (prices.index >= f'{split.val_start}-01-01') &
        (prices.index <= f'{split.val_end}-12-31')
    ]
    
    # Run all configs on validation period
    val_returns: dict[str, pd.Series] = {}
    val_metrics: dict[str, dict] = {}
    
    for name, cfg in configs:
        if name not in train_metrics:
            continue  # Skip failed configs
        
        try:
            trades, _ = run_walkforward_backtest(
                val_prices, cfg,
                start_year=split.val_start + 1,
                end_year=split.val_end,
            )
            
            total_pnl = sum(t['pnl'] for t in trades)
            daily_pnl = _trades_to_daily_returns(trades, val_prices.index)
            
            val_returns[name] = daily_pnl
            val_metrics[name] = {
                'pnl': total_pnl,
                'trades': len(trades),
            }
            
        except Exception:
            continue
    
    # Build returns matrix for CSCV
    # Shape: (n_days, n_strategies)
    common_names = [n for n in train_returns.keys() if n in val_returns]
    
    if len(common_names) < 3:
        logger.error("Not enough configurations for CSCV analysis")
        return result
    
    # Combine train and val for CSCV partitioning
    all_dates = prices[
        (prices.index >= f'{split.train_start}-01-01') &
        (prices.index <= f'{split.val_end}-12-31')
    ].index
    
    returns_matrix = np.zeros((len(all_dates), len(common_names)))
    for i, name in enumerate(common_names):
        # Combine train and val returns
        combined = pd.concat([train_returns[name], val_returns[name]]).reindex(all_dates).fillna(0)
        returns_matrix[:, i] = combined.values
    
    # Run CSCV
    cscv_result = run_cscv_analysis(
        returns_matrix,
        n_partitions=n_cscv_partitions,
        strategy_names=common_names,
    )
    result.cscv_result = cscv_result
    
    # Calculate Deflated Sharpe for best config
    best_train_name = max(train_metrics.keys(), key=lambda x: train_metrics[x]['pnl'])
    best_train_pnl = train_metrics[best_train_name]['pnl']
    n_years = (split.train_end - split.train_start)
    annual_return = best_train_pnl / base_config.initial_capital / n_years
    sharpe_approx = annual_return / 0.10  # Assume 10% vol
    
    dsr, p_value = calculate_deflated_sharpe(
        sharpe_approx,
        n_trials=len(configs),
        backtest_years=n_years,
    )
    result.deflated_sharpe = dsr
    result.dsr_p_value = p_value
    
    if verbose:
        print_cscv_report(cscv_result)
        print(f"\nDeflated Sharpe Ratio: {dsr:.2f} (p={p_value:.4f})")
        print(f"\n{result.recommendation}")
    
    # Select best config based on VALIDATION performance
    best_val_name = max(val_metrics.keys(), key=lambda x: val_metrics[x]['pnl'])
    result.best_config_name = best_val_name
    result.best_config = train_metrics[best_val_name]['config']
    
    if verbose:
        print(f"\nBest config (by Val PnL): {best_val_name}")
        print(f"  Train PnL: ${train_metrics[best_val_name]['pnl']:,.0f}")
        print(f"  Val PnL: ${val_metrics[best_val_name]['pnl']:,.0f}")
    
    # ==========================================================================
    # PHASE 3: TEST - Final Evaluation (ONLY if not overfit)
    # ==========================================================================
    
    if result.is_overfit:
        if verbose:
            print("\n" + "=" * 70)
            print("⚠️  STOPPING: Strategy appears OVERFIT")
            print("    PBO > 0.50 or DSR < 0")
            print("    DO NOT proceed to test phase")
            print("=" * 70)
        return result
    
    if verbose:
        print(f"\nPhase 3: TEST ({split.test_start}-{split.test_end})")
        print("=" * 70)
        print("⚠️  FINAL UNBIASED EVALUATION")
        print("    Do NOT iterate based on these results!")
        print("=" * 70)
    
    test_prices = prices[
        (prices.index >= f'{split.test_start}-01-01') &
        (prices.index <= f'{split.test_end}-12-31')
    ]
    
    trades, _ = run_walkforward_backtest(
        test_prices, 
        result.best_config,
        start_year=split.test_start + 1,
        end_year=split.test_end,
    )
    
    result.test_pnl = sum(t['pnl'] for t in trades)
    result.test_trades = len(trades)
    result.test_win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) if trades else 0
    
    # Calculate test Sharpe
    n_test_years = split.test_end - split.test_start
    test_annual = result.test_pnl / base_config.initial_capital / n_test_years
    result.test_sharpe = test_annual / 0.10 if test_annual else 0
    
    if verbose:
        print("\nTEST RESULTS:")
        print(f"  PnL: ${result.test_pnl:,.2f}")
        print(f"  Trades: {result.test_trades}")
        print(f"  Win Rate: {result.test_win_rate*100:.1f}%")
        print(f"  Sharpe (approx): {result.test_sharpe:.2f}")
    
    return result


def _trades_to_daily_returns(
    trades: list[dict],
    date_index: pd.DatetimeIndex,
) -> pd.Series:
    """Convert trade list to daily PnL series."""
    daily_pnl = pd.Series(0.0, index=date_index)
    
    for trade in trades:
        entry = pd.Timestamp(trade['entry_date'])
        exit_date = pd.Timestamp(trade['exit_date'])
        holding_days = trade.get('holding_days', 1) or 1
        daily_trade_pnl = trade['pnl'] / holding_days
        
        # Add to each day in holding period
        for day in pd.date_range(entry, exit_date):
            if day in daily_pnl.index:
                daily_pnl[day] += daily_trade_pnl
    
    return daily_pnl


# =============================================================================
# QUICK CSCV VALIDATION (for existing backtest)
# =============================================================================

def validate_existing_backtest(
    trades_df: pd.DataFrame,
    n_configs_tested: int = 10,
    backtest_years: float = 14.0,
    initial_capital: float = 50000.0,
) -> dict:
    """
    Quick CSCV validation for an existing backtest result.
    
    This is a simplified version when you only have one configuration.
    It calculates Deflated Sharpe Ratio to assess overfitting.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        Trades with columns: pnl, entry_date, exit_date, holding_days
    n_configs_tested : int
        Estimate of how many configurations were tried
    backtest_years : float
        Length of backtest in years
    initial_capital : float
        Initial capital
        
    Returns
    -------
    dict
        Validation results
    """
    total_pnl = trades_df['pnl'].sum()
    n_trades = len(trades_df)
    win_rate = (trades_df['pnl'] > 0).mean()
    
    # Calculate Sharpe approximation
    annual_return = total_pnl / initial_capital / backtest_years
    
    # Daily volatility from trades
    trades_df = trades_df.copy()
    trades_df['daily_pnl'] = trades_df['pnl'] / trades_df['holding_days']
    daily_std = trades_df['daily_pnl'].std()
    annual_vol = daily_std * np.sqrt(252) / initial_capital
    
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Deflated Sharpe
    dsr, p_value = calculate_deflated_sharpe(
        sharpe,
        n_trials=n_configs_tested,
        backtest_years=backtest_years,
    )
    
    return {
        'total_pnl': total_pnl,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'deflated_sharpe': dsr,
        'p_value': p_value,
        'n_configs_tested': n_configs_tested,
        'is_overfit': dsr < 0,
        'recommendation': "✅ PASS" if dsr > 0 else "❌ LIKELY OVERFIT",
    }
