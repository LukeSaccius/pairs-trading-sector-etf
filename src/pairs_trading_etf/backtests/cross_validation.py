"""
Cross-validation framework for pairs trading backtests.

This module provides:
- Proper train/validation/test split
- Parameter selection on validation set
- Unbiased final evaluation on test set

References:
- Bailey et al. (2015) "The Probability of Backtest Overfitting"
- Pardo (2008) "The Evaluation and Optimization of Trading Strategies"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BacktestConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DATA SPLIT CONFIGURATION
# =============================================================================

@dataclass
class BacktestSplit:
    """
    Define train/validation/test periods for unbiased evaluation.
    
    Default periods:
    - Train: 2009-2016 (8 years) - Explore parameters freely
    - Validation: 2017-2020 (4 years) - Select final config
    - Test: 2021-2024 (4 years) - Final unbiased evaluation
    """
    train_start: str = "2009-01-01"
    train_end: str = "2016-12-31"
    val_start: str = "2017-01-01"
    val_end: str = "2020-12-31"
    test_start: str = "2021-01-01"
    test_end: str = "2024-12-31"
    
    def __post_init__(self):
        # Convert to dates for comparison
        self._train_start = pd.Timestamp(self.train_start)
        self._train_end = pd.Timestamp(self.train_end)
        self._val_start = pd.Timestamp(self.val_start)
        self._val_end = pd.Timestamp(self.val_end)
        self._test_start = pd.Timestamp(self.test_start)
        self._test_end = pd.Timestamp(self.test_end)
        
        # Validate no overlap
        assert self._train_end < self._val_start, "Train/val overlap!"
        assert self._val_end < self._test_start, "Val/test overlap!"
    
    @property
    def train_period(self) -> Tuple[str, str]:
        return (self.train_start, self.train_end)
    
    @property
    def val_period(self) -> Tuple[str, str]:
        return (self.val_start, self.val_end)
    
    @property
    def test_period(self) -> Tuple[str, str]:
        return (self.test_start, self.test_end)
    
    def get_train_years(self) -> List[int]:
        return list(range(self._train_start.year, self._train_end.year + 1))
    
    def get_val_years(self) -> List[int]:
        return list(range(self._val_start.year, self._val_end.year + 1))
    
    def get_test_years(self) -> List[int]:
        return list(range(self._test_start.year, self._test_end.year + 1))
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'train_start': self.train_start,
            'train_end': self.train_end,
            'val_start': self.val_start,
            'val_end': self.val_end,
            'test_start': self.test_start,
            'test_end': self.test_end,
        }


# =============================================================================
# CROSS-VALIDATION RESULTS
# =============================================================================

@dataclass
class CVResult:
    """Results from cross-validated backtest."""
    config_name: str
    config: BacktestConfig
    
    # Train period results
    train_pnl: float = 0.0
    train_sharpe: float = 0.0
    train_win_rate: float = 0.0
    train_n_trades: int = 0
    
    # Validation period results
    val_pnl: float = 0.0
    val_sharpe: float = 0.0
    val_win_rate: float = 0.0
    val_n_trades: int = 0
    
    # Test period results (only filled after final evaluation)
    test_pnl: Optional[float] = None
    test_sharpe: Optional[float] = None
    test_win_rate: Optional[float] = None
    test_n_trades: Optional[int] = None
    
    # Overfitting detection
    train_to_val_pnl_ratio: float = 0.0
    train_to_val_sharpe_ratio: float = 0.0
    
    def calculate_ratios(self):
        """Calculate train-to-validation performance ratios."""
        if self.val_pnl != 0:
            self.train_to_val_pnl_ratio = self.train_pnl / self.val_pnl
        if self.val_sharpe != 0:
            self.train_to_val_sharpe_ratio = self.train_sharpe / self.val_sharpe
    
    @property
    def is_overfit(self) -> bool:
        """
        Detect potential overfitting.
        
        Signs of overfitting:
        1. Train >> Val (ratio > 3.0)
        2. Val PnL negative
        3. Significant Sharpe decay
        """
        if self.val_pnl <= 0:
            return True
        if self.train_to_val_pnl_ratio > 3.0:
            return True
        if self.val_sharpe < 0.3 and self.train_sharpe > 1.0:
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'config_name': self.config_name,
            'train': {
                'pnl': self.train_pnl,
                'sharpe': self.train_sharpe,
                'win_rate': self.train_win_rate,
                'n_trades': self.train_n_trades,
            },
            'validation': {
                'pnl': self.val_pnl,
                'sharpe': self.val_sharpe,
                'win_rate': self.val_win_rate,
                'n_trades': self.val_n_trades,
            },
            'test': {
                'pnl': self.test_pnl,
                'sharpe': self.test_sharpe,
                'win_rate': self.test_win_rate,
                'n_trades': self.test_n_trades,
            } if self.test_pnl is not None else None,
            'ratios': {
                'train_to_val_pnl': self.train_to_val_pnl_ratio,
                'train_to_val_sharpe': self.train_to_val_sharpe_ratio,
            },
            'is_overfit': self.is_overfit,
        }


# =============================================================================
# CROSS-VALIDATION RUNNER
# =============================================================================

def run_cross_validated_backtest(
    prices: pd.DataFrame,
    config: BacktestConfig,
    split: BacktestSplit,
    config_name: str = "unnamed",
) -> CVResult:
    """
    Run backtest with proper train/validation/test split.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Full price data covering all periods
    config : BacktestConfig
        Configuration to test
    split : BacktestSplit
        Period definitions
    config_name : str
        Name for this configuration
        
    Returns
    -------
    CVResult
        Results across all periods
    """
    from .engine import run_walkforward_backtest
    
    result = CVResult(config_name=config_name, config=config)
    
    # Filter prices to train period only
    train_prices = prices[
        (prices.index >= split.train_start) & 
        (prices.index <= split.train_end)
    ]
    
    if len(train_prices) < 252:
        logger.warning(f"Insufficient train data: {len(train_prices)} days")
        return result
    
    # Run backtest on train period
    logger.info(f"Running train period: {split.train_start} to {split.train_end}")
    train_result = run_walkforward_backtest(
        prices=train_prices,
        config=config,
        verbose=False,
    )
    
    if train_result and 'summary' in train_result:
        summary = train_result['summary']
        result.train_pnl = summary.get('total_pnl', 0)
        result.train_sharpe = summary.get('sharpe_ratio', 0) or 0
        result.train_win_rate = summary.get('win_rate', 0)
        result.train_n_trades = summary.get('total_trades', 0)
    
    # Run backtest on validation period
    val_prices = prices[
        (prices.index >= split.val_start) & 
        (prices.index <= split.val_end)
    ]
    
    if len(val_prices) < 252:
        logger.warning(f"Insufficient validation data: {len(val_prices)} days")
        return result
    
    logger.info(f"Running validation period: {split.val_start} to {split.val_end}")
    val_result = run_walkforward_backtest(
        prices=val_prices,
        config=config,
        verbose=False,
    )
    
    if val_result and 'summary' in val_result:
        summary = val_result['summary']
        result.val_pnl = summary.get('total_pnl', 0)
        result.val_sharpe = summary.get('sharpe_ratio', 0) or 0
        result.val_win_rate = summary.get('win_rate', 0)
        result.val_n_trades = summary.get('total_trades', 0)
    
    # Calculate overfitting ratios
    result.calculate_ratios()
    
    return result


def evaluate_on_test_set(
    prices: pd.DataFrame,
    result: CVResult,
    split: BacktestSplit,
) -> CVResult:
    """
    Final evaluation on test set.
    
    IMPORTANT: Only call this ONCE after selecting the best config
    from validation performance.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Full price data
    result : CVResult
        Result from train/val evaluation
    split : BacktestSplit
        Period definitions
        
    Returns
    -------
    CVResult
        Updated result with test metrics
    """
    from .engine import run_walkforward_backtest
    
    test_prices = prices[
        (prices.index >= split.test_start) & 
        (prices.index <= split.test_end)
    ]
    
    if len(test_prices) < 252:
        logger.warning(f"Insufficient test data: {len(test_prices)} days")
        return result
    
    logger.info(f"Running TEST period: {split.test_start} to {split.test_end}")
    logger.info("=" * 60)
    logger.info("FINAL UNBIASED EVALUATION - DO NOT ITERATE ON THESE RESULTS")
    logger.info("=" * 60)
    
    test_result = run_walkforward_backtest(
        prices=test_prices,
        config=result.config,
        verbose=True,
    )
    
    if test_result and 'summary' in test_result:
        summary = test_result['summary']
        result.test_pnl = summary.get('total_pnl', 0)
        result.test_sharpe = summary.get('sharpe_ratio', 0) or 0
        result.test_win_rate = summary.get('win_rate', 0)
        result.test_n_trades = summary.get('total_trades', 0)
    
    return result


# =============================================================================
# CONFIG SELECTION
# =============================================================================

def select_best_config(
    results: List[CVResult],
    metric: str = 'sharpe',
    min_trades: int = 30,
) -> Optional[CVResult]:
    """
    Select best configuration based on validation performance.
    
    Parameters
    ----------
    results : list
        List of CVResult from different configs
    metric : str
        'sharpe', 'pnl', or 'win_rate'
    min_trades : int
        Minimum trades required in validation period
        
    Returns
    -------
    CVResult or None
        Best performing config on validation set
    """
    valid_results = [
        r for r in results 
        if r.val_n_trades >= min_trades and not r.is_overfit
    ]
    
    if not valid_results:
        logger.warning("No configs passed validation criteria")
        # Fall back to non-overfit filter only
        valid_results = [r for r in results if not r.is_overfit]
    
    if not valid_results:
        logger.error("All configs appear overfit")
        return None
    
    if metric == 'sharpe':
        return max(valid_results, key=lambda r: r.val_sharpe)
    elif metric == 'pnl':
        return max(valid_results, key=lambda r: r.val_pnl)
    elif metric == 'win_rate':
        return max(valid_results, key=lambda r: r.val_win_rate)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# =============================================================================
# REPORTING
# =============================================================================

def print_cv_summary(results: List[CVResult]):
    """Print summary of cross-validation results."""
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Config':<25} {'Train PnL':>12} {'Val PnL':>12} {'Val Sharpe':>12} {'Overfit?':>10}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x.val_sharpe, reverse=True):
        overfit_flag = "YES" if r.is_overfit else "NO"
        print(f"{r.config_name:<25} ${r.train_pnl:>10,.0f} ${r.val_pnl:>10,.0f} "
              f"{r.val_sharpe:>11.2f} {overfit_flag:>10}")
    
    print("\n")


def save_cv_results(
    results: List[CVResult],
    output_path: str,
    split: BacktestSplit,
):
    """Save cross-validation results to JSON."""
    output = {
        'split': split.to_dict(),
        'results': [r.to_dict() for r in results],
        'best_config': None,
    }
    
    best = select_best_config(results)
    if best:
        output['best_config'] = best.config_name
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    logger.info(f"Saved CV results to {output_path}")


# =============================================================================
# PARAMETER GRID SEARCH
# =============================================================================

def generate_parameter_grid(
    base_config: BacktestConfig,
    param_ranges: Dict[str, List[Any]],
) -> List[Tuple[str, BacktestConfig]]:
    """
    Generate config grid for parameter search.
    
    Parameters
    ----------
    base_config : BacktestConfig
        Base configuration
    param_ranges : dict
        Parameter name -> list of values
        
    Returns
    -------
    list
        List of (name, config) tuples
    """
    from itertools import product
    
    configs = []
    
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    for values in product(*param_values):
        cfg_dict = base_config.to_dict()
        name_parts = []
        
        for param, value in zip(param_names, values):
            cfg_dict[param] = value
            name_parts.append(f"{param}={value}")
        
        name = "_".join(name_parts)
        configs.append((name, BacktestConfig(**cfg_dict)))
    
    return configs


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_cv_workflow():
    """
    Example cross-validation workflow.
    
    This demonstrates the proper way to evaluate a trading strategy:
    1. Explore parameters on TRAIN set only
    2. Select best config based on VALIDATION performance
    3. Evaluate ONCE on TEST set
    """
    example_code = '''
    # 1. Define split
    split = BacktestSplit(
        train_start="2009-01-01", train_end="2016-12-31",
        val_start="2017-01-01", val_end="2020-12-31",
        test_start="2021-01-01", test_end="2024-12-31",
    )
    
    # 2. Load data
    prices = load_prices(...)
    
    # 3. Generate parameter grid (ONLY use train period for exploration!)
    base_config = BacktestConfig.default()
    param_ranges = {
        'entry_zscore': [2.5, 2.8, 3.0],
        'vol_size_min': [0.3, 0.5, 0.7],
    }
    configs = generate_parameter_grid(base_config, param_ranges)
    
    # 4. Run CV on each config
    results = []
    for name, cfg in configs:
        result = run_cross_validated_backtest(prices, cfg, split, name)
        results.append(result)
    
    # 5. Select best config based on VALIDATION
    best = select_best_config(results, metric='sharpe')
    print(f"Best config: {best.config_name}")
    print(f"Validation Sharpe: {best.val_sharpe:.2f}")
    
    # 6. FINAL evaluation on test set (only once!)
    final = evaluate_on_test_set(prices, best, split)
    print(f"TEST Sharpe: {final.test_sharpe:.2f}")
    print(f"TEST PnL: ${final.test_pnl:,.0f}")
    '''
    print(example_code)
