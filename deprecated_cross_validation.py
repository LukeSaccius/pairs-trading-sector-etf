"""
Cross-validation framework for pairs trading backtests.

This module provides:
- Proper train/validation/test split
- Parameter selection on validation set
- Unbiased final evaluation on test set
- CSCV (Combinatorially Symmetric Cross-Validation) for overfitting detection
- Deflated Sharpe Ratio for multiple testing adjustment

References:
- Bailey et al. (2014) "The Probability of Backtest Overfitting"
- Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
- Pardo (2008) "The Evaluation and Optimization of Trading Strategies"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import norm

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
# CSCV - COMBINATORIALLY SYMMETRIC CROSS-VALIDATION
# =============================================================================
# Implementation based on Bailey & López de Prado (2014)
# "The Probability of Backtest Overfitting"

@dataclass
class CSCVResult:
    """Results from CSCV analysis.
    
    Key metrics from Bailey & López de Prado (2014):
    - PBO: Probability of Backtest Overfitting
    - Degradation: Expected performance decay OOS
    - Stochastic Dominance: How often best IS beats median OOS
    """
    
    n_strategies: int
    n_partitions: int
    n_combinations: int
    
    # Core PBO metric
    pbo: float  # Probability of Backtest Overfitting [0, 1]
    
    # Performance degradation
    is_mean: float  # Mean in-sample return
    oos_mean: float  # Mean out-of-sample return
    degradation: float  # (IS - OOS) / IS as percentage
    
    # Distribution statistics
    logit_distribution: List[float] = field(default_factory=list)
    rank_correlation: float = 0.0  # Spearman correlation between IS and OOS ranks
    
    # Deflated Sharpe Ratio components
    sharpe_is: float = 0.0
    sharpe_oos: float = 0.0
    n_trials: int = 0  # Number of strategies tested
    
    @property
    def is_overfit(self) -> bool:
        """PBO > 0.50 suggests severe overfitting."""
        return self.pbo > 0.50
    
    @property
    def pbo_interpretation(self) -> str:
        """Human-readable PBO interpretation."""
        if self.pbo < 0.25:
            return "Low overfitting risk"
        elif self.pbo < 0.50:
            return "Moderate overfitting risk"
        elif self.pbo < 0.75:
            return "High overfitting risk"
        else:
            return "Severe overfitting - likely spurious"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_strategies': self.n_strategies,
            'n_partitions': self.n_partitions,
            'n_combinations': self.n_combinations,
            'pbo': self.pbo,
            'pbo_interpretation': self.pbo_interpretation,
            'is_overfit': self.is_overfit,
            'is_mean': self.is_mean,
            'oos_mean': self.oos_mean,
            'degradation_pct': self.degradation * 100,
            'rank_correlation': self.rank_correlation,
        }


def _generate_cscv_combinations(n_partitions: int) -> List[Tuple[List[int], List[int]]]:
    """
    Generate all C(n, n/2) combinations for CSCV.
    
    For n=16 partitions: C(16,8) = 12,870 combinations
    Each combination splits partitions into IS and OOS sets.
    
    Parameters
    ----------
    n_partitions : int
        Number of data partitions (must be even)
        
    Returns
    -------
    list
        List of (is_indices, oos_indices) tuples
    """
    assert n_partitions % 2 == 0, "n_partitions must be even"
    half = n_partitions // 2
    
    all_indices = list(range(n_partitions))
    combinations_list = []
    
    for is_indices in combinations(all_indices, half):
        is_set = set(is_indices)
        oos_indices = tuple(i for i in all_indices if i not in is_set)
        combinations_list.append((list(is_indices), list(oos_indices)))
    
    return combinations_list


def run_cscv_analysis(
    returns_matrix: np.ndarray,
    n_partitions: int = 16,
    strategy_names: Optional[List[str]] = None,
    max_combinations: Optional[int] = None,
    random_seed: int = 42,
) -> CSCVResult:
    """
    Run Combinatorially Symmetric Cross-Validation (CSCV).
    
    This implements the PBO methodology from Bailey & López de Prado (2014).
    
    Algorithm:
    1. Partition returns data into S equal blocks
    2. Generate all C(S, S/2) train/test combinations
    3. For each combination:
       - Identify best strategy on IS (in-sample)
       - Check if best IS strategy beats median on OOS
    4. PBO = fraction of times best IS underperforms median OOS
    
    Parameters
    ----------
    returns_matrix : np.ndarray
        Matrix of shape (n_periods, n_strategies)
        Each column is a strategy's return series
    n_partitions : int
        Number of time partitions (default 16, giving C(16,8)=12,870 tests)
    strategy_names : list, optional
        Names for each strategy column
    max_combinations : int, optional
        Limit number of combinations to test (for speed)
    random_seed : int
        Random seed for reproducibility if subsampling
        
    Returns
    -------
    CSCVResult
        Comprehensive overfitting analysis results
        
    Example
    -------
    >>> # Build returns matrix from backtest results
    >>> # Each column = one parameter configuration
    >>> returns = np.column_stack([
    ...     strategy_1_daily_returns,
    ...     strategy_2_daily_returns,
    ...     strategy_3_daily_returns,
    ... ])
    >>> result = run_cscv_analysis(returns, n_partitions=16)
    >>> print(f"PBO = {result.pbo:.2%}")  # e.g., "PBO = 35.2%"
    
    References
    ----------
    Bailey, D., Borwein, J., López de Prado, M., & Zhu, Q. (2014).
    "The Probability of Backtest Overfitting."
    Journal of Computational Finance.
    """
    n_periods, n_strategies = returns_matrix.shape
    
    if strategy_names is None:
        strategy_names = [f"Strategy_{i}" for i in range(n_strategies)]
    
    # Partition data into equal blocks
    partition_size = n_periods // n_partitions
    if partition_size < 5:
        logger.warning(f"Partition size too small: {partition_size} periods. Consider fewer partitions.")
    
    # Truncate to ensure equal partition sizes
    n_periods_used = partition_size * n_partitions
    returns_matrix = returns_matrix[:n_periods_used, :]
    
    # Split into partitions
    partitions = []
    for i in range(n_partitions):
        start = i * partition_size
        end = (i + 1) * partition_size
        partitions.append(returns_matrix[start:end, :])
    
    # Generate all combinations
    all_combinations = _generate_cscv_combinations(n_partitions)
    n_combinations = len(all_combinations)
    
    logger.info(f"CSCV: {n_strategies} strategies, {n_partitions} partitions, {n_combinations} combinations")
    
    # Optionally subsample combinations for speed
    rng = np.random.default_rng(random_seed)
    if max_combinations and n_combinations > max_combinations:
        indices = rng.choice(n_combinations, size=max_combinations, replace=False)
        combinations_to_test = [all_combinations[i] for i in indices]
        n_combinations = max_combinations
        logger.info(f"Subsampling to {max_combinations} combinations")
    else:
        combinations_to_test = all_combinations
    
    # CSCV analysis
    pbo_count = 0  # Count times best IS underperforms median OOS
    logit_values = []  # For logit distribution
    
    is_returns_all = []
    oos_returns_all = []
    
    for is_indices, oos_indices in combinations_to_test:
        # Combine partitions for IS and OOS
        is_data = np.vstack([partitions[i] for i in is_indices])
        oos_data = np.vstack([partitions[i] for i in oos_indices])
        
        # Calculate mean returns for each strategy
        is_means = is_data.mean(axis=0)  # Shape: (n_strategies,)
        oos_means = oos_data.mean(axis=0)
        
        # Find best strategy in IS
        best_is_idx = np.argmax(is_means)
        best_is_return = is_means[best_is_idx]
        
        # Check OOS performance of best IS strategy
        best_oos_return = oos_means[best_is_idx]
        median_oos_return = np.median(oos_means)
        
        # PBO: Does best IS underperform median OOS?
        if best_oos_return < median_oos_return:
            pbo_count += 1
        
        # Logit for distribution analysis
        # λ = rank(best_oos) / n_strategies
        # logit(λ) = log(λ / (1-λ))
        oos_rank = scipy_stats.rankdata(oos_means)[best_is_idx]
        lambda_val = oos_rank / n_strategies
        # Avoid log(0) or log(inf)
        lambda_val = np.clip(lambda_val, 0.01, 0.99)
        logit_val = np.log(lambda_val / (1 - lambda_val))
        logit_values.append(logit_val)
        
        is_returns_all.append(best_is_return)
        oos_returns_all.append(best_oos_return)
    
    # Calculate PBO
    pbo = pbo_count / n_combinations
    
    # Calculate overall statistics
    is_mean = np.mean(is_returns_all)
    oos_mean = np.mean(oos_returns_all)
    degradation = (is_mean - oos_mean) / abs(is_mean) if is_mean != 0 else 0
    
    # Rank correlation between IS and OOS performance
    # High correlation = less overfitting
    is_overall = returns_matrix[:n_periods_used//2, :].mean(axis=0)
    oos_overall = returns_matrix[n_periods_used//2:, :].mean(axis=0)
    rank_corr, _ = scipy_stats.spearmanr(is_overall, oos_overall)
    
    # Sharpe ratios
    daily_returns = returns_matrix.mean(axis=1)
    sharpe_is = daily_returns[:n_periods_used//2].mean() / (daily_returns[:n_periods_used//2].std() + 1e-8) * np.sqrt(252)
    sharpe_oos = daily_returns[n_periods_used//2:].mean() / (daily_returns[n_periods_used//2:].std() + 1e-8) * np.sqrt(252)
    
    result = CSCVResult(
        n_strategies=n_strategies,
        n_partitions=n_partitions,
        n_combinations=n_combinations,
        pbo=pbo,
        is_mean=is_mean,
        oos_mean=oos_mean,
        degradation=degradation,
        logit_distribution=logit_values,
        rank_correlation=rank_corr if not np.isnan(rank_corr) else 0.0,
        sharpe_is=sharpe_is,
        sharpe_oos=sharpe_oos,
        n_trials=n_strategies,
    )
    
    return result


def calculate_deflated_sharpe(
    sharpe_observed: float,
    n_trials: int,
    returns_skewness: float = 0.0,
    returns_kurtosis: float = 3.0,
    backtest_years: float = 1.0,
) -> Tuple[float, float]:
    """
    Calculate Deflated Sharpe Ratio per Bailey & López de Prado.
    
    The Deflated Sharpe Ratio adjusts for multiple testing (n_trials)
    and non-normal return distributions.
    
    DSR = (SR_observed - E[max(SR)]) / std(SR)
    
    Where E[max(SR)] is the expected max Sharpe from n_trials random strategies.
    
    Parameters
    ----------
    sharpe_observed : float
        Observed annualized Sharpe ratio
    n_trials : int
        Number of strategy configurations tested
    returns_skewness : float
        Skewness of returns (0 for normal)
    returns_kurtosis : float
        Kurtosis of returns (3 for normal)
    backtest_years : float
        Length of backtest in years
        
    Returns
    -------
    tuple
        (deflated_sharpe, p_value)
        
    References
    ----------
    Bailey, D. & López de Prado, M. (2014).
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, 
     Backtest Overfitting, and Non-Normality."
    """
    if n_trials <= 1:
        return sharpe_observed, 0.0
    
    # Expected maximum Sharpe from n_trials random strategies
    # E[max(SR)] ≈ (1 - γ) * Φ^(-1)(1 - 1/n) + γ * Φ^(-1)(1 - 1/(n*e))
    # where γ ≈ 0.5772 (Euler-Mascheroni constant)
    gamma = 0.5772156649
    
    # Standard deviation of Sharpe ratio estimator
    # var(SR) ≈ (1 + 0.5*SR^2 - skew*SR + (kurt-3)/4 * SR^2) / T
    T = int(252 * backtest_years)  # Number of observations
    sr = sharpe_observed
    var_sr = (1 + 0.5 * sr**2 - returns_skewness * sr + 
              (returns_kurtosis - 3) / 4 * sr**2) / T
    std_sr = np.sqrt(max(var_sr, 1e-8))
    
    # Expected max Sharpe under null (all strategies are random)
    if n_trials > 1:
        try:
            z1 = norm.ppf(1 - 1/n_trials)
            z2 = norm.ppf(1 - 1/(n_trials * np.e))
            expected_max_sr = (1 - gamma) * z1 + gamma * z2
            expected_max_sr *= std_sr  # Scale by estimator std
        except Exception:
            expected_max_sr = 0
    else:
        expected_max_sr = 0
    
    # Deflated Sharpe Ratio
    dsr = (sharpe_observed - expected_max_sr) / std_sr if std_sr > 0 else 0
    
    # P-value: probability of observing this DSR under null
    p_value = 1 - norm.cdf(dsr)
    
    return float(dsr), float(p_value)


def build_returns_matrix_from_trades(
    all_trades: List[Dict],
    date_range: pd.DatetimeIndex,
    config_labels: List[str],
) -> np.ndarray:
    """
    Build returns matrix from backtest trade results.
    
    Parameters
    ----------
    all_trades : list
        List of trade dictionaries with 'entry_date', 'exit_date', 'pnl'
    date_range : pd.DatetimeIndex
        Full date range for the returns matrix
    config_labels : list
        Labels identifying which config each trade belongs to
        
    Returns
    -------
    np.ndarray
        Matrix of shape (n_dates, n_configs) with daily returns
    """
    n_dates = len(date_range)
    unique_configs = list(set(config_labels))
    n_configs = len(unique_configs)
    config_to_idx = {cfg: i for i, cfg in enumerate(unique_configs)}
    
    # Initialize returns matrix
    returns_matrix = np.zeros((n_dates, n_configs))
    
    # Date to index mapping
    date_to_idx = {d: i for i, d in enumerate(date_range)}
    
    # Distribute PnL across holding period for each trade
    for trade, config_label in zip(all_trades, config_labels):
        config_idx = config_to_idx[config_label]
        entry = pd.Timestamp(trade['entry_date'])
        exit_date = pd.Timestamp(trade['exit_date'])
        pnl = trade['pnl']
        holding_days = trade.get('holding_days', 1) or 1
        
        # Daily PnL
        daily_pnl = pnl / holding_days
        
        # Add to each day in holding period
        for day in pd.date_range(entry, exit_date):
            if day in date_to_idx:
                returns_matrix[date_to_idx[day], config_idx] += daily_pnl
    
    return returns_matrix


def print_cscv_report(result: CSCVResult) -> None:
    """Print formatted CSCV analysis report."""
    print("\n" + "=" * 70)
    print("CSCV ANALYSIS REPORT - Probability of Backtest Overfitting")
    print("=" * 70)
    print(f"\nConfigurations tested: {result.n_strategies}")
    print(f"Data partitions: {result.n_partitions}")
    print(f"CSCV combinations: {result.n_combinations:,}")
    
    print("\n" + "-" * 70)
    print("KEY METRICS")
    print("-" * 70)
    
    # PBO with color coding
    pbo_pct = result.pbo * 100
    print(f"\n  📊 PBO (Probability of Backtest Overfitting): {pbo_pct:.1f}%")
    print(f"     Interpretation: {result.pbo_interpretation}")
    
    if result.is_overfit:
        print("     ⚠️  WARNING: High overfitting detected!")
    else:
        print("     ✅ Overfitting risk appears acceptable")
    
    print("\n  📈 Performance Comparison:")
    print(f"     In-Sample mean return: {result.is_mean:.4%}")
    print(f"     Out-of-Sample mean return: {result.oos_mean:.4%}")
    print(f"     Degradation: {result.degradation*100:.1f}%")
    
    print(f"\n  🔗 Rank Correlation (IS vs OOS): {result.rank_correlation:.2f}")
    if result.rank_correlation > 0.5:
        print("     Good: IS performance predicts OOS performance")
    elif result.rank_correlation > 0:
        print("     Moderate: Some predictive power")
    else:
        print("     Poor: IS performance does NOT predict OOS")
    
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS")
    print("-" * 70)
    
    if result.pbo < 0.25:
        print("\n  ✅ Strategy selection appears robust")
        print("  • Proceed with selected configuration")
        print("  • Continue monitoring OOS performance")
    elif result.pbo < 0.50:
        print("\n  ⚠️  Moderate overfitting risk")
        print("  • Consider simplifying strategy (fewer parameters)")
        print("  • Expand training data if possible")
        print("  • Use more conservative position sizing")
    else:
        print("\n  ❌ Severe overfitting detected")
        print("  • DO NOT deploy this strategy")
        print("  • Reduce number of parameters tested")
        print("  • Use longer lookback periods")
        print("  • Consider different strategy altogether")
    
    print("\n" + "=" * 70)
