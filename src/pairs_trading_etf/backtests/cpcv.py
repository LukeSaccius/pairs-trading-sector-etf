"""
Combinatorial Purged Cross-Validation (CPCV) - Strict Implementation

This module implements CPCV as described in:
- López de Prado (2018) "Advances in Financial Machine Learning" Ch.7
- Bailey et al. (2014) "The Probability of Backtest Overfitting"

Key improvements over standard CSCV:
1. PURGING: Remove observations near train/test boundary to prevent leakage
2. EMBARGO: Add gap after test period to prevent forward-looking bias  
3. TEMPORAL ORDERING: Respect time series nature of financial data
4. MULTIPLE METRICS: PBO, DSR, Rank Stability, Deflated Returns

Usage:
------
>>> from cpcv import CPCVAnalyzer
>>> analyzer = CPCVAnalyzer(n_splits=10, purge_window=5, embargo_window=5)
>>> result = analyzer.analyze(returns_matrix, strategy_names)
>>> print(result.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import norm, spearmanr, kendalltau

# Import shared statistical functions
from ..utils.statistics import expected_max_sharpe, calculate_dsr

logger = logging.getLogger(__name__)


# =============================================================================
# CPCV CONFIGURATION
# =============================================================================

@dataclass
class CPCVConfig:
    """Configuration for CPCV analysis.
    
    Parameters
    ----------
    n_splits : int
        Number of time blocks to split data into (must be even, >= 6)
        More splits = more combinations but smaller blocks
        Recommended: 10-16 for daily data over 5+ years
        
    purge_window : int
        Number of observations to remove before test period
        Prevents information leakage from overlapping trades
        Rule of thumb: max(holding_period) days
        
    embargo_window : int
        Number of observations to skip after test period
        Prevents forward-looking bias in sequential trades
        Rule of thumb: purge_window / 2
        
    min_observations_per_split : int
        Minimum observations required per split
        Smaller = more splits possible but noisier estimates
        
    max_combinations : int, optional
        Cap on number of combinations to test (for speed)
        None = test all C(n, n/2) combinations
        
    random_seed : int
        For reproducibility when subsampling combinations
    """
    n_splits: int = 10
    purge_window: int = 5
    embargo_window: int = 3
    min_observations_per_split: int = 20
    max_combinations: Optional[int] = None
    random_seed: int = 42
    
    def __post_init__(self):
        assert self.n_splits >= 6, "Need at least 6 splits for meaningful CPCV"
        assert self.n_splits % 2 == 0, "n_splits must be even"
        assert self.purge_window >= 0, "purge_window must be non-negative"
        assert self.embargo_window >= 0, "embargo_window must be non-negative"
    
    @property
    def n_test_splits(self) -> int:
        """Number of splits used for testing in each combination."""
        return self.n_splits // 2
    
    @property
    def theoretical_combinations(self) -> int:
        """Theoretical number of C(n, n/2) combinations."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)


# =============================================================================
# CPCV RESULTS
# =============================================================================

@dataclass
class CPCVResult:
    """Comprehensive results from CPCV analysis.
    
    Contains multiple overfitting detection metrics:
    - PBO: Probability of Backtest Overfitting
    - DSR: Deflated Sharpe Ratio
    - Rank stability measures
    - Performance degradation statistics
    """
    
    # Configuration used
    config: CPCVConfig
    n_strategies: int
    n_observations: int
    n_combinations_tested: int
    
    # ==========================================================================
    # PRIMARY METRICS
    # ==========================================================================
    
    # PBO - Probability of Backtest Overfitting [0, 1]
    # Probability that the best in-sample strategy underperforms median out-of-sample
    pbo: float
    
    # PBO Confidence Interval (via bootstrap)
    pbo_ci_lower: float = 0.0
    pbo_ci_upper: float = 1.0
    
    # ==========================================================================
    # DEFLATED SHARPE RATIO
    # ==========================================================================
    
    # Raw observed Sharpe (best strategy)
    sharpe_observed: float = 0.0
    
    # Expected max Sharpe under null (all strategies random)
    sharpe_expected_max: float = 0.0
    
    # Deflated Sharpe Ratio
    dsr: float = 0.0
    dsr_pvalue: float = 1.0
    
    # ==========================================================================
    # PERFORMANCE DEGRADATION
    # ==========================================================================
    
    # Mean return of best IS strategy
    is_mean_return: float = 0.0
    is_std_return: float = 0.0
    
    # Mean OOS return of best IS strategy
    oos_mean_return: float = 0.0
    oos_std_return: float = 0.0
    
    # Degradation ratio: (IS - OOS) / |IS|
    degradation_ratio: float = 0.0
    
    # ==========================================================================
    # RANK STABILITY
    # ==========================================================================
    
    # Spearman correlation between IS and OOS rankings
    rank_correlation_spearman: float = 0.0
    rank_correlation_pvalue: float = 1.0
    
    # Kendall's tau (more robust to outliers)
    rank_correlation_kendall: float = 0.0
    
    # Probability best IS is in top quartile OOS
    prob_top_quartile_oos: float = 0.0
    
    # ==========================================================================
    # DISTRIBUTION OF OUTCOMES
    # ==========================================================================
    
    # Logit values for all combinations (for distribution analysis)
    logit_distribution: List[float] = field(default_factory=list)
    
    # OOS performance rank of best IS (for each combination)
    oos_ranks: List[int] = field(default_factory=list)
    
    # ==========================================================================
    # STRATEGY-LEVEL DETAILS
    # ==========================================================================
    
    # Strategy names
    strategy_names: List[str] = field(default_factory=list)
    
    # How often each strategy was selected as "best" in IS
    selection_frequency: Dict[str, int] = field(default_factory=dict)
    
    # Average OOS performance for each strategy (across all combinations)
    strategy_oos_mean: Dict[str, float] = field(default_factory=dict)
    
    # ==========================================================================
    # INTERPRETATION
    # ==========================================================================
    
    @property
    def is_overfit(self) -> bool:
        """Conservative overfitting detection."""
        return self.pbo > 0.40 or self.dsr_pvalue > 0.05
    
    @property
    def risk_level(self) -> str:
        """Risk level interpretation."""
        if self.pbo < 0.20 and self.dsr > 0 and self.dsr_pvalue < 0.05:
            return "LOW"
        elif self.pbo < 0.40 and self.degradation_ratio < 0.50:
            return "MODERATE"
        elif self.pbo < 0.60:
            return "HIGH"
        else:
            return "SEVERE"
    
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        levels = {
            "LOW": "Strategy selection appears robust. Proceed with caution.",
            "MODERATE": "Some overfitting detected. Use conservative sizing.",
            "HIGH": "Significant overfitting. Consider simpler strategy.",
            "SEVERE": "Severe overfitting. Do NOT deploy this strategy.",
        }
        return levels[self.risk_level]
    
    def summary(self) -> str:
        """Generate formatted summary string."""
        lines = [
            "",
            "=" * 70,
            "CPCV ANALYSIS REPORT - Combinatorial Purged Cross-Validation",
            "=" * 70,
            "",
            "Configuration:",
            f"  • Strategies tested: {self.n_strategies}",
            f"  • Observations: {self.n_observations:,}",
            f"  • Splits: {self.config.n_splits} (purge={self.config.purge_window}, embargo={self.config.embargo_window})",
            f"  • Combinations tested: {self.n_combinations_tested:,}",
            "",
            "-" * 70,
            "PRIMARY METRICS",
            "-" * 70,
            "",
            "  📊 PBO (Probability of Backtest Overfitting):",
            f"     Value: {self.pbo:.1%}  (95% CI: [{self.pbo_ci_lower:.1%}, {self.pbo_ci_upper:.1%}])",
            f"     Risk Level: {self.risk_level}",
            "",
            "  📈 Deflated Sharpe Ratio:",
            f"     Observed Sharpe: {self.sharpe_observed:.2f}",
            f"     Expected Max (null): {self.sharpe_expected_max:.2f}",
            f"     DSR: {self.dsr:.2f} (p-value: {self.dsr_pvalue:.3f})",
            "",
            "  📉 Performance Degradation:",
            f"     In-Sample mean: {self.is_mean_return:.4%}",
            f"     Out-of-Sample mean: {self.oos_mean_return:.4%}",
            f"     Degradation: {self.degradation_ratio:.1%}",
            "",
            "  🔗 Rank Stability:",
            f"     Spearman ρ: {self.rank_correlation_spearman:.2f} (p={self.rank_correlation_pvalue:.3f})",
            f"     Kendall τ: {self.rank_correlation_kendall:.2f}",
            f"     P(top quartile OOS): {self.prob_top_quartile_oos:.1%}",
            "",
            "-" * 70,
            "INTERPRETATION",
            "-" * 70,
            "",
            f"  {self.interpretation}",
            "",
        ]
        
        # Recommendations
        lines.extend([
            "-" * 70,
            "RECOMMENDATIONS",
            "-" * 70,
            "",
        ])
        
        if self.risk_level == "LOW":
            lines.extend([
                "  ✅ Good to proceed:",
                "  • Deploy with standard position sizing",
                "  • Monitor OOS performance closely",
                "  • Re-validate quarterly",
            ])
        elif self.risk_level == "MODERATE":
            lines.extend([
                "  ⚠️ Proceed with caution:",
                "  • Use 50% of intended position size",
                "  • Set tight drawdown limits",
                "  • Paper trade for 1 month first",
            ])
        elif self.risk_level == "HIGH":
            lines.extend([
                "  ⚠️ High risk - reconsider:",
                "  • Reduce number of parameters",
                "  • Use longer lookback periods",
                "  • Consider simpler entry/exit rules",
            ])
        else:
            lines.extend([
                "  ❌ Do NOT deploy:",
                "  • Strategy is likely spurious",
                "  • Results are not statistically significant",
                "  • Start over with fundamentally different approach",
            ])
        
        lines.extend(["", "=" * 70, ""])
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'config': {
                'n_splits': self.config.n_splits,
                'purge_window': self.config.purge_window,
                'embargo_window': self.config.embargo_window,
            },
            'n_strategies': self.n_strategies,
            'n_observations': self.n_observations,
            'n_combinations_tested': self.n_combinations_tested,
            'pbo': self.pbo,
            'pbo_ci': [self.pbo_ci_lower, self.pbo_ci_upper],
            'sharpe_observed': self.sharpe_observed,
            'sharpe_expected_max': self.sharpe_expected_max,
            'dsr': self.dsr,
            'dsr_pvalue': self.dsr_pvalue,
            'is_mean_return': self.is_mean_return,
            'oos_mean_return': self.oos_mean_return,
            'degradation_ratio': self.degradation_ratio,
            'rank_correlation_spearman': self.rank_correlation_spearman,
            'rank_correlation_kendall': self.rank_correlation_kendall,
            'prob_top_quartile_oos': self.prob_top_quartile_oos,
            'risk_level': self.risk_level,
            'is_overfit': self.is_overfit,
        }


# =============================================================================
# CPCV ANALYZER
# =============================================================================

class CPCVAnalyzer:
    """
    Combinatorial Purged Cross-Validation Analyzer.
    
    Implements strict CPCV with purging and embargo to detect backtest overfitting.
    
    Example
    -------
    >>> # Build returns matrix: rows=days, columns=strategies
    >>> returns = build_returns_matrix(trades_list, date_index)
    >>> 
    >>> # Run CPCV analysis
    >>> analyzer = CPCVAnalyzer(n_splits=10, purge_window=5, embargo_window=3)
    >>> result = analyzer.analyze(returns, strategy_names=['A', 'B', 'C'])
    >>> 
    >>> # Check results
    >>> print(result.summary())
    >>> if result.is_overfit:
    ...     print("WARNING: Strategy appears overfit!")
    """
    
    def __init__(
        self,
        n_splits: int = 10,
        purge_window: int = 5,
        embargo_window: int = 3,
        max_combinations: Optional[int] = None,
        random_seed: int = 42,
    ):
        """
        Initialize CPCV analyzer.
        
        Parameters
        ----------
        n_splits : int
            Number of time blocks (must be even, >= 6)
        purge_window : int
            Observations to remove before test period
        embargo_window : int
            Observations to skip after test period  
        max_combinations : int, optional
            Cap on combinations to test
        random_seed : int
            For reproducibility
        """
        self.config = CPCVConfig(
            n_splits=n_splits,
            purge_window=purge_window,
            embargo_window=embargo_window,
            max_combinations=max_combinations,
            random_seed=random_seed,
        )
        self.rng = np.random.default_rng(random_seed)
    
    def _create_split_indices(
        self, 
        n_observations: int
    ) -> List[Tuple[int, int]]:
        """
        Create split boundaries (start, end) for each block.
        
        Returns list of (start_idx, end_idx) for each split.
        """
        split_size = n_observations // self.config.n_splits
        splits = []
        
        for i in range(self.config.n_splits):
            start = i * split_size
            end = (i + 1) * split_size if i < self.config.n_splits - 1 else n_observations
            splits.append((start, end))
        
        return splits
    
    def _generate_combinations(self) -> List[Tuple[List[int], List[int]]]:
        """
        Generate all C(n, n/2) train/test split combinations.
        
        Returns list of (train_split_indices, test_split_indices).
        """
        all_indices = list(range(self.config.n_splits))
        n_test = self.config.n_test_splits
        
        combos = []
        for test_indices in combinations(all_indices, n_test):
            test_set = set(test_indices)
            train_indices = [i for i in all_indices if i not in test_set]
            combos.append((train_indices, list(test_indices)))
        
        return combos
    
    def _apply_purge_embargo(
        self,
        train_indices: List[int],
        test_indices: List[int],
        split_boundaries: List[Tuple[int, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply purging and embargo to create clean train/test masks.
        
        Purging: Remove observations at end of train splits that are
                 within purge_window of any test split start
        
        Embargo: Remove observations at start of train splits that are
                 within embargo_window of any test split end
        
        Returns boolean masks for (train_mask, test_mask).
        """
        n_obs = split_boundaries[-1][1]
        train_mask = np.zeros(n_obs, dtype=bool)
        test_mask = np.zeros(n_obs, dtype=bool)
        
        # Get test period boundaries
        test_starts = [split_boundaries[i][0] for i in test_indices]
        test_ends = [split_boundaries[i][1] for i in test_indices]
        
        # Build test mask (straightforward)
        for i in test_indices:
            start, end = split_boundaries[i]
            test_mask[start:end] = True
        
        # Build train mask with purging and embargo
        for i in train_indices:
            start, end = split_boundaries[i]
            
            # Check if this train split needs purging
            # (if any test split starts within purge_window after this split ends)
            purge_end = end
            for test_start in test_starts:
                if 0 < test_start - end <= self.config.purge_window:
                    # Need to purge observations at end of this train split
                    purge_amount = self.config.purge_window - (test_start - end)
                    purge_end = max(start, end - purge_amount)
            
            # Check if this train split needs embargo
            # (if any test split ends within embargo_window before this split starts)
            embargo_start = start
            for test_end in test_ends:
                if 0 < start - test_end <= self.config.embargo_window:
                    # Need embargo at start of this train split
                    embargo_amount = self.config.embargo_window - (start - test_end)
                    embargo_start = min(end, start + embargo_amount)
            
            # Apply to mask
            train_mask[embargo_start:purge_end] = True
        
        return train_mask, test_mask
    
    def _calculate_metrics(
        self,
        returns_matrix: np.ndarray,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Calculate mean returns for each strategy on train and test sets.
        
        Returns (train_means, test_means, best_train_idx, best_test_rank).
        """
        train_data = returns_matrix[train_mask, :]
        test_data = returns_matrix[test_mask, :]
        
        train_means = train_data.mean(axis=0)
        test_means = test_data.mean(axis=0)
        
        best_train_idx = np.argmax(train_means)
        
        # Rank of best train strategy in test (1 = best)
        test_ranks = scipy_stats.rankdata(-test_means)  # Negative for descending
        best_test_rank = int(test_ranks[best_train_idx])
        
        return train_means, test_means, best_train_idx, best_test_rank
    
    def analyze(
        self,
        returns_matrix: np.ndarray,
        strategy_names: Optional[List[str]] = None,
        calculate_ci: bool = True,
        n_bootstrap: int = 1000,
    ) -> CPCVResult:
        """
        Run full CPCV analysis.
        
        Parameters
        ----------
        returns_matrix : np.ndarray
            Shape (n_observations, n_strategies)
            Each column is daily returns for one strategy
        strategy_names : list, optional
            Names for each strategy
        calculate_ci : bool
            Whether to calculate confidence intervals (slower)
        n_bootstrap : int
            Bootstrap samples for CI estimation
            
        Returns
        -------
        CPCVResult
            Comprehensive overfitting analysis results
        """
        n_observations, n_strategies = returns_matrix.shape
        
        if strategy_names is None:
            strategy_names = [f"Strategy_{i}" for i in range(n_strategies)]
        
        # Validate
        min_obs = self.config.n_splits * self.config.min_observations_per_split
        if n_observations < min_obs:
            raise ValueError(
                f"Insufficient data: {n_observations} observations, need >= {min_obs}"
            )
        
        logger.info(f"CPCV Analysis: {n_strategies} strategies, {n_observations} observations")
        
        # Create split boundaries
        split_boundaries = self._create_split_indices(n_observations)
        
        # Generate combinations
        all_combinations = self._generate_combinations()
        n_total_combinations = len(all_combinations)
        
        # Optionally subsample
        if self.config.max_combinations and n_total_combinations > self.config.max_combinations:
            indices = self.rng.choice(
                n_total_combinations, 
                size=self.config.max_combinations, 
                replace=False
            )
            combinations_to_test = [all_combinations[i] for i in indices]
            logger.info(f"Subsampling from {n_total_combinations} to {self.config.max_combinations} combinations")
        else:
            combinations_to_test = all_combinations
        
        n_combinations = len(combinations_to_test)
        logger.info(f"Testing {n_combinations} combinations...")
        
        # Tracking variables
        pbo_count = 0
        logit_values = []
        oos_ranks = []
        
        is_returns_best = []
        oos_returns_best = []
        
        selection_counts = {name: 0 for name in strategy_names}
        strategy_oos_totals = {name: [] for name in strategy_names}
        
        all_is_rankings = []
        all_oos_rankings = []
        
        # Main CPCV loop
        for train_indices, test_indices in combinations_to_test:
            # Apply purging and embargo
            train_mask, test_mask = self._apply_purge_embargo(
                train_indices, test_indices, split_boundaries
            )
            
            if train_mask.sum() < 10 or test_mask.sum() < 10:
                logger.warning("Skipping combination with insufficient data after purge/embargo")
                continue
            
            # Calculate metrics
            train_means, test_means, best_idx, best_rank = self._calculate_metrics(
                returns_matrix, train_mask, test_mask
            )
            
            # Track selection
            selection_counts[strategy_names[best_idx]] += 1
            
            # Track OOS performance
            for i, name in enumerate(strategy_names):
                strategy_oos_totals[name].append(test_means[i])
            
            is_returns_best.append(train_means[best_idx])
            oos_returns_best.append(test_means[best_idx])
            oos_ranks.append(best_rank)
            
            # PBO check: does best IS underperform median OOS?
            median_oos = np.median(test_means)
            if test_means[best_idx] < median_oos:
                pbo_count += 1
            
            # Logit for distribution
            lambda_val = best_rank / n_strategies
            lambda_val = np.clip(lambda_val, 0.01, 0.99)
            logit_val = np.log(lambda_val / (1 - lambda_val))
            logit_values.append(logit_val)
            
            # Track rankings for correlation
            all_is_rankings.append(scipy_stats.rankdata(-train_means))
            all_oos_rankings.append(scipy_stats.rankdata(-test_means))
        
        # ==========================================================================
        # CALCULATE FINAL METRICS
        # ==========================================================================
        
        n_valid = len(is_returns_best)
        if n_valid == 0:
            raise ValueError("No valid combinations after purge/embargo")
        
        # PBO
        pbo = pbo_count / n_valid
        
        # Confidence interval via bootstrap
        pbo_ci_lower, pbo_ci_upper = 0.0, 1.0
        if calculate_ci and n_valid >= 30:
            pbo_bootstrap = []
            for _ in range(n_bootstrap):
                sample = self.rng.choice(oos_ranks, size=n_valid, replace=True)
                pbo_sample = np.mean(sample > n_strategies // 2)
                pbo_bootstrap.append(pbo_sample)
            pbo_ci_lower = np.percentile(pbo_bootstrap, 2.5)
            pbo_ci_upper = np.percentile(pbo_bootstrap, 97.5)
        
        # Performance statistics
        is_mean = np.mean(is_returns_best)
        is_std = np.std(is_returns_best)
        oos_mean = np.mean(oos_returns_best)
        oos_std = np.std(oos_returns_best)
        
        degradation = (is_mean - oos_mean) / abs(is_mean) if is_mean != 0 else 0
        
        # Rank correlation (average across combinations)
        avg_is_ranking = np.mean(all_is_rankings, axis=0)
        avg_oos_ranking = np.mean(all_oos_rankings, axis=0)
        
        spearman_corr, spearman_pval = spearmanr(avg_is_ranking, avg_oos_ranking)
        kendall_corr, _ = kendalltau(avg_is_ranking, avg_oos_ranking)
        
        # Probability best IS is in top quartile OOS
        top_quartile = n_strategies // 4
        prob_top_quartile = np.mean([r <= top_quartile for r in oos_ranks])
        
        # Deflated Sharpe Ratio
        # Overall Sharpe of best strategy
        best_overall_idx = np.argmax(returns_matrix.mean(axis=0))
        best_returns = returns_matrix[:, best_overall_idx]
        sharpe_observed = (best_returns.mean() / (best_returns.std() + 1e-8)) * np.sqrt(252)
        
        dsr, dsr_pvalue = self._calculate_dsr(
            sharpe_observed, 
            n_strategies, 
            returns_matrix,
            n_observations / 252  # years
        )
        
        # Expected max Sharpe under null
        sharpe_expected_max = self._expected_max_sharpe(n_strategies, n_observations)
        
        # Strategy-level OOS means
        strategy_oos_mean = {
            name: np.mean(vals) if vals else 0.0 
            for name, vals in strategy_oos_totals.items()
        }
        
        return CPCVResult(
            config=self.config,
            n_strategies=n_strategies,
            n_observations=n_observations,
            n_combinations_tested=n_valid,
            pbo=pbo,
            pbo_ci_lower=pbo_ci_lower,
            pbo_ci_upper=pbo_ci_upper,
            sharpe_observed=sharpe_observed,
            sharpe_expected_max=sharpe_expected_max,
            dsr=dsr,
            dsr_pvalue=dsr_pvalue,
            is_mean_return=is_mean,
            is_std_return=is_std,
            oos_mean_return=oos_mean,
            oos_std_return=oos_std,
            degradation_ratio=degradation,
            rank_correlation_spearman=spearman_corr if not np.isnan(spearman_corr) else 0.0,
            rank_correlation_pvalue=spearman_pval if not np.isnan(spearman_pval) else 1.0,
            rank_correlation_kendall=kendall_corr if not np.isnan(kendall_corr) else 0.0,
            prob_top_quartile_oos=prob_top_quartile,
            logit_distribution=logit_values,
            oos_ranks=oos_ranks,
            strategy_names=strategy_names,
            selection_frequency=selection_counts,
            strategy_oos_mean=strategy_oos_mean,
        )
    
    def _expected_max_sharpe(self, n_trials: int, n_obs: int) -> float:
        """Expected maximum Sharpe from n_trials random strategies."""
        # Use shared utility function to avoid duplication
        return expected_max_sharpe(n_trials, n_obs)
    
    def _calculate_dsr(
        self,
        sharpe_observed: float,
        n_trials: int,
        returns_matrix: np.ndarray,
        years: float,
    ) -> Tuple[float, float]:
        """Calculate Deflated Sharpe Ratio."""
        if n_trials <= 1:
            return sharpe_observed, 0.0
        
        # Return statistics
        daily_returns = returns_matrix.mean(axis=1)
        skewness = scipy_stats.skew(daily_returns)
        kurtosis = scipy_stats.kurtosis(daily_returns) + 3  # Convert to raw kurtosis
        
        T = len(daily_returns)
        sr = sharpe_observed
        
        # Variance of Sharpe estimator (Lo 2002)
        var_sr = (1 + 0.5 * sr**2 - skewness * sr + 
                  (kurtosis - 3) / 4 * sr**2) / T
        std_sr = np.sqrt(max(var_sr, 1e-8))
        
        # Expected max under null
        expected_max = self._expected_max_sharpe(n_trials, T)
        
        # DSR
        dsr = (sharpe_observed - expected_max) / std_sr if std_sr > 0 else 0
        
        # P-value
        p_value = 1 - norm.cdf(dsr)
        
        return float(dsr), float(p_value)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def build_returns_matrix_from_trades(
    trades_by_config: Dict[str, List[Dict]],
    date_range: pd.DatetimeIndex,
    initial_capital: float = 50000.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build returns matrix from multiple configurations' trade results.
    
    Parameters
    ----------
    trades_by_config : dict
        {config_name: [trade_dict, ...]}
        Each trade_dict must have: entry_date, exit_date, pnl
    date_range : pd.DatetimeIndex
        Date index for the returns matrix
    initial_capital : float
        Starting capital for return calculation
        
    Returns
    -------
    tuple
        (returns_matrix, config_names)
        returns_matrix shape: (n_dates, n_configs)
    """
    config_names = list(trades_by_config.keys())
    n_dates = len(date_range)
    n_configs = len(config_names)
    
    # Initialize with zeros (no position = 0 return)
    returns_matrix = np.zeros((n_dates, n_configs))
    
    # Date to index mapping
    date_to_idx = {d: i for i, d in enumerate(date_range)}
    
    for config_idx, config_name in enumerate(config_names):
        trades = trades_by_config[config_name]
        
        for trade in trades:
            entry = pd.Timestamp(trade['entry_date'])
            exit_date = pd.Timestamp(trade['exit_date'])
            pnl = trade['pnl']
            holding_days = trade.get('holding_days', 1) or 1
            
            # Daily return (as fraction of capital)
            daily_return = (pnl / holding_days) / initial_capital
            
            # Add to each day in holding period
            for day in pd.date_range(entry, exit_date):
                if day in date_to_idx:
                    returns_matrix[date_to_idx[day], config_idx] += daily_return
    
    return returns_matrix, config_names


def quick_cpcv_check(
    returns_matrix: np.ndarray,
    strategy_names: Optional[List[str]] = None,
) -> str:
    """
    Quick CPCV check with default parameters.
    
    Returns formatted summary string.
    """
    analyzer = CPCVAnalyzer(
        n_splits=10,
        purge_window=5,
        embargo_window=3,
    )
    
    result = analyzer.analyze(returns_matrix, strategy_names)
    return result.summary()


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    # Simulate 5 strategies over 1000 days
    n_days = 1000
    n_strategies = 5
    
    # Some strategies are random, some have slight edge
    returns = np.random.randn(n_days, n_strategies) * 0.01
    returns[:, 0] += 0.0005  # Strategy 0 has slight edge
    
    analyzer = CPCVAnalyzer(n_splits=10, purge_window=5, embargo_window=3)
    result = analyzer.analyze(returns, [f"Strat_{i}" for i in range(n_strategies)])
    
    print(result.summary())
