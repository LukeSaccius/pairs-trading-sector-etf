"""
Combinatorial Purged Cross-Validation (CPCV) - CORRECT Implementation

KEY DIFFERENCE FROM CSCV:
- CSCV: ALL C(n, n/2) combinations allowed (train/test can swap)
- CPCV: Only TEMPORALLY VALID combinations (train BEFORE test always)

Reference: Bailey et al. (2016) - "The Probability of Backtest Overfitting"
López de Prado (2018) - "Advances in Financial Machine Learning" Chapter 7

PURGING & EMBARGO:
- Purge: Remove data at END of train that's too close to test START
- Embargo: Remove data at START of train that's too close to test END
  (Only applies when train comes AFTER test in time)

For pairs trading:
- embargo_window = ceil(average_holding_time_days)
- purge_window = embargo_window (or slightly larger)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import spearmanr

# Import shared statistical functions to avoid duplication
from ..utils.statistics import (
    expected_max_sharpe as _expected_max_sharpe,
    calculate_dsr as _calculate_dsr,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CPCVConfig:
    """Configuration for CPCV.
    
    Parameters
    ----------
    n_splits : int
        Number of time blocks (must be >= 4)
        
    purge_window : int
        Observations to remove at end of train period before test starts
        Rule: ceil(max_holding_period)
        
    embargo_window : int  
        Observations to skip at start of train period after test ends
        Rule: ceil(average_holding_period)
        
    test_ratio : float
        Fraction of splits to use for testing (default 0.5)
    """
    n_splits: int = 10
    purge_window: int = 5
    embargo_window: int = 5  # Should match avg holding period
    test_ratio: float = 0.5
    random_seed: int = 42
    
    def __post_init__(self):
        assert self.n_splits >= 4, "Need at least 4 splits"
        assert 0 < self.test_ratio < 1, "test_ratio must be between 0 and 1"
        assert self.purge_window >= 0
        assert self.embargo_window >= 0
    
    @property
    def n_test_splits(self) -> int:
        """Number of consecutive splits for test."""
        return max(1, int(self.n_splits * self.test_ratio))


# =============================================================================
# RESULT
# =============================================================================

@dataclass
class CPCVResult:
    """Results from CPCV analysis."""
    
    config: CPCVConfig
    n_strategies: int
    n_observations: int
    n_combinations_tested: int
    
    # PBO - Probability of Backtest Overfitting [0, 1]
    pbo: float
    pbo_ci_lower: float = 0.0
    pbo_ci_upper: float = 1.0
    
    # Deflated Sharpe
    sharpe_observed: float = 0.0
    sharpe_expected_max: float = 0.0
    dsr: float = 0.0
    dsr_pvalue: float = 1.0
    
    # Performance
    is_mean_return: float = 0.0
    oos_mean_return: float = 0.0
    degradation_ratio: float = 0.0
    
    # Rank stability
    rank_correlation: float = 0.0
    prob_top_quartile: float = 0.0
    
    # Details
    oos_ranks: List[int] = field(default_factory=list)
    strategy_names: List[str] = field(default_factory=list)
    selection_frequency: Dict[str, int] = field(default_factory=dict)
    
    @property
    def risk_level(self) -> str:
        if self.pbo < 0.20:
            return "LOW"
        elif self.pbo < 0.40:
            return "MODERATE"
        elif self.pbo < 0.60:
            return "HIGH"
        else:
            return "SEVERE"
    
    @property
    def is_overfit(self) -> bool:
        return self.pbo > 0.40
    
    def summary(self) -> str:
        lines = [
            "",
            "=" * 70,
            "CPCV ANALYSIS - Combinatorial Purged Cross-Validation",
            "=" * 70,
            "",
            f"  Strategies: {self.n_strategies}",
            f"  Observations: {self.n_observations:,}",
            f"  Splits: {self.config.n_splits} (purge={self.config.purge_window}, embargo={self.config.embargo_window})",
            f"  Combinations tested: {self.n_combinations_tested}",
            "",
            "-" * 70,
            "METRICS",
            "-" * 70,
            "",
            f"  PBO: {self.pbo:.1%} (95% CI: [{self.pbo_ci_lower:.1%}, {self.pbo_ci_upper:.1%}])",
            f"  Risk Level: {self.risk_level}",
            "",
            f"  Observed Sharpe: {self.sharpe_observed:.2f}",
            f"  Deflated Sharpe (DSR): {self.dsr:.2f} (p={self.dsr_pvalue:.3f})",
            "",
            f"  IS mean return: {self.is_mean_return:.4%}",
            f"  OOS mean return: {self.oos_mean_return:.4%}",
            f"  Degradation: {self.degradation_ratio:.1%}",
            "",
            f"  Rank correlation: {self.rank_correlation:.2f}",
            f"  P(top quartile): {self.prob_top_quartile:.1%}",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pbo': self.pbo,
            'pbo_ci': [self.pbo_ci_lower, self.pbo_ci_upper],
            'dsr': self.dsr,
            'dsr_pvalue': self.dsr_pvalue,
            'degradation': self.degradation_ratio,
            'rank_correlation': self.rank_correlation,
            'risk_level': self.risk_level,
            'is_overfit': self.is_overfit,
        }


# =============================================================================
# CPCV ANALYZER - CORRECT IMPLEMENTATION
# =============================================================================

class CPCVAnalyzer:
    """
    Combinatorial Purged Cross-Validation - CORRECT for time series.
    
    KEY PRINCIPLE:
    - Train data MUST be from BEFORE test period (no future leakage)
    - Purge: remove data at boundary between train/test
    - Embargo: gap after test ends before next train can use that data
    
    VALID SPLITS ONLY:
    For n_splits=10, test_ratio=0.5:
    - Test blocks must be CONTIGUOUS (e.g., [3,4,5,6,7])
    - Train blocks are all blocks BEFORE test start
    - Data AFTER test end is NOT used (to prevent look-ahead)
    
    This is fundamentally different from CSCV which allows
    arbitrary train/test mixing!
    """
    
    def __init__(
        self,
        n_splits: int = 10,
        purge_window: int = 5,
        embargo_window: int = 5,
        random_seed: int = 42,
    ):
        self.config = CPCVConfig(
            n_splits=n_splits,
            purge_window=purge_window,
            embargo_window=embargo_window,
            random_seed=random_seed,
        )
        self.rng = np.random.default_rng(random_seed)
    
    def _create_split_boundaries(self, n_obs: int) -> List[Tuple[int, int]]:
        """Create (start, end) for each split."""
        split_size = n_obs // self.config.n_splits
        splits = []
        for i in range(self.config.n_splits):
            start = i * split_size
            end = (i + 1) * split_size if i < self.config.n_splits - 1 else n_obs
            splits.append((start, end))
        return splits
    
    def _generate_valid_splits(
        self
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Generate TEMPORALLY VALID train/test splits.
        
        CRITICAL DIFFERENCE FROM CSCV:
        - Test must be CONTIGUOUS blocks
        - Train must be BEFORE test (temporally)
        - No mixing of future into past
        
        Example with n_splits=6, n_test=3:
        Valid: train=[0,1,2], test=[3,4,5]  ✓ (train before test)
        Valid: train=[0,1], test=[2,3,4]    ✓ (train before test, some unused)
        Valid: train=[0], test=[1,2,3]      ✓
        
        INVALID: train=[0,3,5], test=[1,2,4] ✗ (non-contiguous, mixing)
        """
        n_test = self.config.n_test_splits
        valid_splits = []
        
        # Slide test window through splits
        # Train = all splits before test window
        for test_start in range(self.config.n_splits - n_test + 1):
            test_end = test_start + n_test
            
            # Test blocks
            test_indices = list(range(test_start, test_end))
            
            # Train blocks: everything BEFORE test
            train_indices = list(range(0, test_start))
            
            # Must have at least 1 train block
            if len(train_indices) >= 1:
                valid_splits.append((train_indices, test_indices))
        
        return valid_splits
    
    def _apply_purge_embargo(
        self,
        train_indices: List[int],
        test_indices: List[int],
        split_boundaries: List[Tuple[int, int]],
        n_obs: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply purging and embargo.
        
        Purge: Remove last `purge_window` observations from train
               (before test starts)
               
        Embargo: NOT applicable here because train is ALWAYS before test.
                 Embargo is for when train comes AFTER test (walk-forward).
        
        For walk-forward with re-training, embargo applies differently.
        """
        train_mask = np.zeros(n_obs, dtype=bool)
        test_mask = np.zeros(n_obs, dtype=bool)
        
        # Test mask: straightforward
        for i in test_indices:
            start, end = split_boundaries[i]
            test_mask[start:end] = True
        
        # Train mask with purging
        for i in train_indices:
            start, end = split_boundaries[i]
            
            # If this is the last train split (adjacent to test)
            if i == max(train_indices):
                # Apply purge: don't use last purge_window observations
                purge_end = max(start, end - self.config.purge_window)
                train_mask[start:purge_end] = True
            else:
                # Not adjacent to test, use all
                train_mask[start:end] = True
        
        return train_mask, test_mask
    
    def analyze(
        self,
        returns_matrix: np.ndarray,
        strategy_names: Optional[List[str]] = None,
        n_bootstrap: int = 1000,
    ) -> CPCVResult:
        """
        Run CPCV analysis.
        
        Parameters
        ----------
        returns_matrix : np.ndarray
            Shape (n_observations, n_strategies)
            Each column = daily returns for one strategy
        strategy_names : list, optional
            Names for strategies
        n_bootstrap : int
            Bootstrap samples for confidence intervals
        """
        n_obs, n_strategies = returns_matrix.shape
        
        if strategy_names is None:
            strategy_names = [f"S{i}" for i in range(n_strategies)]
        
        logger.info(f"CPCV: {n_strategies} strategies, {n_obs} observations")
        
        split_boundaries = self._create_split_boundaries(n_obs)
        valid_splits = self._generate_valid_splits()
        
        logger.info(f"Generated {len(valid_splits)} valid temporal splits")
        
        # Tracking
        pbo_count = 0
        oos_ranks = []
        is_returns = []
        oos_returns = []
        selection_counts = {name: 0 for name in strategy_names}
        all_is_rankings = []
        all_oos_rankings = []
        
        for train_indices, test_indices in valid_splits:
            train_mask, test_mask = self._apply_purge_embargo(
                train_indices, test_indices, split_boundaries, n_obs
            )
            
            if train_mask.sum() < 10 or test_mask.sum() < 10:
                continue
            
            train_data = returns_matrix[train_mask, :]
            test_data = returns_matrix[test_mask, :]
            
            train_means = train_data.mean(axis=0)
            test_means = test_data.mean(axis=0)
            
            best_train_idx = np.argmax(train_means)
            
            # Rank of best train strategy in test
            test_ranks = scipy_stats.rankdata(-test_means)
            best_test_rank = int(test_ranks[best_train_idx])
            
            selection_counts[strategy_names[best_train_idx]] += 1
            oos_ranks.append(best_test_rank)
            is_returns.append(train_means[best_train_idx])
            oos_returns.append(test_means[best_train_idx])
            
            # PBO: best IS underperforms median OOS?
            if test_means[best_train_idx] < np.median(test_means):
                pbo_count += 1
            
            all_is_rankings.append(scipy_stats.rankdata(-train_means))
            all_oos_rankings.append(scipy_stats.rankdata(-test_means))
        
        n_valid = len(oos_ranks)
        if n_valid == 0:
            raise ValueError("No valid combinations")
        
        # PBO
        pbo = pbo_count / n_valid
        
        # Bootstrap CI
        pbo_ci_lower, pbo_ci_upper = 0.0, 1.0
        if n_valid >= 10:
            pbo_samples = []
            for _ in range(n_bootstrap):
                sample = self.rng.choice(oos_ranks, size=n_valid, replace=True)
                pbo_samples.append(np.mean(sample > n_strategies // 2))
            pbo_ci_lower = np.percentile(pbo_samples, 2.5)
            pbo_ci_upper = np.percentile(pbo_samples, 97.5)
        
        # Performance
        is_mean = np.mean(is_returns)
        oos_mean = np.mean(oos_returns)
        degradation = (is_mean - oos_mean) / abs(is_mean) if is_mean != 0 else 0
        
        # Rank correlation
        avg_is = np.mean(all_is_rankings, axis=0)
        avg_oos = np.mean(all_oos_rankings, axis=0)
        rank_corr, _ = spearmanr(avg_is, avg_oos) if len(avg_is) > 1 else (0, 1)
        
        # Top quartile
        top_q = n_strategies // 4
        prob_top = np.mean([r <= max(1, top_q) for r in oos_ranks])
        
        # Sharpe & DSR
        best_idx = np.argmax(returns_matrix.mean(axis=0))
        best_returns = returns_matrix[:, best_idx]
        sharpe_obs = (best_returns.mean() / (best_returns.std() + 1e-8)) * np.sqrt(252)
        dsr, dsr_p = self._calculate_dsr(sharpe_obs, n_strategies, n_obs)
        expected_max = self._expected_max_sharpe(n_strategies, n_obs)
        
        return CPCVResult(
            config=self.config,
            n_strategies=n_strategies,
            n_observations=n_obs,
            n_combinations_tested=n_valid,
            pbo=pbo,
            pbo_ci_lower=pbo_ci_lower,
            pbo_ci_upper=pbo_ci_upper,
            sharpe_observed=sharpe_obs,
            sharpe_expected_max=expected_max,
            dsr=dsr,
            dsr_pvalue=dsr_p,
            is_mean_return=is_mean,
            oos_mean_return=oos_mean,
            degradation_ratio=degradation,
            rank_correlation=rank_corr if not np.isnan(rank_corr) else 0.0,
            prob_top_quartile=prob_top,
            oos_ranks=oos_ranks,
            strategy_names=strategy_names,
            selection_frequency=selection_counts,
        )
    
    def _expected_max_sharpe(self, n_trials: int, n_obs: int) -> float:
        """Expected max Sharpe from random strategies."""
        return _expected_max_sharpe(n_trials, n_obs)
    
    def _calculate_dsr(
        self,
        sharpe_obs: float,
        n_trials: int,
        n_obs: int,
    ) -> Tuple[float, float]:
        """Deflated Sharpe Ratio (wrapper for shared helper)."""
        return _calculate_dsr(sharpe_obs, n_trials, n_obs)


# =============================================================================
# WALK-FORWARD CPCV (for pairs trading)
# =============================================================================

class WalkForwardCPCV:
    """
    Walk-Forward validation with Purge and Embargo - CORRECT for trading.
    
    This matches how pairs trading ACTUALLY works:
    1. Use historical data to select pairs/parameters
    2. Trade forward in time
    3. Re-calibrate periodically
    
    Structure (Bailey et al. 2016):
    |---- Train ----|-- Purge --|---- Test ----|
    
    Purge: Remove last N days of training data before test starts.
           This prevents data leakage from overlapping trades.
           purge_days = ceil(max_holding_days) is recommended.
    
    Embargo (for next iteration):
           When using walk-forward with re-training, embargo prevents
           training on data too close to previous test period.
           embargo_days = ceil(avg_holding_days) is recommended.
    
    NOTE: In simple walk-forward (train year Y-1, test year Y), embargo
    is naturally satisfied by the year boundary. The purge is the key.
    """
    
    def __init__(
        self,
        train_years: int = 1,
        test_years: int = 1,
        purge_days: int = 21,   # Should be >= max_holding_days
        embargo_days: int = 10,  # Should be >= avg_holding_days
    ):
        self.train_years = train_years
        self.test_years = test_years
        self.purge_days = purge_days
        self.embargo_days = embargo_days
    
    def generate_splits(
        self,
        dates: pd.DatetimeIndex,
    ) -> List[Tuple[np.ndarray, np.ndarray, int, int]]:
        """
        Generate walk-forward splits with purge and embargo.
        
        Purge: Remove last purge_days from train (before test)
        Embargo: In year-based walk-forward, embargo is implicitly handled
                 by the year boundary. We explicitly enforce it here.
        
        Returns: List of (train_mask, test_mask, train_year, test_year)
        """
        dates = pd.DatetimeIndex(dates)
        n_obs = len(dates)
        
        years = dates.year.unique()
        min_year = years.min()
        max_year = years.max()
        
        splits = []
        
        for test_year in range(min_year + self.train_years, max_year + 1):
            train_year_end = test_year - 1
            train_year_start = train_year_end - self.train_years + 1
            
            if train_year_start < min_year:
                continue
            
            # =====================================================
            # TRAIN PERIOD with PURGE
            # =====================================================
            train_mask = np.zeros(n_obs, dtype=bool)
            for y in range(train_year_start, train_year_end + 1):
                year_mask = dates.year == y
                train_mask |= year_mask
            
            # Apply PURGE: remove last purge_days from train
            # This prevents data leakage from trades that span train/test boundary
            train_indices = np.where(train_mask)[0]
            if len(train_indices) > self.purge_days:
                purge_indices = train_indices[-self.purge_days:]
                train_mask[purge_indices] = False
                logger.debug(
                    f"Year {train_year_end}: Purged {len(purge_indices)} days "
                    f"({dates[purge_indices[0]].date()} to {dates[purge_indices[-1]].date()})"
                )
            
            # =====================================================
            # TEST PERIOD with EMBARGO at START
            # =====================================================
            test_mask = (dates.year == test_year)
            
            # Apply EMBARGO at START of test:
            # This represents the realistic delay between:
            # 1. End of formation period (pair selection complete)
            # 2. Start of actual trading (market adjustment, implementation delay)
            test_indices = np.where(test_mask)[0]
            if len(test_indices) > self.embargo_days:
                embargo_indices = test_indices[:self.embargo_days]
                test_mask[embargo_indices] = False
                logger.debug(
                    f"Year {test_year}: Embargo {len(embargo_indices)} days at start "
                    f"({dates[embargo_indices[0]].date()} to {dates[embargo_indices[-1]].date()})"
                )
            
            # Validate: no overlap between train and test
            overlap = train_mask & test_mask
            if overlap.any():
                logger.error(f"DATA LEAK! Train/test overlap detected for year {test_year}")
                continue
            
            if train_mask.sum() > 0 and test_mask.sum() > 0:
                splits.append((train_mask, test_mask, train_year_end, test_year))
                logger.debug(
                    f"Split: Train {train_year_start}-{train_year_end} ({train_mask.sum()} days) "
                    f"-> Test {test_year} ({test_mask.sum()} days)"
                )
        
        return splits
    
    def analyze(
        self,
        returns_matrix: np.ndarray,
        dates: pd.DatetimeIndex,
        strategy_names: Optional[List[str]] = None,
    ) -> CPCVResult:
        """
        Run walk-forward analysis with proper purge/embargo.
        """
        n_obs, n_strategies = returns_matrix.shape
        
        if strategy_names is None:
            strategy_names = [f"S{i}" for i in range(n_strategies)]
        
        splits = self.generate_splits(dates)
        
        # Validation logging
        logger.info("Walk-Forward CPCV Analysis:")
        logger.info(f"  - {len(splits)} train/test periods")
        logger.info(f"  - Purge window: {self.purge_days} days")
        logger.info(f"  - Embargo window: {self.embargo_days} days")
        logger.info(f"  - {n_strategies} strategies being evaluated")
        
        if len(splits) == 0:
            raise ValueError("No valid splits generated. Check date range and purge/embargo settings.")
        
        pbo_count = 0
        oos_ranks = []
        is_returns = []
        oos_returns = []
        selection_counts = {name: 0 for name in strategy_names}
        
        # Track for detailed reporting
        yearly_results = []
        
        for train_mask, test_mask, train_year, test_year in splits:
            train_data = returns_matrix[train_mask, :]
            test_data = returns_matrix[test_mask, :]
            
            # Sanity check: no overlap
            if (train_mask & test_mask).any():
                logger.error(f"CRITICAL: Data leak detected in {train_year}->{test_year}!")
                continue
            
            train_means = train_data.mean(axis=0)
            test_means = test_data.mean(axis=0)
            
            best_train_idx = np.argmax(train_means)
            test_ranks = scipy_stats.rankdata(-test_means)
            best_test_rank = int(test_ranks[best_train_idx])
            
            selection_counts[strategy_names[best_train_idx]] += 1
            oos_ranks.append(best_test_rank)
            is_returns.append(train_means[best_train_idx])
            oos_returns.append(test_means[best_train_idx])
            
            if test_means[best_train_idx] < np.median(test_means):
                pbo_count += 1
            
            yearly_results.append({
                'train_year': train_year,
                'test_year': test_year,
                'best_is_strategy': strategy_names[best_train_idx],
                'is_return': train_means[best_train_idx],
                'oos_return': test_means[best_train_idx],
                'oos_rank': best_test_rank,
                'train_days': train_mask.sum(),
                'test_days': test_mask.sum(),
            })
            
            logger.debug(
                f"Year {train_year}->{test_year}: "
                f"Best IS = {strategy_names[best_train_idx]}, "
                f"IS ret = {train_means[best_train_idx]:.4%}, "
                f"OOS ret = {test_means[best_train_idx]:.4%}, "
                f"OOS rank = {best_test_rank}/{n_strategies}"
            )
        
        n_valid = len(oos_ranks)
        pbo = pbo_count / n_valid if n_valid > 0 else 0
        
        is_mean = np.mean(is_returns) if is_returns else 0
        oos_mean = np.mean(oos_returns) if oos_returns else 0
        degradation = (is_mean - oos_mean) / abs(is_mean) if is_mean != 0 else 0
        
        # Sharpe
        best_idx = np.argmax(returns_matrix.mean(axis=0))
        best_returns = returns_matrix[:, best_idx]
        sharpe_obs = (best_returns.mean() / (best_returns.std() + 1e-8)) * np.sqrt(252)
        
        config = CPCVConfig(
            n_splits=len(splits) * 2,
            purge_window=self.purge_days,
            embargo_window=self.embargo_days,
        )
        
        return CPCVResult(
            config=config,
            n_strategies=n_strategies,
            n_observations=n_obs,
            n_combinations_tested=n_valid,
            pbo=pbo,
            sharpe_observed=sharpe_obs,
            is_mean_return=is_mean,
            oos_mean_return=oos_mean,
            degradation_ratio=degradation,
            prob_top_quartile=np.mean([r <= max(1, n_strategies//4) for r in oos_ranks]) if oos_ranks else 0,
            oos_ranks=oos_ranks,
            strategy_names=strategy_names,
            selection_frequency=selection_counts,
        )


# =============================================================================
# CSCV - For comparison / overfitting detection ONLY
# =============================================================================

class CSCVAnalyzer:
    """
    Combinatorial Symmetric Cross-Validation (Bailey et al., 2016).
    
    WARNING: CSCV intentionally allows train/test permutations to estimate
    the probability of backtest overfitting (PBO). Use it as a diagnostic,
    not as a deployment validation tool.
    """
    
    def __init__(self, n_splits: int = 10, random_seed: int = 42):
        assert n_splits >= 4 and n_splits % 2 == 0
        self.n_splits = n_splits
        self.n_test = n_splits // 2
        self.rng = np.random.default_rng(random_seed)
    
    def analyze(
        self,
        returns_matrix: np.ndarray,
        strategy_names: Optional[List[str]] = None,
        max_combinations: Optional[int] = 252,
    ) -> CPCVResult:
        """Run CSCV analysis and return a CPCVResult-style report."""
        n_obs, n_strategies = returns_matrix.shape
        
        if strategy_names is None:
            strategy_names = [f"S{i}" for i in range(n_strategies)]
        
        split_size = n_obs // self.n_splits
        split_boundaries = [
            (i * split_size, (i+1) * split_size if i < self.n_splits - 1 else n_obs)
            for i in range(self.n_splits)
        ]
        
        all_combos = list(combinations(range(self.n_splits), self.n_test))
        if max_combinations and len(all_combos) > max_combinations:
            indices = self.rng.choice(len(all_combos), size=max_combinations, replace=False)
            combos = [all_combos[i] for i in indices]
        else:
            combos = all_combos
        
        pbo_count = 0
        oos_ranks = []
        is_returns = []
        oos_returns = []
        selection_counts = {name: 0 for name in strategy_names}
        all_is_rankings = []
        all_oos_rankings = []
        
        for test_indices in combos:
            test_set = set(test_indices)
            train_indices = [i for i in range(self.n_splits) if i not in test_set]
            
            train_mask = np.zeros(n_obs, dtype=bool)
            test_mask = np.zeros(n_obs, dtype=bool)
            
            for i in train_indices:
                s, e = split_boundaries[i]
                train_mask[s:e] = True
            for i in test_indices:
                s, e = split_boundaries[i]
                test_mask[s:e] = True
            
            train_data = returns_matrix[train_mask, :]
            test_data = returns_matrix[test_mask, :]
            if train_data.size == 0 or test_data.size == 0:
                continue
            
            train_means = train_data.mean(axis=0)
            test_means = test_data.mean(axis=0)
            
            best_train_idx = np.argmax(train_means)
            best_name = strategy_names[best_train_idx]
            test_ranks = scipy_stats.rankdata(-test_means)
            best_rank = int(test_ranks[best_train_idx])
            
            selection_counts[best_name] += 1
            oos_ranks.append(best_rank)
            is_returns.append(train_means[best_train_idx])
            oos_returns.append(test_means[best_train_idx])
            
            if test_means[best_train_idx] < np.median(test_means):
                pbo_count += 1
            
            all_is_rankings.append(scipy_stats.rankdata(-train_means))
            all_oos_rankings.append(test_ranks)
        
        n_valid = len(oos_ranks)
        if n_valid == 0:
            raise ValueError("CSCV produced no valid combinations")
        
        pbo = pbo_count / n_valid
        is_mean = float(np.mean(is_returns))
        oos_mean = float(np.mean(oos_returns))
        degradation = (is_mean - oos_mean) / abs(is_mean) if is_mean != 0 else 0.0
        
        avg_is = np.mean(all_is_rankings, axis=0)
        avg_oos = np.mean(all_oos_rankings, axis=0)
        rank_corr, _ = spearmanr(avg_is, avg_oos) if len(avg_is) > 1 else (0.0, 1.0)
        
        top_q = max(1, n_strategies // 4)
        prob_top = np.mean([r <= top_q for r in oos_ranks])
        
        best_idx = np.argmax(returns_matrix.mean(axis=0))
        best_returns = returns_matrix[:, best_idx]
        sharpe_obs = (best_returns.mean() / (best_returns.std() + 1e-8)) * np.sqrt(252)
        dsr, dsr_p = _calculate_dsr(sharpe_obs, n_strategies, n_obs)
        expected_max = _expected_max_sharpe(n_strategies, n_obs)
        
        config = CPCVConfig(
            n_splits=self.n_splits,
            purge_window=0,
            embargo_window=0,
            test_ratio=self.n_test / self.n_splits,
        )
        
        return CPCVResult(
            config=config,
            n_strategies=n_strategies,
            n_observations=n_obs,
            n_combinations_tested=n_valid,
            pbo=pbo,
            pbo_ci_lower=0.0,
            pbo_ci_upper=1.0,
            sharpe_observed=sharpe_obs,
            sharpe_expected_max=expected_max,
            dsr=dsr,
            dsr_pvalue=dsr_p,
            is_mean_return=is_mean,
            oos_mean_return=oos_mean,
            degradation_ratio=degradation,
            rank_correlation=float(rank_corr if not np.isnan(rank_corr) else 0.0),
            prob_top_quartile=float(prob_top),
            oos_ranks=oos_ranks,
            strategy_names=strategy_names,
            selection_frequency=selection_counts,
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def build_returns_matrix_from_trades(
    trades_by_config: Dict[str, List[Dict]],
    date_range: pd.DatetimeIndex,
    initial_capital: float = 50000.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build returns matrix from trade results.
    
    Parameters
    ----------
    trades_by_config : dict
        {config_name: [trade_dict, ...]}
    date_range : DatetimeIndex
        Full date range
    initial_capital : float
        Starting capital
    
    Returns
    -------
    returns_matrix : np.ndarray
        (n_dates, n_configs) daily returns
    strategy_names : list
        Config names
    """
    n_dates = len(date_range)
    strategy_names = list(trades_by_config.keys())
    n_strategies = len(strategy_names)
    
    returns_matrix = np.zeros((n_dates, n_strategies))
    date_to_idx = {d: i for i, d in enumerate(date_range)}
    
    for col, name in enumerate(strategy_names):
        trades = trades_by_config[name]
        capital = initial_capital
        
        for trade in trades:
            exit_date = pd.Timestamp(trade['exit_date'])
            pnl = trade['pnl']
            
            if exit_date not in date_to_idx:
                continue
            
            idx = date_to_idx[exit_date]
            
            # Daily return on exit day
            returns_matrix[idx, col] += pnl / capital
            capital += pnl
    
    return returns_matrix, strategy_names


# =============================================================================
# COMPARISON TEST
# =============================================================================

def compare_cscv_vs_cpcv(
    returns_matrix: np.ndarray,
    dates: pd.DatetimeIndex,
    strategy_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare CSCV vs CPCV results to highlight the difference.
    """
    # CSCV (with data leakage)
    cscv = CSCVAnalyzer(n_splits=10)
    cscv_result = cscv.analyze(returns_matrix, strategy_names)
    
    # CPCV (proper temporal ordering)
    cpcv = CPCVAnalyzer(n_splits=10, purge_window=5, embargo_window=5)
    cpcv_result = cpcv.analyze(returns_matrix, strategy_names)
    
    # Walk-forward (most realistic)
    wf = WalkForwardCPCV(train_years=1, test_years=1, purge_days=21, embargo_days=10)
    wf_result = wf.analyze(returns_matrix, dates, strategy_names)
    
    return {
        'cscv_pbo': cscv_result.pbo,
        'cpcv_pbo': cpcv_result.pbo,
        'walkforward_pbo': wf_result.pbo,
        'cpcv_degradation': cpcv_result.degradation_ratio,
        'wf_degradation': wf_result.degradation_ratio,
        'message': (
            f"CSCV PBO: {cscv_result.pbo:.1%} (diagnostic only)\n"
            f"CPCV PBO: {cpcv_result.pbo:.1%} (proper temporal ordering)\n"
            f"Walk-Forward PBO: {wf_result.pbo:.1%} (most realistic)"
        ),
    }


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_days = 2520  # ~10 years
    n_strategies = 9
    
    dates = pd.date_range("2010-01-01", periods=n_days, freq='B')
    
    # Synthetic returns with slight trend (strategy 0 is slightly better)
    returns = np.random.randn(n_days, n_strategies) * 0.01
    returns[:, 0] += 0.0001  # Small edge
    
    print("Testing CPCV (correct implementation)...")
    analyzer = CPCVAnalyzer(n_splits=10, purge_window=5, embargo_window=5)
    result = analyzer.analyze(returns)
    print(result.summary())
    
    print("\nTesting Walk-Forward CPCV...")
    wf = WalkForwardCPCV(train_years=1, test_years=1)
    wf_result = wf.analyze(returns, dates)
    print(wf_result.summary())
    
    print("\nComparing CSCV vs CPCV...")
    comparison = compare_cscv_vs_cpcv(returns, dates)
    print(comparison['message'])
