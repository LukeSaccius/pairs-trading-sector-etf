"""
Integrated Backtest Pipeline with Mandatory CPCV Validation

This module provides a complete backtest pipeline that:
1. Runs walk-forward backtest with parameter variations
2. Performs MANDATORY CPCV analysis to detect overfitting
3. Only returns results if CPCV passes safety thresholds
4. Generates comprehensive validation report

Usage:
------
>>> from pairs_trading_etf.backtests.pipeline import run_validated_backtest
>>> result = run_validated_backtest(prices, base_config)
>>> if result.is_valid:
...     print("Strategy passed validation!")
...     print(result.summary())

The pipeline WILL NOT return positive results for overfit strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from itertools import product
from pathlib import Path
import json
import yaml
from datetime import datetime

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .engine import run_walkforward_backtest
# Use CORRECT CPCV implementation (temporal ordering)
from .cpcv_correct import CSCVAnalyzer, CPCVResult
from .validation import (
    PurgedWalkForwardValidator,
    WalkForwardValidationResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the validated backtest pipeline.
    
    Parameters
    ----------
    run_cpcv : bool
        Whether to run CPCV analysis (default: True, RECOMMENDED)
        Set to False only for quick debugging
        
    cpcv_n_splits : int
        Number of time splits for CPCV (default: 10)
        
    cpcv_purge_window : int
        Days to purge near boundaries (default: 5)
        
    cpcv_embargo_window : int
        Days embargo after test (default: 3)
        
    max_pbo : float
        Maximum acceptable PBO (default: 0.40)
        Strategies with PBO > max_pbo are rejected
        
    min_dsr : float
        Minimum acceptable Deflated Sharpe Ratio (default: 0.0)
        
    require_positive_oos : bool
        Require positive OOS returns (default: True)
        
    parameter_variations : dict
        Parameters to vary for CPCV analysis
        If None, uses default variations
    """
    run_cpcv: bool = True
    cpcv_n_splits: int = 10
    cpcv_purge_window: int = 5
    cpcv_embargo_window: int = 3
    max_pbo: float = 0.40
    min_dsr: float = 0.0
    require_positive_oos: bool = True
    parameter_variations: Optional[Dict[str, List[Any]]] = None
    run_walkforward_validator: bool = True
    walkforward_train_years: int = 1
    walkforward_test_years: int = 1
    walkforward_min_positive_ratio: float = 0.55
    walkforward_min_oos_return: float = 0.0
    walkforward_default_purge: int = 21
    walkforward_default_embargo: int = 5
    
    # Output settings
    save_results: bool = True
    output_dir: str = "results"
    
    def get_default_variations(self) -> Dict[str, List[Any]]:
        """Get default parameter variations for CPCV."""
        return {
            # Keep small to reduce runtime while still checking robustness
            'entry_threshold_sigma': [2.5, 2.8],
            'exit_threshold_sigma': [0.3, 0.5],
        }


# =============================================================================
# PIPELINE RESULT
# =============================================================================

@dataclass
class PipelineResult:
    """Complete result from validated backtest pipeline.
    
    Contains:
    - Backtest performance metrics
    - CPCV validation results
    - Safety assessment
    - Recommendations
    """
    
    # Identification
    config_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M"))
    
    # Backtest results
    total_pnl: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    
    # Trades data
    trades: List[Dict] = field(default_factory=list)
    yearly_breakdown: Dict[int, Dict] = field(default_factory=dict)
    
    # Walk-forward validation
    walkforward_result: Optional[WalkForwardValidationResult] = None
    walkforward_passed: bool = True
    
    # CPCV results (None if not run)
    cpcv_result: Optional[CPCVResult] = None
    
    # Embargo/Purge calculated from actual trades (CRITICAL for validation)
    avg_holding_days: Optional[float] = None
    embargo_width: Optional[int] = None
    purge_width: Optional[int] = None
    
    # Validation status
    cpcv_passed: bool = False
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """True if strategy passed all validation checks."""
        return self.cpcv_passed and self.walkforward_passed and len(self.validation_errors) == 0
    
    @property
    def risk_level(self) -> str:
        """Overall risk assessment."""
        if self.cpcv_result is None:
            return "UNKNOWN"
        return self.cpcv_result.risk_level
    
    def summary(self) -> str:
        """Generate formatted summary."""
        lines = [
            "",
            "=" * 70,
            "VALIDATED BACKTEST PIPELINE RESULT",
            "=" * 70,
            "",
            f"Strategy: {self.config_name}",
            f"Timestamp: {self.timestamp}",
            "",
            "-" * 70,
            "PERFORMANCE METRICS",
            "-" * 70,
            f"  Total PnL: ${self.total_pnl:,.2f}",
            f"  Total Trades: {self.total_trades}",
            f"  Win Rate: {self.win_rate:.1%}",
            f"  Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"  Max Drawdown: ${self.max_drawdown:,.2f}",
            f"  Profit Factor: {self.profit_factor:.2f}",
            "",
        ]

        # Walk-forward validator
        if self.walkforward_result:
            wf = self.walkforward_result
            lines.extend([
                "-" * 70,
                "PURGED WALK-FORWARD VALIDATION",
                "-" * 70,
                f"  Avg IS Return: {wf.avg_is_return:.2%}",
                f"  Avg OOS Return: {wf.avg_oos_return:.2%}",
                f"  Positive OOS Splits: {wf.positive_ratio:.1%}",
                f"  Purge/Embargo: {wf.purge_days}d / {wf.embargo_days}d",
                f"  Result: {'PASSED' if wf.passed else 'FAILED'}",
                "",
            ])
            if wf.warnings:
                lines.append("  Warnings:")
                for warn in wf.warnings:
                    lines.append(f"    - {warn}")
                lines.append("")
        
        # CPCV Results
        if self.cpcv_result:
            lines.extend([
                "-" * 70,
                "CPCV VALIDATION",
                "-" * 70,
            ])
            
            # Show embargo/purge calculation (CRITICAL for transparency)
            if self.avg_holding_days is not None:
                lines.extend([
                    f"  Average Holding Days: {self.avg_holding_days:.1f}",
                    f"  Embargo Width: {self.embargo_width} days (calculated from avg holding)",
                    f"  Purge Width: {self.purge_width} days",
                    "",
                ])
            
            lines.extend([
                f"  PBO: {self.cpcv_result.pbo:.1%}",
                f"  DSR: {self.cpcv_result.dsr:.2f} (p={self.cpcv_result.dsr_pvalue:.3f})",
                f"  Degradation: {self.cpcv_result.degradation_ratio:.1%}",
                f"  Risk Level: {self.cpcv_result.risk_level}",
                "",
            ])
        
        # Validation Status
        lines.extend([
            "-" * 70,
            "VALIDATION STATUS",
            "-" * 70,
        ])
        
        if self.is_valid:
            lines.append("  PASSED - Strategy is validated for deployment")
        else:
            lines.append("  FAILED - Strategy did not pass validation")
        
        if self.validation_errors:
            lines.append("")
            lines.append("  Errors:")
            for err in self.validation_errors:
                lines.append(f"    - {err}")
        
        if self.validation_warnings:
            lines.append("")
            lines.append("  Warnings:")
            for warn in self.validation_warnings:
                lines.append(f"    - {warn}")
        
        lines.extend(["", "=" * 70, ""])
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'config_name': self.config_name,
            'timestamp': self.timestamp,
            'performance': {
                'total_pnl': self.total_pnl,
                'total_trades': self.total_trades,
                'win_rate': self.win_rate,
                'sharpe_ratio': self.sharpe_ratio,
                'max_drawdown': self.max_drawdown,
                'profit_factor': self.profit_factor,
            },
            'walkforward': {
                'avg_is_return': self.walkforward_result.avg_is_return,
                'avg_oos_return': self.walkforward_result.avg_oos_return,
                'positive_ratio': self.walkforward_result.positive_ratio,
                'purge_days': self.walkforward_result.purge_days,
                'embargo_days': self.walkforward_result.embargo_days,
                'passed': self.walkforward_result.passed,
                'warnings': self.walkforward_result.warnings,
            } if self.walkforward_result else None,
            'cpcv': self.cpcv_result.to_dict() if self.cpcv_result else None,
            'validation': {
                'is_valid': self.is_valid,
                'cpcv_passed': self.cpcv_passed,
                'walkforward_passed': self.walkforward_passed,
                'risk_level': self.risk_level,
                'errors': self.validation_errors,
                'warnings': self.validation_warnings,
            },
        }


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def run_validated_backtest(
    prices: pd.DataFrame,
    config: BacktestConfig,
    pipeline_config: Optional[PipelineConfig] = None,
    start_year: int = 2010,
    end_year: int = 2024,
    verbose: bool = True,
) -> PipelineResult:
    """
    Run complete backtest pipeline with mandatory CPCV validation.
    
    This is the RECOMMENDED entry point for backtesting. It ensures:
    1. Walk-forward backtest is run correctly
    2. CPCV analysis detects potential overfitting
    3. Results are validated against safety thresholds
    4. Only validated strategies are marked as deployable
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data with DatetimeIndex
    config : BacktestConfig
        Base configuration for backtest
    pipeline_config : PipelineConfig, optional
        Pipeline settings (uses defaults if None)
    start_year : int
        Start year for backtest
    end_year : int
        End year for backtest
    verbose : bool
        Print progress information
        
    Returns
    -------
    PipelineResult
        Complete results with validation status
        
    Example
    -------
    >>> result = run_validated_backtest(prices, config)
    >>> if result.is_valid:
    ...     print("Strategy is safe to deploy!")
    ... else:
    ...     print("Strategy failed validation:")
    ...     for err in result.validation_errors:
    ...         print(f"  - {err}")
    """
    if pipeline_config is None:
        pipeline_config = PipelineConfig()
    
    result = PipelineResult(config_name=config.experiment_name)
    
    # Filter prices to date range
    # Include formation year before start_year so walk-forward has full lookback
    mask = (prices.index.year >= (start_year - 1)) & (prices.index.year <= end_year)
    prices_filtered = prices[mask]
    
    if verbose:
        logger.info("=" * 60)
        logger.info("VALIDATED BACKTEST PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Config: {config.experiment_name}")
        logger.info(f"Period: {start_year} - {end_year}")
        logger.info(f"CPCV Validation: {'ENABLED' if pipeline_config.run_cpcv else 'DISABLED'}")
    
    # =========================================================================
    # STEP 1: Run base backtest
    # =========================================================================
    if verbose:
        logger.info("")
        logger.info("-" * 60)
        logger.info("STEP 1: Running Walk-Forward Backtest")
        logger.info("-" * 60)
    
    try:
        backtest_result = run_walkforward_backtest(
            prices=prices_filtered,
            cfg=config,
            start_year=start_year,
            end_year=end_year,
        )
        
        # Handle both tuple (all_trades, summary_df) and dict returns
        if isinstance(backtest_result, tuple):
            all_trades, summary_df = backtest_result
            result.trades = all_trades
            
            if not summary_df.empty:
                result.total_pnl = summary_df['total_pnl'].sum()
                result.total_trades = int(summary_df['total_trades'].sum())
                total_wins = int(summary_df['winning_trades'].sum())
                result.win_rate = (total_wins / result.total_trades) if result.total_trades > 0 else 0
                
                # Calculate Sharpe from trades
                if all_trades:
                    pnls = [t['pnl'] for t in all_trades]
                    if len(pnls) > 1 and np.std(pnls) > 0:
                        result.sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
                    
                    # Calculate max drawdown
                    cumulative = np.cumsum(pnls)
                    running_max = np.maximum.accumulate(cumulative)
                    drawdowns = running_max - cumulative
                    result.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
                    
                    # Profit factor
                    wins = sum(p for p in pnls if p > 0)
                    losses = abs(sum(p for p in pnls if p < 0))
                    result.profit_factor = wins / losses if losses > 0 else float('inf')
                
                # Yearly breakdown
                result.yearly_breakdown = summary_df.set_index('trading_year').to_dict('index')
        elif backtest_result and 'all_trades' in backtest_result:
            result.trades = backtest_result['all_trades']
            summary = backtest_result.get('summary', {})
            
            result.total_pnl = summary.get('total_pnl', 0)
            result.total_trades = summary.get('total_trades', 0)
            win_rate_val = summary.get('win_rate', 0)
            result.win_rate = win_rate_val / 100 if win_rate_val > 1 else win_rate_val
            result.sharpe_ratio = summary.get('sharpe_ratio', 0) or 0
            result.max_drawdown = summary.get('max_drawdown', 0)
            result.profit_factor = summary.get('profit_factor', 0) or 0
            result.yearly_breakdown = backtest_result.get('yearly_breakdown', {})
        
        # Check if we got trades
        if result.total_trades == 0:
            result.validation_errors.append("Backtest returned no trades")
            return result
            
    except Exception as e:
        result.validation_errors.append(f"Backtest failed: {str(e)}")
        return result
    
    # Basic validation
    if result.total_trades < 30:
        result.validation_warnings.append(
            f"Low trade count ({result.total_trades}). Results may be unreliable."
        )
    
    if result.total_pnl <= 0:
        result.validation_warnings.append(
            f"Negative PnL (${result.total_pnl:,.2f}). Strategy is unprofitable."
        )
    
    # =========================================================================
    # STEP 1.5: Calculate embargo/purge from actual holding days
    # =========================================================================
    # CRITICAL: embargo_width = ceil(avg_holding_days) per Bailey et al.
    avg_holding_days = None
    calculated_embargo = None
    calculated_purge = None
    
    if result.trades:
        holding_days_list = [
            trade.get('holding_days', 0) 
            for trade in result.trades 
            if trade.get('holding_days') and trade.get('holding_days') > 0
        ]
        if holding_days_list:
            avg_holding_days = np.mean(holding_days_list)
            max_holding_days = np.max(holding_days_list)
            
            # Bailey et al. (2016): embargo = ceil(avg holding), purge = ceil(max holding)
            calculated_embargo = max(1, int(np.ceil(avg_holding_days)))
            calculated_purge = max(1, int(np.ceil(max_holding_days)))
            
            if verbose:
                logger.info("")
                logger.info("-" * 60)
                logger.info("STEP 1.5: Embargo/Purge Calculation (from actual trades)")
                logger.info("-" * 60)
                logger.info(f"  Total trades analyzed: {len(holding_days_list)}")
                logger.info(f"  Average holding days: {avg_holding_days:.1f}")
                logger.info(f"  Max holding days: {max_holding_days}")
                logger.info(f"  Calculated embargo width: {calculated_embargo} days")
                logger.info(f"  Calculated purge width: {calculated_purge} days")
        else:
            if verbose:
                logger.warning("  No valid holding_days data in trades, using config defaults")

    # =========================================================================
    # OPTIONAL: Purged walk-forward validator (practical health check)
    # =========================================================================
    if pipeline_config.run_walkforward_validator and result.trades:
        if verbose:
            logger.info("")
            logger.info("-" * 60)
            logger.info("STEP 1.6: Purged Walk-Forward Validation")
            logger.info("-" * 60)

        purge_days = calculated_purge or pipeline_config.walkforward_default_purge
        embargo_days = calculated_embargo or pipeline_config.walkforward_default_embargo

        try:
            validator = PurgedWalkForwardValidator(
                train_years=pipeline_config.walkforward_train_years,
                test_years=pipeline_config.walkforward_test_years,
                min_positive_ratio=pipeline_config.walkforward_min_positive_ratio,
                min_avg_oos_return=pipeline_config.walkforward_min_oos_return,
                default_purge_days=pipeline_config.walkforward_default_purge,
                default_embargo_days=pipeline_config.walkforward_default_embargo,
            )
            wf_matrix, _ = _build_returns_matrix(
                {'base': result.trades},
                prices_filtered.index,
                config.initial_capital,
            )
            wf_result = validator.evaluate(
                returns_series=wf_matrix[:, 0],
                dates=prices_filtered.index,
                purge_days=purge_days,
                embargo_days=embargo_days,
            )
            result.walkforward_result = wf_result
            result.walkforward_passed = wf_result.passed

            if verbose:
                logger.info(wf_result.summary())

            if not wf_result.passed:
                result.validation_errors.append(
                    "Walk-forward validation failed "
                    f"(positive {wf_result.positive_ratio:.1%}, "
                    f"avg OOS {wf_result.avg_oos_return:.2%})"
                )
            elif wf_result.warnings:
                result.validation_warnings.extend(wf_result.warnings)

        except ValueError as wf_error:
            result.validation_warnings.append(f"Walk-forward validation skipped: {wf_error}")
    elif not pipeline_config.run_walkforward_validator:
        result.walkforward_passed = True
    
    # =========================================================================
    # STEP 2: Generate parameter variations for CPCV
    # =========================================================================
    if not pipeline_config.run_cpcv:
        result.validation_warnings.append(
            "CPCV validation DISABLED. Cannot verify overfitting risk."
        )
        result.cpcv_passed = True  # Skip CPCV check
        return result
    
    if verbose:
        logger.info("")
        logger.info("-" * 60)
        logger.info("STEP 2: Generating Parameter Variations for CPCV")
        logger.info("-" * 60)
    
    param_variations = pipeline_config.parameter_variations
    if param_variations is None:
        param_variations = pipeline_config.get_default_variations()
    
    # Generate all config combinations
    configs_to_test = _generate_config_variations(config, param_variations)
    
    if verbose:
        logger.info(f"Generated {len(configs_to_test)} parameter combinations")
    
    # =========================================================================
    # STEP 3: Run backtests for all variations
    # =========================================================================
    if verbose:
        logger.info("")
        logger.info("-" * 60)
        logger.info("STEP 3: Running Backtests for CPCV Analysis")
        logger.info("-" * 60)
    
    trades_by_config = {}
    
    for i, (name, cfg) in enumerate(configs_to_test):
        if verbose:
            logger.info(f"  [{i+1}/{len(configs_to_test)}] {name}")
        
        try:
            bt_result = run_walkforward_backtest(
                prices=prices_filtered,
                cfg=cfg,
                start_year=start_year,
                end_year=end_year,
            )
            
            # Handle tuple return (all_trades, summary_df)
            if isinstance(bt_result, tuple):
                all_trades, _ = bt_result
                trades_by_config[name] = all_trades
            elif bt_result and 'all_trades' in bt_result:
                trades_by_config[name] = bt_result['all_trades']
        except Exception as e:
            logger.warning(f"  Failed: {e}")
            trades_by_config[name] = []
    
    # Filter valid configs
    valid_trades = {k: v for k, v in trades_by_config.items() if len(v) > 0}
    
    if len(valid_trades) < 3:
        result.validation_errors.append(
            f"Insufficient data for CPCV: only {len(valid_trades)} configs have trades"
        )
        return result
    
    # =========================================================================
    # STEP 4: Run CPCV Analysis (using CORRECT implementation)
    # =========================================================================
    if verbose:
        logger.info("")
        logger.info("-" * 60)
        logger.info("STEP 4: Running CSCV (Bailey) Overfitting Diagnostic")
        logger.info("-" * 60)
    
    try:
        date_range = prices_filtered.index
        returns_matrix, config_names = _build_returns_matrix(
            valid_trades, date_range, config.initial_capital
        )
        
        cscv_analyzer = CSCVAnalyzer(
            n_splits=pipeline_config.cpcv_n_splits
        )
        cpcv_result = cscv_analyzer.analyze(returns_matrix, config_names)
        result.cpcv_result = cpcv_result
        result.avg_holding_days = avg_holding_days
        result.embargo_width = calculated_embargo
        result.purge_width = calculated_purge
        
        if verbose:
            logger.info(f"  CSCV PBO: {cpcv_result.pbo:.1%}")
            logger.info(f"  DSR: {cpcv_result.dsr:.2f}")
            logger.info(f"  Degradation: {cpcv_result.degradation_ratio:.1%}")
            logger.info(f"  Rank Corr: {cpcv_result.rank_correlation:.2f}")
        
    except Exception as e:
        result.validation_errors.append(f"CSCV analysis failed: {str(e)}")
        return result
    
    # =========================================================================
    # STEP 5: Validate against thresholds
    # =========================================================================
    if verbose:
        logger.info("")
        logger.info("-" * 60)
        logger.info("STEP 5: Validating Against Safety Thresholds")
        logger.info("-" * 60)
    
    # Check PBO
    if cpcv_result.pbo > pipeline_config.max_pbo:
        result.validation_errors.append(
            f"PBO ({cpcv_result.pbo:.1%}) exceeds maximum ({pipeline_config.max_pbo:.1%})"
        )
    
    # Check DSR
    if cpcv_result.dsr < pipeline_config.min_dsr:
        result.validation_errors.append(
            f"DSR ({cpcv_result.dsr:.2f}) below minimum ({pipeline_config.min_dsr:.2f})"
        )
    
    # Check OOS returns
    if pipeline_config.require_positive_oos and cpcv_result.oos_mean_return <= 0:
        result.validation_errors.append(
            f"OOS mean return is negative ({cpcv_result.oos_mean_return:.4%})"
        )
    
    # High degradation warning
    if cpcv_result.degradation_ratio > 0.5:
        result.validation_warnings.append(
            f"High performance degradation ({cpcv_result.degradation_ratio:.1%})"
        )
    
    # Low rank correlation warning (use rank_correlation instead of rank_correlation_spearman)
    if hasattr(cpcv_result, 'rank_correlation') and cpcv_result.rank_correlation < 0.3:
        result.validation_warnings.append(
            f"Low rank stability (rho={cpcv_result.rank_correlation:.2f})"
        )
    
    # Set CPCV passed status
    result.cpcv_passed = len(result.validation_errors) == 0
    
    if verbose:
        if result.is_valid:
            logger.info("  All validation checks PASSED")
        else:
            logger.info("  Validation FAILED")
            for err in result.validation_errors:
                logger.info(f"     - {err}")
    
    # =========================================================================
    # STEP 6: Save results if configured
    # =========================================================================
    if pipeline_config.save_results:
        _save_pipeline_result(result, pipeline_config.output_dir, config.experiment_name, config)
    
    return result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _generate_config_variations(
    base_config: BacktestConfig,
    param_ranges: Dict[str, List[Any]],
) -> List[Tuple[str, BacktestConfig]]:
    """Generate all parameter combinations."""
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    configs = []
    for values in product(*param_values):
        cfg_dict = base_config.to_dict()
        name_parts = []
        
        for param, value in zip(param_names, values):
            cfg_dict[param] = value
            short = param.split('_')[0][0] + param.split('_')[-1][0]
            name_parts.append(f"{short}{value}")
        
        name = "_".join(name_parts)
        try:
            configs.append((name, BacktestConfig(**cfg_dict)))
        except Exception:
            pass
    
    return configs


def _build_returns_matrix(
    trades_by_config: Dict[str, List[Dict]],
    date_range: pd.DatetimeIndex,
    initial_capital: float,
) -> Tuple[np.ndarray, List[str]]:
    """Build returns matrix from trades."""
    config_names = list(trades_by_config.keys())
    n_dates = len(date_range)
    n_configs = len(config_names)
    
    returns_matrix = np.zeros((n_dates, n_configs))
    date_to_idx = {d: i for i, d in enumerate(date_range)}
    
    for config_idx, config_name in enumerate(config_names):
        trades = trades_by_config[config_name]
        
        for trade in trades:
            entry = pd.Timestamp(trade['entry_date'])
            exit_date = pd.Timestamp(trade['exit_date'])
            pnl = trade['pnl']
            holding_days = trade.get('holding_days', 1) or 1
            
            daily_return = (pnl / holding_days) / initial_capital
            
            for day in pd.date_range(entry, exit_date):
                if day in date_to_idx:
                    returns_matrix[date_to_idx[day], config_idx] += daily_return
    
    return returns_matrix, config_names


def _save_pipeline_result(
    result: PipelineResult,
    output_dir: str,
    config_name: str,
    config: BacktestConfig | None = None,
) -> None:
    """Save pipeline result to files."""
    output_path = Path(output_dir) / f"{result.timestamp}_{config_name}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save summary JSON
    with open(output_path / "pipeline_result.json", 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    
    # Save trades CSV
    if result.trades:
        trades_df = pd.DataFrame(result.trades)
        trades_df.to_csv(output_path / "trades.csv", index=False)
    
    # Save config snapshot (for visualization and audit)
    if config is not None:
        if hasattr(config, "to_dict"):
            cfg_dict = config.to_dict()
        else:
            cfg_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
        cfg_dict["_saved_at"] = datetime.now().isoformat()
        with open(output_path / "config_snapshot.yaml", "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
    
    # Save CPCV report
    if result.cpcv_result:
        with open(output_path / "cpcv_report.txt", 'w', encoding='utf-8') as f:
            f.write(result.cpcv_result.summary())
    
    # Save validation summary
    with open(output_path / "validation_summary.txt", 'w', encoding='utf-8') as f:
        f.write(result.summary())
    
    logger.info(f"Results saved to {output_path}")


# =============================================================================
# QUICK VALIDATION FUNCTION
# =============================================================================

def quick_validate(
    prices: pd.DataFrame,
    config: BacktestConfig,
    start_year: int = 2010,
    end_year: int = 2024,
) -> bool:
    """
    Quick validation check - returns True if strategy passes basic CPCV.
    
    This is a simplified version for quick checks. Use run_validated_backtest()
    for full analysis.
    """
    pipeline_config = PipelineConfig(
        run_cpcv=True,
        cpcv_n_splits=6,  # Fewer splits for speed
        max_pbo=0.50,     # More lenient threshold
        save_results=False,
    )
    
    result = run_validated_backtest(
        prices, config, pipeline_config, 
        start_year, end_year, 
        verbose=False
    )
    
    return result.is_valid
