"""
Backtesting module for pairs trading.

This module provides:
- BacktestConfig: Unified configuration management
- run_walkforward_backtest: Walk-forward backtest engine
- CSCV: Combinatorially Symmetric Cross-Validation for overfitting detection
- Performance metrics and reporting utilities
"""

from .config import (
    BacktestConfig,
    load_config,
    merge_configs,
    get_conservative_config,
    get_aggressive_config,
    get_europe_only_config,
)

from .engine import (
    run_engle_granger_test,
    select_pairs,
    run_trading_simulation,
    run_walkforward_backtest,
    PairBlacklist,
)

from .metrics import (
    calculate_performance_metrics,
    pnl_by_exit_reason,
    pnl_by_sector,
    print_backtest_report,
    save_results,
)

from .cross_validation import (
    BacktestSplit,
    CVResult,
    CSCVResult,
    run_cross_validated_backtest,
    evaluate_on_test_set,
    select_best_config,
    run_cscv_analysis,
    calculate_deflated_sharpe,
    print_cscv_report,
)

from .cscv_backtest import (
    CSCVBacktestSplit,
    ParameterGrid,
    CSCVBacktestResult,
    run_cscv_backtest,
    validate_existing_backtest,
)

# fast_backtest.py has been removed - functionality merged into main engine

__all__ = [
    # Config
    "BacktestConfig",
    "load_config",
    "merge_configs",
    "get_conservative_config",
    "get_aggressive_config",
    "get_europe_only_config",
    # Engine
    "run_engle_granger_test",
    "select_pairs",
    "run_trading_simulation",
    "run_walkforward_backtest",
    "PairBlacklist",
    # Metrics
    "calculate_performance_metrics",
    "pnl_by_exit_reason",
    "pnl_by_sector",
    "print_backtest_report",
    "save_results",
    # Cross-validation & CSCV
    "BacktestSplit",
    "CVResult",
    "CSCVResult",
    "run_cross_validated_backtest",
    "evaluate_on_test_set",
    "select_best_config",
    "run_cscv_analysis",
    "calculate_deflated_sharpe",
    "print_cscv_report",
    # CSCV-integrated backtest
    "CSCVBacktestSplit",
    "ParameterGrid",
    "CSCVBacktestResult",
    "run_cscv_backtest",
    "validate_existing_backtest",
]
