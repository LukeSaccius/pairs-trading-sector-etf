"""
Backtesting module for pairs trading.

This module provides:
- BacktestConfig: Unified configuration management
- run_walkforward_backtest: Walk-forward backtest engine
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
]
