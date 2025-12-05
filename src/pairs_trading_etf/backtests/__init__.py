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

from .validation import (
    PurgedWalkForwardValidator,
    WalkForwardValidationResult,
)

# NOTE: cross_validation module has been deprecated and moved to project root
# Its functionality is now provided by cpcv_correct.py and cscv_backtest.py

# OLD CPCV (has logic issues - kept for backward compatibility)
from .cpcv import (
    CPCVConfig,
    CPCVResult,
    CPCVAnalyzer,
    build_returns_matrix_from_trades,
    quick_cpcv_check,
)

# NEW CORRECT CPCV (use this for validation)
from .cpcv_correct import (
    CPCVConfig as CPCVConfigCorrect,
    CPCVResult as CPCVResultCorrect,
    CPCVAnalyzer as CPCVAnalyzerCorrect,
    WalkForwardCPCV,
    CSCVAnalyzer,
    compare_cscv_vs_cpcv,
)

# NOTE: cscv_backtest module is deprecated (depends on removed cross_validation.py)
# Use pipeline.py with cpcv_correct.py instead
# from .cscv_backtest import (
#     CSCVBacktestSplit,
#     ParameterGrid,
#     CSCVBacktestResult,
#     run_cscv_backtest,
#     validate_existing_backtest,
# )

from .pipeline import (
    PipelineConfig,
    PipelineResult,
    run_validated_backtest,
    quick_validate,
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
    # Validation helpers
    "PurgedWalkForwardValidator",
    "WalkForwardValidationResult",
    # CPCV (Combinatorial Purged Cross-Validation)
    "CPCVConfig",
    "CPCVResult", 
    "CPCVAnalyzer",
    "build_returns_matrix_from_trades",
    "quick_cpcv_check",
    # NEW CORRECT CPCV (RECOMMENDED)
    "CPCVConfigCorrect",
    "CPCVResultCorrect",
    "CPCVAnalyzerCorrect",
    "WalkForwardCPCV",
    "CSCVAnalyzer",
    "compare_cscv_vs_cpcv",
    # Integrated Pipeline (RECOMMENDED)
    "PipelineConfig",
    "PipelineResult",
    "run_validated_backtest",
    "quick_validate",
]
