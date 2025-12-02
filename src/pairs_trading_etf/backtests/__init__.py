"""Backtesting utilities."""

from pairs_trading_etf.backtests.pairs_backtester import (
    PairsBacktester,
    BacktestConfig,
    BacktestResult,
    BacktestMetrics,
    Trade,
    TradeDirection,
    run_backtest,
    compare_configs,
)

__all__ = [
    "PairsBacktester",
    "BacktestConfig",
    "BacktestResult",
    "BacktestMetrics",
    "Trade",
    "TradeDirection",
    "run_backtest",
    "compare_configs",
]
