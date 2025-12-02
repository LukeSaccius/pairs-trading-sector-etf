"""Signal generation utilities."""

from pairs_trading_etf.signals.zscore import (
    Position,
    TradeSignal,
    SignalConfig,
    calculate_z_score,
    generate_signals,
    signals_to_dataframe,
    summarize_signals,
)
