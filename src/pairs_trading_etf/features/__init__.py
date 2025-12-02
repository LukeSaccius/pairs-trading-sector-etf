"""Feature generation utilities for ETF analytics."""

from pairs_trading_etf.features.hedging import (
    HedgeRatioEstimate,
    RollingHedgeConfig,
    estimate_hedge_ratio_ols,
    rolling_hedge_ratio,
    calculate_dynamic_spread,
    hedge_ratio_stability,
)

from pairs_trading_etf.features.kalman_hedge import (
    KalmanHedgeResult,
    kalman_filter_hedge,
    kalman_filter_hedge_with_regime,
)

__all__ = [
    # Hedging
    "HedgeRatioEstimate",
    "RollingHedgeConfig",
    "estimate_hedge_ratio_ols",
    "rolling_hedge_ratio",
    "calculate_dynamic_spread",
    "hedge_ratio_stability",
    # Kalman filter
    "KalmanHedgeResult",
    "kalman_filter_hedge",
    "kalman_filter_hedge_with_regime",
]