"""Dynamic hedge ratio estimation using Kalman filter.

Implements adaptive hedge ratio tracking for pairs trading, allowing the
relationship between assets to evolve over time.

The model assumes:
    price_x_t = β_t * price_y_t + α_t + ε_t

Where β_t (hedge ratio) and α_t (intercept) follow random walks:
    β_t = β_{t-1} + w_β
    α_t = α_{t-1} + w_α

References:
- Elliott, R.J., van der Hoek, J., Malcolm, W.P. (2005). "Pairs trading"
- Montana, G., Triantafyllopoulos, K. (2009). "Dynamic modeling of hedge ratios"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KalmanHedgeResult:
    """Container for Kalman filter hedge ratio estimation results."""
    
    hedge_ratios: pd.Series        # Time-varying hedge ratios β_t
    intercepts: pd.Series          # Time-varying intercepts α_t
    hedge_ratio_var: pd.Series     # Variance of β_t estimates
    spread: pd.Series              # Dynamic spread: x - β_t * y - α_t
    final_hedge_ratio: float       # Last hedge ratio estimate
    final_intercept: float         # Last intercept estimate
    measurement_std: float         # Estimated measurement noise
    
    def to_dataframe(self) -> pd.DataFrame:
        """Combine all series into a single DataFrame."""
        return pd.DataFrame({
            "hedge_ratio": self.hedge_ratios,
            "intercept": self.intercepts,
            "hedge_ratio_var": self.hedge_ratio_var,
            "spread": self.spread,
        })


def kalman_filter_hedge(
    price_x: pd.Series,
    price_y: pd.Series,
    delta: float = 1e-4,
    Ve: float | None = None,
    use_log: bool = True,
    initial_hedge_ratio: float | None = None,
    initial_intercept: float | None = None,
) -> KalmanHedgeResult:
    """Estimate time-varying hedge ratio using Kalman filter.
    
    The filter tracks the linear relationship between two price series,
    allowing the hedge ratio to adapt to regime changes.
    
    State: x_t = [α_t, β_t]' (intercept and hedge ratio)
    Observation: price_x_t = [1, price_y_t] @ x_t + v_t
    State evolution: x_t = x_{t-1} + w_t (random walk)
    
    Parameters
    ----------
    price_x : pd.Series
        Price series for the dependent variable (long leg).
    price_y : pd.Series
        Price series for the independent variable (short leg).
    delta : float, default=1e-4
        Process noise variance. Controls how quickly the hedge ratio adapts.
        Smaller values = slower adaptation, more stable estimates.
        Larger values = faster adaptation, more responsive to changes.
        Typical range: 1e-5 to 1e-3.
    Ve : float | None, default=None
        Measurement noise variance. If None, estimated from data.
    use_log : bool, default=True
        Whether to use log prices. Recommended for scale invariance.
    initial_hedge_ratio : float | None, default=None
        Initial hedge ratio estimate. If None, uses OLS from first 20 obs.
    initial_intercept : float | None, default=None
        Initial intercept estimate. If None, uses OLS from first 20 obs.
        
    Returns
    -------
    KalmanHedgeResult
        Container with time-varying hedge ratios, spreads, and diagnostics.
        
    Examples
    --------
    >>> result = kalman_filter_hedge(spy_prices, qqq_prices)
    >>> current_hedge = result.final_hedge_ratio
    >>> spread = result.spread
    >>> z_score = (spread - spread.mean()) / spread.std()
    """
    # Align and prepare data
    df = pd.concat([price_x, price_y], axis=1, join="inner").dropna()
    if df.shape[0] < 30:
        raise ValueError(f"Need at least 30 observations, got {df.shape[0]}")
    
    x = df.iloc[:, 0].copy()
    y = df.iloc[:, 1].copy()
    
    if use_log:
        x = np.log(x.where(x > 0)).replace([np.inf, -np.inf], np.nan)
        y = np.log(y.where(y > 0)).replace([np.inf, -np.inf], np.nan)
        df = pd.concat([x, y], axis=1, join="inner").dropna()
        x, y = df.iloc[:, 0], df.iloc[:, 1]
    
    n = len(x)
    index = x.index
    
    # Initialize state using OLS on first observations
    init_period = min(20, n // 4)
    if initial_hedge_ratio is None or initial_intercept is None:
        X_init = np.column_stack([np.ones(init_period), y.iloc[:init_period].values])
        y_init = x.iloc[:init_period].values
        try:
            beta_init = np.linalg.lstsq(X_init, y_init, rcond=None)[0]
            if initial_intercept is None:
                initial_intercept = beta_init[0]
            if initial_hedge_ratio is None:
                initial_hedge_ratio = beta_init[1]
        except np.linalg.LinAlgError:
            initial_intercept = initial_intercept or 0.0
            initial_hedge_ratio = initial_hedge_ratio or 1.0
    
    # State vector: [alpha, beta]
    state = np.array([initial_intercept, initial_hedge_ratio])
    
    # State covariance matrix
    P = np.eye(2)
    
    # Process noise covariance (Vw)
    Vw = delta / (1 - delta) * np.eye(2)
    
    # Estimate measurement noise if not provided
    if Ve is None:
        # Use residual variance from initial OLS
        X_full = np.column_stack([np.ones(n), y.values])
        y_full = x.values
        beta_ols = np.linalg.lstsq(X_full, y_full, rcond=None)[0]
        residuals = y_full - X_full @ beta_ols
        Ve = np.var(residuals)
    
    # Storage for results
    hedge_ratios = np.zeros(n)
    intercepts = np.zeros(n)
    hedge_ratio_vars = np.zeros(n)
    spreads = np.zeros(n)
    
    for t in range(n):
        # Observation vector: [1, y_t]
        F = np.array([1.0, y.iloc[t]])
        
        # Prediction
        state_pred = state
        P_pred = P + Vw
        
        # Observation
        y_obs = x.iloc[t]
        y_pred = F @ state_pred
        
        # Innovation
        e = y_obs - y_pred
        
        # Innovation variance
        Q = F @ P_pred @ F.T + Ve
        
        # Kalman gain
        K = P_pred @ F.T / Q
        
        # State update
        state = state_pred + K * e
        
        # Covariance update (Joseph form for numerical stability)
        I_KF = np.eye(2) - np.outer(K, F)
        P = I_KF @ P_pred @ I_KF.T + np.outer(K, K) * Ve
        
        # Store results
        intercepts[t] = state[0]
        hedge_ratios[t] = state[1]
        hedge_ratio_vars[t] = P[1, 1]
        spreads[t] = e  # Innovation is the spread
    
    return KalmanHedgeResult(
        hedge_ratios=pd.Series(hedge_ratios, index=index, name="hedge_ratio"),
        intercepts=pd.Series(intercepts, index=index, name="intercept"),
        hedge_ratio_var=pd.Series(hedge_ratio_vars, index=index, name="hedge_ratio_var"),
        spread=pd.Series(spreads, index=index, name="spread"),
        final_hedge_ratio=float(hedge_ratios[-1]),
        final_intercept=float(intercepts[-1]),
        measurement_std=float(np.sqrt(Ve)),
    )


def kalman_filter_hedge_with_regime(
    price_x: pd.Series,
    price_y: pd.Series,
    regime_indicator: pd.Series | None = None,
    deltas: Sequence[float] = (1e-5, 1e-3),
    Ve: float | None = None,
    use_log: bool = True,
) -> KalmanHedgeResult:
    """Kalman filter with regime-dependent process noise.
    
    Allows different adaptation speeds in different market regimes.
    Higher delta during high-volatility regimes = faster adaptation.
    
    Parameters
    ----------
    price_x : pd.Series
        Price series for dependent variable.
    price_y : pd.Series
        Price series for independent variable.
    regime_indicator : pd.Series | None, default=None
        Binary series (0/1) indicating regime. 0 = low vol, 1 = high vol.
        If None, uses rolling volatility to determine regime.
    deltas : Sequence[float], default=(1e-5, 1e-3)
        Process noise for [low_vol_regime, high_vol_regime].
    Ve : float | None, default=None
        Measurement noise variance.
    use_log : bool, default=True
        Whether to use log prices.
        
    Returns
    -------
    KalmanHedgeResult
        Time-varying hedge ratios adapted to market regime.
    """
    # Align data
    df = pd.concat([price_x, price_y], axis=1, join="inner").dropna()
    if df.shape[0] < 60:
        raise ValueError(f"Need at least 60 observations for regime detection, got {df.shape[0]}")
    
    x = df.iloc[:, 0].copy()
    y = df.iloc[:, 1].copy()
    
    if use_log:
        x = np.log(x.where(x > 0)).replace([np.inf, -np.inf], np.nan)
        y = np.log(y.where(y > 0)).replace([np.inf, -np.inf], np.nan)
        df = pd.concat([x, y], axis=1, join="inner").dropna()
        x, y = df.iloc[:, 0], df.iloc[:, 1]
    
    n = len(x)
    index = x.index
    
    # Determine regime if not provided
    if regime_indicator is None:
        # Use rolling volatility of spread for regime detection
        simple_spread = x - y  # Approximate spread
        vol_20 = simple_spread.rolling(window=20, min_periods=10).std()
        vol_60 = simple_spread.rolling(window=60, min_periods=30).std()
        regime_indicator = (vol_20 > vol_60 * 1.5).astype(int)
    
    # Align regime indicator
    regime_indicator = regime_indicator.reindex(index).fillna(0).astype(int)
    
    # Initialize state
    init_period = min(20, n // 4)
    X_init = np.column_stack([np.ones(init_period), y.iloc[:init_period].values])
    y_init = x.iloc[:init_period].values
    beta_init = np.linalg.lstsq(X_init, y_init, rcond=None)[0]
    
    state = np.array([beta_init[0], beta_init[1]])
    P = np.eye(2)
    
    # Estimate measurement noise
    if Ve is None:
        X_full = np.column_stack([np.ones(n), y.values])
        residuals = x.values - X_full @ np.linalg.lstsq(X_full, x.values, rcond=None)[0]
        Ve = np.var(residuals)
    
    # Storage
    hedge_ratios = np.zeros(n)
    intercepts = np.zeros(n)
    hedge_ratio_vars = np.zeros(n)
    spreads = np.zeros(n)
    
    for t in range(n):
        # Select delta based on regime
        regime = regime_indicator.iloc[t]
        delta = deltas[min(regime, len(deltas) - 1)]
        Vw = delta / (1 - delta) * np.eye(2)
        
        F = np.array([1.0, y.iloc[t]])
        
        # Kalman filter steps
        state_pred = state
        P_pred = P + Vw
        
        y_obs = x.iloc[t]
        y_pred = F @ state_pred
        e = y_obs - y_pred
        
        Q = F @ P_pred @ F.T + Ve
        K = P_pred @ F.T / Q
        
        state = state_pred + K * e
        I_KF = np.eye(2) - np.outer(K, F)
        P = I_KF @ P_pred @ I_KF.T + np.outer(K, K) * Ve
        
        intercepts[t] = state[0]
        hedge_ratios[t] = state[1]
        hedge_ratio_vars[t] = P[1, 1]
        spreads[t] = e
    
    return KalmanHedgeResult(
        hedge_ratios=pd.Series(hedge_ratios, index=index, name="hedge_ratio"),
        intercepts=pd.Series(intercepts, index=index, name="intercept"),
        hedge_ratio_var=pd.Series(hedge_ratio_vars, index=index, name="hedge_ratio_var"),
        spread=pd.Series(spreads, index=index, name="spread"),
        final_hedge_ratio=float(hedge_ratios[-1]),
        final_intercept=float(intercepts[-1]),
        measurement_std=float(np.sqrt(Ve)),
    )


def estimate_hedge_ratio_ols(
    price_x: pd.Series,
    price_y: pd.Series,
    use_log: bool = True,
    window: int | None = None,
) -> pd.Series | float:
    """Simple OLS hedge ratio estimation (static or rolling).
    
    Parameters
    ----------
    price_x : pd.Series
        Dependent variable prices.
    price_y : pd.Series  
        Independent variable prices.
    use_log : bool, default=True
        Whether to use log prices.
    window : int | None, default=None
        If provided, compute rolling hedge ratio.
        
    Returns
    -------
    pd.Series | float
        Rolling hedge ratios if window provided, else single static estimate.
    """
    df = pd.concat([price_x, price_y], axis=1, join="inner").dropna()
    x = df.iloc[:, 0].copy()
    y = df.iloc[:, 1].copy()
    
    if use_log:
        x = np.log(x.where(x > 0)).replace([np.inf, -np.inf], np.nan)
        y = np.log(y.where(y > 0)).replace([np.inf, -np.inf], np.nan)
        df = pd.concat([x, y], axis=1, join="inner").dropna()
        x, y = df.iloc[:, 0], df.iloc[:, 1]
    
    if window is None:
        # Static OLS
        X = np.column_stack([np.ones(len(y)), y.values])
        beta = np.linalg.lstsq(X, x.values, rcond=None)[0]
        return float(beta[1])
    else:
        # Rolling OLS
        hedge_ratios = pd.Series(index=x.index, dtype=float)
        for i in range(window, len(x) + 1):
            window_x = x.iloc[i - window:i]
            window_y = y.iloc[i - window:i]
            X = np.column_stack([np.ones(window), window_y.values])
            beta = np.linalg.lstsq(X, window_x.values, rcond=None)[0]
            hedge_ratios.iloc[i - 1] = beta[1]
        return hedge_ratios
