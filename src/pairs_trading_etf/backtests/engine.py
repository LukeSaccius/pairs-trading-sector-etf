"""
Backtest execution engine for pairs trading.

This module provides the core trading simulation loop, including:
- Cointegration testing
- Pair selection with sector diversification
- Z-score signal generation
- Position management and trade execution
- Dynamic hedge ratio updates
- Vidyamurthy Framework: SNR, Zero-Crossing Rate, Time-based Stops
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint

from .config import BacktestConfig
from ..utils.sectors import get_sector, are_same_sector

logger = logging.getLogger(__name__)


# =============================================================================
# VIDYAMURTHY FRAMEWORK - SNR & TRADABILITY METRICS
# =============================================================================

def calculate_snr(spread: pd.Series, half_life: float) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR) per Vidyamurthy Ch.6.
    
    SNR = sigma_stationary / sigma_nonstationary
    
    For a mean-reverting spread:
    - sigma_stationary = standard deviation of the spread
    - sigma_nonstationary = standard deviation of innovations (changes)
    
    Higher SNR indicates stronger cointegration (mean-reverting vs noise).
    
    Parameters
    ----------
    spread : pd.Series
        Cointegration spread (residuals)
    half_life : float
        Half-life in days
        
    Returns
    -------
    float
        SNR ratio (typically want SNR >= 2.0)
    """
    if len(spread) < 30:
        return 0.0
    
    # Standard deviation of spread (stationary component)
    sigma_stationary = spread.std()
    
    # Standard deviation of changes (non-stationary/noise component)
    spread_diff = spread.diff().dropna()
    sigma_noise = spread_diff.std()
    
    if sigma_noise == 0 or np.isnan(sigma_noise):
        return 0.0
    
    snr = sigma_stationary / sigma_noise
    return float(snr)


def calculate_zero_crossing_rate(spread: pd.Series, lookback: int = 252) -> Tuple[float, float]:
    """
    Calculate Zero-Crossing Rate per Vidyamurthy Ch.7.
    
    The zero-crossing rate measures how frequently the spread crosses 
    its equilibrium (mean). Higher rate = more tradeable.
    
    Also calculates expected holding period:
    E[holding_period] =~ trading_days / (2 * zero_crossings)
    
    Parameters
    ----------
    spread : pd.Series
        Cointegration spread
    lookback : int
        Period for calculation (default 252 = 1 year)
        
    Returns
    -------
    tuple
        (zero_crossing_rate_per_year, expected_holding_days)
    """
    if len(spread) < 30:
        return 0.0, float('inf')
    
    # Use last N days
    s = spread.iloc[-lookback:] if len(spread) > lookback else spread
    
    # Demean the spread
    demeaned = s - s.mean()
    
    # Count zero crossings
    signs = np.sign(demeaned.values)
    signs[signs == 0] = 1  # Treat exactly 0 as positive
    
    crossings = np.sum(signs[1:] != signs[:-1])
    
    # Annualize
    n_days = len(s)
    zcr_annual = crossings * (252 / n_days)
    
    # Expected holding period (time between entries and exits)
    # Vidyamurthy: E[T] ~= N / (2 * crossings) where N is number of observations
    if crossings > 0:
        expected_holding = n_days / (2.0 * crossings)
    else:
        expected_holding = float('inf')
    
    return float(zcr_annual), float(expected_holding)


def bootstrap_holding_period(spread: pd.Series, n_simulations: int = 1000) -> Dict[str, float]:
    """
    Bootstrap estimate of holding period distribution per Vidyamurthy Ch.7.
    
    Resamples the time between zero crossings to estimate
    the distribution of expected holding periods.
    
    Parameters
    ----------
    spread : pd.Series
        Cointegration spread
    n_simulations : int
        Number of bootstrap samples
        
    Returns
    -------
    dict
        {'mean': mean_holding, 'median': median, 'p25': 25th, 'p75': 75th}
    """
    if len(spread) < 30:
        return {'mean': float('inf'), 'median': float('inf'), 'p25': float('inf'), 'p75': float('inf')}
    
    # Find crossing times
    demeaned = spread - spread.mean()
    signs = np.sign(demeaned.values)
    signs[signs == 0] = 1
    
    crossing_indices = np.where(signs[1:] != signs[:-1])[0]
    
    if len(crossing_indices) < 3:
        return {'mean': float('inf'), 'median': float('inf'), 'p25': float('inf'), 'p75': float('inf')}
    
    # Calculate inter-crossing times
    inter_crossing_times = np.diff(crossing_indices)
    
    # Bootstrap
    bootstrap_means = []
    n = len(inter_crossing_times)
    
    for _ in range(n_simulations):
        sample = np.random.choice(inter_crossing_times, size=n, replace=True)
        bootstrap_means.append(sample.mean())
    
    bootstrap_means = np.array(bootstrap_means)
    
    return {
        'mean': float(np.mean(bootstrap_means)),
        'median': float(np.median(bootstrap_means)),
        'p25': float(np.percentile(bootstrap_means, 25)),
        'p75': float(np.percentile(bootstrap_means, 75)),
    }


def calculate_factor_correlation(series_x: pd.Series, series_y: pd.Series) -> float:
    """
    Calculate common factor correlation per Vidyamurthy APT model.
    
    High correlation between price series indicates they share
    common factor exposure (good for pairs trading).
    
    Parameters
    ----------
    series_x : pd.Series
        First price series
    series_y : pd.Series
        Second price series
        
    Returns
    -------
    float
        Correlation coefficient (want >= 0.85)
    """
    aligned = pd.concat([series_x, series_y], axis=1, join='inner').dropna()
    if len(aligned) < 30:
        return 0.0
    
    # Use log returns for correlation
    returns_x = np.log(aligned.iloc[:, 0]).diff().dropna()
    returns_y = np.log(aligned.iloc[:, 1]).diff().dropna()
    
    corr = returns_x.corr(returns_y)
    return float(corr) if not np.isnan(corr) else 0.0


def calculate_time_based_stop(
    entry_z: float,
    current_z: float,
    holding_days: int,
    half_life: float,
    base_stop_zscore: float,
    tightening_rate: float = 0.15,
) -> Tuple[bool, float]:
    """
    Time-based stop tightening per Vidyamurthy insight.
    
    "The mere passage of time represents an increase in risk"
    
    As holding period exceeds half-life, the stop loss tightens,
    because the probability of mean reversion decreases.
    
    Parameters
    ----------
    entry_z : float
        Z-score at entry
    current_z : float
        Current z-score
    holding_days : int
        Days held
    half_life : float
        Expected half-life
    base_stop_zscore : float
        Base stop-loss threshold
    tightening_rate : float
        Rate of stop tightening per half-life elapsed
        
    Returns
    -------
    tuple
        (should_stop, effective_stop_zscore)
    """
    # Calculate how many half-lives have passed
    half_lives_passed = holding_days / max(half_life, 1)
    
    # Only start tightening after 1 full half-life has passed
    if half_lives_passed < 1.0:
        return False, base_stop_zscore
    
    # Tighten stop as more half-lives pass (starts after 1 HL)
    # After 1 HL: start tightening
    # After 2 HL: stop tightens by tightening_rate
    excess_hl = half_lives_passed - 1.0
    tightening = excess_hl * tightening_rate * base_stop_zscore
    
    # Effective stop gets closer to entry z
    effective_stop = base_stop_zscore - tightening
    effective_stop = max(effective_stop, 1.5)  # Floor at z=1.5 (less aggressive)
    
    # Check if stop triggered - the position is getting WORSE (diverging)
    # For long spread (entered at negative z): stop if z goes MORE negative
    # For short spread (entered at positive z): stop if z goes MORE positive
    direction = np.sign(entry_z)  # -1 for long spread, +1 for short spread
    
    if direction < 0:  # Long spread (entered at negative z)
        # Stop if z is MORE negative than effective_stop (diverging)
        should_stop = current_z <= -effective_stop
    else:  # Short spread (entered at positive z)
        # Stop if z is MORE positive than effective_stop (diverging)
        should_stop = current_z >= effective_stop
    
    return should_stop, effective_stop


# =============================================================================
# KALMAN FILTER FOR DYNAMIC HEDGE RATIO
# =============================================================================
# Implementation based on Palomar (2025) Chapter 15.6 "Kalman Filtering for Pairs Trading"
# and the extended momentum model from Section 15.6.3 equation (15.4)

def estimate_kalman_hedge_ratio(
    series_x: pd.Series,
    series_y: pd.Series,
    use_log: bool = True,
    delta: float = 0.00001,
    vw: float = 0.001,
    use_momentum: bool = True,
) -> pd.DataFrame:
    """
    Estimate time-varying hedge ratio using Kalman Filter.
    
    Based on Palomar (2025) Chapter 15.6 "Kalman Filtering for Pairs Trading".
    
    Basic Model (Section 15.6.3, Eq. 15.3):
        y_t = [1, x_t] @ [mu_t, gamma_t]' + epsilon_t    (observation)
        [mu_{t+1}, gamma_{t+1}]' = I @ [mu_t, gamma_t]' + eta_t  (random walk state)
    
    Extended Momentum Model (Eq. 15.4):
        State: [mu_t, gamma_t, gamma_dot_t]' where gamma_dot is hedge ratio velocity
        Observation: y_t = [1, x_t, 0] @ state + epsilon_t
        State transition:
            mu_{t+1} = mu_t + eta_mu
            gamma_{t+1} = gamma_t + gamma_dot_t + eta_gamma  (local linear trend)
            gamma_dot_{t+1} = gamma_dot_t + eta_gamma_dot
    
    The momentum model produces SMOOTHER hedge ratio estimates because it
    models the rate of change, acting as regularization.
    
    Parameters
    ----------
    series_x : pd.Series
        Independent variable (price series)
    series_y : pd.Series
        Dependent variable (price series)
    use_log : bool
        Whether to use log prices (recommended per book)
    delta : float
        Process noise scaling factor. Controls how fast state can change.
        Typical values: 1e-5 to 1e-6 (smaller = more stable)
    vw : float
        Initial observation noise variance (will be adapted online)
    use_momentum : bool
        If True, use the extended momentum model (Eq. 15.4)
        If False, use the basic 2-state model (Eq. 15.3)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: hedge_ratio, intercept, spread
        
    References
    ----------
    - Palomar (2025) "Portfolio Optimization", Chapter 15
    - Chan (2013) "Algorithmic Trading", Chapter 3
    - Vidyamurthy (2004) "Pairs Trading"
    """
    # Align series
    aligned = pd.concat([series_x, series_y], axis=1, join='inner').dropna()
    if len(aligned) < 30:
        return None
    
    x = aligned.iloc[:, 0].values.astype(float)
    y = aligned.iloc[:, 1].values.astype(float)
    
    if use_log:
        x = np.log(x)
        y = np.log(y)
    
    n = len(x)
    
    # Initialize with OLS estimate (as recommended by Palomar Section 15.6.3)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    residuals = y - intercept - slope * x
    sigma_epsilon = np.std(residuals)
    
    if use_momentum:
        # Extended Momentum Model (Eq. 15.4)
        # State: [intercept, hedge_ratio, hedge_ratio_velocity]
        return _kalman_momentum_model(
            x, y, n, intercept, slope, sigma_epsilon, delta, vw, aligned.index
        )
    else:
        # Basic 2-State Model (Eq. 15.3)
        return _kalman_basic_model(
            x, y, n, intercept, slope, sigma_epsilon, delta, vw, aligned.index
        )


def _kalman_basic_model(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    intercept: float,
    slope: float,
    sigma_epsilon: float,
    delta: float,
    vw: float,
    index: pd.Index,
) -> pd.DataFrame:
    """
    Basic 2-state Kalman filter (Palomar Eq. 15.3).
    
    State: [mu_t, gamma_t]' (intercept, hedge_ratio)
    Observation: y_t = [1, x_t] @ state + epsilon_t
    State transition: Random walk (identity matrix)
    """
    # State: [intercept, hedge_ratio]
    theta = np.array([[intercept], [slope]])
    
    # State covariance - initialize based on OLS variance (per Palomar)
    P = np.eye(2)
    P[0, 0] = sigma_epsilon ** 2  # Variance of intercept
    P[1, 1] = sigma_epsilon ** 2 / np.var(x)  # Variance of slope
    
    # Process noise - standard formulation Q = delta * I
    # (Note: old code used delta/(1-delta) which is non-standard)
    Q = delta * np.eye(2)
    
    # Observation noise - will be adapted online
    R = vw
    
    # Storage
    hedge_ratios = np.zeros(n)
    intercepts = np.zeros(n)
    spreads = np.zeros(n)
    
    for t in range(n):
        # Observation matrix: y_t = [1, x_t] @ [intercept, slope]'
        F = np.array([[1.0, x[t]]])
        
        # === Prediction Step ===
        # State prediction: theta_{t|t-1} = theta_{t-1} (random walk)
        # Covariance prediction: P_{t|t-1} = P_{t-1} + Q
        P = P + Q
        
        # === Update Step ===
        # Innovation (prediction error)
        y_pred = (F @ theta)[0, 0]
        innovation = y[t] - y_pred
        
        # Innovation covariance
        S = (F @ P @ F.T)[0, 0] + R
        
        # Kalman gain
        K = (P @ F.T) / S
        
        # State update
        theta = theta + K * innovation
        
        # Covariance update (Joseph form for numerical stability)
        I_KF = np.eye(2) - K @ F
        P = I_KF @ P @ I_KF.T + (K * R) @ K.T
        
        # Adaptive observation variance (exponential smoothing)
        # This helps the filter adapt to changing market volatility
        R = 0.99 * R + 0.01 * (innovation ** 2)
        R = max(R, 1e-8)  # Prevent numerical issues
        
        # Store results
        intercepts[t] = theta[0, 0]
        hedge_ratios[t] = theta[1, 0]
        # Spread = y - intercept - hedge_ratio * x (residual, should be mean-zero)
        spreads[t] = y[t] - theta[0, 0] - theta[1, 0] * x[t]
    
    return pd.DataFrame({
        'hedge_ratio': hedge_ratios,
        'intercept': intercepts,
        'spread': spreads,
    }, index=index)


def _kalman_momentum_model(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    intercept: float,
    slope: float,
    sigma_epsilon: float,
    delta: float,
    vw: float,
    index: pd.Index,
) -> pd.DataFrame:
    """
    Extended Kalman filter with momentum (Palomar Eq. 15.4).
    
    This model tracks the VELOCITY of the hedge ratio, producing smoother
    estimates. This is the "local linear trend" model commonly used in
    time series analysis.
    
    State: [mu_t, gamma_t, gamma_dot_t]' (intercept, hedge_ratio, hedge_ratio_velocity)
    
    State Transition Matrix:
        A = [[1, 0, 0],    # mu_{t+1} = mu_t
             [0, 1, 1],    # gamma_{t+1} = gamma_t + gamma_dot_t
             [0, 0, 1]]    # gamma_dot_{t+1} = gamma_dot_t
             
    Observation: y_t = [1, x_t, 0] @ state + epsilon_t
    
    The key insight is that hedge ratio changes are modeled as having
    MOMENTUM - if it's been increasing, it will likely continue increasing.
    This smooths out noise while still adapting to regime changes.
    """
    # State: [intercept, hedge_ratio, hedge_ratio_velocity]
    theta = np.array([[intercept], [slope], [0.0]])
    
    # State transition matrix (local linear trend)
    A = np.array([
        [1.0, 0.0, 0.0],  # intercept: random walk
        [0.0, 1.0, 1.0],  # hedge_ratio: random walk + velocity
        [0.0, 0.0, 1.0],  # velocity: random walk
    ])
    
    # State covariance
    P = np.eye(3)
    P[0, 0] = sigma_epsilon ** 2
    P[1, 1] = sigma_epsilon ** 2 / max(np.var(x), 1e-8)
    P[2, 2] = delta  # Small initial velocity variance
    
    # Process noise - different scales for each state component
    Q = np.diag([
        delta,          # intercept noise
        delta,          # hedge ratio noise
        delta * 0.1,    # velocity noise (smaller for smoothness)
    ])
    
    # Observation noise
    R = vw
    
    # Storage
    hedge_ratios = np.zeros(n)
    intercepts = np.zeros(n)
    spreads = np.zeros(n)
    velocities = np.zeros(n)
    
    for t in range(n):
        # Observation matrix: y_t = [1, x_t, 0] @ [intercept, hedge_ratio, velocity]'
        F = np.array([[1.0, x[t], 0.0]])
        
        # === Prediction Step ===
        # State prediction with transition
        theta = A @ theta
        # Covariance prediction
        P = A @ P @ A.T + Q
        
        # === Update Step ===
        # Innovation
        y_pred = (F @ theta)[0, 0]
        innovation = y[t] - y_pred
        
        # Innovation covariance
        S = (F @ P @ F.T)[0, 0] + R
        
        # Kalman gain
        K = (P @ F.T) / S
        
        # State update
        theta = theta + K * innovation
        
        # Covariance update (Joseph form)
        I_KF = np.eye(3) - K @ F
        P = I_KF @ P @ I_KF.T + (K * R) @ K.T
        
        # Adaptive observation variance
        R = 0.99 * R + 0.01 * (innovation ** 2)
        R = max(R, 1e-8)
        
        # Store results
        intercepts[t] = theta[0, 0]
        hedge_ratios[t] = theta[1, 0]
        velocities[t] = theta[2, 0]
        spreads[t] = y[t] - theta[0, 0] - theta[1, 0] * x[t]
    
    return pd.DataFrame({
        'hedge_ratio': hedge_ratios,
        'intercept': intercepts,
        'spread': spreads,
        'hedge_velocity': velocities,
    }, index=index)


# =============================================================================
# VIX REGIME FILTER
# =============================================================================

def check_vix_regime(
    vix_data: Optional[pd.Series],
    current_date: pd.Timestamp,
    vix_threshold: float = 30.0,
    lookback_days: int = 5,
) -> Dict[str, Any]:
    """
    Check if current market regime is high volatility based on VIX.
    
    Parameters
    ----------
    vix_data : pd.Series or None
        VIX closing prices indexed by date
    current_date : pd.Timestamp
        Current trading date
    vix_threshold : float
        VIX level above which to flag high volatility regime
    lookback_days : int
        Number of days to average VIX over
        
    Returns
    -------
    dict
        Contains: is_high_vol, current_vix, avg_vix
    """
    if vix_data is None or len(vix_data) == 0:
        return {
            'is_high_vol': False,
            'current_vix': None,
            'avg_vix': None,
        }
    
    try:
        # Get VIX data up to current date
        available = vix_data[vix_data.index <= current_date]
        
        if len(available) == 0:
            return {'is_high_vol': False, 'current_vix': None, 'avg_vix': None}
        
        current_vix = available.iloc[-1]
        avg_vix = available.iloc[-lookback_days:].mean() if len(available) >= lookback_days else available.mean()
        
        is_high_vol = current_vix > vix_threshold or avg_vix > vix_threshold
        
        return {
            'is_high_vol': is_high_vol,
            'current_vix': float(current_vix),
            'avg_vix': float(avg_vix),
        }
    except Exception as e:
        logger.debug(f"VIX check failed: {e}")
        return {'is_high_vol': False, 'current_vix': None, 'avg_vix': None}


# =============================================================================
# VOLATILITY-ADJUSTED POSITION SIZING
# =============================================================================

def calculate_volatility_adjusted_size(
    base_capital: float,
    spread_volatility: float,
    target_volatility: float = 0.02,
    min_scale: float = 0.25,
    max_scale: float = 2.0,
) -> float:
    """
    Calculate position size adjusted for spread volatility.
    
    Position is scaled inversely to volatility:
    - High volatility spread -> smaller position
    - Low volatility spread -> larger position
    
    Parameters
    ----------
    base_capital : float
        Base capital allocation for this trade
    spread_volatility : float
        Daily volatility of the spread
    target_volatility : float
        Target daily volatility for position (default 2%)
    min_scale : float
        Minimum position size as fraction of base (0.25 = 25%)
    max_scale : float
        Maximum position size as fraction of base (2.0 = 200%)
        
    Returns
    -------
    float
        Volatility-adjusted position size
    """
    if spread_volatility <= 0 or np.isnan(spread_volatility):
        return base_capital
    
    # Scale factor: target_vol / actual_vol
    scale = target_volatility / spread_volatility
    
    # Clamp to min/max
    scale = max(min_scale, min(max_scale, scale))
    
    return base_capital * scale


# =============================================================================
# COINTEGRATION TESTING
# =============================================================================

def run_engle_granger_test(
    series_x: pd.Series,
    series_y: pd.Series,
    use_log: bool = True,
    pvalue_threshold: float = 0.05,
    min_half_life: float = 5.0,
    max_half_life: float = 30.0,
) -> Optional[Dict[str, float]]:
    """
    Run Engle-Granger cointegration test on two price series.
    
    
    Uses statsmodels.coint() which implements proper MacKinnon critical values
    for cointegration (NOT standard ADF critical values).
    
    Parameters
    ----------
    series_x : pd.Series
        First price series
    series_y : pd.Series
        Second price series
    use_log : bool
        Whether to use log prices (recommended)
    pvalue_threshold : float
        Maximum p-value for cointegration
    min_half_life : float
        Minimum half-life in days
    max_half_life : float
        Maximum half-life in days
        
    Returns
    -------
    dict or None
        Dictionary with hedge_ratio, pvalue, half_life, spread stats
        None if pair doesn't pass cointegration test
    """
    try:
        # Align series
        aligned = pd.concat([series_x, series_y], axis=1, join='inner').dropna()
        if len(aligned) < 60:
            return None
        
        x = aligned.iloc[:, 0]
        y = aligned.iloc[:, 1]
        
        if use_log:
            x = np.log(x)
            y = np.log(y)
        
        # Engle-Granger test using statsmodels
        test_stat, pvalue, crit_values = coint(x, y, trend='c', maxlag=1, autolag='aic')
        
        if pvalue > pvalue_threshold:
            return None
        
        # Calculate hedge ratio via OLS
        slope, intercept, r_value, _, std_err = stats.linregress(y, x)
        hedge_ratio = slope
        
        # Calculate spread
        spread = x - (intercept + hedge_ratio * y)
        
        # Estimate half-life using OU model
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        common_idx = spread_lag.index.intersection(spread_diff.index)
        spread_lag = spread_lag.loc[common_idx]
        spread_diff = spread_diff.loc[common_idx]
        
        if len(spread_lag) < 30:
            return None
        
        slope_hl, _, _, _, _ = stats.linregress(spread_lag, spread_diff)
        
        if slope_hl >= 0:
            return None
        
        phi = 1 + slope_hl
        if phi <= 0 or phi >= 1:
            return None
        
        half_life = -np.log(2) / np.log(phi)
        
        if not (min_half_life <= half_life <= max_half_life):
            return None
        
        # Spread statistics
        spread_std = spread.std()
        spread_range = spread.max() - spread.min()
        
        # Vidyamurthy metrics
        snr = calculate_snr(spread, half_life)
        zcr, expected_holding = calculate_zero_crossing_rate(spread)
        
        return {
            'hedge_ratio': float(hedge_ratio),
            'intercept': float(intercept),
            'pvalue': float(pvalue),
            'test_stat': float(test_stat),
            'half_life': float(half_life),
            'spread_mean': float(spread.mean()),
            'spread_std': float(spread_std),
            'spread_range': float(spread_range),
            'r_squared': float(r_value ** 2),
            # Vidyamurthy metrics
            'snr': float(snr),
            'zero_crossing_rate': float(zcr),
            'expected_holding': float(expected_holding),
        }
        
    except Exception as e:
        logger.debug(f"Cointegration test failed: {e}")
        return None


def update_hedge_ratio(
    prices: pd.DataFrame,
    pair: Tuple[str, str],
    lookback: int = 63,
    use_log: bool = True,
) -> Tuple[float, float]:
    """
    Update hedge ratio using recent price data.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    pair : tuple
        (ticker_x, ticker_y) pair
    lookback : int
        Days of data to use
    use_log : bool
        Whether to use log prices
        
    Returns
    -------
    tuple
        (hedge_ratio, intercept)
    """
    leg_x, leg_y = pair
    
    x = prices[leg_x].iloc[-lookback:]
    y = prices[leg_y].iloc[-lookback:]
    
    if use_log:
        x = np.log(x)
        y = np.log(y)
    
    slope, intercept, _, _, _ = stats.linregress(y, x)
    return slope, intercept


# =============================================================================
# PAIR SELECTION
# =============================================================================

def select_pairs(
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    blacklist: Optional[set] = None,
) -> Tuple[List[Tuple[str, str]], Dict, Dict, Dict]:
    """
    Select cointegrated pairs from price data.
    
    Process:
    1. Filter by correlation
    2. Filter by sector (if sector_focus enabled)
    3. Test for cointegration
    4. Validate pair stability (if enabled)
    5. Score and rank pairs
    6. Apply diversification limits
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data with tickers as columns
    cfg : BacktestConfig
        Configuration object
    blacklist : set, optional
        Pairs to exclude
        
    Returns
    -------
    tuple
        (selected_pairs, hedge_ratios, half_lives, formation_stats)
    """
    tickers = list(prices.columns)
    n_tickers = len(tickers)
    logger.info(f"Selecting pairs from {n_tickers} tickers")
    
    # Import validation functions if available
    try:
        from .validation import check_rolling_consistency
        validation_available = True
    except ImportError:
        validation_available = False
        logger.debug("Validation module not available")
    
    # Step 1: Correlation filter
    returns = prices.pct_change().dropna()
    corr_matrix = returns.corr()
    
    candidate_pairs = []
    for i in range(n_tickers):
        for j in range(i + 1, n_tickers):
            corr = corr_matrix.iloc[i, j]
            if cfg.min_correlation <= corr <= cfg.max_correlation:
                # Sector filter
                if cfg.sector_focus:
                    if are_same_sector(tickers[i], tickers[j]):
                        sector = get_sector(tickers[i])
                        if sector not in cfg.exclude_sectors:
                            candidate_pairs.append((tickers[i], tickers[j]))
                else:
                    candidate_pairs.append((tickers[i], tickers[j]))
    
    logger.info(f"Pairs with corr {cfg.min_correlation:.2f}-{cfg.max_correlation:.2f}: {len(candidate_pairs)}")
    
    # Blacklist filter
    if blacklist:
        before = len(candidate_pairs)
        candidate_pairs = [
            p for p in candidate_pairs 
            if p not in blacklist and (p[1], p[0]) not in blacklist
        ]
        logger.info(f"After blacklist: {len(candidate_pairs)} (removed {before - len(candidate_pairs)})")
    
    # Step 2: Cointegration test
    cointegrated = []
    results = {}
    validation_scores = {}  # Store stability scores for ranking
    
    for pair in candidate_pairs:
        leg_x, leg_y = pair
        result = run_engle_granger_test(
            prices[leg_x],
            prices[leg_y],
            use_log=cfg.use_log_prices,
            pvalue_threshold=cfg.pvalue_threshold,
            min_half_life=cfg.min_half_life,
            max_half_life=cfg.max_half_life,
        )
        
        if result is not None:
            if result['spread_range'] >= cfg.min_spread_range_pct:
                # Hedge ratio filter - avoid imbalanced positions
                hr = abs(result['hedge_ratio'])
                if cfg.min_hedge_ratio <= hr <= cfg.max_hedge_ratio:
                    # Vidyamurthy filters: SNR and Zero-Crossing Rate
                    snr_ok = result.get('snr', 0) >= getattr(cfg, 'min_snr', 0)
                    zcr_ok = result.get('zero_crossing_rate', 0) >= getattr(cfg, 'min_zero_crossing_rate', 0)
                    
                    if snr_ok and zcr_ok:
                        cointegrated.append(pair)
                        results[pair] = result
                        validation_scores[pair] = 1.0  # Default score
    
    logger.info(f"Cointegrated pairs: {len(cointegrated)}")
    
    if not cointegrated:
        return [], {}, {}, {}
    
    # Step 2.5: Rolling consistency validation (if enabled)
    if validation_available and getattr(cfg, 'rolling_consistency', False):
        n_windows = getattr(cfg, 'n_rolling_windows', 4)
        min_passing = getattr(cfg, 'min_passing_windows', 2)
        
        logger.info(f"Running rolling consistency check ({n_windows} windows, {min_passing} required)")
        validated_pairs = []
        
        for pair in cointegrated:
            rc_result = check_rolling_consistency(
                prices=prices,
                pair=pair,
                use_log=cfg.use_log_prices,
                pvalue_threshold=cfg.pvalue_threshold,
                min_half_life=cfg.min_half_life,
                max_half_life=cfg.max_half_life,
                n_windows=n_windows,
                min_passing=min_passing,
            )
            
            if rc_result.get('passes', False):
                validated_pairs.append(pair)
                # Update validation score for ranking
                validation_scores[pair] = rc_result.get('score', 1.0)
                logger.debug(f"  {pair}: PASSED ({rc_result['passing_windows']}/{n_windows} windows)")
            else:
                logger.debug(f"  {pair}: FAILED ({rc_result.get('passing_windows', 0)}/{n_windows} windows)")
        
        removed = len(cointegrated) - len(validated_pairs)
        logger.info(f"After rolling consistency: {len(validated_pairs)} pairs (removed {removed})")
        cointegrated = validated_pairs
    
    if not cointegrated:
        logger.warning("No pairs passed rolling consistency check")
        return [], {}, {}, {}
    
    # Step 3: Scoring - include Vidyamurthy metrics and validation scores
    scores = {}
    for pair in cointegrated:
        r = results[pair]
        pvalue_score = min(-np.log(max(r['pvalue'], 1e-10)) / 7.0, 1.0)
        hl_score = max(0, 1 - abs(r['half_life'] - 15) / 15)
        range_score = min(r['spread_range'] / 0.10, 1.0)
        
        # Hedge ratio quality score
        hr = abs(r['hedge_ratio'])
        hr_score = 1.0 - abs(hr - 1.0) / 1.0
        hr_score = max(0, min(1, hr_score))
        
        # Vidyamurthy metrics in scoring
        snr = r.get('snr', 1.0)
        zcr = r.get('zero_crossing_rate', 0)
        snr_score = min(snr / 3.0, 1.0)  # Normalize to ~3.0 max
        zcr_score = min(zcr / 20.0, 1.0)  # Normalize to ~20 crossings/year
        
        # Validation/stability score
        stability_score = validation_scores.get(pair, 1.0)
        
        # Updated weights to include Vidyamurthy metrics and stability
        scores[pair] = (
            0.20 * pvalue_score + 
            0.15 * hl_score + 
            0.10 * range_score + 
            0.10 * hr_score +
            0.15 * snr_score +
            0.15 * zcr_score +
            0.15 * stability_score  # Stability matters!
        )
    
    sorted_pairs = sorted(cointegrated, key=lambda p: scores[p], reverse=True)
    
    # Step 4: Diversification (skip limits if unlimited_pairs)
    selected = []
    sector_counts = defaultdict(int)
    etf_counts = defaultdict(int)
    
    for pair in sorted_pairs:
        # Check pair limit (unless unlimited)
        if not cfg.unlimited_pairs and len(selected) >= cfg.top_pairs:
            break
        
        leg_x, leg_y = pair
        sector = get_sector(leg_x)
        
        # Apply diversification limits only if not unlimited
        if not cfg.unlimited_pairs:
            if sector_counts[sector] >= cfg.max_pairs_per_sector:
                continue
            if etf_counts[leg_x] >= cfg.max_pairs_per_etf or etf_counts[leg_y] >= cfg.max_pairs_per_etf:
                continue
        
        selected.append(pair)
        sector_counts[sector] += 1
        etf_counts[leg_x] += 1
        etf_counts[leg_y] += 1
    
    logger.info(f"Selected {len(selected)} pairs")
    
    # Log top pairs
    for i, pair in enumerate(selected[:5], 1):
        r = results[pair]
        sector = get_sector(pair[0])
        logger.info(f"  {i}. {pair} [{sector}]: p={r['pvalue']:.4f}, HL={r['half_life']:.1f}, range={r['spread_range']:.3f}")
    
    # Build output
    hedge_ratios = {p: results[p]['hedge_ratio'] for p in selected}
    half_lives = {p: results[p]['half_life'] for p in selected}
    formation_stats = {p: (results[p]['spread_mean'], results[p]['spread_std']) for p in selected}
    
    return selected, hedge_ratios, half_lives, formation_stats


# =============================================================================
# TRADING SIMULATION
# =============================================================================

def run_trading_simulation(
    prices: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    hedge_ratios: Dict,
    half_lives: Dict,
    cfg: BacktestConfig,
    current_capital: float = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Run trading simulation on selected pairs.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Trading period price data
    pairs : list
        Selected pairs to trade
    hedge_ratios : dict
        Hedge ratios for each pair
    half_lives : dict
        Half-lives for each pair
    cfg : BacktestConfig
        Configuration
    current_capital : float, optional
        Current capital (for compounding mode)
        
    Returns
    -------
    tuple
        (trades_list, ending_capital)
    """
    trades = []
    n_dates = len(prices)
    n_pairs = len(pairs)
    use_adaptive = getattr(cfg, 'use_adaptive_lookback', False)
    lb_min = getattr(cfg, 'adaptive_lookback_min', 30)
    lb_max = getattr(cfg, 'adaptive_lookback_max', cfg.zscore_lookback)
    warmup = max(30, lb_max if use_adaptive else cfg.zscore_lookback)
    
    if n_pairs == 0 or n_dates <= warmup:
        return trades, current_capital if current_capital else cfg.initial_capital
    
    # Capital management
    if current_capital is None:
        current_capital = cfg.initial_capital
    
    # Dynamic capital per pair: divide available among max positions
    max_pos = cfg.max_positions if cfg.max_positions > 0 else len(pairs)
    # Create pair name mapping
    pair_names = {pair: f"{pair[0]}_{pair[1]}" for pair in pairs}
    
    # State tracking
    position_state = {pair: 0 for pair in pairs}
    entry_data = {pair: {} for pair in pairs}
    current_hr = dict(hedge_ratios)
    
    # Calculate initial spreads with NaN/zero price handling
    spreads = pd.DataFrame(index=prices.index)
    for pair in pairs:
        leg_x, leg_y = pair
        hr = current_hr[pair]

        # Check for invalid prices (NaN, zero, negative)
        px = prices[leg_x]
        py = prices[leg_y]
        if (px <= 0).any() or (py <= 0).any():
            logger.warning(f"Invalid prices detected for pair {pair_names[pair]}, skipping")
            spreads[pair_names[pair]] = np.nan
            continue

        log_x = np.log(px)
        log_y = np.log(py)
        spread = log_x - hr * log_y
        spreads[pair_names[pair]] = spread
    
    # Rolling z-score with adaptive lookback per-pair (QMA compliance)
    # If use_adaptive_lookback=True, each pair gets lookback = f(half_life)
    if use_adaptive:
        # Per-pair lookback based on half-life
        mult = getattr(cfg, 'adaptive_lookback_multiplier', 4.0)
        
        rolling_mean = pd.DataFrame(index=prices.index)
        rolling_std = pd.DataFrame(index=prices.index)
        zscores = pd.DataFrame(index=prices.index)
        pair_lookbacks = {}  # Store for reference
        
        for pair in pairs:
            pair_name = pair_names[pair]
            hl = half_lives[pair]
            # Compute adaptive lookback: clamp(mult * hl, min, max)
            lookback = int(max(lb_min, min(lb_max, mult * hl)))
            pair_lookbacks[pair] = lookback
            
            # min_periods must be <= window
            min_per = min(lookback, 30)
            
            spread_series = spreads[pair_name]
            rolling_mean[pair_name] = spread_series.rolling(window=lookback, min_periods=min_per).mean()
            rolling_std[pair_name] = spread_series.rolling(window=lookback, min_periods=min_per).std()
            std_vals = rolling_std[pair_name]
            zscores[pair_name] = (spread_series - rolling_mean[pair_name]) / std_vals.where(std_vals > 0, np.nan)
        
        logger.debug(f"Adaptive lookbacks: min={min(pair_lookbacks.values())}, max={max(pair_lookbacks.values())}")
    else:
        # Original: single lookback for all pairs
        rolling_mean = spreads.rolling(window=cfg.zscore_lookback, min_periods=30).mean()
        rolling_std = spreads.rolling(window=cfg.zscore_lookback, min_periods=30).std()
        zscores = (spreads - rolling_mean) / rolling_std
    
    # Determine hedge ratio methodology
    hedge_method = getattr(cfg, 'hedge_ratio_method', 'auto')
    if isinstance(hedge_method, str):
        method_normalized = hedge_method.strip().lower()
    else:
        method_normalized = 'auto'
    use_kalman = getattr(cfg, 'use_kalman_hedge', False)
    use_dynamic_ols = getattr(cfg, 'dynamic_hedge', False)
    if method_normalized != 'auto':
        if method_normalized == 'kalman':
            use_kalman = True
            use_dynamic_ols = False
        elif method_normalized in {'rolling', 'ols'}:
            use_dynamic_ols = True
            use_kalman = False
        elif method_normalized == 'fixed':
            use_dynamic_ols = False
            use_kalman = False
        logger.info("Hedge ratio method override: %s", method_normalized)
    
    # Pre-compute Kalman hedge ratios if enabled
    kalman_results = {}
    if use_kalman:
        use_momentum = getattr(cfg, 'kalman_use_momentum', True)
        model_type = "momentum" if use_momentum else "basic"
        logger.info(f"Computing Kalman filter hedge ratios ({model_type} model)...")
        for pair in pairs:
            leg_x, leg_y = pair
            kalman_df = estimate_kalman_hedge_ratio(
                prices[leg_x],
                prices[leg_y],
                use_log=cfg.use_log_prices,
                delta=getattr(cfg, 'kalman_delta', 0.00001),
                vw=getattr(cfg, 'kalman_vw', 0.001),
                use_momentum=use_momentum,
            )
            if kalman_df is not None:
                kalman_results[pair] = kalman_df
        
        # Replace spreads with Kalman-adjusted spreads
        # Option 1: Use full Palomar formula (mean-zero spread, small variance)
        # Option 2: Use only Kalman hedge ratio with OLS-style spread (no intercept)
        # We use option 2 for compatibility with rolling z-score
        if kalman_results:
            logger.info("Updating spreads with Kalman hedge ratios (OLS-style)...")
            for pair in pairs:
                if pair in kalman_results:
                    pair_name = pair_names[pair]
                    leg_x, leg_y = pair
                    
                    kalman_df = kalman_results[pair]
                    kalman_hr = kalman_df['hedge_ratio']
                    
                    # Get log prices
                    log_x = np.log(prices[leg_x])
                    log_y = np.log(prices[leg_y])
                    
                    # Compute spread with time-varying hedge ratio only (no intercept)
                    # IMPORTANT: Use same convention as normal spread: log_x - hr * log_y
                    # (Bug fix: was reversed as log_y - hr * log_x which inverted signals)
                    common_idx = spreads.index.intersection(kalman_df.index)
                    new_spread = log_x.loc[common_idx] - kalman_hr.loc[common_idx] * log_y.loc[common_idx]
                    spreads.loc[common_idx, pair_name] = new_spread.values
            
            # Recalculate z-scores with Kalman-adjusted spreads
            rolling_mean = spreads.rolling(window=cfg.zscore_lookback, min_periods=30).mean()
            rolling_std = spreads.rolling(window=cfg.zscore_lookback, min_periods=30).std()
            zscores = (spreads - rolling_mean) / rolling_std
    
    dates = prices.index.tolist()
    
    # FIX LOOK-AHEAD BIAS: Start from warmup+1 so we have t-1 for signals
    for t in range(warmup + 1, n_dates):
        current_date = dates[t]        # Execution date (today)
        signal_date = dates[t - 1]      # Signal date (yesterday's close)
        
        # Kalman hedge ratio update (if enabled)
        # Spreads are already pre-computed with Kalman hedge ratios
        # Only update current_hr for position management
        if use_kalman:
            for pair in pairs:
                if pair in kalman_results and current_date in kalman_results[pair].index:
                    if position_state[pair] == 0:  # Only update when not in position
                        kalman_hr = kalman_results[pair].loc[current_date, 'hedge_ratio']
                        if not np.isnan(kalman_hr):
                            current_hr[pair] = kalman_hr
            # Z-scores already computed from Kalman-adjusted spreads
        
        # Dynamic hedge ratio update (OLS-based, if not using Kalman)
        elif use_dynamic_ols and t % cfg.hedge_update_days == 0 and t > cfg.hedge_update_days:
            for pair in pairs:
                try:
                    new_hr, _ = update_hedge_ratio(
                        prices.iloc[:t], pair,
                        lookback=cfg.hedge_update_days,
                        use_log=cfg.use_log_prices
                    )
                    if position_state[pair] == 0:
                        current_hr[pair] = new_hr
                        leg_x, leg_y = pair
                        log_x = np.log(prices[leg_x].iloc[:t])
                        log_y = np.log(prices[leg_y].iloc[:t])
                        spreads[pair_names[pair]] = log_x - new_hr * log_y
                except Exception:
                    pass
            
            # Recalculate rolling stats (respecting adaptive lookback setting)
            if use_adaptive:
                for pair in pairs:
                    pair_name = pair_names[pair]
                    lookback = pair_lookbacks[pair]
                    min_per = min(lookback, 30)  # min_periods must be <= window
                    spread_series = spreads[pair_name]
                    rolling_mean[pair_name] = spread_series.rolling(window=lookback, min_periods=min_per).mean()
                    rolling_std[pair_name] = spread_series.rolling(window=lookback, min_periods=min_per).std()
                    std_vals = rolling_std[pair_name]
                    zscores[pair_name] = (spread_series - rolling_mean[pair_name]) / std_vals.where(std_vals > 0, np.nan)
            else:
                rolling_mean = spreads.rolling(window=cfg.zscore_lookback, min_periods=30).mean()
                rolling_std = spreads.rolling(window=cfg.zscore_lookback, min_periods=30).std()
                zscores = (spreads - rolling_mean) / rolling_std
        
        # Check exits
        # FIX LOOK-AHEAD BIAS: Signal from t-1 (signal_date), execution at t (current_date)
        for pair in pairs:
            if position_state[pair] == 0:
                continue
            
            pair_name = pair_names[pair]
            direction = position_state[pair]
            entry = entry_data[pair]
            
            # QMA Level 2: Complete fixed-parameter exit
            # Use hr_entry, mu_entry, sigma_entry from entry time
            # This prevents "Rolling Beta Trap" where exit uses different distribution
            if getattr(cfg, 'use_fixed_exit_params', True):
                # Recalculate spread using hr from entry time
                # SIGNAL uses signal_date (t-1) prices for z-score calculation
                leg_x, leg_y = pair
                log_x = np.log(prices.loc[signal_date, leg_x])
                log_y = np.log(prices.loc[signal_date, leg_y])
                hr_entry = entry.get('hr', current_hr[pair])  # Use entry hr
                spread = log_x - hr_entry * log_y
                
                # Use mu and sigma from entry time for consistent exit evaluation
                mu_entry = entry.get('mu_entry', rolling_mean.loc[signal_date, pair_name])
                sigma_entry = entry.get('sigma_entry', rolling_std.loc[signal_date, pair_name])
                if sigma_entry > 0:
                    z = (spread - mu_entry) / sigma_entry
                else:
                    z = 0.0
            else:
                # Original behavior: use rolling z-score with rolling hr
                # SIGNAL uses signal_date (t-1)
                spread = spreads.loc[signal_date, pair_name]
                z = zscores.loc[signal_date, pair_name]
            
            if pd.isna(z):
                continue
            
            should_exit = False
            exit_reason = None
            
            holding_days = t - entry['t']
            hl = half_lives[pair]
            
            # Dynamic max holding based on half-life
            if cfg.dynamic_max_holding:
                # Scale max holding by half-life: faster mean-reversion = shorter holding
                max_hold = int(np.ceil(getattr(cfg, 'max_holding_multiplier', 3.0) * hl))
                dyn_cap = getattr(cfg, 'max_dynamic_holding_days', 0)
                if dyn_cap > 0:
                    max_hold = min(max_hold, dyn_cap)
                max_hold = max(1, max_hold)
            else:
                max_hold = cfg.max_holding_days
            
            if holding_days >= max_hold:
                should_exit = True
                exit_reason = "max_holding"
            
            # Dynamic Z-score exit: exit early if Z is diverging after sufficient time
            if not should_exit and getattr(cfg, 'use_dynamic_z_exit', False):
                hl_ratio_threshold = getattr(cfg, 'dynamic_z_exit_hl_ratio', 1.5)
                z_threshold = getattr(cfg, 'dynamic_z_exit_threshold', 0.0)
                
                # Only check after hl_ratio_threshold * half_life days
                if holding_days >= hl_ratio_threshold * hl:
                    entry_z_abs = abs(entry.get('z', 0))
                    current_z_abs = abs(z)
                    
                    # Exit if current |Z| >= entry |Z| + threshold (Z is diverging)
                    if current_z_abs >= entry_z_abs + z_threshold:
                        should_exit = True
                        exit_reason = "z_diverging"
            
            # Slow convergence exit: exit if Z hasn't converged enough after threshold
            if not should_exit and getattr(cfg, 'use_slow_convergence_exit', False):
                sc_hl_ratio = getattr(cfg, 'slow_conv_hl_ratio', 1.5)
                sc_z_pct_threshold = getattr(cfg, 'slow_conv_z_pct', 0.50)  # Exit if >50% of entry Z remains
                
                # Only check after sc_hl_ratio * half_life days
                if holding_days >= sc_hl_ratio * hl:
                    entry_z_abs = abs(entry.get('z', 0))
                    current_z_abs = abs(z)
                    
                    # Calculate what % of entry Z remains
                    z_pct_remaining = current_z_abs / entry_z_abs if entry_z_abs > 0 else 0
                    
                    # Exit if Z hasn't converged enough (too much Z remaining)
                    if z_pct_remaining > sc_z_pct_threshold:
                        should_exit = True
                        exit_reason = "slow_convergence"
            
            if not should_exit and getattr(cfg, 'use_kalman_hedge', False) and getattr(cfg, 'kalman_zscore_regime', True):
                # For Kalman: use z-score based regime break
                # Since Kalman spread is mean-zero by construction, sign changes are too frequent
                # Instead, check if z-score moves significantly against entry direction
                kalman_regime_z = getattr(cfg, 'kalman_regime_zscore', 3.0)
                if direction == 1:  # Long spread (entered when z < -entry_z)
                    # Regime break if z goes way more negative (trend continues against us)
                    if z <= entry.get('z', 0) - kalman_regime_z:
                        should_exit = True
                        exit_reason = "regime_break"
                else:  # Short spread (entered when z > entry_z)
                    # Regime break if z goes way more positive (trend continues against us)
                    if z >= entry.get('z', 0) + kalman_regime_z:
                        should_exit = True
                        exit_reason = "regime_break"
            elif entry['spread'] * spread < 0:
                should_exit = True
                exit_reason = "regime_break"
            
            # Check convergence and stop-loss
            if not should_exit:
                # Get exit tolerance (Ch.8: exit if within tolerance band of exit threshold)
                exit_tol = getattr(cfg, 'exit_tolerance_sigma', 0.1)
                exit_thresh = getattr(cfg, 'exit_threshold_sigma', cfg.exit_zscore if hasattr(cfg, 'exit_zscore') else 0.0)
                
                # Adaptive stop-loss based on half-life
                # Faster mean-reversion (short HL) -> tighter stop (should recover quickly)
                # Slower mean-reversion (long HL) -> wider stop (needs more time)
                base_stop = getattr(cfg, 'stop_loss_sigma', 4.0)
                use_adaptive_stop = getattr(cfg, 'use_adaptive_stop_loss', False)
                
                if use_adaptive_stop:
                    # Scale stop with half-life: base + 0.5 * (HL/10 - 1)
                    # HL=5:  3.5 + 0.5*(-0.5) = 3.25 sigma
                    # HL=10: 3.5 + 0.5*(0)    = 3.5 sigma
                    # HL=20: 3.5 + 0.5*(1)    = 4.0 sigma
                    # HL=30: 3.5 + 0.5*(2)    = 4.5 sigma
                    hl_factor = (hl / 10.0) - 1.0
                    stop_sigma = base_stop + 0.5 * hl_factor
                    stop_sigma = max(3.0, min(stop_sigma, 5.0))  # Clamp [3.0, 5.0]
                else:
                    stop_sigma = base_stop
                
                if direction == 1:  # Long spread
                    # Exit if z >= -(exit_threshold + tolerance) i.e. within tolerance of mean
                    if z >= -(exit_thresh + exit_tol):
                        should_exit = True
                        exit_reason = "convergence"
                    else:
                        # Check stop loss with optional time-based tightening
                        use_time_stops = getattr(cfg, 'time_based_stops', True)  # Default True per Ch.8
                        if use_time_stops:
                            tightening_rate = getattr(cfg, 'stop_tightening_rate', 0.15)
                            time_stop, effective_stop = calculate_time_based_stop(
                                entry['z'], z, holding_days, hl,
                                stop_sigma, tightening_rate
                            )
                            if time_stop:
                                should_exit = True
                                exit_reason = "stop_loss_time"
                        else:
                            if z <= -stop_sigma:
                                should_exit = True
                                exit_reason = "stop_loss"
                else:  # Short spread
                    # Exit if z <= (exit_threshold + tolerance) i.e. within tolerance of mean
                    if z <= (exit_thresh + exit_tol):
                        should_exit = True
                        exit_reason = "convergence"
                    else:
                        # Check stop loss with optional time-based tightening
                        use_time_stops = getattr(cfg, 'time_based_stops', True)  # Default True per Ch.8
                        if use_time_stops:
                            tightening_rate = getattr(cfg, 'stop_tightening_rate', 0.15)
                            time_stop, effective_stop = calculate_time_based_stop(
                                entry['z'], z, holding_days, hl,
                                stop_sigma, tightening_rate
                            )
                            if time_stop:
                                should_exit = True
                                exit_reason = "stop_loss_time"
                        else:
                            if z >= stop_sigma:
                                should_exit = True
                                exit_reason = "stop_loss"
            
            if should_exit:
                leg_x, leg_y = pair
                px = prices.loc[current_date, leg_x]
                py = prices.loc[current_date, leg_y]
                
                pnl_x = entry['qty_x'] * (px - entry['px'])
                pnl_y = entry['qty_y'] * (py - entry['py'])
                pnl = pnl_x + pnl_y
                
                entry_notional = abs(entry['qty_x']) * entry['px'] + abs(entry['qty_y']) * entry['py']
                exit_notional = abs(entry['qty_x']) * px + abs(entry['qty_y']) * py
                cost = (entry_notional + exit_notional) * (cfg.transaction_cost_bps / 10000)
                pnl -= cost
                
                trades.append({
                    'pair': pair,
                    'leg_x': leg_x,
                    'leg_y': leg_y,
                    'sector': get_sector(leg_x),
                    'direction': 'LONG' if direction == 1 else 'SHORT',
                    'entry_date': entry['date'],
                    'exit_date': current_date,
                    'holding_days': holding_days,
                    'entry_z': entry['z'],
                    'exit_z': z,
                    'hedge_ratio': entry['hr'],
                    'half_life': hl,
                    'pnl': pnl,
                    'exit_reason': exit_reason,
                    'capital_at_entry': entry.get('capital', current_capital),
                    'qty_x': entry['qty_x'],
                    'qty_y': entry['qty_y'],
                    'entry_px': entry['px'],
                    'entry_py': entry['py'],
                })
                
                # Update capital for compounding
                if cfg.compounding:
                    current_capital += pnl
                
                position_state[pair] = 0
                entry_data[pair] = {}
        
        # Check entries
        n_active = sum(1 for p in pairs if position_state[p] != 0)
        
        # Handle unlimited positions (max_positions == 0)
        max_pos_limit = cfg.max_positions if cfg.max_positions > 0 else len(pairs)
        
        # VIX regime filter - skip new entries in high volatility regime
        skip_entries = False
        if getattr(cfg, 'use_vix_filter', False):
            # Try to get VIX from prices (if included) or skip check
            if 'VIX' in prices.columns or '^VIX' in prices.columns:
                vix_col = 'VIX' if 'VIX' in prices.columns else '^VIX'
                vix_data = prices[vix_col]
                vix_info = check_vix_regime(
                    vix_data, current_date,
                    getattr(cfg, 'vix_threshold', 30.0),
                    getattr(cfg, 'vix_lookback_days', 5)
                )
                skip_entries = vix_info['is_high_vol']
        
        if n_active < max_pos_limit and not skip_entries:
            for pair in pairs:
                if position_state[pair] != 0:
                    continue
                if n_active >= max_pos_limit:
                    break
                
                pair_name = pair_names[pair]
                # FIX LOOK-AHEAD BIAS: Signal from signal_date (t-1), execution at current_date (t)
                z = zscores.loc[signal_date, pair_name]      # Signal from yesterday
                spread = spreads.loc[signal_date, pair_name]  # Signal from yesterday
                
                if pd.isna(z):
                    continue
                
                leg_x, leg_y = pair
                # EXECUTION prices from current_date (today)
                px = prices.loc[current_date, leg_x]
                py = prices.loc[current_date, leg_y]
                hr = current_hr[pair]
                
                # Use dynamic capital if compounding, else use config
                if cfg.compounding:
                    # Recalculate available capital based on current equity
                    # Use at least 5 as divisor to avoid over-concentration
                    max_pos = cfg.max_positions if cfg.max_positions > 0 else max(5, len(pairs))
                    position_capital = (current_capital * cfg.leverage) / max(1, max_pos)
                    
                    # Apply max capital per trade limit if set
                    if cfg.max_capital_per_trade > 0:
                        position_capital = min(position_capital, cfg.max_capital_per_trade)
                else:
                    position_capital = cfg.capital_per_pair * cfg.leverage
                
                # Volatility-adjusted position sizing
                if getattr(cfg, 'use_vol_sizing', False):
                    # Calculate recent spread volatility using diff() not pct_change()
                    # (Bug fix: spread oscillates around 0, pct_change gives extreme values)
                    spread_series = spreads[pair_name]
                    spread_changes = spread_series.diff().dropna()
                    if len(spread_changes) >= 20:
                        spread_vol = spread_changes.iloc[-20:].std()
                        position_capital = calculate_volatility_adjusted_size(
                            position_capital,
                            spread_vol,
                            getattr(cfg, 'target_daily_vol', 0.02),
                            getattr(cfg, 'vol_size_min', 0.25),
                            getattr(cfg, 'vol_size_max', 2.0),
                        )
                
                notional_x = position_capital / (1 + abs(hr))
                notional_y = abs(hr) * notional_x
                
                # Get entry threshold (backwards compatible)
                entry_thresh = getattr(cfg, 'entry_threshold_sigma', cfg.entry_zscore if hasattr(cfg, 'entry_zscore') else 0.75)
                
                if z <= -entry_thresh:
                    position_state[pair] = 1
                    entry_data[pair] = {
                        't': t,
                        'date': current_date,  # Execution date
                        'signal_date': signal_date,  # Signal date for reference
                        'z': z,  # Z-score from signal_date
                        'spread': spread,  # Spread from signal_date
                        'px': px,  # Execution price (current_date)
                        'py': py,  # Execution price (current_date)
                        'hr': hr,
                        'qty_x': notional_x / px,
                        'qty_y': -notional_y / py,
                        'capital': current_capital,
                        # QMA Level 2: Save mu and sigma at SIGNAL time for fixed exit z-score
                        'mu_entry': rolling_mean.loc[signal_date, pair_name],
                        'sigma_entry': rolling_std.loc[signal_date, pair_name],
                    }
                    n_active += 1
                elif z >= entry_thresh:
                    position_state[pair] = -1
                    entry_data[pair] = {
                        't': t,
                        'date': current_date,  # Execution date
                        'signal_date': signal_date,  # Signal date for reference
                        'z': z,  # Z-score from signal_date
                        'spread': spread,  # Spread from signal_date
                        'px': px,  # Execution price (current_date)
                        'py': py,  # Execution price (current_date)
                        'hr': hr,
                        'qty_x': -notional_x / px,
                        'qty_y': notional_y / py,
                        'capital': current_capital,
                        # QMA Level 2: Save mu and sigma at SIGNAL time for fixed exit z-score
                        'mu_entry': rolling_mean.loc[signal_date, pair_name],
                        'sigma_entry': rolling_std.loc[signal_date, pair_name],
                    }
                    n_active += 1
    
    # Close remaining positions
    last_date = dates[-1]
    for pair in pairs:
        if position_state[pair] == 0:
            continue
        
        direction = position_state[pair]
        entry = entry_data[pair]
        leg_x, leg_y = pair
        pair_name = pair_names[pair]
        
        px = prices.loc[last_date, leg_x]
        py = prices.loc[last_date, leg_y]
        
        # QMA Level 2: Use fixed exit params (hr_entry, mu_entry, sigma_entry) for period_end z-score
        if getattr(cfg, 'use_fixed_exit_params', True):
            log_x = np.log(px)
            log_y = np.log(py)
            hr_entry = entry.get('hr', current_hr[pair])
            spread = log_x - hr_entry * log_y
            
            mu_entry = entry.get('mu_entry', rolling_mean.loc[last_date, pair_name])
            sigma_entry = entry.get('sigma_entry', rolling_std.loc[last_date, pair_name])
            if sigma_entry > 0:
                z = (spread - mu_entry) / sigma_entry
            else:
                z = 0.0
        else:
            z_val = zscores.loc[last_date, pair_name]
            z = z_val if not pd.isna(z_val) else 0
        
        pnl_x = entry['qty_x'] * (px - entry['px'])
        pnl_y = entry['qty_y'] * (py - entry['py'])
        pnl = pnl_x + pnl_y
        
        entry_notional = abs(entry['qty_x']) * entry['px'] + abs(entry['qty_y']) * entry['py']
        exit_notional = abs(entry['qty_x']) * px + abs(entry['qty_y']) * py
        cost = (entry_notional + exit_notional) * (cfg.transaction_cost_bps / 10000)
        pnl -= cost

        # Ensure holding_days is at least 1 to avoid edge cases
        holding_days = max(1, len(prices) - 1 - entry['t'])
        
        trades.append({
            'pair': pair,
            'leg_x': leg_x,
            'leg_y': leg_y,
            'sector': get_sector(leg_x),
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'entry_date': entry['date'],
            'exit_date': last_date,
            'holding_days': holding_days,
            'entry_z': entry['z'],
            'exit_z': z,
            'hedge_ratio': entry['hr'],
            'half_life': half_lives[pair],
            'pnl': pnl,
            'exit_reason': 'period_end',
            'capital_at_entry': entry.get('capital', current_capital),
            'qty_x': entry['qty_x'],
            'qty_y': entry['qty_y'],
            'entry_px': entry['px'],
            'entry_py': entry['py'],
        })
        
        # Update capital for compounding
        if cfg.compounding:
            current_capital += pnl
    
    return trades, current_capital


# =============================================================================
# BLACKLIST MANAGEMENT
# =============================================================================

class PairBlacklist:
    """Manages pairs that should be excluded due to poor performance."""
    
    def __init__(self, threshold: float = 0.30, min_trades: int = 3):
        self.threshold = threshold
        self.min_trades = min_trades
        self.blacklist = set()
        self.pair_stats = defaultdict(lambda: {'trades': 0, 'stop_losses': 0})
    
    def update(self, trades: List[Dict]) -> None:
        """Update blacklist based on new trades."""
        for trade in trades:
            pair = trade['pair']
            self.pair_stats[pair]['trades'] += 1
            if trade['exit_reason'] == 'stop_loss':
                self.pair_stats[pair]['stop_losses'] += 1
        
        for pair, pair_stats in self.pair_stats.items():
            if pair_stats['trades'] >= self.min_trades:
                sl_rate = pair_stats['stop_losses'] / pair_stats['trades']
                if sl_rate > self.threshold and pair not in self.blacklist:
                    logger.info(f"Blacklisting {pair}: {sl_rate:.1%} stop-loss rate")
                    self.blacklist.add(pair)


# =============================================================================
# WALK-FORWARD BACKTEST
# =============================================================================

def run_walkforward_backtest(
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    start_year: int = 2010,
    end_year: int = 2024,
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Run walk-forward backtest across multiple years.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Full price data
    cfg : BacktestConfig
        Configuration
    start_year : int
        First trading year
    end_year : int
        Last trading year
        
    Returns
    -------
    tuple
        (all_trades, yearly_summary)
    """
    all_trades = []
    year_results = []
    
    blacklist = PairBlacklist(cfg.blacklist_stoploss_rate, cfg.blacklist_min_trades)
    
    # Track capital for compounding
    current_capital = cfg.initial_capital
    
    for trading_year in range(start_year, end_year + 1):
        formation_year = trading_year - 1
        
        logger.info("=" * 60)
        logger.info(f"Year {trading_year}: Formation {formation_year}")
        if cfg.compounding:
            logger.info(f"Current Capital: ${current_capital:,.2f}")
        logger.info("=" * 60)
        
        # Formation period
        formation_start = pd.Timestamp(f'{formation_year}-01-01')
        formation_end = pd.Timestamp(f'{formation_year}-12-31')
        
        mask = (prices.index >= formation_start) & (prices.index <= formation_end)
        # Allow small gaps in formation data: keep columns with <=20% missing, then fill
        formation_prices = prices.loc[mask]
        if formation_prices.isna().values.any():
            missing = formation_prices.isna().mean()
            cols = missing[missing <= 0.20].index
            formation_prices = formation_prices[cols].ffill().bfill()
        else:
            formation_prices = formation_prices.copy()
        
        if len(formation_prices) < cfg.formation_days * 0.8:
            logger.warning(f"Insufficient formation data for {trading_year}")
            continue
        
        # Select pairs
        t0 = time.time()
        pairs, hedge_ratios, half_lives, formation_stats = select_pairs(
            formation_prices, cfg, blacklist.blacklist
        )
        logger.info(f"Pair selection: {time.time() - t0:.2f}s")
        
        if not pairs:
            logger.warning(f"No pairs selected for {trading_year}")
            continue
        
        # Trading period
        trading_start = pd.Timestamp(f'{trading_year}-01-01')
        trading_end = pd.Timestamp(f'{trading_year}-12-31')
        
        mask = (prices.index >= trading_start) & (prices.index <= trading_end)
        trading_prices = prices.loc[mask]
        
        # Keep valid tickers
        valid_tickers = set()
        for pair in pairs:
            valid_tickers.add(pair[0])
            valid_tickers.add(pair[1])
        valid_tickers = [t for t in valid_tickers if t in trading_prices.columns]
        trading_prices = trading_prices[valid_tickers].dropna(axis=1, how='any')
        
        pairs = [p for p in pairs if p[0] in trading_prices.columns and p[1] in trading_prices.columns]
        
        if not pairs:
            continue
        
        # Check minimum pairs for risk diversification
        if len(pairs) < cfg.min_pairs_for_trading:
            logger.warning(f"Only {len(pairs)} pairs selected (min: {cfg.min_pairs_for_trading}), skipping {trading_year}")
            continue
        
        # Run simulation with capital tracking
        trades, current_capital = run_trading_simulation(
            trading_prices, pairs, hedge_ratios, half_lives, cfg, current_capital
        )
        
        blacklist.update(trades)
        
        # Calculate stats
        n_trades = len(trades)
        n_wins = sum(1 for t in trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in trades)
        
        exit_reasons = defaultdict(int)
        for t in trades:
            exit_reasons[t['exit_reason']] += 1
        
        logger.info(f"Pairs: {len(pairs)}, Trades: {n_trades}, PnL: ${total_pnl:.2f}")
        logger.info(f"Exit reasons: {dict(exit_reasons)}")
        
        year_results.append({
            'trading_year': trading_year,
            'pairs_selected': len(pairs),
            'total_trades': n_trades,
            'winning_trades': n_wins,
            'win_rate': n_wins / n_trades * 100 if n_trades > 0 else 0,
            'total_pnl': total_pnl,
            'ending_capital': current_capital if cfg.compounding else None,
            **{f'{k}_exits': v for k, v in exit_reasons.items()},
        })
        
        all_trades.extend(trades)
    
    summary_df = pd.DataFrame(year_results)
    return all_trades, summary_df
