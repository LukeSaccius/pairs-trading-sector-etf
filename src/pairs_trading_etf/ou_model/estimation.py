"""Ornstein-Uhlenbeck model parameter estimation.

Implements ML/OLS estimation for OU process parameters from spread time series.
The OU process models mean-reverting spreads as:
    dS_t = θ(μ - S_t)dt + σ dW_t

Parameters:
    θ (theta): Mean reversion speed (>0 for mean reversion)
    μ (mu): Long-run equilibrium level
    σ (sigma): Volatility of the process
    half_life: Time to revert halfway to mean = ln(2)/θ

References:
- Uhlenbeck, G. & Ornstein, L. (1930). "On the Theory of Brownian Motion"
- Avellaneda, M. & Lee, J.H. (2010). "Statistical Arbitrage in the US Equities Market"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OUParameters:
    """Container for estimated OU process parameters."""
    
    theta: float          # Mean reversion speed
    mu: float             # Long-run mean
    sigma: float          # Process volatility
    half_life: float      # ln(2) / theta in same time units as data
    r_squared: float      # Goodness of fit from AR(1) regression
    residual_std: float   # Standard deviation of regression residuals
    n_observations: int   # Number of observations used
    
    # Standard errors for hypothesis testing
    theta_se: float | None = None
    mu_se: float | None = None
    
    # Statistical significance
    theta_tstat: float | None = None
    theta_pvalue: float | None = None
    
    def is_mean_reverting(self, significance: float = 0.05) -> bool:
        """Check if theta is significantly positive (mean-reverting)."""
        if self.theta_pvalue is not None:
            return self.theta > 0 and self.theta_pvalue < significance
        return self.theta > 0
    
    def as_dict(self) -> Mapping[str, float | int | None]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "theta": self.theta,
            "mu": self.mu,
            "sigma": self.sigma,
            "half_life": self.half_life,
            "r_squared": self.r_squared,
            "residual_std": self.residual_std,
            "n_observations": self.n_observations,
            "theta_se": self.theta_se,
            "theta_tstat": self.theta_tstat,
            "theta_pvalue": self.theta_pvalue,
        }


def estimate_ou_parameters(
    spread: pd.Series,
    dt: float = 1.0,
    min_observations: int = 30,
) -> OUParameters:
    """Estimate OU process parameters using AR(1) regression.
    
    Uses the discrete-time AR(1) representation:
        S_t = α + β * S_{t-1} + ε_t
    
    Where the continuous OU parameters are recovered as:
        θ = -ln(β) / dt
        μ = α / (1 - β)
        σ = std(ε) * sqrt(-2*ln(β) / (dt*(1-β²)))
    
    Parameters
    ----------
    spread : pd.Series
        The spread time series (e.g., log price spread between two assets).
    dt : float, default=1.0
        Time step size. For daily data, dt=1 gives half-life in days.
    min_observations : int, default=30
        Minimum required observations for estimation.
        
    Returns
    -------
    OUParameters
        Estimated parameters with standard errors and goodness-of-fit metrics.
        
    Raises
    ------
    ValueError
        If insufficient observations or non-stationary process detected.
        
    Examples
    --------
    >>> spread = pd.Series([...])  # Your spread series
    >>> params = estimate_ou_parameters(spread)
    >>> print(f"Half-life: {params.half_life:.1f} days")
    >>> print(f"Mean-reverting: {params.is_mean_reverting()}")
    """
    # Clean the spread series
    spread = spread.dropna()
    n = len(spread)
    
    if n < min_observations:
        raise ValueError(
            f"Insufficient observations: {n} < {min_observations} required"
        )
    
    # Prepare AR(1) regression: S_t = α + β * S_{t-1} + ε
    y = spread.iloc[1:].values  # S_t
    x = spread.iloc[:-1].values  # S_{t-1}
    
    # Add intercept term
    X = np.column_stack([np.ones(len(x)), x])
    
    # OLS estimation
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y
    
    alpha, beta = beta_hat[0], beta_hat[1]
    
    # Residuals and fit statistics
    residuals = y - X @ beta_hat
    residual_std = np.std(residuals, ddof=2)  # Account for 2 estimated params
    
    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Check for stationarity: β must be in (0, 1) for mean reversion
    if beta <= 0:
        logger.warning("β ≤ 0 detected: Process is not mean-reverting (explosive)")
        # Return parameters but flag as non-mean-reverting
        return OUParameters(
            theta=np.nan,
            mu=spread.mean(),  # Use sample mean as fallback
            sigma=spread.std(),
            half_life=np.inf,
            r_squared=r_squared,
            residual_std=residual_std,
            n_observations=n,
            theta_se=None,
            theta_tstat=None,
            theta_pvalue=1.0,  # Not significant
        )
    
    if beta >= 1:
        logger.warning("β ≥ 1 detected: Process has unit root (non-stationary)")
        return OUParameters(
            theta=0.0,
            mu=np.nan,  # Long-run mean undefined for unit root
            sigma=residual_std,
            half_life=np.inf,
            r_squared=r_squared,
            residual_std=residual_std,
            n_observations=n,
            theta_se=None,
            theta_tstat=None,
            theta_pvalue=1.0,
        )
    
    # Convert to continuous OU parameters
    theta = -np.log(beta) / dt
    mu = alpha / (1 - beta)
    
    # OU process volatility (continuous-time)
    # σ_OU = σ_ε * sqrt(-2*ln(β) / (dt*(1-β²)))
    beta_sq = beta ** 2
    if beta_sq < 1:
        sigma = residual_std * np.sqrt(-2 * np.log(beta) / (dt * (1 - beta_sq)))
    else:
        sigma = residual_std
    
    # Half-life calculation
    half_life = np.log(2) / theta
    
    # Standard errors via delta method
    # SE(β) from OLS
    mse = ss_res / (n - 3)  # degrees of freedom
    var_beta = mse * XtX_inv[1, 1]
    se_beta = np.sqrt(var_beta)
    
    # SE(θ) via delta method: θ = -ln(β)/dt, so dθ/dβ = -1/(β*dt)
    theta_se = se_beta / (beta * dt)
    
    # t-statistic and p-value for testing θ > 0 (mean reversion)
    # Under H0: β = 1 (unit root), test if β < 1
    theta_tstat = (1 - beta) / se_beta  # Testing β < 1
    # One-sided p-value for mean reversion
    theta_pvalue = 1 - stats.t.cdf(theta_tstat, df=n - 3)
    
    logger.debug(
        "OU estimation: θ=%.4f, μ=%.4f, σ=%.4f, HL=%.1f, R²=%.3f",
        theta, mu, sigma, half_life, r_squared
    )
    
    return OUParameters(
        theta=float(theta),
        mu=float(mu),
        sigma=float(sigma),
        half_life=float(half_life),
        r_squared=float(r_squared),
        residual_std=float(residual_std),
        n_observations=n,
        theta_se=float(theta_se),
        theta_tstat=float(theta_tstat),
        theta_pvalue=float(theta_pvalue),
    )


def estimate_ou_from_prices(
    price_x: pd.Series,
    price_y: pd.Series,
    hedge_ratio: float,
    use_log: bool = True,
    dt: float = 1.0,
    min_observations: int = 30,
) -> OUParameters:
    """Estimate OU parameters from two price series and a hedge ratio.
    
    Constructs the spread as:
        spread = log(price_x) - hedge_ratio * log(price_y)  [if use_log]
        spread = price_x - hedge_ratio * price_y            [otherwise]
    
    Parameters
    ----------
    price_x : pd.Series
        Price series for the first asset (long leg).
    price_y : pd.Series
        Price series for the second asset (short leg).
    hedge_ratio : float
        The hedge ratio β from cointegration regression.
    use_log : bool, default=True
        Whether to use log prices for spread construction.
    dt : float, default=1.0
        Time step size.
    min_observations : int, default=30
        Minimum required observations.
        
    Returns
    -------
    OUParameters
        Estimated OU process parameters.
    """
    # Align and clean prices
    df = pd.concat([price_x, price_y], axis=1, join="inner").dropna()
    if df.shape[0] < min_observations:
        raise ValueError(f"Insufficient overlapping data: {df.shape[0]} < {min_observations}")
    
    px, py = df.iloc[:, 0], df.iloc[:, 1]
    
    if use_log:
        px = np.log(px.where(px > 0)).replace([np.inf, -np.inf], np.nan)
        py = np.log(py.where(py > 0)).replace([np.inf, -np.inf], np.nan)
        # Re-align after log transform
        df = pd.concat([px, py], axis=1, join="inner").dropna()
        px, py = df.iloc[:, 0], df.iloc[:, 1]
    
    spread = px - hedge_ratio * py
    spread.name = "spread"
    
    return estimate_ou_parameters(spread, dt=dt, min_observations=min_observations)


def rolling_ou_estimation(
    spread: pd.Series,
    window: int = 252,
    min_periods: int | None = None,
    dt: float = 1.0,
) -> pd.DataFrame:
    """Estimate OU parameters over rolling windows.
    
    Useful for tracking regime changes and parameter stability.
    
    Parameters
    ----------
    spread : pd.Series
        The spread time series.
    window : int, default=252
        Rolling window size (e.g., 252 for 1 year of daily data).
    min_periods : int | None, default=None
        Minimum periods required. If None, uses window size.
    dt : float, default=1.0
        Time step size.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: theta, mu, sigma, half_life, r_squared
        indexed by the end date of each estimation window.
    """
    if min_periods is None:
        min_periods = window
    
    results = []
    spread = spread.dropna()
    
    for i in range(min_periods, len(spread) + 1):
        window_start = max(0, i - window)
        window_spread = spread.iloc[window_start:i]
        
        try:
            params = estimate_ou_parameters(window_spread, dt=dt, min_observations=30)
            results.append({
                "date": spread.index[i - 1],
                "theta": params.theta,
                "mu": params.mu,
                "sigma": params.sigma,
                "half_life": params.half_life,
                "r_squared": params.r_squared,
                "theta_pvalue": params.theta_pvalue,
            })
        except ValueError:
            # Not enough data in window
            results.append({
                "date": spread.index[i - 1],
                "theta": np.nan,
                "mu": np.nan,
                "sigma": np.nan,
                "half_life": np.nan,
                "r_squared": np.nan,
                "theta_pvalue": np.nan,
            })
    
    df = pd.DataFrame(results).set_index("date")
    return df


def estimate_ou_with_kalman(
    spread: pd.Series,
    Q: float = 1e-5,
    R: float = 1e-3,
    initial_theta: float = 0.1,
    initial_mu: float | None = None,
    dt: float = 1.0,
) -> tuple[pd.DataFrame, OUParameters]:
    """Estimate time-varying OU parameters using Kalman filter.
    
    Models the spread as an OU process with time-varying parameters.
    State vector: [mu_t, theta_t]
    Measurement: S_t
    
    Parameters
    ----------
    spread : pd.Series
        The spread time series.
    Q : float, default=1e-5
        Process noise variance for state evolution.
    R : float, default=1e-3
        Measurement noise variance.
    initial_theta : float, default=0.1
        Initial guess for mean reversion speed.
    initial_mu : float | None, default=None
        Initial guess for equilibrium level. If None, uses first spread value.
    dt : float, default=1.0
        Time step size.
        
    Returns
    -------
    tuple[pd.DataFrame, OUParameters]
        - DataFrame with time-varying parameter estimates
        - Final OUParameters from last estimation window
    """
    spread = spread.dropna()
    n = len(spread)
    
    if n < 30:
        raise ValueError(f"Need at least 30 observations, got {n}")
    
    # Initialize state and covariance
    if initial_mu is None:
        initial_mu = spread.iloc[0]
    
    state = np.array([initial_mu, initial_theta])  # [mu, theta]
    P = np.eye(2) * 0.1  # Initial state covariance
    
    # Process and measurement noise
    Q_matrix = np.eye(2) * Q
    R_matrix = R
    
    # Storage for filtered estimates
    filtered_states = np.zeros((n, 2))
    filtered_vars = np.zeros((n, 2))
    
    for t in range(n):
        # Prediction step (assume parameters evolve slowly - random walk)
        state_pred = state
        P_pred = P + Q_matrix
        
        # Get current spread value
        s_t = spread.iloc[t]
        
        # Expected spread given current state
        # Under OU: E[S_t | S_{t-1}] = mu + (S_{t-1} - mu) * exp(-theta * dt)
        if t > 0:
            s_prev = spread.iloc[t - 1]
            mu_t, theta_t = state_pred
            
            # Linearized observation model
            # H = d(expected_spread)/d(state)
            exp_neg_theta_dt = np.exp(-max(theta_t, 1e-10) * dt)
            H = np.array([
                1 - exp_neg_theta_dt,  # d/d_mu
                (s_prev - mu_t) * dt * exp_neg_theta_dt  # d/d_theta
            ])
            
            # Innovation
            expected_s = mu_t + (s_prev - mu_t) * exp_neg_theta_dt
            innovation = s_t - expected_s
            
            # Kalman gain
            S = H @ P_pred @ H.T + R_matrix
            K = P_pred @ H.T / S
            
            # Update step
            state = state_pred + K * innovation
            P = (np.eye(2) - np.outer(K, H)) @ P_pred
        else:
            state = state_pred
            P = P_pred
        
        # Ensure theta stays positive
        state[1] = max(state[1], 1e-6)
        
        filtered_states[t] = state
        filtered_vars[t] = np.diag(P)
    
    # Build results DataFrame
    results = pd.DataFrame({
        "date": spread.index,
        "mu": filtered_states[:, 0],
        "theta": filtered_states[:, 1],
        "mu_var": filtered_vars[:, 0],
        "theta_var": filtered_vars[:, 1],
    }).set_index("date")
    
    # Calculate half-life and sigma from final estimates
    final_theta = filtered_states[-1, 1]
    final_mu = filtered_states[-1, 0]
    half_life = np.log(2) / final_theta if final_theta > 0 else np.inf
    
    # Estimate sigma from residuals
    residuals = []
    for t in range(1, n):
        mu_t, theta_t = filtered_states[t - 1]
        exp_decay = np.exp(-theta_t * dt)
        expected = mu_t + (spread.iloc[t - 1] - mu_t) * exp_decay
        residuals.append(spread.iloc[t] - expected)
    
    sigma = np.std(residuals) * np.sqrt(2 * final_theta)
    
    final_params = OUParameters(
        theta=float(final_theta),
        mu=float(final_mu),
        sigma=float(sigma) if not np.isnan(sigma) else 0.0,
        half_life=float(half_life),
        r_squared=np.nan,  # Not directly applicable to Kalman filter
        residual_std=float(np.std(residuals)),
        n_observations=n,
    )
    
    return results, final_params
