"""Half-life estimation for mean-reverting processes.

This module provides functions for estimating the half-life of a 
mean-reverting spread, which is critical for pairs trading viability.

Half-life represents the expected time for the spread to revert halfway 
to its long-run mean. For trading purposes:
- HL < 5 days: Too fast (likely noise, execution risk)
- HL 5-120 days: Tradeable range
- HL > 120 days: Too slow (capital locked too long)

Mathematical Background:
------------------------
For an AR(1) process:
    spread_t = c + φ * spread_{t-1} + ε_t

The half-life in discrete time is:
    HL = -ln(2) / ln(φ)

This is derived from the decay factor φ^k where k is the number of periods.
After k periods: deviation = φ^k * initial_deviation
At half-life: φ^HL = 0.5 → HL = ln(0.5) / ln(φ) = -ln(2) / ln(φ)

For mean reversion, we need 0 < φ < 1:
- φ close to 1: slow reversion (large HL)
- φ close to 0: fast reversion (small HL)
- φ >= 1: non-stationary (unit root or explosive)
- φ <= 0: oscillatory behavior

References:
-----------
- Uhlenbeck, G. & Ornstein, L. (1930). "On the Theory of Brownian Motion"
- Avellaneda, M. & Lee, J.H. (2010). "Statistical Arbitrage"

Bug Fix History:
---------------
2025-12-03: Fixed critical bug in half-life calculation
  - OLD (WRONG): HL = -ln(2) / b where b is slope from OLS without intercept
  - NEW (CORRECT): HL = -ln(2) / ln(1+b) where b is slope from OLS with intercept
  - The bug caused half-lives to be 100-1000x larger than actual values
  - Example: EWA-EWC showed HL=∞ before fix, HL=24 days after fix
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def estimate_half_life(spread: pd.Series, min_observations: int = 30) -> float | None:
    """Estimate half-life of mean reversion using AR(1) regression.
    
    Uses the error-correction model representation:
        delta_spread_t = a + b * spread_{t-1} + error_t
    
    Where:
        - a is the intercept
        - b is the speed of adjustment (should be negative for mean reversion)
        - phi = 1 + b is the AR(1) coefficient
        - Half-life = -ln(2) / ln(phi)
    
    Parameters
    ----------
    spread : pd.Series
        The spread time series to analyze. Should be stationary for
        meaningful half-life estimation.
    min_observations : int, default=30
        Minimum number of observations required for reliable estimation.
        
    Returns
    -------
    float | None
        Estimated half-life in the same time units as the input data
        (e.g., days for daily data). Returns None if:
        - Insufficient observations
        - Process is not mean-reverting (b >= 0)
        - AR(1) coefficient outside valid range (phi <= 0 or phi >= 1)
        
    Examples
    --------
    >>> spread = pd.Series([...])  # Daily spread series
    >>> hl = estimate_half_life(spread)
    >>> if hl is not None and 5 <= hl <= 120:
    ...     print(f"Tradeable half-life: {hl:.1f} days")
    
    Notes
    -----
    The estimation uses OLS with an intercept term, which is important
    because the spread may not be centered at zero. Omitting the intercept
    leads to biased estimates.
    
    For small samples (<100 observations), the OLS estimator has downward
    bias, causing half-life estimates to be slightly shorter than true values.
    This is acceptable for screening purposes but should be noted for
    precise trading decisions.
    """
    spread = spread.dropna()
    if spread.shape[0] < min_observations:
        logger.debug(f"Insufficient observations: {spread.shape[0]} < {min_observations}")
        return None

    # Prepare lagged spread and first difference
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    
    # Align the series
    aligned = pd.concat([lagged, delta], axis=1, join="inner").dropna()
    if aligned.empty or len(aligned) < min_observations:
        logger.debug(f"Insufficient aligned observations: {len(aligned)}")
        return None

    # OLS regression: delta = a + b * spread_lag + error
    x = aligned.iloc[:, 0].values  # spread_lag (t-1)
    y = aligned.iloc[:, 1].values  # delta (t - t-1)
    
    # CRITICAL: Include intercept in regression
    # Without intercept, estimates are biased for non-centered spreads
    X = np.column_stack([np.ones(len(x)), x])
    
    try:
        # Solve normal equations: β = (X'X)^(-1) X'y
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        a = beta[0]  # Intercept
        b = beta[1]  # Slope (speed of adjustment)
        
        logger.debug(f"OLS estimates: intercept={a:.6f}, slope={b:.6f}")
        
        # For mean reversion, b should be negative
        if b >= 0:
            logger.debug(f"Non-mean-reverting: slope b={b:.6f} >= 0")
            return None
        
        # AR(1) coefficient: φ = 1 + b
        # From the relationship: spread_t = c + φ * spread_{t-1} + ε
        # Taking differences: Δspread = c(1-φ) + (φ-1) * spread_{t-1} = a + b * spread_{t-1}
        # Therefore: b = φ - 1, so φ = 1 + b
        phi = 1 + b
        
        # For valid half-life, need 0 < φ < 1
        if phi <= 0:
            logger.debug(f"Invalid AR(1) coefficient: φ={phi:.6f} <= 0 (oscillatory)")
            return None
        if phi >= 1:
            logger.debug(f"Non-stationary: φ={phi:.6f} >= 1 (unit root)")
            return None
        
        # Half-life in discrete time
        # From φ^HL = 0.5, solve for HL: HL = ln(0.5) / ln(φ) = -ln(2) / ln(φ)
        half_life = -np.log(2) / np.log(phi)
        
        logger.debug(f"Half-life estimate: φ={phi:.6f}, HL={half_life:.2f}")
        
        return float(half_life)
        
    except Exception as e:
        logger.warning(f"Half-life estimation failed: {e}")
        return None


def estimate_half_life_with_stats(
    spread: pd.Series, 
    min_observations: int = 30
) -> tuple[float | None, dict]:
    """Estimate half-life with additional statistics.
    
    Extended version that returns regression statistics for diagnostics.
    
    Parameters
    ----------
    spread : pd.Series
        The spread time series to analyze.
    min_observations : int, default=30
        Minimum observations required.
        
    Returns
    -------
    tuple[float | None, dict]
        - Half-life estimate (or None if invalid)
        - Dictionary with additional statistics:
            - intercept: OLS intercept
            - slope: OLS slope (b)
            - phi: AR(1) coefficient (1 + b)
            - r_squared: Regression R²
            - n_obs: Number of observations used
            - residual_std: Standard deviation of residuals
    """
    stats = {
        "intercept": None,
        "slope": None,
        "phi": None,
        "r_squared": None,
        "n_obs": None,
        "residual_std": None,
    }
    
    spread = spread.dropna()
    if spread.shape[0] < min_observations:
        return None, stats

    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    aligned = pd.concat([lagged, delta], axis=1, join="inner").dropna()
    
    if aligned.empty or len(aligned) < min_observations:
        return None, stats

    x = aligned.iloc[:, 0].values
    y = aligned.iloc[:, 1].values
    n = len(x)
    
    X = np.column_stack([np.ones(n), x])
    
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        a, b = beta[0], beta[1]
        
        # Compute residuals and R²
        y_pred = X @ beta
        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        stats["intercept"] = float(a)
        stats["slope"] = float(b)
        stats["n_obs"] = n
        stats["r_squared"] = float(r_squared)
        stats["residual_std"] = float(np.std(residuals, ddof=2))
        
        if b >= 0:
            stats["phi"] = float(1 + b)
            return None, stats
        
        phi = 1 + b
        stats["phi"] = float(phi)
        
        if phi <= 0 or phi >= 1:
            return None, stats
        
        half_life = -np.log(2) / np.log(phi)
        
        return float(half_life), stats
        
    except Exception:
        return None, stats


def validate_half_life_for_trading(
    half_life: float | None,
    min_hl: float = 5.0,
    max_hl: float = 120.0,
) -> bool:
    """Check if half-life falls within tradeable range.
    
    Parameters
    ----------
    half_life : float | None
        Estimated half-life in trading days.
    min_hl : float, default=5.0
        Minimum acceptable half-life (days). Below this is too noisy.
    max_hl : float, default=120.0
        Maximum acceptable half-life (days). Above this ties up capital.
        
    Returns
    -------
    bool
        True if half-life is within tradeable range.
    """
    if half_life is None:
        return False
    if not np.isfinite(half_life):
        return False
    return min_hl <= half_life <= max_hl
