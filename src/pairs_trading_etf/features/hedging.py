"""Rolling hedge ratio estimation for pairs trading.

Implements dynamic hedge ratio updates using rolling OLS regression.
This helps adapt to regime changes in the cointegration relationship.

References:
- Palomar, D. (2025). "Pairs Trading", HKUST
- Chan, E. (2013). "Algorithmic Trading", Chapter 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


@dataclass
class HedgeRatioEstimate:
    """A single hedge ratio estimate at a point in time."""
    date: pd.Timestamp
    hedge_ratio: float
    intercept: float
    r_squared: float
    spread_std: float


@dataclass
class RollingHedgeConfig:
    """Configuration for rolling hedge ratio estimation."""
    window_months: int = 12       # Lookback window in months
    update_frequency: str = 'M'   # 'M' = monthly, 'W' = weekly, 'D' = daily
    min_observations: int = 60    # Minimum observations required
    use_log_prices: bool = True   # Estimate on log prices (recommended)


def estimate_hedge_ratio_ols(
    price_x: pd.Series,
    price_y: pd.Series,
    use_log: bool = True,
) -> tuple[float, float, float]:
    """Estimate hedge ratio using OLS regression.
    
    Model: log(P_x) = alpha + beta * log(P_y) + epsilon
    
    Parameters
    ----------
    price_x, price_y : pd.Series
        Price series for the two legs.
    use_log : bool
        If True, use log prices for estimation.
    
    Returns
    -------
    tuple[float, float, float]
        (hedge_ratio/beta, intercept/alpha, r_squared)
    """
    # Align series
    df = pd.concat([price_x, price_y], axis=1, join='inner').dropna()
    if df.shape[0] < 30:
        raise ValueError("Need at least 30 observations for hedge ratio estimation")
    
    x_vals = df.iloc[:, 0]
    y_vals = df.iloc[:, 1]
    
    if use_log:
        x_vals = np.log(x_vals)
        y_vals = np.log(y_vals)
    
    # OLS: x = alpha + beta * y
    Y = sm.add_constant(y_vals)
    model = sm.OLS(x_vals, Y).fit()
    
    intercept = model.params.iloc[0]
    hedge_ratio = model.params.iloc[1]
    r_squared = model.rsquared
    
    return hedge_ratio, intercept, r_squared


def rolling_hedge_ratio(
    price_x: pd.Series,
    price_y: pd.Series,
    config: RollingHedgeConfig | None = None,
) -> pd.DataFrame:
    """Calculate rolling hedge ratios over time.
    
    Parameters
    ----------
    price_x, price_y : pd.Series
        Price series with datetime index.
    config : RollingHedgeConfig, optional
        Configuration for the rolling estimation.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, hedge_ratio, intercept, r_squared, spread
    """
    if config is None:
        config = RollingHedgeConfig()
    
    # Align series
    df = pd.concat([price_x, price_y], axis=1, join='inner').dropna()
    df.columns = ['x', 'y']
    
    if config.use_log_prices:
        df['log_x'] = np.log(df['x'])
        df['log_y'] = np.log(df['y'])
        x_col, y_col = 'log_x', 'log_y'
    else:
        x_col, y_col = 'x', 'y'
    
    # Determine rebalance dates
    if config.update_frequency == 'M':
        rebalance_dates = df.resample('ME').last().index
    elif config.update_frequency == 'W':
        rebalance_dates = df.resample('W').last().index
    else:
        rebalance_dates = df.index
    
    results = []
    
    for rebal_date in rebalance_dates:
        # Get lookback window
        start_date = rebal_date - pd.DateOffset(months=config.window_months)
        window_df = df[(df.index > start_date) & (df.index <= rebal_date)]
        
        if len(window_df) < config.min_observations:
            logger.debug(
                "Skipping %s: insufficient observations (%d < %d)",
                rebal_date.date(), len(window_df), config.min_observations
            )
            continue
        
        try:
            # OLS regression
            Y = sm.add_constant(window_df[y_col])
            model = sm.OLS(window_df[x_col], Y).fit()
            
            intercept = model.params.iloc[0]
            hedge_ratio = model.params.iloc[1]
            r_squared = model.rsquared
            
            # Calculate spread with this hedge ratio
            spread = window_df[x_col] - hedge_ratio * window_df[y_col]
            spread_std = spread.std()
            
            results.append(HedgeRatioEstimate(
                date=rebal_date,
                hedge_ratio=hedge_ratio,
                intercept=intercept,
                r_squared=r_squared,
                spread_std=spread_std,
            ))
            
        except Exception as e:
            logger.warning("Failed to estimate hedge ratio at %s: %s", rebal_date.date(), e)
    
    if not results:
        return pd.DataFrame(columns=['date', 'hedge_ratio', 'intercept', 'r_squared', 'spread_std'])
    
    return pd.DataFrame([
        {
            'date': r.date,
            'hedge_ratio': r.hedge_ratio,
            'intercept': r.intercept,
            'r_squared': r.r_squared,
            'spread_std': r.spread_std,
        }
        for r in results
    ]).set_index('date')


def calculate_dynamic_spread(
    price_x: pd.Series,
    price_y: pd.Series,
    hedge_ratios: pd.DataFrame,
    use_log: bool = True,
) -> pd.Series:
    """Calculate spread using time-varying hedge ratios.
    
    Parameters
    ----------
    price_x, price_y : pd.Series
        Price series with datetime index.
    hedge_ratios : pd.DataFrame
        DataFrame from rolling_hedge_ratio() with hedge_ratio column.
    use_log : bool
        Whether to use log prices for spread calculation.
    
    Returns
    -------
    pd.Series
        Dynamic spread series.
    """
    # Align prices
    df = pd.concat([price_x, price_y], axis=1, join='inner').dropna()
    df.columns = ['x', 'y']
    
    if use_log:
        df['x'] = np.log(df['x'])
        df['y'] = np.log(df['y'])
    
    # Forward-fill hedge ratios to daily frequency
    hr_daily = hedge_ratios['hedge_ratio'].reindex(df.index, method='ffill')
    
    # Calculate spread
    spread = df['x'] - hr_daily * df['y']
    
    return spread


def hedge_ratio_stability(hedge_ratios: pd.DataFrame) -> dict:
    """Assess stability of hedge ratio over time.
    
    Returns metrics indicating whether the cointegration relationship is stable.
    """
    hr = hedge_ratios['hedge_ratio']
    
    if len(hr) < 3:
        return {'stable': False, 'reason': 'insufficient_data'}
    
    mean_hr = hr.mean()
    std_hr = hr.std()
    cv = std_hr / abs(mean_hr) if mean_hr != 0 else np.inf  # Coefficient of variation
    
    # Check for drift
    first_half = hr.iloc[:len(hr)//2].mean()
    second_half = hr.iloc[len(hr)//2:].mean()
    drift = abs(second_half - first_half) / abs(mean_hr) if mean_hr != 0 else np.inf
    
    return {
        'mean_hedge_ratio': mean_hr,
        'std_hedge_ratio': std_hr,
        'coefficient_of_variation': cv,
        'drift_pct': drift * 100,
        'stable': cv < 0.20 and drift < 0.20,  # Thresholds for stability
        'mean_r_squared': hedge_ratios['r_squared'].mean(),
    }
