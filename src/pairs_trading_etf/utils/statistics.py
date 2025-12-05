"""
Statistical utilities for backtesting validation.

This module contains shared statistical functions used across different
backtesting modules to avoid code duplication.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple


def expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """
    Expected maximum Sharpe from random strategies (Bailey et al.).

    Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio:
    correcting for selection bias, backtest overfitting and non-normality.
    Journal of Portfolio Management, 40(5), 94-107.

    Parameters
    ----------
    n_trials : int
        Number of strategies tested (for selection bias adjustment)
    n_obs : int
        Number of observations in backtest

    Returns
    -------
    float
        Expected maximum Sharpe ratio from random trials
    """
    if n_trials <= 1:
        return 0.0

    # Euler-Mascheroni constant
    gamma = 0.5772156649

    try:
        z1 = norm.ppf(1 - 1 / n_trials)
        z2 = norm.ppf(1 - 1 / (n_trials * np.e))
        expected_max = (1 - gamma) * z1 + gamma * z2
        # Adjust for sample size (annualized)
        return expected_max * np.sqrt(252 / n_obs)
    except Exception:
        return 0.0


def calculate_dsr(
    sharpe_obs: float,
    n_trials: int,
    n_obs: int,
) -> Tuple[float, float]:
    """
    Deflated Sharpe Ratio (DSR) and p-value.

    The DSR adjusts the observed Sharpe ratio for selection bias from
    multiple testing and finite sample size.

    Parameters
    ----------
    sharpe_obs : float
        Observed Sharpe ratio from backtest
    n_trials : int
        Number of strategies tested
    n_obs : int
        Number of observations

    Returns
    -------
    Tuple[float, float]
        (DSR, p-value) where:
        - DSR > 0 suggests genuine skill after deflation
        - p-value < 0.05 indicates statistical significance
    """
    expected_max = expected_max_sharpe(n_trials, n_obs)

    # Standard error of Sharpe ratio (annualized)
    se = 1 / np.sqrt(max(n_obs, 1)) * np.sqrt(252)

    # Deflated Sharpe Ratio
    if se > 0:
        dsr = (sharpe_obs - expected_max) / se
    else:
        dsr = 0.0

    # P-value for H0: true Sharpe <= expected_max
    p_val = 1 - norm.cdf(dsr)

    return float(dsr), float(p_val)


def calculate_pbo(
    in_sample_sharpes: np.ndarray,
    out_sample_sharpes: np.ndarray,
) -> float:
    """
    Probability of Backtest Overfitting (PBO).

    Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014).
    Probability of backtest overfitting. Journal of Computational Finance.

    Parameters
    ----------
    in_sample_sharpes : np.ndarray
        In-sample Sharpe ratios for each parameter configuration
    out_sample_sharpes : np.ndarray
        Out-of-sample Sharpe ratios for same configurations

    Returns
    -------
    float
        PBO in [0, 1]. Values > 0.5 indicate likely overfitting.
        PBO > 0.7 is severe overfitting.
    """
    if len(in_sample_sharpes) != len(out_sample_sharpes):
        raise ValueError("In-sample and out-of-sample arrays must have same length")

    if len(in_sample_sharpes) == 0:
        return 1.0

    # Find the parameter with best in-sample performance
    best_is_idx = np.argmax(in_sample_sharpes)

    # Count configurations where IS rank > median but OOS underperforms
    median_is = np.median(in_sample_sharpes)
    above_median_mask = in_sample_sharpes >= median_is

    if not above_median_mask.any():
        return 1.0

    # For configurations above median IS, check if OOS < median OOS
    median_oos = np.median(out_sample_sharpes)
    overfit_count = np.sum(
        (in_sample_sharpes >= median_is) & (out_sample_sharpes < median_oos)
    )

    total_above_median = np.sum(above_median_mask)
    pbo = overfit_count / max(total_above_median, 1)

    return float(pbo)


def calculate_probability_loss(returns: np.ndarray) -> float:
    """
    Probability that strategy has negative returns.

    Uses Student's t-distribution to account for finite sample size.

    Parameters
    ----------
    returns : np.ndarray
        Array of strategy returns

    Returns
    -------
    float
        Probability in [0, 1] that true mean return is negative
    """
    if len(returns) == 0:
        return 1.0

    from scipy.stats import t

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    if std_ret == 0:
        return 0.0 if mean_ret > 0 else 1.0

    # t-statistic
    t_stat = mean_ret / (std_ret / np.sqrt(len(returns)))

    # P(mean < 0)
    prob_loss = t.cdf(-abs(t_stat), df=len(returns) - 1) if mean_ret > 0 else 1.0

    return float(prob_loss)
