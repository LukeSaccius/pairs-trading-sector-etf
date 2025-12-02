"""Optimized Walk-Forward Pairs Trading Backtest.

Performance optimizations:
1. Parallel cointegration testing with joblib
2. Vectorized z-score calculation for all pairs simultaneously
3. NumPy-based daily simulation loop
4. Pre-computed spread matrices
5. Batch processing for cointegration tests

Usage:
    python scripts/optimized_backtest.py [--workers N] [--start YEAR] [--end YEAR]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import time
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pairs_trading_etf.ou_model.half_life import estimate_half_life

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OptimizedConfig:
    """Backtest configuration with optimization settings."""
    
    # Window parameters
    formation_days: int = 252
    trading_days: int = 252
    
    # Cointegration thresholds
    pvalue_threshold: float = 0.05  # Stricter threshold
    min_half_life: float = 5
    max_half_life: float = 60  # Tighter range for faster mean reversion
    
    # Pair selection
    top_pairs: int = 20  # Only trade top N pairs ranked by quality score
    
    # Correlation filter
    min_corr: float = 0.60
    max_corr: float = 0.95  # Exclude near-perfect correlation (same index)
    
    # Trading signals
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_loss_z: float = 4.0
    zscore_lookback: int = 60
    
    # Position sizing
    capital_per_pair: float = 10000.0
    max_positions: int = 10
    
    # Transaction costs
    cost_bps: float = 10.0
    
    # Data
    use_log: bool = True
    
    # Optimization settings
    n_workers: int = -1  # -1 = all cores


# ============================================================================
# OPTIMIZATION 1: Fast Cointegration Testing with Pure NumPy
# ============================================================================

# Pre-computed ADF critical values (from statsmodels)
# Keys: (n_samples, significance) -> critical value
# We use interpolation for sample sizes
ADF_CRITICAL_VALUES = {
    # Sample size 100
    (100, 0.01): -3.51,
    (100, 0.05): -2.89,
    (100, 0.10): -2.58,
    # Sample size 250
    (250, 0.01): -3.46,
    (250, 0.05): -2.87,
    (250, 0.10): -2.57,
    # Sample size 500
    (500, 0.01): -3.44,
    (500, 0.05): -2.87,
    (500, 0.10): -2.57,
}


def _fast_adf_test(series: np.ndarray, maxlag: int = 1) -> tuple[float, float]:
    """Fast ADF test using pure NumPy.
    
    Returns:
        (t_statistic, approximate_pvalue)
        
    Note: p-value is computed using MacKinnon (1994) critical values.
    For Engle-Granger residual test with constant, sample size 250.
    """
    n = len(series)
    if n < 50:
        return 0.0, 1.0
    
    # Compute first differences
    diff = np.diff(series)
    lagged = series[:-1]
    
    # Build regression: diff[t] = alpha + beta*series[t-1] + gamma*diff[t-1] + epsilon
    if maxlag > 0 and len(diff) > maxlag:
        # Include lagged differences
        y = diff[maxlag:]
        X = np.column_stack([
            np.ones(len(y)),
            lagged[maxlag:],
            diff[:-maxlag] if maxlag == 1 else np.column_stack([diff[i:-maxlag+i] for i in range(maxlag)])
        ])
    else:
        y = diff
        X = np.column_stack([np.ones(len(y)), lagged])
    
    try:
        # OLS estimation
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        beta = XtX_inv @ (X.T @ y)
        
        # Residuals and standard error
        residuals = y - X @ beta
        sse = np.sum(residuals ** 2)
        dof = len(y) - X.shape[1]
        mse = sse / dof
        
        # Standard error of beta coefficient (for lagged level)
        se_beta = np.sqrt(mse * XtX_inv[1, 1])
        
        # t-statistic for beta (coefficient on lagged level)
        t_stat = beta[1] / se_beta
        
        # MacKinnon (1994) approximate p-values for ADF with constant
        # For Engle-Granger residual test (no trend), asymptotic critical values:
        # 1%: -3.43, 5%: -2.86, 10%: -2.57
        # Linear interpolation between critical values
        if t_stat < -4.5:
            pvalue = 0.0001
        elif t_stat < -4.0:
            pvalue = 0.001
        elif t_stat < -3.43:
            # Between 1% and 0.1%
            pvalue = 0.01 - (t_stat + 3.43) / (-4.0 + 3.43) * 0.009
        elif t_stat < -2.86:
            # Between 5% and 1% - linear interpolation
            pvalue = 0.05 - (t_stat + 2.86) / (-3.43 + 2.86) * 0.04
        elif t_stat < -2.57:
            # Between 10% and 5%
            pvalue = 0.10 - (t_stat + 2.57) / (-2.86 + 2.57) * 0.05
        elif t_stat < -1.95:
            # Between 30% and 10%
            pvalue = 0.30 - (t_stat + 1.95) / (-2.57 + 1.95) * 0.20
        elif t_stat < -1.62:
            pvalue = 0.50 - (t_stat + 1.62) / (-1.95 + 1.62) * 0.20
        else:
            # Linear extrapolation for high p-values
            pvalue = min(0.9, 0.5 + 0.2 * (t_stat + 1.62))
        
        return float(t_stat), float(pvalue)
        
    except Exception:
        return 0.0, 1.0

def _fast_coint_test(
    leg_x: str,
    leg_y: str,
    px: np.ndarray,
    py: np.ndarray,
    use_log: bool,
    pvalue_threshold: float,
    min_half_life: float,
    max_half_life: float,
) -> dict | None:
    """Ultra-fast cointegration test using pure NumPy.
    
    Replaces statsmodels.coint with direct ADF implementation.
    """
    try:
        # Apply log if needed
        if use_log:
            mask = (px > 0) & (py > 0)
            px_clean = np.log(px[mask])
            py_clean = np.log(py[mask])
        else:
            px_clean = px
            py_clean = py
        
        if len(px_clean) < 50:
            return None
        
        # Step 1: OLS regression y = a + b*x
        # Calculate hedge ratio
        X = np.column_stack([np.ones(len(py_clean)), py_clean])
        try:
            beta = np.linalg.lstsq(X, px_clean, rcond=None)[0]
            intercept, hedge_ratio = beta
        except Exception:
            return None
        
        # Step 2: Calculate residuals (spread)
        spread = px_clean - (intercept + hedge_ratio * py_clean)
        
        # Step 3: ADF test on residuals
        t_stat, pvalue = _fast_adf_test(spread, maxlag=1)
        
        # Use > instead of >= to include boundary cases
        if pvalue > pvalue_threshold:
            return None
        
        # Step 4: Estimate half-life using AR(1)
        lagged = spread[:-1]
        delta = np.diff(spread)
        
        if len(lagged) < 30:
            return None
        
        # OLS with intercept: delta = a + b * spread_lag
        X = np.column_stack([np.ones(len(lagged)), lagged])
        try:
            beta = np.linalg.lstsq(X, delta, rcond=None)[0]
            b = beta[1]
            
            if b >= 0:
                return None
            
            phi = 1 + b
            if phi <= 0 or phi >= 1:
                return None
            
            half_life = -np.log(2) / np.log(phi)
            
            if not (min_half_life <= half_life <= max_half_life):
                return None
            
            return {
                "pair": (leg_x, leg_y),
                "hedge_ratio": hedge_ratio,
                "half_life": float(half_life),
                "pvalue": float(pvalue),
                "spread_mean": float(spread.mean()),
                "spread_std": float(spread.std()),
            }
        except Exception:
            return None
            
    except Exception:
        return None


def parallel_cointegration_test(
    prices: pd.DataFrame,
    candidate_pairs: list[tuple[str, str]],
    cfg: OptimizedConfig,
    n_workers: int = -1,
) -> tuple[list[tuple[str, str]], dict, dict, dict, dict]:
    """Test pairs for cointegration using joblib for parallelization.
    
    Returns:
        (selected_pairs, hedge_ratios, half_lives, formation_stats, pvalues)
    """
    if not candidate_pairs:
        return [], {}, {}, {}, {}
    
    # Prepare data - extract numpy arrays once
    price_arrays = {col: prices[col].values for col in prices.columns}
    
    # Use joblib for parallel processing
    results = Parallel(n_jobs=n_workers, prefer="threads")(
        delayed(_fast_coint_test)(
            leg_x, leg_y,
            price_arrays[leg_x], price_arrays[leg_y],
            cfg.use_log,
            cfg.pvalue_threshold,
            cfg.min_half_life,
            cfg.max_half_life,
        )
        for leg_x, leg_y in candidate_pairs
    )
    
    # Collect results
    selected_pairs = []
    hedge_ratios = {}
    half_lives = {}
    formation_stats = {}
    pvalues = {}
    
    for result in results:
        if result is not None:
            pair = result["pair"]
            selected_pairs.append(pair)
            hedge_ratios[pair] = result["hedge_ratio"]
            half_lives[pair] = result["half_life"]
            formation_stats[pair] = (result["spread_mean"], result["spread_std"])
            pvalues[pair] = result["pvalue"]
    
    return selected_pairs, hedge_ratios, half_lives, formation_stats, pvalues


# ============================================================================
# OPTIMIZATION 2: Vectorized Z-Score Calculation
# ============================================================================

def vectorized_zscore_all_pairs(
    prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    hedge_ratios: dict,
    lookback: int = 60,
    use_log: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate z-scores for all pairs simultaneously using vectorized operations.
    
    Returns:
        (zscore_df, spread_df) - DataFrames with pairs as columns
    """
    n_dates = len(prices)
    n_pairs = len(pairs)
    
    if n_pairs == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Pre-allocate arrays
    spreads = np.zeros((n_dates, n_pairs))
    
    # Calculate all spreads at once
    for i, (leg_x, leg_y) in enumerate(pairs):
        if leg_x not in prices.columns or leg_y not in prices.columns:
            spreads[:, i] = np.nan
            continue
        
        hr = hedge_ratios.get((leg_x, leg_y), 1.0)
        
        if use_log:
            px = np.log(prices[leg_x].values)
            py = np.log(prices[leg_y].values)
        else:
            px = prices[leg_x].values
            py = prices[leg_y].values
        
        spreads[:, i] = px - hr * py
    
    # Create DataFrame
    spread_df = pd.DataFrame(spreads, index=prices.index, columns=pairs)
    
    # Vectorized rolling z-score calculation
    rolling_mean = spread_df.rolling(window=lookback).mean()
    rolling_std = spread_df.rolling(window=lookback).std()
    
    zscore_df = (spread_df - rolling_mean) / rolling_std
    
    return zscore_df, spread_df


# ============================================================================
# OPTIMIZATION 3: Vectorized Trading Simulation
# ============================================================================

@dataclass
class TradeRecord:
    """Lightweight trade record for fast simulation."""
    pair_idx: int
    direction: int
    entry_idx: int
    entry_z: float
    entry_spread: float
    qty_x: float
    qty_y: float
    entry_price_x: float
    entry_price_y: float


def vectorized_trading_simulation(
    prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    zscores: pd.DataFrame,
    spreads: pd.DataFrame,
    hedge_ratios: dict,
    cfg: OptimizedConfig,
) -> list[dict]:
    """Vectorized trading simulation with NumPy arrays.
    
    Uses state machines for each pair instead of per-day iteration.
    """
    trades = []
    n_dates = len(prices)
    n_pairs = len(pairs)
    warmup = cfg.zscore_lookback
    
    if n_pairs == 0 or n_dates <= warmup:
        return trades
    
    # Convert to numpy for speed
    z_matrix = zscores.values  # (n_dates, n_pairs)
    spread_matrix = spreads.values
    
    # Pre-extract price arrays for all legs
    price_x_matrix = np.zeros((n_dates, n_pairs))
    price_y_matrix = np.zeros((n_dates, n_pairs))
    hr_array = np.zeros(n_pairs)
    
    for i, (leg_x, leg_y) in enumerate(pairs):
        if leg_x in prices.columns and leg_y in prices.columns:
            price_x_matrix[:, i] = prices[leg_x].values
            price_y_matrix[:, i] = prices[leg_y].values
            hr_array[i] = hedge_ratios.get((leg_x, leg_y), 1.0)
        else:
            price_x_matrix[:, i] = np.nan
            price_y_matrix[:, i] = np.nan
    
    # State tracking
    # 0 = flat, 1 = long spread, -1 = short spread
    position_state = np.zeros(n_pairs, dtype=np.int8)
    entry_idx = np.zeros(n_pairs, dtype=np.int32)
    entry_z = np.zeros(n_pairs)
    entry_spread = np.zeros(n_pairs)
    entry_price_x = np.zeros(n_pairs)
    entry_price_y = np.zeros(n_pairs)
    qty_x = np.zeros(n_pairs)
    qty_y = np.zeros(n_pairs)
    
    # Count active positions
    n_active = 0
    
    dates = prices.index
    
    for t in range(warmup, n_dates):
        z = z_matrix[t, :]
        spread = spread_matrix[t, :]
        px = price_x_matrix[t, :]
        py = price_y_matrix[t, :]
        
        # Skip if all NaN
        if np.all(np.isnan(z)):
            continue
        
        # ===== Check exits for open positions =====
        open_mask = position_state != 0
        
        if np.any(open_mask):
            for i in np.where(open_mask)[0]:
                if np.isnan(z[i]):
                    continue
                
                direction = position_state[i]
                should_exit = False
                exit_reason = None
                
                if direction == 1:  # Long spread
                    if z[i] >= -cfg.exit_z:
                        should_exit = True
                        exit_reason = "convergence"
                    elif z[i] <= -cfg.stop_loss_z:
                        should_exit = True
                        exit_reason = "stop_loss"
                else:  # Short spread
                    if z[i] <= cfg.exit_z:
                        should_exit = True
                        exit_reason = "convergence"
                    elif z[i] >= cfg.stop_loss_z:
                        should_exit = True
                        exit_reason = "stop_loss"
                
                if should_exit:
                    # Calculate PnL
                    pnl_x = qty_x[i] * (px[i] - entry_price_x[i])
                    pnl_y = qty_y[i] * (py[i] - entry_price_y[i])
                    pnl = pnl_x + pnl_y
                    
                    # Transaction costs
                    entry_notional = abs(qty_x[i]) * entry_price_x[i] + abs(qty_y[i]) * entry_price_y[i]
                    exit_notional = abs(qty_x[i]) * px[i] + abs(qty_y[i]) * py[i]
                    cost = (entry_notional + exit_notional) * (cfg.cost_bps / 10000)
                    pnl -= cost
                    
                    leg_x, leg_y = pairs[i]
                    trades.append({
                        "pair": (leg_x, leg_y),
                        "direction": "LONG" if direction == 1 else "SHORT",
                        "entry_date": dates[entry_idx[i]],
                        "exit_date": dates[t],
                        "entry_z": entry_z[i],
                        "exit_z": z[i],
                        "entry_spread": entry_spread[i],
                        "exit_spread": spread[i],
                        "hedge_ratio": hr_array[i],
                        "pnl": pnl,
                        "exit_reason": exit_reason,
                    })
                    
                    # Reset state
                    position_state[i] = 0
                    n_active -= 1
        
        # ===== Check entries for new positions =====
        if n_active < cfg.max_positions:
            flat_mask = position_state == 0
            
            # Find entry signals
            long_signal = (z <= -cfg.entry_z) & flat_mask & ~np.isnan(z)
            short_signal = (z >= cfg.entry_z) & flat_mask & ~np.isnan(z)
            
            # Process long entries
            for i in np.where(long_signal)[0]:
                if n_active >= cfg.max_positions:
                    break
                
                hr = hr_array[i]
                notional_x = cfg.capital_per_pair / (1 + abs(hr))
                notional_y = abs(hr) * notional_x
                
                position_state[i] = 1
                entry_idx[i] = t
                entry_z[i] = z[i]
                entry_spread[i] = spread[i]
                entry_price_x[i] = px[i]
                entry_price_y[i] = py[i]
                qty_x[i] = notional_x / px[i]  # Long X
                qty_y[i] = -notional_y / py[i]  # Short Y
                n_active += 1
            
            # Process short entries
            for i in np.where(short_signal)[0]:
                if n_active >= cfg.max_positions:
                    break
                
                hr = hr_array[i]
                notional_x = cfg.capital_per_pair / (1 + abs(hr))
                notional_y = abs(hr) * notional_x
                
                position_state[i] = -1
                entry_idx[i] = t
                entry_z[i] = z[i]
                entry_spread[i] = spread[i]
                entry_price_x[i] = px[i]
                entry_price_y[i] = py[i]
                qty_x[i] = -notional_x / px[i]  # Short X
                qty_y[i] = notional_y / py[i]  # Long Y
                n_active += 1
    
    # Close remaining positions at end
    last_idx = n_dates - 1
    open_mask = position_state != 0
    
    for i in np.where(open_mask)[0]:
        direction = position_state[i]
        px_exit = price_x_matrix[last_idx, i]
        py_exit = price_y_matrix[last_idx, i]
        z_exit = z_matrix[last_idx, i]
        spread_exit = spread_matrix[last_idx, i]
        
        pnl_x = qty_x[i] * (px_exit - entry_price_x[i])
        pnl_y = qty_y[i] * (py_exit - entry_price_y[i])
        pnl = pnl_x + pnl_y
        
        entry_notional = abs(qty_x[i]) * entry_price_x[i] + abs(qty_y[i]) * entry_price_y[i]
        exit_notional = abs(qty_x[i]) * px_exit + abs(qty_y[i]) * py_exit
        cost = (entry_notional + exit_notional) * (cfg.cost_bps / 10000)
        pnl -= cost
        
        leg_x, leg_y = pairs[i]
        trades.append({
            "pair": (leg_x, leg_y),
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_date": dates[entry_idx[i]],
            "exit_date": dates[last_idx],
            "entry_z": entry_z[i],
            "exit_z": z_exit,
            "entry_spread": entry_spread[i],
            "exit_spread": spread_exit,
            "hedge_ratio": hr_array[i],
            "pnl": pnl,
            "exit_reason": "period_end",
        })
    
    return trades


# ============================================================================
# Main Backtest Functions
# ============================================================================

def compute_pair_score(pvalue: float, half_life: float, optimal_hl: float = 25.0) -> float:
    """Compute quality score for a pair.
    
    Score formula:
    - Lower p-value → higher score (use -log(pvalue))
    - Half-life closer to optimal → higher score
    
    Both components are normalized to [0, 1] range then combined.
    """
    # P-value component: -log(pvalue) normalized
    # p=0.001 → 6.9, p=0.01 → 4.6, p=0.05 → 3.0
    # Normalize to [0, 1] by capping at -log(0.001) ≈ 6.9
    pvalue_score = min(-np.log(max(pvalue, 1e-10)), 7.0) / 7.0
    
    # Half-life component: prefer values close to optimal
    # Score decays as half-life moves away from optimal
    hl_deviation = abs(half_life - optimal_hl) / optimal_hl
    hl_score = max(0, 1 - hl_deviation)  # 1.0 at optimal, 0 at 2x deviation
    
    # Combined score (weight p-value more heavily)
    score = 0.6 * pvalue_score + 0.4 * hl_score
    
    return score


def select_pairs_optimized(
    prices: pd.DataFrame,
    formation_start: pd.Timestamp,
    formation_end: pd.Timestamp,
    cfg: OptimizedConfig,
) -> tuple[list[tuple[str, str]], dict, dict, dict]:
    """Select TOP pairs with optimized parallel testing and ranking."""
    
    # Slice formation period
    mask = (prices.index >= formation_start) & (prices.index <= formation_end)
    formation_prices = prices.loc[mask].dropna(axis=1, how='any')
    
    if formation_prices.shape[0] < cfg.formation_days * 0.8:
        logger.warning(f"Insufficient formation data: {formation_prices.shape[0]} days")
        return [], {}, {}, {}
    
    tickers = list(formation_prices.columns)
    logger.info(f"Formation: {formation_start.date()} to {formation_end.date()}, {len(tickers)} tickers")
    
    # Step 1: Vectorized correlation filter
    returns = formation_prices.pct_change().dropna()
    corr_matrix = returns.corr().values
    
    # Find pairs meeting correlation criteria (vectorized)
    candidate_pairs = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            corr = corr_matrix[i, j]
            if cfg.min_corr <= corr <= cfg.max_corr:
                candidate_pairs.append((tickers[i], tickers[j]))
    
    logger.info(f"Pairs with {cfg.min_corr} <= corr <= {cfg.max_corr}: {len(candidate_pairs)}")
    
    # Step 2: Parallel cointegration test
    all_pairs, hedge_ratios, half_lives, formation_stats, pvalues = parallel_cointegration_test(
        formation_prices, candidate_pairs, cfg, n_workers=cfg.n_workers
    )
    
    logger.info(f"Cointegrated pairs (p<{cfg.pvalue_threshold}, HL {cfg.min_half_life}-{cfg.max_half_life}): {len(all_pairs)}")
    
    # Step 3: Rank pairs by quality score and select top N
    if len(all_pairs) > cfg.top_pairs:
        # Calculate scores for all passing pairs
        pair_scores = []
        for pair in all_pairs:
            score = compute_pair_score(pvalues[pair], half_lives[pair])
            pair_scores.append((pair, score, pvalues[pair], half_lives[pair]))
        
        # Sort by score descending
        pair_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N
        selected_pairs = [p[0] for p in pair_scores[:cfg.top_pairs]]
        
        # Log top pairs info
        logger.info(f"Selected TOP {cfg.top_pairs} pairs by quality score:")
        for i, (pair, score, pval, hl) in enumerate(pair_scores[:min(5, cfg.top_pairs)]):
            logger.info(f"  {i+1}. {pair}: score={score:.3f}, p={pval:.4f}, HL={hl:.1f}")
        
        # Filter dictionaries to only include selected pairs
        hedge_ratios = {p: hedge_ratios[p] for p in selected_pairs}
        half_lives = {p: half_lives[p] for p in selected_pairs}
        formation_stats = {p: formation_stats[p] for p in selected_pairs}
    else:
        selected_pairs = all_pairs
        logger.info(f"Using all {len(all_pairs)} cointegrated pairs (below top_pairs={cfg.top_pairs})")
    
    return selected_pairs, hedge_ratios, half_lives, formation_stats


def run_optimized_backtest(
    prices: pd.DataFrame,
    cfg: OptimizedConfig,
    start_year: int = 2015,
    end_year: int = 2024,
) -> tuple[list, pd.DataFrame]:
    """Run optimized walk-forward backtest."""
    
    year_results = []
    
    for year in range(start_year, end_year + 1):
        formation_year = year - 1
        trading_year = year
        
        formation_start = pd.Timestamp(f"{formation_year}-01-01")
        formation_end = pd.Timestamp(f"{formation_year}-12-31")
        trading_start = pd.Timestamp(f"{trading_year}-01-01")
        trading_end = pd.Timestamp(f"{trading_year}-12-31")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Year {trading_year}: Formation {formation_year}")
        logger.info(f"{'='*60}")
        
        t0 = time.time()
        
        # Step 1: Select pairs (parallel)
        selected_pairs, hedge_ratios, half_lives, formation_stats = select_pairs_optimized(
            prices, formation_start, formation_end, cfg
        )
        
        t1 = time.time()
        logger.info(f"Pair selection: {t1-t0:.2f}s")
        
        if not selected_pairs:
            logger.warning(f"No pairs selected for {trading_year}")
            continue
        
        # Step 2: Get trading period data
        mask = (prices.index >= trading_start) & (prices.index <= trading_end)
        trading_prices = prices.loc[mask]
        
        # Step 3: Vectorized z-score calculation
        zscores, spreads = vectorized_zscore_all_pairs(
            trading_prices, selected_pairs, hedge_ratios,
            lookback=cfg.zscore_lookback, use_log=cfg.use_log
        )
        
        t2 = time.time()
        logger.info(f"Z-score calculation: {t2-t1:.2f}s")
        
        # Step 4: Vectorized trading simulation
        trades = vectorized_trading_simulation(
            trading_prices, selected_pairs, zscores, spreads, hedge_ratios, cfg
        )
        
        t3 = time.time()
        logger.info(f"Trading simulation: {t3-t2:.2f}s")
        
        # Calculate metrics
        total_pnl = sum(t["pnl"] for t in trades)
        capital = cfg.capital_per_pair * cfg.max_positions
        return_pct = (total_pnl / capital) * 100 if capital > 0 else 0.0
        
        winning = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = winning / len(trades) * 100 if trades else 0
        
        year_results.append({
            "trading_year": trading_year,
            "pairs_selected": len(selected_pairs),
            "total_trades": len(trades),
            "winning_trades": winning,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "return_pct": return_pct,
            "trades": trades,
        })
        
        logger.info(f"Pairs: {len(selected_pairs)}, Trades: {len(trades)}, Return: {return_pct:.2f}%")
        logger.info(f"Total time: {t3-t0:.2f}s")
    
    # Build summary
    summary_df = pd.DataFrame([
        {k: v for k, v in yr.items() if k != "trades"}
        for yr in year_results
    ])
    
    return year_results, summary_df


def main():
    """Run optimized backtest with timing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized pairs trading backtest")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--start", type=int, default=2008, help="Start year (trading)")
    parser.add_argument("--end", type=int, default=2024, help="End year (trading)")
    parser.add_argument("--data", type=str, default="data/raw/etf_prices_fresh.csv", help="Price data file")
    args = parser.parse_args()
    
    # Load prices
    logger.info(f"Loading price data from {args.data}...")
    prices_df = pd.read_csv(args.data, index_col="Date", parse_dates=True)
    prices = prices_df.dropna(axis=1, thresh=int(len(prices_df) * 0.8))
    logger.info(f"Loaded {prices.shape[1]} ETFs, {prices.shape[0]} days")
    
    # Configuration - use defaults from OptimizedConfig
    # p-value < 0.05, half-life 5-60 days for better pair selection
    cfg = OptimizedConfig(
        n_workers=args.workers,
        # Thresholds from dataclass defaults:
        # pvalue_threshold=0.05
        # min_half_life=5, max_half_life=60
    )
    
    # Run backtest
    logger.info(f"\nRunning backtest with {cfg.n_workers} workers...")
    t_start = time.time()
    
    year_results, summary_df = run_optimized_backtest(
        prices, cfg, start_year=args.start, end_year=args.end
    )
    
    t_total = time.time() - t_start
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZED WALK-FORWARD BACKTEST RESULTS")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    print("\n" + "-"*40)
    print("PERFORMANCE METRICS")
    print("-"*40)
    print(f"Total PnL: ${summary_df['total_pnl'].sum():,.2f}")
    print(f"Total Trades: {summary_df['total_trades'].sum()}")
    print(f"Average Annual Return: {summary_df['return_pct'].mean():.2f}%")
    print(f"Average Win Rate: {summary_df['win_rate'].mean():.1f}%")
    print(f"\nTotal Execution Time: {t_total:.2f}s")
    
    # Save results
    summary_df.to_csv("results/optimized_backtest_summary.csv", index=False)
    
    # Save all trades
    all_trades = []
    for yr in year_results:
        for t in yr["trades"]:
            t["trading_year"] = yr["trading_year"]
            t["leg_x"] = t["pair"][0]
            t["leg_y"] = t["pair"][1]
            del t["pair"]
            all_trades.append(t)
    
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv("results/optimized_backtest_trades.csv", index=False)
    
    logger.info("\nResults saved to results/optimized_backtest_*.csv")


if __name__ == "__main__":
    main()
