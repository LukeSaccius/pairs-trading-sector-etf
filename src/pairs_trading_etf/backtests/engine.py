"""
Backtest execution engine for pairs trading.

This module provides the core trading simulation loop, including:
- Cointegration testing
- Pair selection with sector diversification
- Z-score signal generation
- Position management and trade execution
- Dynamic hedge ratio updates
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint

from .config import BacktestConfig
from ..utils.sectors import get_sector, are_same_sector

logger = logging.getLogger(__name__)


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
    4. Score and rank pairs
    5. Apply diversification limits
    
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
                # NEW: Hedge ratio filter - avoid imbalanced positions
                hr = abs(result['hedge_ratio'])
                if cfg.min_hedge_ratio <= hr <= cfg.max_hedge_ratio:
                    cointegrated.append(pair)
                    results[pair] = result
    
    logger.info(f"Cointegrated pairs: {len(cointegrated)}")
    
    if not cointegrated:
        return [], {}, {}, {}
    
    # Step 3: Scoring - add hedge ratio quality score
    scores = {}
    for pair in cointegrated:
        r = results[pair]
        pvalue_score = min(-np.log(max(r['pvalue'], 1e-10)) / 7.0, 1.0)
        hl_score = max(0, 1 - abs(r['half_life'] - 15) / 15)
        range_score = min(r['spread_range'] / 0.10, 1.0)
        # NEW: Prefer hedge ratios closer to 1.0 (balanced positions)
        hr = abs(r['hedge_ratio'])
        hr_score = 1.0 - abs(hr - 1.0) / 1.0  # 1.0 at HR=1, 0.5 at HR=0.5 or 1.5
        hr_score = max(0, min(1, hr_score))
        scores[pair] = 0.30 * pvalue_score + 0.30 * hl_score + 0.25 * range_score + 0.15 * hr_score
    
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
    warmup = cfg.zscore_lookback
    
    if n_pairs == 0 or n_dates <= warmup:
        return trades, current_capital if current_capital else cfg.initial_capital
    
    # Capital management
    if current_capital is None:
        current_capital = cfg.initial_capital
    
    available_capital = current_capital * cfg.leverage
    
    # Dynamic capital per pair: divide available among max positions
    max_pos = cfg.max_positions if cfg.max_positions > 0 else len(pairs)
    capital_per_position = available_capital / max_pos
    
    # Create pair name mapping
    pair_names = {pair: f"{pair[0]}_{pair[1]}" for pair in pairs}
    
    # State tracking
    position_state = {pair: 0 for pair in pairs}
    entry_data = {pair: {} for pair in pairs}
    current_hr = dict(hedge_ratios)
    
    # Calculate initial spreads
    spreads = pd.DataFrame(index=prices.index)
    for pair in pairs:
        leg_x, leg_y = pair
        hr = current_hr[pair]
        log_x = np.log(prices[leg_x])
        log_y = np.log(prices[leg_y])
        spread = log_x - hr * log_y
        spreads[pair_names[pair]] = spread
    
    # Rolling z-score
    rolling_mean = spreads.rolling(window=cfg.zscore_lookback, min_periods=30).mean()
    rolling_std = spreads.rolling(window=cfg.zscore_lookback, min_periods=30).std()
    zscores = (spreads - rolling_mean) / rolling_std
    
    dates = prices.index.tolist()
    
    for t in range(warmup, n_dates):
        current_date = dates[t]
        
        # Dynamic hedge ratio update
        if cfg.dynamic_hedge and t % cfg.hedge_update_days == 0 and t > cfg.hedge_update_days:
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
            
            rolling_mean = spreads.rolling(window=cfg.zscore_lookback, min_periods=30).mean()
            rolling_std = spreads.rolling(window=cfg.zscore_lookback, min_periods=30).std()
            zscores = (spreads - rolling_mean) / rolling_std
        
        # Check exits
        for pair in pairs:
            if position_state[pair] == 0:
                continue
            
            pair_name = pair_names[pair]
            z = zscores.loc[current_date, pair_name]
            spread = spreads.loc[current_date, pair_name]
            
            if pd.isna(z):
                continue
            
            direction = position_state[pair]
            entry = entry_data[pair]
            
            should_exit = False
            exit_reason = None
            
            holding_days = t - entry['t']
            hl = half_lives[pair]
            
            # Dynamic max holding based on half-life
            if cfg.dynamic_max_holding:
                # Scale max holding by half-life: faster mean-reversion = shorter holding
                max_hold = min(int(cfg.max_holding_multiplier * hl), cfg.max_holding_days)
            else:
                max_hold = cfg.max_holding_days
            
            if holding_days >= max_hold:
                should_exit = True
                exit_reason = "max_holding"
            elif entry['spread'] * spread < 0:
                should_exit = True
                exit_reason = "regime_break"
            elif direction == 1:  # Long spread
                if z >= -cfg.exit_zscore:
                    should_exit = True
                    exit_reason = "convergence"
                elif z <= -cfg.stop_loss_zscore:
                    should_exit = True
                    exit_reason = "stop_loss"
            else:  # Short spread
                if z <= cfg.exit_zscore:
                    should_exit = True
                    exit_reason = "convergence"
                elif z >= cfg.stop_loss_zscore:
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
        
        if n_active < max_pos_limit:
            for pair in pairs:
                if position_state[pair] != 0:
                    continue
                if n_active >= max_pos_limit:
                    break
                
                pair_name = pair_names[pair]
                z = zscores.loc[current_date, pair_name]
                spread = spreads.loc[current_date, pair_name]
                
                if pd.isna(z):
                    continue
                
                leg_x, leg_y = pair
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
                
                notional_x = position_capital / (1 + abs(hr))
                notional_y = abs(hr) * notional_x
                
                if z <= -cfg.entry_zscore:
                    position_state[pair] = 1
                    entry_data[pair] = {
                        't': t,
                        'date': current_date,
                        'z': z,
                        'spread': spread,
                        'px': px,
                        'py': py,
                        'hr': hr,
                        'qty_x': notional_x / px,
                        'qty_y': -notional_y / py,
                        'capital': current_capital,
                    }
                    n_active += 1
                elif z >= cfg.entry_zscore:
                    position_state[pair] = -1
                    entry_data[pair] = {
                        't': t,
                        'date': current_date,
                        'z': z,
                        'spread': spread,
                        'px': px,
                        'py': py,
                        'hr': hr,
                        'qty_x': -notional_x / px,
                        'qty_y': notional_y / py,
                        'capital': current_capital,
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
        z_val = zscores.loc[last_date, pair_name]
        z = z_val if not pd.isna(z_val) else 0
        
        pnl_x = entry['qty_x'] * (px - entry['px'])
        pnl_y = entry['qty_y'] * (py - entry['py'])
        pnl = pnl_x + pnl_y
        
        entry_notional = abs(entry['qty_x']) * entry['px'] + abs(entry['qty_y']) * entry['py']
        exit_notional = abs(entry['qty_x']) * px + abs(entry['qty_y']) * py
        cost = (entry_notional + exit_notional) * (cfg.transaction_cost_bps / 10000)
        pnl -= cost
        
        holding_days = len(prices) - 1 - entry['t']
        
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
        
        for pair, stats in self.pair_stats.items():
            if stats['trades'] >= self.min_trades:
                sl_rate = stats['stop_losses'] / stats['trades']
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
        formation_prices = prices.loc[mask].dropna(axis=1, how='any')
        
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
