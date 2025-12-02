"""Walk-Forward Pairs Trading Backtest - Tokat & Hayrullahoglu (2021) Methodology.

Implements the methodology from:
"Pairs Trading: Is it Applicable to Exchange-Traded Funds?"
Tokat & Hayrullahoglu (2021), Borsa Istanbul Review

Key methodology:
1. Formation Period: 252 trading days (1 year)
2. Trading Period: 252 trading days (1 year) 
3. Annual rebalancing with fresh parameter estimation
4. Cointegration via Engle-Granger
5. Trading signals via Bollinger Bands / Z-score

Paper Results (benchmark):
- Annual Return: 15% (after 10bps costs)
- Sharpe Ratio: 1.43
- Crisis Performance: 41% (2008-2009)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from pairs_trading_etf.cointegration.engle_granger import run_engle_granger
from pairs_trading_etf.data.loader import build_price_frame
from pairs_trading_etf.features.pair_generation import enumerate_pairs, score_pairs
from pairs_trading_etf.features.kalman_hedge import kalman_filter_hedge

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TokatConfig:
    """Configuration following Tokat paper methodology."""
    
    # Window parameters (Paper: 252-day formation, 252-day trading)
    formation_days: int = 252
    trading_days: int = 252
    
    # Cointegration thresholds
    pvalue_threshold: float = 0.10
    min_half_life: float = 5        # Paper uses broader range
    max_half_life: float = 252      # Allow up to 1 year
    
    # Correlation filter
    min_corr: float = 0.80          # Paper: correlation > 0.80
    
    # Trading signals (Bollinger Bands based)
    entry_z: float = 2.0            # Enter when z > 2 or z < -2
    exit_z: float = 0.5             # Exit when |z| < 0.5 (partial convergence)
    exit_on_mean: bool = False      # Don't exit early on mean cross
    stop_loss_z: float = 4.0        # Stop loss at 4 std (looser)
    zscore_lookback: int = 60       # Z-score rolling window (longer = more stable)
    
    # Kalman filter for dynamic hedge ratio
    use_kalman: bool = False        # Use Kalman filter for adaptive hedge ratio
    kalman_delta: float = 1e-4      # Kalman process noise (smaller = more stable)
    
    # Position sizing
    capital_per_pair: float = 10000.0
    max_positions: int = 10         # Max concurrent pairs
    
    # Transaction costs (Paper: 10 bps one-way)
    cost_bps: float = 10.0
    
    # Data
    use_log: bool = True


@dataclass
class PairTrade:
    """Record of a single pair trade."""
    pair: tuple[str, str]
    direction: int  # 1 = long spread, -1 = short spread
    entry_date: pd.Timestamp
    entry_z: float
    entry_spread: float
    hedge_ratio: float
    exit_date: pd.Timestamp | None = None
    exit_z: float | None = None
    exit_spread: float | None = None
    exit_reason: str | None = None
    pnl: float = 0.0
    
    @property
    def is_open(self) -> bool:
        return self.exit_date is None


@dataclass 
class YearResult:
    """Results for a single formation-trading year."""
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp
    
    # Pair selection
    pairs_tested: int
    pairs_cointegrated: int
    pairs_selected: list[tuple[str, str]]
    
    # Trading results
    trades: list[PairTrade]
    total_pnl: float
    return_pct: float
    sharpe: float | None
    
    # Parameters
    hedge_ratios: dict[tuple[str, str], float]
    half_lives: dict[tuple[str, str], float]


def select_pairs_for_period(
    prices: pd.DataFrame,
    formation_start: pd.Timestamp,
    formation_end: pd.Timestamp,
    cfg: TokatConfig,
) -> tuple[list[tuple[str, str]], dict, dict, dict]:
    """Select cointegrated pairs from formation period.
    
    Returns:
        (selected_pairs, hedge_ratios, half_lives, formation_stats)
        formation_stats: {pair: (mean, std)} for z-score calculation
    """
    # Slice formation period
    mask = (prices.index >= formation_start) & (prices.index <= formation_end)
    formation_prices = prices.loc[mask].dropna(axis=1, how='any')
    
    if formation_prices.shape[0] < cfg.formation_days * 0.8:
        logger.warning(f"Insufficient data for formation period: {formation_prices.shape[0]} days")
        return [], {}, {}, {}
    
    tickers = list(formation_prices.columns)
    logger.info(f"Formation period: {formation_start.date()} to {formation_end.date()}, {len(tickers)} tickers")
    
    # Step 1: Correlation filter
    returns = formation_prices.pct_change().dropna()
    corr_matrix = returns.corr()
    
    candidate_pairs = []
    for i, t1 in enumerate(tickers):
        for t2 in tickers[i+1:]:
            corr = corr_matrix.loc[t1, t2]
            if corr >= cfg.min_corr:
                candidate_pairs.append((t1, t2))
    
    logger.info(f"Pairs with correlation >= {cfg.min_corr}: {len(candidate_pairs)}")
    
    # Step 2: Cointegration test
    selected_pairs = []
    hedge_ratios = {}
    half_lives = {}
    formation_stats = {}  # NEW: Store spread mean/std from formation period
    
    for leg_x, leg_y in candidate_pairs:
        px = formation_prices[leg_x]
        py = formation_prices[leg_y]
        
        try:
            result = run_engle_granger(px, py, use_log=cfg.use_log)
            
            if result.pvalue < cfg.pvalue_threshold:
                hl = result.half_life
                if hl and cfg.min_half_life <= hl <= cfg.max_half_life:
                    selected_pairs.append((leg_x, leg_y))
                    hedge_ratios[(leg_x, leg_y)] = result.hedge_ratio
                    half_lives[(leg_x, leg_y)] = hl
                    
                    # Compute formation period spread statistics
                    if cfg.use_log:
                        spread = np.log(px) - result.hedge_ratio * np.log(py)
                    else:
                        spread = px - result.hedge_ratio * py
                    formation_stats[(leg_x, leg_y)] = (spread.mean(), spread.std())
        except Exception as e:
            continue
    
    logger.info(f"Cointegrated pairs (p < {cfg.pvalue_threshold}): {len(selected_pairs)}")
    return selected_pairs, hedge_ratios, half_lives, formation_stats


def compute_spread_zscore(
    prices: pd.DataFrame,
    pair: tuple[str, str],
    hedge_ratio: float,
    lookback: int = 20,
    use_log: bool = True,
    formation_mean: float | None = None,
    formation_std: float | None = None,
) -> pd.Series:
    """Compute z-score for spread.
    
    If formation_mean/std provided, uses fixed statistics (Tokat method).
    Otherwise uses rolling statistics (Bollinger-style).
    """
    leg_x, leg_y = pair
    
    if use_log:
        px = np.log(prices[leg_x])
        py = np.log(prices[leg_y])
    else:
        px = prices[leg_x]
        py = prices[leg_y]
    
    spread = px - hedge_ratio * py
    
    if formation_mean is not None and formation_std is not None:
        # Use fixed formation period statistics (Tokat methodology)
        zscore = (spread - formation_mean) / formation_std
    else:
        # Rolling z-score (Bollinger-style fallback)
        rolling_mean = spread.rolling(window=lookback).mean()
        rolling_std = spread.rolling(window=lookback).std()
        zscore = (spread - rolling_mean) / rolling_std
    
    return zscore


@dataclass
class OpenPosition:
    """Track actual position quantities for proper PnL calculation."""
    trade: PairTrade
    qty_x: float  # Units of asset X (positive = long)
    qty_y: float  # Units of asset Y (positive = long)
    entry_price_x: float
    entry_price_y: float


def simulate_trading_period(
    prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    hedge_ratios: dict,
    half_lives: dict,
    formation_stats: dict,  # NEW: {pair: (mean, std)} from formation period
    trading_start: pd.Timestamp,
    trading_end: pd.Timestamp,
    cfg: TokatConfig,
) -> list[PairTrade]:
    """Simulate trading for a single period with CORRECT PnL calculation.
    
    PnL is calculated from actual price changes, not spread changes.
    For a pairs trade:
    - Long spread: Long X, Short Y (hedged)
    - Short spread: Short X, Long Y (hedged)
    
    PnL = qty_X * (exit_price_X - entry_price_X) + qty_Y * (exit_price_Y - entry_price_Y)
    """
    
    mask = (prices.index >= trading_start) & (prices.index <= trading_end)
    trading_prices = prices.loc[mask]
    
    if trading_prices.empty:
        return []
    
    trades: list[PairTrade] = []
    open_positions: dict[tuple[str, str], OpenPosition] = {}
    
    # Compute z-scores for all pairs using ROLLING statistics
    # Note: Formation stats are available but rolling is more adaptive
    zscores = {}
    spreads = {}
    for pair in pairs:
        leg_x, leg_y = pair
        if leg_x not in trading_prices.columns or leg_y not in trading_prices.columns:
            continue
        
        hr = hedge_ratios[pair]
        
        # Optionally use Kalman filter for dynamic hedge ratio
        if cfg.use_kalman:
            try:
                kalman_result = kalman_filter_hedge(
                    trading_prices[leg_x], trading_prices[leg_y],
                    delta=cfg.kalman_delta, use_log=cfg.use_log
                )
                # Use Kalman spread directly (it's the innovation/residual)
                spreads[pair] = kalman_result.spread
                dynamic_hr = kalman_result.hedge_ratios
                
                # Z-score on Kalman spread
                rolling_mean = spreads[pair].rolling(window=cfg.zscore_lookback).mean()
                rolling_std = spreads[pair].rolling(window=cfg.zscore_lookback).std()
                zscores[pair] = (spreads[pair] - rolling_mean) / rolling_std
                
                # Store dynamic hedge ratio for position sizing (use latest)
                hedge_ratios[pair] = kalman_result.final_hedge_ratio
            except Exception:
                # Fallback to static hedge ratio
                zscore = compute_spread_zscore(
                    trading_prices, pair, hr, 
                    lookback=cfg.zscore_lookback, use_log=cfg.use_log,
                    formation_mean=None, formation_std=None
                )
                zscores[pair] = zscore
                if cfg.use_log:
                    spreads[pair] = np.log(trading_prices[leg_x]) - hr * np.log(trading_prices[leg_y])
                else:
                    spreads[pair] = trading_prices[leg_x] - hr * trading_prices[leg_y]
        else:
            # Use rolling z-score with configurable lookback
            zscore = compute_spread_zscore(
                trading_prices, pair, hr, 
                lookback=cfg.zscore_lookback, use_log=cfg.use_log,
                formation_mean=None, formation_std=None  # Rolling mode
            )
            zscores[pair] = zscore
            
            if cfg.use_log:
                spreads[pair] = np.log(trading_prices[leg_x]) - hr * np.log(trading_prices[leg_y])
            else:
                spreads[pair] = trading_prices[leg_x] - hr * trading_prices[leg_y]
    
    # Daily simulation - skip warmup based on lookback
    warmup = cfg.zscore_lookback
    for date in trading_prices.index[warmup:]:
        
        # Check exits for open positions
        for pair, pos in list(open_positions.items()):
            if pair not in zscores:
                continue
            
            leg_x, leg_y = pair
            z = zscores[pair].loc[date]
            spread = spreads[pair].loc[date]
            
            if pd.isna(z):
                continue
            
            trade = pos.trade
            should_exit = False
            exit_reason = None
            
            # Exit conditions
            # LONG spread: entered when z < -entry_z, profit when z RISES toward 0
            # SHORT spread: entered when z > +entry_z, profit when z FALLS toward 0
            if trade.direction == 1:  # Long spread (entered at z < -2)
                # Take profit: z rises to -exit_z or above
                if z >= -cfg.exit_z:
                    should_exit = True
                    exit_reason = "convergence"
                # Exit on mean cross (z crosses above 0)
                elif cfg.exit_on_mean and z >= 0:
                    should_exit = True
                    exit_reason = "mean_cross"
                # Stop loss: z falls further (spread diverges more)
                elif z <= -cfg.stop_loss_z:
                    should_exit = True
                    exit_reason = "stop_loss"
            else:  # Short spread (entered at z > +2)
                # Take profit: z falls to +exit_z or below
                if z <= cfg.exit_z:
                    should_exit = True
                    exit_reason = "convergence"
                # Exit on mean cross (z crosses below 0)
                elif cfg.exit_on_mean and z <= 0:
                    should_exit = True
                    exit_reason = "mean_cross"
                # Stop loss: z rises further (spread diverges more)
                elif z >= cfg.stop_loss_z:
                    should_exit = True
                    exit_reason = "stop_loss"
            
            if should_exit:
                exit_price_x = trading_prices[leg_x].loc[date]
                exit_price_y = trading_prices[leg_y].loc[date]
                
                trade.exit_date = date
                trade.exit_z = z
                trade.exit_spread = spread
                trade.exit_reason = exit_reason
                
                # Calculate PnL from ACTUAL price changes
                # Long position profit = qty * (exit - entry)
                # Short position profit = qty * (entry - exit)
                pnl_x = pos.qty_x * (exit_price_x - pos.entry_price_x)
                pnl_y = pos.qty_y * (exit_price_y - pos.entry_price_y)
                trade.pnl = pnl_x + pnl_y
                
                # Subtract transaction costs
                # Entry: trade both legs, Exit: trade both legs
                # Cost = notional * cost_bps for each trade
                entry_notional = abs(pos.qty_x) * pos.entry_price_x + abs(pos.qty_y) * pos.entry_price_y
                exit_notional = abs(pos.qty_x) * exit_price_x + abs(pos.qty_y) * exit_price_y
                cost = (entry_notional + exit_notional) * (cfg.cost_bps / 10000)
                trade.pnl -= cost
                
                trades.append(trade)
                del open_positions[pair]
        
        # Check entries for new positions
        if len(open_positions) < cfg.max_positions:
            for pair in pairs:
                if pair in open_positions:
                    continue
                if pair not in zscores:
                    continue
                
                leg_x, leg_y = pair
                z = zscores[pair].loc[date]
                spread = spreads[pair].loc[date]
                
                if pd.isna(z):
                    continue
                
                price_x = trading_prices[leg_x].loc[date]
                price_y = trading_prices[leg_y].loc[date]
                hr = hedge_ratios[pair]
                
                # Entry signals with HEDGE RATIO adjusted position sizing
                # Spread = log(X) - hr * log(Y)
                # For proper hedging: notional_Y = hr * notional_X
                # Total capital split: notional_X + |hr| * notional_X = capital
                # So: notional_X = capital / (1 + |hr|)
                
                if z >= cfg.entry_z:  # Spread too high -> short spread (short X, long Y)
                    # Short spread: SHORT X, LONG Y (expect spread to fall)
                    notional_x = cfg.capital_per_pair / (1 + abs(hr))
                    notional_y = abs(hr) * notional_x
                    
                    qty_x = -notional_x / price_x  # Short X
                    qty_y = notional_y / price_y   # Long Y (hedged amount)
                    
                    trade = PairTrade(
                        pair=pair,
                        direction=-1,
                        entry_date=date,
                        entry_z=z,
                        entry_spread=spread,
                        hedge_ratio=hr,
                    )
                    open_positions[pair] = OpenPosition(
                        trade=trade,
                        qty_x=qty_x,
                        qty_y=qty_y,
                        entry_price_x=price_x,
                        entry_price_y=price_y,
                    )
                    
                elif z <= -cfg.entry_z:  # Spread too low -> long spread (long X, short Y)
                    # Long spread: LONG X, SHORT Y (expect spread to rise)
                    notional_x = cfg.capital_per_pair / (1 + abs(hr))
                    notional_y = abs(hr) * notional_x
                    
                    qty_x = notional_x / price_x    # Long X
                    qty_y = -notional_y / price_y   # Short Y (hedged amount)
                    
                    trade = PairTrade(
                        pair=pair,
                        direction=1,
                        entry_date=date,
                        entry_z=z,
                        entry_spread=spread,
                        hedge_ratio=hr,
                    )
                    open_positions[pair] = OpenPosition(
                        trade=trade,
                        qty_x=qty_x,
                        qty_y=qty_y,
                        entry_price_x=price_x,
                        entry_price_y=price_y,
                    )
                
                if len(open_positions) >= cfg.max_positions:
                    break
    
    # Close any remaining open positions at end of trading period
    last_date = trading_prices.index[-1]
    for pair, pos in open_positions.items():
        if pair not in zscores:
            continue
        
        leg_x, leg_y = pair
        trade = pos.trade
        
        z = zscores[pair].iloc[-1]
        spread = spreads[pair].iloc[-1]
        exit_price_x = trading_prices[leg_x].iloc[-1]
        exit_price_y = trading_prices[leg_y].iloc[-1]
        
        trade.exit_date = last_date
        trade.exit_z = z
        trade.exit_spread = spread
        trade.exit_reason = "period_end"
        
        pnl_x = pos.qty_x * (exit_price_x - pos.entry_price_x)
        pnl_y = pos.qty_y * (exit_price_y - pos.entry_price_y)
        trade.pnl = pnl_x + pnl_y
        
        entry_notional = abs(pos.qty_x) * pos.entry_price_x + abs(pos.qty_y) * pos.entry_price_y
        exit_notional = abs(pos.qty_x) * exit_price_x + abs(pos.qty_y) * exit_price_y
        cost = (entry_notional + exit_notional) * (cfg.cost_bps / 10000)
        trade.pnl -= cost
        
        trades.append(trade)
    
    return trades


def run_walk_forward_backtest(
    prices: pd.DataFrame,
    cfg: TokatConfig,
    start_year: int = 2015,
    end_year: int = 2024,
) -> tuple[list[YearResult], pd.DataFrame]:
    """Run full walk-forward backtest with annual rebalancing.
    
    Methodology:
    - Year N formation: Jan 1 to Dec 31 of year N
    - Year N+1 trading: Jan 1 to Dec 31 of year N+1
    - Repeat for each year
    
    Returns:
        (year_results, summary_df)
    """
    year_results = []
    
    for year in range(start_year, end_year + 1):
        formation_year = year - 1
        trading_year = year
        
        # Formation period: full previous year
        formation_start = pd.Timestamp(f"{formation_year}-01-01")
        formation_end = pd.Timestamp(f"{formation_year}-12-31")
        
        # Trading period: full current year
        trading_start = pd.Timestamp(f"{trading_year}-01-01")
        trading_end = pd.Timestamp(f"{trading_year}-12-31")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Year {trading_year}: Formation {formation_year}, Trading {trading_year}")
        logger.info(f"{'='*60}")
        
        # Step 1: Select pairs from formation period
        selected_pairs, hedge_ratios, half_lives, formation_stats = select_pairs_for_period(
            prices, formation_start, formation_end, cfg
        )
        
        if not selected_pairs:
            logger.warning(f"No pairs selected for {trading_year}")
            continue
        
        # Step 2: Trade during trading period
        trades = simulate_trading_period(
            prices, selected_pairs, hedge_ratios, half_lives, formation_stats,
            trading_start, trading_end, cfg
        )
        
        # Calculate metrics
        total_pnl = sum(t.pnl for t in trades)
        capital = cfg.capital_per_pair * cfg.max_positions
        return_pct = (total_pnl / capital) * 100 if capital > 0 else 0.0
        
        # Simple Sharpe (annualized)
        if trades:
            trade_returns = [t.pnl / cfg.capital_per_pair for t in trades]
            if len(trade_returns) > 1 and np.std(trade_returns) > 0:
                sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(len(trade_returns))
            else:
                sharpe = None
        else:
            sharpe = None
        
        year_result = YearResult(
            formation_start=formation_start,
            formation_end=formation_end,
            trading_start=trading_start,
            trading_end=trading_end,
            pairs_tested=len(enumerate_pairs(list(prices.columns))),
            pairs_cointegrated=len(selected_pairs),
            pairs_selected=selected_pairs,
            trades=trades,
            total_pnl=total_pnl,
            return_pct=return_pct,
            sharpe=sharpe,
            hedge_ratios=hedge_ratios,
            half_lives=half_lives,
        )
        
        year_results.append(year_result)
        
        logger.info(f"Selected pairs: {len(selected_pairs)}")
        logger.info(f"Total trades: {len(trades)}")
        logger.info(f"Total PnL: ${total_pnl:,.2f}")
        logger.info(f"Return: {return_pct:.2f}%")
        if sharpe:
            logger.info(f"Sharpe: {sharpe:.2f}")
    
    # Build summary DataFrame
    summary_data = []
    for yr in year_results:
        winning = sum(1 for t in yr.trades if t.pnl > 0)
        losing = sum(1 for t in yr.trades if t.pnl <= 0)
        
        summary_data.append({
            "trading_year": yr.trading_start.year,
            "pairs_cointegrated": yr.pairs_cointegrated,
            "total_trades": len(yr.trades),
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": winning / len(yr.trades) * 100 if yr.trades else 0,
            "total_pnl": yr.total_pnl,
            "return_pct": yr.return_pct,
            "sharpe": yr.sharpe,
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    return year_results, summary_df


def main():
    """Run the Tokat methodology backtest."""
    
    # Load prices
    logger.info("Loading price data...")
    prices_df = pd.read_csv("data/raw/etf_prices.csv", index_col="Date", parse_dates=True)
    
    # Use all available ETFs
    prices = prices_df.dropna(axis=1, thresh=int(len(prices_df) * 0.8))
    logger.info(f"Loaded {prices.shape[1]} ETFs, {prices.shape[0]} days")
    
    # Configuration
    cfg = TokatConfig(
        formation_days=252,
        trading_days=252,
        min_corr=0.80,
        pvalue_threshold=0.10,
        min_half_life=5,
        max_half_life=252,
        entry_z=2.0,
        exit_z=0.5,
        cost_bps=10.0,  # Paper: 10 bps
        max_positions=10,
        capital_per_pair=10000.0,
    )
    
    # Run backtest
    year_results, summary_df = run_walk_forward_backtest(
        prices,
        cfg,
        start_year=2015,  # Formation 2014 -> Trading 2015
        end_year=2024,    # Formation 2023 -> Trading 2024
    )
    
    # Print summary
    print("\n" + "="*80)
    print("WALK-FORWARD BACKTEST RESULTS (Tokat Methodology)")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Overall metrics
    total_pnl = summary_df["total_pnl"].sum()
    total_trades = summary_df["total_trades"].sum()
    avg_return = summary_df["return_pct"].mean()
    
    print("\n" + "-"*40)
    print("OVERALL METRICS")
    print("-"*40)
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Total Trades: {total_trades}")
    print(f"Average Annual Return: {avg_return:.2f}%")
    
    # Compare to paper
    print("\n" + "-"*40)
    print("COMPARISON TO TOKAT PAPER")
    print("-"*40)
    print(f"{'Metric':<25} {'Our Result':<15} {'Paper Result':<15}")
    print(f"{'Avg Annual Return':<25} {avg_return:.1f}%{'':<10} 15.0%")
    
    # Save results
    summary_df.to_csv("results/tokat_backtest_summary.csv", index=False)
    
    # Save detailed trades
    all_trades = []
    for yr in year_results:
        for t in yr.trades:
            all_trades.append({
                "trading_year": yr.trading_start.year,
                "leg_x": t.pair[0],
                "leg_y": t.pair[1],
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_z": t.entry_z,
                "exit_z": t.exit_z,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
            })
    
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv("results/tokat_backtest_trades.csv", index=False)
    
    logger.info(f"\nResults saved to results/tokat_backtest_*.csv")
    
    return year_results, summary_df


if __name__ == "__main__":
    main()
