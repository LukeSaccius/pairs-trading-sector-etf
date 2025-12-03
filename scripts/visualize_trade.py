"""
Trade Visualization - Google Finance Style
Shows % change from starting point for both ETFs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Configuration - Use relative paths from project root
DATA_PATH = project_root / "data/raw/etf_prices_fresh.csv"
TRADES_PATH = project_root / "results/backtest_v4_trades.csv"
OUTPUT_DIR = project_root / "results/figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load price data and trades"""
    prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    trades = pd.read_csv(TRADES_PATH, parse_dates=['entry_date', 'exit_date'])
    return prices, trades


def visualize_trade(trade_row, prices, context_days=30, save=True):
    """
    Visualize a single trade with 3 panels:
    1. Price % change from ENTRY - shows which is LONG (green) and SHORT (red)
    2. Price Spread (normal, not log) and Z-score
    3. PnL evolution during trade
    """
    etf1 = trade_row['leg_x']
    etf2 = trade_row['leg_y']
    entry_date = pd.to_datetime(trade_row['entry_date'])
    exit_date = pd.to_datetime(trade_row['exit_date'])
    direction = trade_row['direction']
    pnl = trade_row['pnl']
    hedge_ratio = trade_row['hedge_ratio']
    exit_reason = trade_row.get('exit_reason', 'unknown')
    
    # Determine which ETF is LONG and which is SHORT
    is_long_spread = (direction == "LONG" or direction == 1)
    
    # Get date range with context
    start_date = entry_date - pd.Timedelta(days=context_days)
    end_date = exit_date + pd.Timedelta(days=context_days)
    
    # Get price data
    mask = (prices.index >= start_date) & (prices.index <= end_date)
    p1 = prices.loc[mask, etf1].dropna()
    p2 = prices.loc[mask, etf2].dropna()
    
    # Align dates
    common_idx = p1.index.intersection(p2.index)
    p1 = p1.loc[common_idx]
    p2 = p2.loc[common_idx]
    
    if len(p1) < 5:
        print(f"Not enough data for {etf1}/{etf2}")
        return None
    
    # Find entry and exit indices
    entry_idx = common_idx.get_indexer([entry_date], method='nearest')[0]
    exit_idx = common_idx.get_indexer([exit_date], method='nearest')[0]
    
    # Calculate % change from ENTRY date (not window start)
    entry_p1 = p1.iloc[entry_idx]
    entry_p2 = p2.iloc[entry_idx]
    pct_change1 = (p1 / entry_p1 - 1) * 100
    pct_change2 = (p2 / entry_p2 - 1) * 100
    
    # Calculate spreads
    # Normal spread (price ratio)
    price_spread = p1 / p2
    # Log spread
    log_spread = np.log(p1) - hedge_ratio * np.log(p2)
    
    # Calculate z-score with lookback
    lookback = min(60, len(log_spread)//2)
    spread_mean = log_spread.rolling(window=lookback, min_periods=10).mean()
    spread_std = log_spread.rolling(window=lookback, min_periods=10).std()
    zscore = (log_spread - spread_mean) / spread_std
    
    # Calculate PnL evolution during trade
    capital = 10000  # Same as backtest
    notional_x = capital / (1 + abs(hedge_ratio))
    notional_y = abs(hedge_ratio) * notional_x
    
    if is_long_spread:
        qty_x = notional_x / entry_p1  # Long X
        qty_y = -notional_y / entry_p2  # Short Y
    else:
        qty_x = -notional_x / entry_p1  # Short X
        qty_y = notional_y / entry_p2  # Long Y
    
    # PnL at each point during trade
    pnl_x = qty_x * (p1 - entry_p1)
    pnl_y = qty_y * (p2 - entry_p2)
    total_pnl = pnl_x + pnl_y
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[1, 1, 0.8])
    
    # Colors based on position
    if is_long_spread:
        color1, color2 = '#27ae60', '#e74c3c'  # Green for LONG, Red for SHORT
        label1 = f'{etf1} (LONG)'
        label2 = f'{etf2} (SHORT)'
    else:
        color1, color2 = '#e74c3c', '#27ae60'  # Red for SHORT, Green for LONG
        label1 = f'{etf1} (SHORT)'
        label2 = f'{etf2} (LONG)'
    
    # ========== PANEL 1: Price % Change from ENTRY ==========
    ax1 = axes[0]
    
    ax1.plot(pct_change1.index, pct_change1.values, 
             label=label1, color=color1, linewidth=2.5)
    ax1.plot(pct_change2.index, pct_change2.values, 
             label=label2, color=color2, linewidth=2.5)
    
    # Shade the trade period
    ax1.axvspan(entry_date, exit_date, alpha=0.15, 
                color='green' if pnl > 0 else 'red')
    
    # Entry marker - Triangle UP
    ax1.scatter([common_idx[entry_idx]], [0], marker='^', s=200, 
                color='blue', edgecolor='black', linewidth=2, zorder=5)
    ax1.scatter([common_idx[entry_idx]], [0], marker='^', s=200, 
                color='blue', edgecolor='black', linewidth=2, zorder=5)
    
    # Exit marker - Triangle DOWN
    exit_y1 = pct_change1.iloc[exit_idx]
    exit_y2 = pct_change2.iloc[exit_idx]
    ax1.scatter([common_idx[exit_idx]], [exit_y1], marker='v', s=200, 
                color=color1, edgecolor='black', linewidth=2, zorder=5)
    ax1.scatter([common_idx[exit_idx]], [exit_y2], marker='v', s=200, 
                color=color2, edgecolor='black', linewidth=2, zorder=5)
    
    # Zero line (entry reference)
    ax1.axhline(0, color='blue', linestyle='--', alpha=0.7, linewidth=1.5, label='Entry (0%)')
    
    # Annotations
    ax1.annotate('ENTRY', xy=(common_idx[entry_idx], 0.5),
                 fontsize=10, ha='center', fontweight='bold', color='blue')
    ax1.annotate(f'{etf1}: {exit_y1:+.1f}%', 
                 xy=(common_idx[exit_idx], exit_y1 + 0.3),
                 fontsize=9, ha='left', color=color1)
    ax1.annotate(f'{etf2}: {exit_y2:+.1f}%', 
                 xy=(common_idx[exit_idx], exit_y2 - 0.5),
                 fontsize=9, ha='left', color=color2)
    
    ax1.set_ylabel('% Change from Entry', fontsize=12)
    ax1.set_title(f'Price Movement from Entry Point', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ========== PANEL 2: Z-Score Evolution ==========
    ax2 = axes[1]
    
    ax2.plot(zscore.index, zscore.values, color='#2c3e50', linewidth=2, label='Z-Score')
    
    # Entry/Exit z-score bands
    ax2.axhline(2, color='red', linestyle='--', alpha=0.7, label='Entry Short (z=+2)')
    ax2.axhline(-2, color='green', linestyle='--', alpha=0.7, label='Entry Long (z=-2)')
    ax2.axhline(0.5, color='orange', linestyle=':', alpha=0.7, label='Exit (z=±0.5)')
    ax2.axhline(-0.5, color='orange', linestyle=':', alpha=0.7)
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
    
    # Shade trade period
    ax2.axvspan(entry_date, exit_date, alpha=0.15, 
                color='green' if pnl > 0 else 'red')
    
    # Entry/Exit markers
    entry_z = zscore.iloc[entry_idx] if not np.isnan(zscore.iloc[entry_idx]) else trade_row['entry_z']
    exit_z = zscore.iloc[exit_idx] if not np.isnan(zscore.iloc[exit_idx]) else trade_row['exit_z']
    
    ax2.scatter([common_idx[entry_idx]], [entry_z], marker='^', s=200, 
                color='blue', edgecolor='black', linewidth=2, zorder=5)
    ax2.scatter([common_idx[exit_idx]], [exit_z], marker='v', s=200, 
                color='purple', edgecolor='black', linewidth=2, zorder=5)
    
    ax2.annotate(f'Entry z={entry_z:.2f}', xy=(common_idx[entry_idx], entry_z + 0.3),
                 fontsize=9, ha='center', color='blue')
    ax2.annotate(f'Exit z={exit_z:.2f}', xy=(common_idx[exit_idx], exit_z - 0.4),
                 fontsize=9, ha='center', color='purple')
    
    ax2.set_ylabel('Z-Score', fontsize=12)
    ax2.set_title(f'Spread Z-Score (log spread)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-4, 4)
    
    # ========== PANEL 3: PnL Evolution ==========
    ax3 = axes[2]
    
    # Only show PnL during and after trade period
    trade_mask = common_idx >= entry_date
    pnl_during = total_pnl[trade_mask]
    
    ax3.fill_between(pnl_during.index, 0, pnl_during.values, 
                     where=pnl_during.values >= 0, 
                     color='green', alpha=0.3, interpolate=True)
    ax3.fill_between(pnl_during.index, 0, pnl_during.values, 
                     where=pnl_during.values < 0, 
                     color='red', alpha=0.3, interpolate=True)
    ax3.plot(pnl_during.index, pnl_during.values, color='black', linewidth=2)
    
    # Mark entry (0) and exit
    ax3.scatter([common_idx[entry_idx]], [0], marker='^', s=200, 
                color='blue', edgecolor='black', linewidth=2, zorder=5)
    exit_pnl = total_pnl.iloc[exit_idx]
    ax3.scatter([common_idx[exit_idx]], [exit_pnl], marker='v', s=200, 
                color='purple', edgecolor='black', linewidth=2, zorder=5)
    
    # Shade trade period
    ax3.axvspan(entry_date, exit_date, alpha=0.15, 
                color='green' if pnl > 0 else 'red')
    ax3.axhline(0, color='gray', linestyle='-', alpha=0.5)
    
    # Calculate PnL as % of capital (before costs)
    pnl_pct_before_cost = (exit_pnl / capital) * 100
    # PnL from CSV is after costs
    pnl_pct_after_cost = (pnl / capital) * 100
    ax3.annotate(f'Before costs: ${exit_pnl:+.0f} ({pnl_pct_before_cost:+.2f}%)\n'
                 f'After costs: ${pnl:+.0f} ({pnl_pct_after_cost:+.2f}%)', 
                 xy=(common_idx[exit_idx], exit_pnl),
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=10, fontweight='bold', color='purple',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_ylabel('PnL ($)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title(f'Cumulative PnL (Capital: ${capital:,})', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Date formatting for all axes
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Main title
    holding_days = (exit_date - entry_date).days
    result_text = "WIN" if pnl > 0 else "LOSS"
    result_color = 'green' if pnl > 0 else 'red'
    pnl_pct_final = (pnl / capital) * 100
    
    fig.suptitle(
        f'{result_text} | {etf1}-{etf2} | '
        f'{entry_date.strftime("%Y-%m-%d")} → {exit_date.strftime("%Y-%m-%d")} ({holding_days}d) | '
        f'PnL: ${pnl:+,.0f} ({pnl_pct_final:+.2f}%) | Exit: {exit_reason}',
        fontsize=14, fontweight='bold', color=result_color, y=0.99
    )
    
    plt.tight_layout()
    
    if save:
        result_str = "WIN" if pnl > 0 else "LOSS"
        filename = f"trade_{result_str}_{etf1}_{etf2}_{entry_date.strftime('%Y%m%d')}.png"
        filepath = OUTPUT_DIR / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Saved: {filepath}")
    
    plt.show()
    return fig


def show_best_and_worst_trades(n=3):
    """Show n best and n worst trades"""
    prices, trades = load_data()
    
    # Sort by PnL
    sorted_trades = trades.sort_values('pnl', ascending=False)
    
    print("\n" + "="*60)
    print("TOP WINNING TRADES")
    print("="*60)
    
    for i, (idx, trade) in enumerate(sorted_trades.head(n).iterrows()):
        print(f"\n--- Win #{i+1}: {trade['leg_x']}/{trade['leg_y']} | PnL: ${trade['pnl']:.2f} ---")
        visualize_trade(trade, prices)
    
    print("\n" + "="*60)
    print("TOP LOSING TRADES")
    print("="*60)
    
    for i, (idx, trade) in enumerate(sorted_trades.tail(n).iterrows()):
        print(f"\n--- Loss #{i+1}: {trade['leg_x']}/{trade['leg_y']} | PnL: ${trade['pnl']:.2f} ---")
        visualize_trade(trade, prices)


def show_specific_trade(etf1, etf2, entry_date_str=None):
    """Show a specific trade by ETF pair and optional date"""
    prices, trades = load_data()
    
    mask = ((trades['leg_x'] == etf1) & (trades['leg_y'] == etf2)) | \
           ((trades['leg_x'] == etf2) & (trades['leg_y'] == etf1))
    
    if entry_date_str:
        entry_date = pd.to_datetime(entry_date_str)
        mask = mask & (trades['entry_date'] == entry_date)
    
    matching = trades[mask]
    
    if len(matching) == 0:
        print(f"No trades found for {etf1}/{etf2}")
        return
    
    for idx, trade in matching.iterrows():
        print(f"\nTrade: {trade['leg_x']}/{trade['leg_y']} | "
              f"Entry: {trade['entry_date']} | PnL: ${trade['pnl']:.2f}")
        visualize_trade(trade, prices)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize pair trades')
    parser.add_argument('--best-worst', '-bw', type=int, default=None,
                        help='Show N best and N worst trades')
    parser.add_argument('--pair', '-p', nargs=2, default=None,
                        help='Show specific pair (e.g., -p XLF XLI)')
    parser.add_argument('--date', '-d', default=None,
                        help='Entry date for specific trade (YYYY-MM-DD)')
    parser.add_argument('--index', '-i', type=int, default=None,
                        help='Show trade at specific index in trades file')
    
    args = parser.parse_args()
    
    if args.best_worst:
        show_best_and_worst_trades(args.best_worst)
    elif args.pair:
        show_specific_trade(args.pair[0], args.pair[1], args.date)
    elif args.index is not None:
        prices, trades = load_data()
        if 0 <= args.index < len(trades):
            visualize_trade(trades.iloc[args.index], prices)
        else:
            print(f"Index out of range. Total trades: {len(trades)}")
    else:
        # Default: show first profitable trade
        prices, trades = load_data()
        profitable = trades[trades['pnl'] > 0]
        if len(profitable) > 0:
            print("Showing first profitable trade. Use --help for options.")
            visualize_trade(profitable.iloc[0], prices)
        else:
            visualize_trade(trades.iloc[0], prices)
