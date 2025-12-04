"""
Debug visualization - show ALL trades to diagnose issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

DATA_PATH = project_root / "data/raw/etf_prices_fresh.csv"
OUTPUT_DIR = project_root / "results/figures/debug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def visualize_all_trades(trades_df, prices, year=None, max_trades=50):
    """Visualize all trades in a compact grid"""
    
    if year:
        trades_df = trades_df[pd.to_datetime(trades_df['entry_date']).dt.year == year]
    
    n_trades = min(len(trades_df), max_trades)
    if n_trades == 0:
        print(f"No trades found for year {year}")
        return
    
    # Calculate grid dimensions
    n_cols = 4
    n_rows = (n_trades + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()
    
    for i, (_, trade) in enumerate(trades_df.head(max_trades).iterrows()):
        ax = axes[i]
        
        etf1 = trade['leg_x']
        etf2 = trade['leg_y']
        entry_date = pd.to_datetime(trade['entry_date'])
        exit_date = pd.to_datetime(trade['exit_date'])
        direction = trade['direction']
        pnl = trade['pnl']
        hedge_ratio = trade.get('hedge_ratio', 1.0)
        exit_reason = trade.get('exit_reason', 'unknown')
        
        # Get price data
        start_date = entry_date - pd.Timedelta(days=10)
        end_date = exit_date + pd.Timedelta(days=10)
        
        mask = (prices.index >= start_date) & (prices.index <= end_date)
        p1 = prices.loc[mask, etf1].dropna() if etf1 in prices.columns else pd.Series()
        p2 = prices.loc[mask, etf2].dropna() if etf2 in prices.columns else pd.Series()
        
        if len(p1) < 5 or len(p2) < 5:
            ax.text(0.5, 0.5, f'No data\n{etf1}/{etf2}', ha='center', va='center')
            ax.set_title(f'#{i+1} - NO DATA')
            continue
        
        # Align
        common_idx = p1.index.intersection(p2.index)
        p1 = p1.loc[common_idx]
        p2 = p2.loc[common_idx]
        
        # Find entry point
        entry_idx = common_idx.get_indexer([entry_date], method='nearest')[0]
        exit_idx = common_idx.get_indexer([exit_date], method='nearest')[0]
        
        # % change from entry
        entry_p1 = p1.iloc[entry_idx]
        entry_p2 = p2.iloc[entry_idx]
        pct1 = (p1 / entry_p1 - 1) * 100
        pct2 = (p2 / entry_p2 - 1) * 100
        
        # Plot
        is_long = direction == 'LONG' or direction == 1
        c1 = 'green' if is_long else 'red'
        c2 = 'red' if is_long else 'green'
        
        ax.plot(pct1.values, color=c1, linewidth=1.5, label=f'{etf1}')
        ax.plot(pct2.values, color=c2, linewidth=1.5, label=f'{etf2}')
        
        # Entry/Exit lines
        ax.axvline(entry_idx, color='blue', linestyle='--', alpha=0.7)
        ax.axvline(exit_idx, color='purple', linestyle='--', alpha=0.7)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        
        # Shade trade period
        ax.axvspan(entry_idx, exit_idx, alpha=0.2, color='green' if pnl > 0 else 'red')
        
        # Title with key info
        title_color = 'green' if pnl > 0 else 'red'
        ax.set_title(f'{etf1}/{etf2}\nPnL: ${pnl:.0f} | {exit_reason}\nHR: {hedge_ratio:.2f} | {direction}',
                    fontsize=9, color=title_color)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Mark final % change at exit
        exit_pct1 = pct1.iloc[exit_idx]
        exit_pct2 = pct2.iloc[exit_idx]
        ax.annotate(f'{exit_pct1:+.1f}%', xy=(exit_idx, exit_pct1), fontsize=7, color=c1)
        ax.annotate(f'{exit_pct2:+.1f}%', xy=(exit_idx, exit_pct2), fontsize=7, color=c2)
    
    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    title = f'All Trades {year}' if year else 'All Trades'
    fig.suptitle(title, fontsize=16, y=1.02)
    
    save_path = OUTPUT_DIR / f'all_trades_{year if year else "all"}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def analyze_trade_logic(trades_df, prices):
    """Deep analysis of trade logic"""
    
    print("\n" + "="*80)
    print("TRADE LOGIC ANALYSIS")
    print("="*80)
    
    issues = []
    
    for i, trade in trades_df.iterrows():
        etf1 = trade['leg_x']
        etf2 = trade['leg_y']
        entry_date = pd.to_datetime(trade['entry_date'])
        exit_date = pd.to_datetime(trade['exit_date'])
        direction = trade['direction']
        pnl = trade['pnl']
        hedge_ratio = trade.get('hedge_ratio', 1.0)
        entry_z = trade.get('entry_z', 0)
        exit_z = trade.get('exit_z', 0)
        
        # Get prices at entry and exit
        if etf1 not in prices.columns or etf2 not in prices.columns:
            continue
            
        try:
            entry_p1 = prices.loc[entry_date:, etf1].iloc[0]
            entry_p2 = prices.loc[entry_date:, etf2].iloc[0]
            exit_p1 = prices.loc[exit_date:, etf1].iloc[0]
            exit_p2 = prices.loc[exit_date:, etf2].iloc[0]
        except:
            continue
        
        # Calculate expected PnL
        is_long = direction == 'LONG' or direction == 1
        
        capital = 10000
        notional_x = capital / (1 + abs(hedge_ratio))
        notional_y = abs(hedge_ratio) * notional_x
        
        if is_long:  # Long spread = Long X, Short Y
            qty_x = notional_x / entry_p1
            qty_y = -notional_y / entry_p2
        else:  # Short spread = Short X, Long Y
            qty_x = -notional_x / entry_p1
            qty_y = notional_y / entry_p2
        
        calc_pnl_x = qty_x * (exit_p1 - entry_p1)
        calc_pnl_y = qty_y * (exit_p2 - entry_p2)
        calc_pnl = calc_pnl_x + calc_pnl_y
        
        # Price changes
        pct1 = (exit_p1 / entry_p1 - 1) * 100
        pct2 = (exit_p2 / entry_p2 - 1) * 100
        
        # Logic check: For LONG spread (long X, short Y)
        # We profit if X outperforms Y
        # For SHORT spread (short X, long Y)
        # We profit if Y outperforms X
        
        if is_long:
            expected_profit = pct1 > pct2  # X should outperform Y
        else:
            expected_profit = pct2 > pct1  # Y should outperform X
        
        actual_profit = pnl > 0
        
        if expected_profit != actual_profit and abs(pnl) > 50:
            issues.append({
                'pair': f"{etf1}/{etf2}",
                'direction': direction,
                'pnl': pnl,
                'calc_pnl': calc_pnl,
                'pct1': pct1,
                'pct2': pct2,
                'expected_profit': expected_profit,
                'actual_profit': actual_profit,
                'entry_date': entry_date,
            })
    
    if issues:
        print(f"\n⚠️  FOUND {len(issues)} POTENTIAL LOGIC ISSUES:\n")
        for issue in issues[:10]:
            print(f"  {issue['pair']} ({issue['direction']})")
            print(f"    PnL: ${issue['pnl']:.2f} | Calc PnL: ${issue['calc_pnl']:.2f}")
            print(f"    Leg X: {issue['pct1']:+.2f}% | Leg Y: {issue['pct2']:+.2f}%")
            print(f"    Expected profit: {issue['expected_profit']} | Actual: {issue['actual_profit']}")
            print()
    else:
        print("✅ No obvious logic issues found")
    
    return issues


def main():
    # Load data
    print("Loading data...")
    prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    # Load latest trades
    trades_path = project_root / "results/2025-12-03_01-27_v9_compounding/trades.csv"
    if not trades_path.exists():
        # Try finding the most recent
        results_dir = project_root / "results"
        trade_files = list(results_dir.glob("*/trades.csv"))
        if trade_files:
            trades_path = sorted(trade_files)[-1]
        else:
            trades_path = project_root / "results/backtest_v4_trades.csv"
    
    print(f"Loading trades from: {trades_path}")
    trades = pd.read_csv(trades_path, parse_dates=['entry_date', 'exit_date'])
    
    print(f"\nTotal trades: {len(trades)}")
    print(f"Years: {pd.to_datetime(trades['entry_date']).dt.year.unique()}")
    
    # Analyze logic
    issues = analyze_trade_logic(trades, prices)
    
    # Summary by year
    print("\n" + "="*80)
    print("SUMMARY BY YEAR")
    print("="*80)
    
    trades['year'] = pd.to_datetime(trades['entry_date']).dt.year
    for year in sorted(trades['year'].unique()):
        year_trades = trades[trades['year'] == year]
        n = len(year_trades)
        wins = (year_trades['pnl'] > 0).sum()
        pnl = year_trades['pnl'].sum()
        print(f"{year}: {n:3d} trades, {wins:3d} wins ({100*wins/n if n > 0 else 0:.0f}%), PnL: ${pnl:,.0f}")
    
    # Visualize each year
    years_with_trades = sorted(trades['year'].unique())
    for year in years_with_trades:
        if (trades['year'] == year).sum() > 0:
            visualize_all_trades(trades, prices, year=year)
    
    # Also visualize all together
    visualize_all_trades(trades, prices, year=None)
    
    print(f"\n✅ Visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
