#!/usr/bin/env python
"""
Sensitivity Analysis: Entry Threshold and Position Sizing
Test impact of different entry_zscore values and position sizes on performance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from itertools import product
import copy

from pairs_trading_etf.backtests import (
    run_walkforward_backtest,
    load_config,
)


def run_sensitivity_analysis():
    """Run sensitivity analysis on entry_zscore and capital_per_pair."""
    
    # Load base config (V15b)
    base_cfg = load_config("configs/experiments/v15b_vix_volsizing.yaml")
    
    # Load price data
    prices = pd.read_csv(base_cfg.price_data_path, index_col=0, parse_dates=True)
    
    # Test parameters
    entry_zscores = [1.5, 2.0, 2.5, 2.8, 3.0]
    capital_per_pairs = [10000, 15000, 20000]
    max_positions_list = [5, 8, 10, 15]
    
    # Results storage
    all_results = []
    
    print("="*80)
    print("SENSITIVITY ANALYSIS: Entry Z-Score, Position Size, Max Positions")
    print("="*80)
    print(f"\nEntry Z-Scores: {entry_zscores}")
    print(f"Capital per Pair: {capital_per_pairs}")
    print(f"Max Positions: {max_positions_list}")
    print(f"\nTotal combinations: {len(entry_zscores) * len(capital_per_pairs) * len(max_positions_list)}")
    print("\n" + "-"*80)
    
    # Baseline (V15b settings)
    print("\nRunning BASELINE (V15b settings)...")
    base_trades, base_summary = run_walkforward_backtest(
        prices=prices,
        cfg=base_cfg,
        start_year=2009,
        end_year=2024
    )
    baseline_pnl = base_summary['total_pnl'].sum()
    baseline_trades = base_summary['total_trades'].sum()
    baseline_winrate = base_summary['win_rate'].mean()
    
    print(f"Baseline: PnL=${baseline_pnl:,.0f}, Trades={baseline_trades}, WinRate={baseline_winrate:.1f}%")
    
    # Test combinations
    for entry_z, capital, max_pos in product(entry_zscores, capital_per_pairs, max_positions_list):
        cfg = copy.deepcopy(base_cfg)
        cfg.entry_zscore = entry_z
        cfg.capital_per_pair = capital
        cfg.max_positions = max_pos
        cfg.timestamped_output = False  # Don't save each run
        
        print(f"\nTesting: entry_z={entry_z}, capital={capital}, max_pos={max_pos}...", end=" ")
        
        try:
            trades_list, summary = run_walkforward_backtest(
                prices=prices,
                cfg=cfg,
                start_year=2009,
                end_year=2024
            )
            
            if summary.empty or len(trades_list) == 0:
                print("No trades")
                continue
            
            # Convert trades list to DataFrame
            trades_df = pd.DataFrame(trades_list)
                
            total_pnl = summary['total_pnl'].sum()
            total_trades = summary['total_trades'].sum()
            win_rate = summary['win_rate'].mean()
            
            # Calculate profit factor from trades
            wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = wins / losses if losses > 0 else 999
            
            # Calculate max drawdown
            cumulative = trades_df['pnl'].cumsum()
            running_max = cumulative.cummax()
            drawdown = running_max - cumulative
            max_dd = drawdown.max()
            
            all_results.append({
                'entry_zscore': entry_z,
                'capital_per_pair': capital,
                'max_positions': max_pos,
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_dd,
                'pnl_per_trade': total_pnl / max(total_trades, 1),
                'annualized_return': ((1 + total_pnl/50000) ** (1/14.26) - 1) * 100
            })
            
            print(f"PnL=${total_pnl:,.0f}, Trades={total_trades}, WR={win_rate:.1f}%")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    # Create results DataFrame
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No results to analyze!")
        return
    
    # Sort by PnL
    df_sorted = df.sort_values('total_pnl', ascending=False)
    
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS BY TOTAL PnL")
    print("="*80)
    print(df_sorted.head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS BY PROFIT FACTOR")
    print("="*80)
    print(df.sort_values('profit_factor', ascending=False).head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("AVERAGE METRICS BY ENTRY Z-SCORE")
    print("="*80)
    print(df.groupby('entry_zscore').agg({
        'total_pnl': 'mean',
        'total_trades': 'mean',
        'win_rate': 'mean',
        'profit_factor': 'mean'
    }).round(2).to_string())
    
    print("\n" + "="*80)
    print("AVERAGE METRICS BY CAPITAL PER PAIR")
    print("="*80)
    print(df.groupby('capital_per_pair').agg({
        'total_pnl': 'mean',
        'total_trades': 'mean',
        'win_rate': 'mean',
        'profit_factor': 'mean'
    }).round(2).to_string())
    
    print("\n" + "="*80)
    print("AVERAGE METRICS BY MAX POSITIONS")
    print("="*80)
    print(df.groupby('max_positions').agg({
        'total_pnl': 'mean',
        'total_trades': 'mean',
        'win_rate': 'mean',
        'profit_factor': 'mean'
    }).round(2).to_string())
    
    # Save results
    df.to_csv("results/sensitivity_entry_position.csv", index=False)
    print("\nResults saved to: results/sensitivity_entry_position.csv")
    
    # Best configuration
    best = df_sorted.iloc[0]
    print("\n" + "="*80)
    print("RECOMMENDED CONFIGURATION")
    print("="*80)
    print(f"Entry Z-Score: {best['entry_zscore']}")
    print(f"Capital per Pair: ${best['capital_per_pair']:,.0f}")
    print(f"Max Positions: {best['max_positions']}")
    print(f"Expected PnL: ${best['total_pnl']:,.0f}")
    print(f"Expected Trades: {best['total_trades']:.0f}")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"Profit Factor: {best['profit_factor']:.2f}")
    print(f"Annualized Return: {best['annualized_return']:.2f}%")


if __name__ == "__main__":
    run_sensitivity_analysis()
