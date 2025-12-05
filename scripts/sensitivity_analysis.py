"""Sensitivity Analysis: Test strategy across different time periods.

This script measures how sensitive the backtest results are to:
1. Different time periods (including/excluding crisis)
2. Different parameter configurations

Purpose: Control for data snooping and assess robustness of findings.

Usage:
    python scripts/sensitivity_analysis.py
    python scripts/sensitivity_analysis.py --config configs/experiments/conservative.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

try:
    from pairs_trading_etf.backtests import (
        BacktestConfig,
        run_walkforward_backtest,
        load_config,
    )
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(SRC_PATH))
    from pairs_trading_etf.backtests import (  # type: ignore[no-redef]
        BacktestConfig,
        run_walkforward_backtest,
        load_config,
    )

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_period_analysis(
    prices: pd.DataFrame,
    periods: list[tuple[str, int, int]],
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """Run backtest across multiple time periods."""
    
    results = []
    for name, start, end in periods:
        print(f"\n📊 Testing: {name} ({start}-{end})...")
        
        trades, summary = run_walkforward_backtest(prices, cfg, start, end)
        
        if summary.empty:
            print("  ⚠️ No trades")
            continue
        
        total_pnl = summary['total_pnl'].sum()
        total_trades = summary['total_trades'].sum()
        avg_win_rate = summary['win_rate'].mean()
        profitable_years = (summary['total_pnl'] > 0).sum()
        total_years = len(summary)
        
        capital = cfg.capital_per_pair * cfg.max_positions
        avg_return = (total_pnl / capital / total_years) * 100 if total_years > 0 else 0
        
        results.append({
            'Period': name,
            'Years': f"{start}-{end}",
            'Avg Return %': round(avg_return, 2),
            'Total PnL': round(total_pnl, 2),
            'Trades': total_trades,
            'Win Rate %': round(avg_win_rate, 1),
            'Profitable Years': f"{profitable_years}/{total_years}",
        })
        
        print(f"  Avg Return: {avg_return:.2f}%")
        print(f"  Total PnL: ${total_pnl:,.0f}")
        print(f"  Win Rate: {avg_win_rate:.1f}%")
    
    return pd.DataFrame(results)


def run_regime_analysis(
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    start: int = 2008,
    end: int = 2021,
) -> None:
    """Compare performance in crisis vs non-crisis periods."""
    
    print("\n" + "="*60)
    print("REGIME ANALYSIS (Crisis vs Normal Markets)")
    print("="*60)
    
    _, summary = run_walkforward_backtest(prices, cfg, start, end)
    
    if summary.empty:
        print("No data for regime analysis")
        return
    
    capital = cfg.capital_per_pair * cfg.max_positions
    summary['return_pct'] = summary['total_pnl'] / capital * 100
    
    # Define regimes
    crisis_years = [2008, 2009, 2020]  # Financial crisis + COVID
    
    crisis = summary[summary['year'].isin(crisis_years)]
    normal = summary[~summary['year'].isin(crisis_years)]
    
    if not crisis.empty:
        print(f"\n🔴 Crisis Years ({crisis_years}):")
        print(f"   Avg Return: {crisis['return_pct'].mean():.2f}%")
        print(f"   Avg Win Rate: {crisis['win_rate'].mean():.1f}%")
        print(f"   Total PnL: ${crisis['total_pnl'].sum():,.0f}")
    
    if not normal.empty:
        print("\n🟢 Normal Years:")
        print(f"   Avg Return: {normal['return_pct'].mean():.2f}%")
        print(f"   Avg Win Rate: {normal['win_rate'].mean():.1f}%")
        print(f"   Total PnL: ${normal['total_pnl'].sum():,.0f}")
    
    # Key finding
    print("\n" + "-"*40)
    if not crisis.empty and not normal.empty:
        if crisis['return_pct'].mean() > normal['return_pct'].mean():
            print("📈 Strategy performs BETTER in crisis periods")
            print("   → Mean-reversion benefits from high volatility")
        else:
            print("📉 Strategy performs BETTER in normal periods")


def main():
    parser = argparse.ArgumentParser(description='Sensitivity Analysis')
    parser.add_argument('--config', '-c', type=str, help='Path to YAML config')
    parser.add_argument('--data', type=str, default='data/raw/etf_prices_fresh.csv',
                        help='Path to price data')
    args = parser.parse_args()
    
    print("="*60)
    print("SENSITIVITY ANALYSIS: Strategy Robustness")
    print("="*60)
    
    # Load data
    prices = pd.read_csv(args.data, index_col=0, parse_dates=True)
    print(f"\n📅 Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"📊 ETFs: {len(prices.columns)}")
    
    # Load config
    if args.config:
        cfg = load_config(args.config)
        print(f"⚙️ Config: {args.config}")
    else:
        cfg = BacktestConfig(
            experiment_name="sensitivity_analysis",
            pvalue_threshold=0.05,
            min_half_life=5.0,
            max_half_life=30.0,
            entry_zscore=2.0,
            exit_zscore=0.5,
            stop_loss_zscore=4.0,
            max_holding_days=45,
            sector_focus=True,
            exclude_sectors=('EMERGING', 'BONDS_GOV', 'US_GROWTH', 'INDUSTRIALS', 'HEALTHCARE'),
        )
        print("⚙️ Config: v4 defaults")
    
    # Define test periods
    periods = [
        ("Full Period (2010-2024)", 2010, 2024),
        ("Pre-COVID (2010-2019)", 2010, 2019),
        ("Post-COVID (2020-2024)", 2020, 2024),
        ("Bull Market (2012-2019)", 2012, 2019),
        ("Recent (2021-2024)", 2021, 2024),
    ]
    
    # Run period analysis
    print("\n" + "="*60)
    print("PERIOD ANALYSIS")
    print("="*60)
    
    results_df = run_period_analysis(prices, periods, cfg)
    
    # Summary table
    if not results_df.empty:
        print("\n" + "="*60)
        print("SUMMARY TABLE")
        print("="*60)
        print(results_df.to_string(index=False))
    
    # Regime analysis
    run_regime_analysis(prices, cfg, 2010, 2024)
    
    # Save results
    output_path = PROJECT_ROOT / "results" / "sensitivity_analysis.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n\n✅ Results saved to: {output_path}")
    
    return results_df


if __name__ == "__main__":
    main()
