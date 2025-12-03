#!/usr/bin/env python
"""
Pairs Trading Walk-Forward Backtest v4

This is a lightweight wrapper around the pairs_trading_etf library.
For new experiments, use run_backtest.py with YAML configs instead.

Usage:
    python scripts/backtest_v4.py
    python scripts/backtest_v4.py --start 2015 --end 2024
    python scripts/backtest_v4.py --no-sector
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pairs_trading_etf.backtests import (
    BacktestConfig,
    run_walkforward_backtest,
    print_backtest_report,
    save_results,
)
from pairs_trading_etf.utils.sectors import DEFAULT_EXCLUDED_SECTORS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Pairs Trading Backtest v4')
    parser.add_argument('--start', type=int, default=2010, help='Start year')
    parser.add_argument('--end', type=int, default=2024, help='End year')
    parser.add_argument('--no-sector', action='store_true', help='Disable sector focus')
    parser.add_argument('--no-dynamic', action='store_true', help='Disable dynamic hedge')
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading price data...")
    prices = pd.read_csv('data/raw/etf_prices_fresh.csv', index_col=0, parse_dates=True)
    logger.info(f"Loaded {prices.shape[1]} ETFs, {prices.shape[0]} days")
    
    # Config - v4 optimized settings
    cfg = BacktestConfig(
        experiment_name="v4_backtest",
        description="V4 backtest with sector focus and dynamic hedge",
        
        # Cointegration
        pvalue_threshold=0.05,
        min_half_life=5.0,
        max_half_life=30.0,
        use_log_prices=True,
        
        # Correlation
        min_correlation=0.75,
        max_correlation=0.95,
        
        # Sector focus
        sector_focus=not args.no_sector,
        exclude_sectors=('EMERGING', 'BONDS_GOV', 'US_GROWTH', 'INDUSTRIALS', 'HEALTHCARE'),
        
        # Trading
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_loss_zscore=4.0,
        max_holding_days=45,
        
        # Position management
        capital_per_pair=10000.0,
        max_positions=10,
        dynamic_hedge=not args.no_dynamic,
        
        # Output
        timestamped_output=False,  # Use fixed output for v4
    )
    
    # Run backtest
    trades, summary = run_walkforward_backtest(prices, cfg, args.start, args.end)
    
    # Print report
    print_backtest_report(trades, summary, "v4 - Sector Focus + Dynamic Hedge")
    
    # Save results (fixed paths for backward compatibility)
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['pair'] = trades_df['pair'].apply(lambda x: f"{x[0]}_{x[1]}")
        trades_df.to_csv('results/backtest_v4_trades.csv', index=False)
        summary.to_csv('results/backtest_v4_summary.csv', index=False)
        logger.info("\nResults saved to results/backtest_v4_*.csv")


if __name__ == "__main__":
    main()
