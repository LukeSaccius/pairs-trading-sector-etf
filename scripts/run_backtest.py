#!/usr/bin/env python
"""
Pairs Trading Backtest Runner

This is the main entry point for running backtests. It loads configuration
from YAML files and uses the pairs_trading_etf library.

Usage:
    python scripts/run_backtest.py --config configs/experiments/default.yaml
    python scripts/run_backtest.py --config configs/experiments/conservative.yaml
    python scripts/run_backtest.py --config configs/experiments/europe_only.yaml
    
    # Override config parameters:
    python scripts/run_backtest.py --config configs/experiments/default.yaml --start 2015 --end 2024

Examples:
    # Run default configuration
    python scripts/run_backtest.py
    
    # Run with specific config
    python scripts/run_backtest.py --config configs/experiments/conservative.yaml
    
    # Quick test on recent years
    python scripts/run_backtest.py --start 2020 --end 2024
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
    load_config,
    run_walkforward_backtest,
    print_backtest_report,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run pairs trading backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/experiments/default.yaml',
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=2010,
        help='Start year for backtest'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=2024,
        help='End year for backtest'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to files'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce logging output'
    )
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    logger.info(f"Loading config from: {config_path}")
    cfg = load_config(str(config_path))
    
    # Load price data
    logger.info(f"Loading price data from: {cfg.price_data_path}")
    prices = pd.read_csv(cfg.price_data_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {prices.shape[1]} ETFs, {prices.shape[0]} trading days")
    
    # Run backtest
    logger.info(f"Running backtest: {args.start} - {args.end}")
    trades, summary = run_walkforward_backtest(
        prices=prices,
        cfg=cfg,
        start_year=args.start,
        end_year=args.end,
    )
    
    # Print report
    print_backtest_report(trades, summary, cfg.experiment_name)
    
    # Save results
    if not args.no_save:
        output_dir = cfg.get_output_path()
        save_results(trades, summary, cfg, output_dir)
    
    # Return PnL for scripting
    total_pnl = sum(t['pnl'] for t in trades) if trades else 0
    return total_pnl


if __name__ == "__main__":
    main()
