#!/usr/bin/env python
"""
Pairs Trading Backtest Runner with Mandatory CPCV Validation

This is the main entry point for running backtests. It includes:
- Walk-forward backtest execution
- MANDATORY CPCV overfitting detection
- Validation against safety thresholds
- Comprehensive reporting

Usage:
    # Standard run with CPCV validation (RECOMMENDED)
    python scripts/run_backtest.py --config configs/experiments/default.yaml
    
    # Quick run without CPCV (for debugging only)
    python scripts/run_backtest.py --config configs/experiments/default.yaml --no-cpcv
    
    # Custom date range
    python scripts/run_backtest.py --config configs/experiments/default.yaml --start 2015 --end 2024

Examples:
    # Run with full validation
    python scripts/run_backtest.py
    
    # Quick debug run
    python scripts/run_backtest.py --no-cpcv --no-save
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

try:
    from pairs_trading_etf.backtests import (
        load_config,
        run_walkforward_backtest,
        print_backtest_report,
        save_results,
        PipelineConfig,
        run_validated_backtest,
    )
except ModuleNotFoundError:  # pragma: no cover - environment-specific path fix
    project_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root))
    sys.path.append(str(project_root / "src"))
    from pairs_trading_etf.backtests import (  # type: ignore[no-redef]
        load_config,
        run_walkforward_backtest,
        print_backtest_report,
        save_results,
        PipelineConfig,
        run_validated_backtest,
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run pairs trading backtest with CPCV validation',
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
        '--no-cpcv',
        action='store_true',
        help='Skip CPCV validation (NOT RECOMMENDED - for debugging only)'
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
    parser.add_argument(
        '--max-pbo',
        type=float,
        default=0.40,
        help='Maximum acceptable PBO (default: 0.40)'
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
    
    # Run with or without CPCV
    if args.no_cpcv:
        # Legacy mode: simple backtest without validation
        logger.warning("=" * 60)
        logger.warning("CPCV VALIDATION DISABLED - RESULTS MAY BE OVERFIT")
        logger.warning("=" * 60)
        
        logger.info(f"Running backtest: {args.start} - {args.end}")
        trades, summary_df = run_walkforward_backtest(
            prices=prices,
            cfg=cfg,
            start_year=args.start,
            end_year=args.end,
        )
        
        if not summary_df.empty:
            print_backtest_report(trades, summary_df, cfg.experiment_name)
            
            if not args.no_save:
                output_dir = cfg.get_output_path()
                save_results(trades, summary_df, cfg, output_dir)
            
            total_pnl = float(summary_df['total_pnl'].sum())
        else:
            logger.warning("No trades generated for the requested period.")
            total_pnl = 0.0
    else:
        # RECOMMENDED: Run with CPCV validation
        pipeline_config = PipelineConfig(
            run_cpcv=True,
            max_pbo=args.max_pbo,
            save_results=not args.no_save,
            output_dir=cfg.output_dir,
        )
        
        result = run_validated_backtest(
            prices=prices,
            config=cfg,
            pipeline_config=pipeline_config,
            start_year=args.start,
            end_year=args.end,
            verbose=not args.quiet,
        )
        
        # Print summary (handle consoles without UTF-8)
        summary_str = result.summary()
        try:
            print(summary_str)
        except UnicodeEncodeError:
            sys.stdout.buffer.write((summary_str + "\n").encode("utf-8", errors="replace"))
        
        # Return status based on validation
        if result.is_valid:
            logger.info("Strategy PASSED validation")
            total_pnl = result.total_pnl
        else:
            logger.warning("Strategy FAILED validation")
            total_pnl = 0  # Don't report PnL for failed strategies
            sys.exit(1)
    
    return total_pnl


if __name__ == "__main__":
    main()
