"""
Run cross-validated backtest with proper train/validation/test split.

This script properly separates parameter tuning from final evaluation
to get an unbiased estimate of true strategy performance.

Usage:
    python run_cv_backtest.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import pandas as pd
import numpy as np

from src.pairs_trading_etf.backtests.config import BacktestConfig
from src.pairs_trading_etf.backtests.cross_validation import (
    BacktestSplit,
    CVResult,
    run_cross_validated_backtest,
    evaluate_on_test_set,
    select_best_config,
    print_cv_summary,
    save_cv_results,
)
from src.pairs_trading_etf.backtests.engine import run_walkforward_backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_prices(data_path: str) -> pd.DataFrame:
    """Load price data."""
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} tickers")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    return df


def create_v17a_config() -> BacktestConfig:
    """Create V17a best configuration."""
    return BacktestConfig(
        # Core parameters
        use_log_prices=True,
        pvalue_threshold=0.10,
        min_half_life=5,
        max_half_life=25,
        
        # Entry/Exit - higher entry threshold + no stop-loss
        entry_zscore=3.0,  # Higher threshold for stronger signals
        exit_zscore=0.5,   # Slightly wider exit
        stop_loss_zscore=99.0,  # Effectively disable stop-loss
        
        # Position sizing
        use_vol_sizing=True,
        vol_size_min=0.50,
        vol_size_max=2.0,
        target_daily_vol=0.02,
        
        # Pair selection
        min_correlation=0.75,
        max_correlation=0.95,
        sector_focus=True,
        
        # Vidyamurthy filters
        min_snr=1.5,
        min_zero_crossing_rate=5.0,
        
        # Portfolio
        max_positions=8,
        top_pairs=12,
        
        # Capital
        initial_capital=100000,
        
        # Longer holding period
        max_holding_days=90,  # Allow more time for convergence
        dynamic_max_holding=True,
        max_holding_multiplier=5.0,  # 5x half-life before timeout
        
        # DISABLE ROLLING CONSISTENCY to compare
        rolling_consistency=False,
        n_rolling_windows=4,
        min_passing_windows=2,
    )


def run_full_cv_pipeline():
    """Run the complete cross-validation pipeline."""
    print("\n" + "=" * 80)
    print("CROSS-VALIDATED BACKTEST")
    print("=" * 80)
    
    # 1. Define periods
    split = BacktestSplit(
        train_start="2009-01-01",
        train_end="2016-12-31",
        val_start="2017-01-01",
        val_end="2020-12-31",
        test_start="2021-01-01",
        test_end="2024-12-31",
    )
    
    print(f"\nData Split:")
    print(f"  TRAIN: {split.train_start} to {split.train_end} (parameter exploration)")
    print(f"  VALIDATION: {split.val_start} to {split.val_end} (config selection)")
    print(f"  TEST: {split.test_start} to {split.test_end} (final unbiased evaluation)")
    
    # 2. Load data
    data_path = project_root / "data" / "raw" / "etf_prices_fresh.csv"
    prices = load_prices(str(data_path))
    
    # 3. Create V17a config
    config = create_v17a_config()
    
    print(f"\nConfiguration: V17a with rolling consistency")
    print(f"  entry_zscore: {config.entry_zscore}")
    print(f"  vol_size_min: {config.vol_size_min}")
    print(f"  rolling_consistency: {config.rolling_consistency}")
    
    # 4. Run on train period (2010-2016, formation uses 2009)
    print("\n" + "-" * 60)
    print("PHASE 1: TRAIN PERIOD (2009-2016)")
    print("-" * 60)
    
    logger.info(f"Train period: {split.train_start} to {split.train_end}")
    
    train_result = run_walkforward_backtest(
        prices=prices,
        cfg=config,
        start_year=2010,  # Formation year 2009
        end_year=2016,
    )
    
    train_summary = None
    if train_result and len(train_result) >= 2:
        all_trades, yearly_df = train_result
        if len(all_trades) > 0:
            wins = sum(1 for t in all_trades if t.get('pnl', 0) > 0)
            total_pnl = sum(t.get('pnl', 0) for t in all_trades)
            train_summary = {
                'total_pnl': total_pnl,
                'total_trades': len(all_trades),
                'win_rate': wins / len(all_trades) if len(all_trades) > 0 else 0,
            }
            print(f"\nTrain Results:")
            print(f"  Total PnL: ${total_pnl:,.0f}")
            print(f"  Total Trades: {len(all_trades)}")
            print(f"  Win Rate: {wins}/{len(all_trades)} = {train_summary['win_rate']*100:.1f}%")
    
    # 5. Run on validation period (2017-2020, formation uses 2016)
    print("\n" + "-" * 60)
    print("PHASE 2: VALIDATION PERIOD (2017-2020)")
    print("-" * 60)
    
    logger.info(f"Validation period: {split.val_start} to {split.val_end}")
    
    val_result = run_walkforward_backtest(
        prices=prices,
        cfg=config,
        start_year=2017,
        end_year=2020,
    )
    
    val_summary = None
    if val_result and len(val_result) >= 2:
        all_trades, yearly_df = val_result
        if len(all_trades) > 0:
            wins = sum(1 for t in all_trades if t.get('pnl', 0) > 0)
            total_pnl = sum(t.get('pnl', 0) for t in all_trades)
            val_summary = {
                'total_pnl': total_pnl,
                'total_trades': len(all_trades),
                'win_rate': wins / len(all_trades) if len(all_trades) > 0 else 0,
            }
            print(f"\nValidation Results:")
            print(f"  Total PnL: ${total_pnl:,.0f}")
            print(f"  Total Trades: {len(all_trades)}")
            print(f"  Win Rate: {wins}/{len(all_trades)} = {val_summary['win_rate']*100:.1f}%")
    
    # Check for overfitting
    train_pnl = train_summary.get('total_pnl', 0) if train_summary else 0
    val_pnl = val_summary.get('total_pnl', 0) if val_summary else 0
    
    if val_pnl > 0:
        ratio = train_pnl / val_pnl
        print(f"\n  Train/Val PnL Ratio: {ratio:.2f}")
        if ratio > 3.0:
            print("  WARNING: High train/val ratio suggests overfitting!")
    elif val_pnl < 0:
        print("\n  WARNING: Negative validation PnL!")
    
    # 6. Run on TEST period (FINAL EVALUATION)
    print("\n" + "=" * 60)
    print("PHASE 3: TEST PERIOD (2021-2024) - FINAL UNBIASED EVALUATION")
    print("=" * 60)
    print("DO NOT iterate on these results! This is the true performance.")
    
    logger.info(f"Test period: {split.test_start} to {split.test_end}")
    
    test_result = run_walkforward_backtest(
        prices=prices,
        cfg=config,
        start_year=2021,
        end_year=2024,
    )
    
    test_summary = None
    if test_result and len(test_result) >= 2:
        all_trades, yearly_df = test_result
        if len(all_trades) > 0:
            wins = sum(1 for t in all_trades if t.get('pnl', 0) > 0)
            total_pnl = sum(t.get('pnl', 0) for t in all_trades)
            test_summary = {
                'total_pnl': total_pnl,
                'total_trades': len(all_trades),
                'win_rate': wins / len(all_trades) if len(all_trades) > 0 else 0,
            }
            print("\n" + "=" * 60)
            print("FINAL TEST RESULTS (TRUE PERFORMANCE)")
            print("=" * 60)
            print(f"  Total PnL: ${total_pnl:,.0f}")
            print(f"  Total Trades: {len(all_trades)}")
            print(f"  Win Rate: {wins}/{len(all_trades)} = {test_summary['win_rate']*100:.1f}%")
    
    # 7. Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    train_m = train_summary or {'total_pnl': 0, 'win_rate': 0, 'total_trades': 0}
    val_m = val_summary or {'total_pnl': 0, 'win_rate': 0, 'total_trades': 0}
    test_m = test_summary or {'total_pnl': 0, 'win_rate': 0, 'total_trades': 0}
    
    print(f"\n{'Period':<15} {'PnL':>12} {'Win Rate':>10} {'Trades':>8}")
    print("-" * 50)
    print(f"{'Train':<15} ${train_m['total_pnl']:>10,.0f} {train_m['win_rate']*100:>9.1f}% {train_m['total_trades']:>8}")
    print(f"{'Validation':<15} ${val_m['total_pnl']:>10,.0f} {val_m['win_rate']*100:>9.1f}% {val_m['total_trades']:>8}")
    print(f"{'TEST':<15} ${test_m['total_pnl']:>10,.0f} {test_m['win_rate']*100:>9.1f}% {test_m['total_trades']:>8}")
    
    # Save results
    output_dir = project_root / "results" / "experiments" / "cross_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame([
        {'period': 'train', **train_m},
        {'period': 'validation', **val_m},
        {'period': 'test', **test_m},
    ])
    results_df.to_csv(output_dir / "cv_results_v17a.csv", index=False)
    print(f"\nResults saved to {output_dir / 'cv_results_v17a.csv'}")
    
    return train_result, val_result, test_result


if __name__ == "__main__":
    run_full_cv_pipeline()
