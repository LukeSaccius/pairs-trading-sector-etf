"""
Run CSCV-Integrated Backtest.

This script runs the full 3-phase backtest:
1. TRAIN: Grid search over parameter configurations (2009-2016)
2. VALIDATION: CSCV analysis to detect overfitting (2017-2020)
3. TEST: Final unbiased evaluation (2021-2024)

Usage:
    python scripts/run_cscv_backtest.py

Output:
    - CSCV analysis results
    - Best configuration recommendation
    - Final test performance (if not overfit)
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import yaml

from src.pairs_trading_etf.backtests import (
    BacktestConfig,
    run_cscv_backtest,
    CSCVBacktestSplit,
    ParameterGrid,
)


def main():
    """Run CSCV-integrated backtest."""
    print("=" * 70)
    print("CSCV-INTEGRATED BACKTEST")
    print("=" * 70)
    
    # Load data
    data_path = project_root / "data" / "raw" / "etf_prices_fresh.csv"
    print(f"\nLoading data from: {data_path}")
    
    prices = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"  Shape: {prices.shape}")
    print(f"  Date range: {prices.index.min()} to {prices.index.max()}")
    
    # Load base config
    config_path = project_root / "configs" / "experiments" / "v16_optimized.yaml"
    print(f"\nLoading base config from: {config_path}")
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    base_config = BacktestConfig(**config_dict)
    
    # Define split
    split = CSCVBacktestSplit(
        train_start=2009,
        train_end=2016,
        val_start=2017,
        val_end=2020,
        test_start=2021,
        test_end=2024,
    )
    
    print(f"\nData split:")
    print(f"  TRAIN: {split.train_start}-{split.train_end}")
    print(f"  VAL:   {split.val_start}-{split.val_end}")
    print(f"  TEST:  {split.test_start}-{split.test_end}")
    
    # Define parameter grid (smaller for faster testing)
    param_grid = ParameterGrid(
        entry_zscore=[2.5, 2.8, 3.0],
        exit_zscore=[0.3, 0.5],
        max_positions=[8, 10],
        max_half_life=[20, 25],
        vol_size_min=[0.3, 0.5],
    )
    
    print(f"\nParameter grid: {param_grid.n_configs} configurations")
    
    # Run CSCV backtest
    result = run_cscv_backtest(
        prices=prices,
        base_config=base_config,
        split=split,
        param_grid=param_grid,
        n_cscv_partitions=8,
        verbose=True,
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\nConfigurations tested: {result.n_configs_tested}")
    print(f"Best config: {result.best_config_name}")
    
    if result.cscv_result:
        print(f"\nCSCV Results:")
        print(f"  PBO: {result.cscv_result.pbo:.3f}")
        print(f"  Deflated Sharpe: {result.deflated_sharpe:.2f}")
        print(f"  Is overfit: {result.is_overfit}")
    
    print(f"\nRecommendation: {result.recommendation}")
    
    if result.test_pnl is not None:
        print(f"\nTest Results:")
        print(f"  PnL: ${result.test_pnl:,.2f}")
        print(f"  Trades: {result.test_trades}")
        print(f"  Win Rate: {result.test_win_rate*100:.1f}%")
        print(f"  Sharpe: {result.test_sharpe:.2f}")
    else:
        print("\n⚠️ Test phase was NOT executed (strategy overfit)")
    
    # Save results
    results_dir = project_root / "results" / "experiments" / "cscv_backtest"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "cscv_results.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(result.to_dict(), f, default_flow_style=False)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
