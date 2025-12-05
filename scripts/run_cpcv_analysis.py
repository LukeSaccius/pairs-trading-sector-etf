#!/usr/bin/env python
"""
Run CPCV (Combinatorial Purged Cross-Validation) Analysis

This script runs comprehensive overfitting detection on backtest results.

Usage:
------
# Test multiple configs
python scripts/run_cpcv_analysis.py --configs configs/experiments/*.yaml

# Single config with parameter sweep
python scripts/run_cpcv_analysis.py --config configs/experiments/vidyamurthy_practical.yaml --sweep

# Quick test
python scripts/run_cpcv_analysis.py --quick
"""

import argparse
import logging
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
try:
    from pairs_trading_etf.backtests.cpcv_correct import (
        CSCVAnalyzer,
        CPCVResult,
        build_returns_matrix_from_trades,
    )
    from pairs_trading_etf.backtests.config import BacktestConfig
    from pairs_trading_etf.backtests.engine import run_walkforward_backtest
    from pairs_trading_etf.backtests.validation import PurgedWalkForwardValidator
except ModuleNotFoundError:  # pragma: no cover
    sys.path.append(str(SRC_PATH))
    from pairs_trading_etf.backtests.cpcv_correct import (  # type: ignore[no-redef]
        CSCVAnalyzer,
        CPCVResult,
        build_returns_matrix_from_trades,
    )
    from pairs_trading_etf.backtests.config import BacktestConfig  # type: ignore[no-redef]
    from pairs_trading_etf.backtests.engine import run_walkforward_backtest  # type: ignore[no-redef]
    from pairs_trading_etf.backtests.validation import PurgedWalkForwardValidator  # type: ignore[no-redef]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PARAMETER SWEEP CONFIGURATIONS
# =============================================================================

# Define parameter ranges for sweep
PARAMETER_SWEEP = {
    'entry_threshold_sigma': [1.5, 2.0, 2.5],
    'exit_threshold_sigma': [0.0, 0.3, 0.5],
    'pvalue_threshold': [0.01, 0.05],
    'max_holding_days': [30, 60],
}

# Simplified sweep for faster testing
QUICK_SWEEP = {
    'entry_threshold_sigma': [1.5, 2.0, 2.5],
    'exit_threshold_sigma': [0.0, 0.5],
}


def load_price_data(config: BacktestConfig) -> pd.DataFrame:
    """Load price data from configured path."""
    data_path = PROJECT_ROOT / config.price_data_path
    prices = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(prices)} days of price data")
    return prices


def generate_config_variations(
    base_config: BacktestConfig,
    param_ranges: Dict[str, List[Any]],
) -> List[tuple]:
    """
    Generate all parameter combinations.
    
    Returns list of (name, config) tuples.
    """
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    configs = []
    for values in product(*param_values):
        cfg_dict = base_config.to_dict()
        name_parts = []
        
        for param, value in zip(param_names, values):
            cfg_dict[param] = value
            # Shorten parameter names for readability
            short_name = {
                'entry_threshold_sigma': 'e',
                'exit_threshold_sigma': 'x',
                'pvalue_threshold': 'p',
                'max_holding_days': 'h',
            }.get(param, param[:3])
            name_parts.append(f"{short_name}{value}")
        
        name = "_".join(name_parts)
        try:
            configs.append((name, BacktestConfig(**cfg_dict)))
        except Exception as e:
            logger.warning(f"Invalid config {name}: {e}")
    
    logger.info(f"Generated {len(configs)} config variations")
    return configs


def run_backtests_for_cpcv(
    prices: pd.DataFrame,
    configs: List[tuple],
    start_year: int = 2010,
    end_year: int = 2024,
) -> Dict[str, List[Dict]]:
    """
    Run backtests for all configs and collect trades.
    
    Returns dict of {config_name: [trades]}
    """
    trades_by_config = {}
    mask = (prices.index.year >= start_year) & (prices.index.year <= end_year)
    period_prices = prices[mask]
    
    for i, (name, config) in enumerate(configs):
        logger.info(f"[{i+1}/{len(configs)}] Running backtest: {name}")
        try:
            bt_result = run_walkforward_backtest(
                prices=period_prices,
                cfg=config,
                start_year=start_year,
                end_year=end_year,
            )
            trades: List[Dict]
            pnl_value = 0.0
            if isinstance(bt_result, tuple):
                trades, summary_df = bt_result
                if summary_df is not None and not summary_df.empty:
                    pnl_value = float(summary_df['total_pnl'].sum())
            elif isinstance(bt_result, dict) and 'all_trades' in bt_result:
                trades = bt_result['all_trades']
                pnl_value = float(bt_result.get('summary', {}).get('total_pnl', 0.0))
            else:
                trades = []
            trades_by_config[name] = trades
            if trades:
                logger.info(f"  -> {len(trades)} trades, PnL: ${pnl_value:,.2f}")
            else:
                logger.warning("  -> No trades for this configuration")
        except Exception as exc:  # pragma: no cover - diagnostics
            logger.error(f"  -> Error while running {name}: {exc}")
            trades_by_config[name] = []
    
    return trades_by_config


def run_cpcv_analysis(
    trades_by_config: Dict[str, List[Dict]],
    date_range: pd.DatetimeIndex,
    initial_capital: float,
    n_splits: int = 10,
    purge_window: int = 5,
    embargo_window: int = 3,
) -> CPCVResult:
    """
    Run CSCV (Bailey) analysis on collected trades.
    """
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING CSCV ANALYSIS")
    logger.info("=" * 60)
    
    # Build returns matrix
    returns_matrix, config_names = build_returns_matrix_from_trades(
        trades_by_config,
        date_range,
        initial_capital=initial_capital,
    )
    
    logger.info(f"Returns matrix: {returns_matrix.shape}")
    
    # Check for valid data
    valid_configs = [name for name, trades in trades_by_config.items() if len(trades) > 0]
    if len(valid_configs) < 3:
        raise ValueError(f"Need at least 3 configs with trades, got {len(valid_configs)}")
    
    analyzer = CSCVAnalyzer(n_splits=n_splits)
    result = analyzer.analyze(returns_matrix, config_names)
    logger.info(
        "CSCV metrics => PBO: %.2f%% | DSR: %.2f | Degradation: %.1f%%",
        result.pbo * 100,
        result.dsr,
        result.degradation_ratio * 100,
    )
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run CPCV overfitting analysis on pairs trading strategy'
    )
    parser.add_argument(
        '--config', '-c', type=str, 
        default='configs/experiments/vidyamurthy_practical.yaml',
        help='Base config file for parameter sweep'
    )
    parser.add_argument(
        '--configs', type=str, nargs='+',
        help='Multiple config files to compare (alternative to sweep)'
    )
    parser.add_argument(
        '--sweep', '-s', action='store_true',
        help='Run parameter sweep around base config'
    )
    parser.add_argument(
        '--quick', '-q', action='store_true',
        help='Quick test with fewer parameters'
    )
    parser.add_argument(
        '--n-splits', type=int, default=10,
        help='Number of time splits for CPCV (default: 10)'
    )
    parser.add_argument(
        '--purge', type=int, default=5,
        help='Purge window (default: 5 days)'
    )
    parser.add_argument(
        '--embargo', type=int, default=3,
        help='Embargo window (default: 3 days)'
    )
    parser.add_argument(
        '--start-year', type=int, default=2010,
        help='Start year for backtest'
    )
    parser.add_argument(
        '--end-year', type=int, default=2024,
        help='End year for backtest'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output path for results (JSON)'
    )
    parser.add_argument(
        '--walk-forward',
        action='store_true',
        help='Run PurgedWalkForwardValidator on a selected config'
    )
    parser.add_argument(
        '--walk-forward-target',
        type=str,
        help='Specific configuration name to validate (defaults to first config)'
    )
    parser.add_argument(
        '--walk-forward-min-positive',
        type=float,
        default=0.55,
        help='Minimum fraction of positive OOS splits for walk-forward validation'
    )
    parser.add_argument(
        '--walk-forward-min-oos',
        type=float,
        default=0.0,
        help='Minimum average OOS return for walk-forward validation'
    )
    
    args = parser.parse_args()
    
    # Load base config
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return 1
    
    base_config = BacktestConfig.from_yaml(str(config_path))
    logger.info(f"Base config: {base_config.experiment_name}")
    
    # Load price data
    prices = load_price_data(base_config)
    
    # Generate configs to test
    if args.configs:
        # Load multiple configs from files
        configs = []
        for cfg_path in args.configs:
            full_path = PROJECT_ROOT / cfg_path
            if full_path.exists():
                cfg = BacktestConfig.from_yaml(str(full_path))
                configs.append((cfg.experiment_name, cfg))
            else:
                logger.warning(f"Config not found: {cfg_path}")
    elif args.sweep:
        # Parameter sweep
        sweep_params = QUICK_SWEEP if args.quick else PARAMETER_SWEEP
        configs = generate_config_variations(base_config, sweep_params)
    else:
        # Just the base config (won't work for CPCV - need multiple)
        logger.error("CPCV requires multiple configs. Use --sweep or --configs")
        return 1
    
    if len(configs) < 3:
        logger.error("Need at least 3 configs for CPCV analysis")
        return 1
    
    # Run backtests
    trades_by_config = run_backtests_for_cpcv(
        prices, configs, 
        start_year=args.start_year, 
        end_year=args.end_year
    )
    
    # Filter valid configs
    valid_trades = {k: v for k, v in trades_by_config.items() if len(v) > 0}
    if len(valid_trades) < 3:
        logger.error(f"Only {len(valid_trades)} configs have trades, need >= 3")
        return 1
    
    # Date range for returns matrix
    mask = (prices.index.year >= args.start_year) & (prices.index.year <= args.end_year)
    date_range = prices[mask].index
    
    # Run CPCV
    try:
        result = run_cpcv_analysis(
            valid_trades,
            date_range,
            initial_capital=base_config.initial_capital,
            n_splits=args.n_splits,
            purge_window=args.purge,
            embargo_window=args.embargo,
        )
        
        # Print results
        print(result.summary())

        if args.walk_forward:
            target_name = args.walk_forward_target or next(iter(valid_trades))
            target_trades = valid_trades.get(target_name)
            if not target_trades:
                logger.error(f"Cannot run walk-forward validation: no trades for {target_name}")
            else:
                wf_matrix, _ = build_returns_matrix_from_trades(
                    {target_name: target_trades},
                    date_range,
                    initial_capital=base_config.initial_capital,
                )
                validator = PurgedWalkForwardValidator(
                    min_positive_ratio=args.walk_forward_min_positive,
                    min_avg_oos_return=args.walk_forward_min_oos,
                    default_purge_days=args.purge,
                    default_embargo_days=args.embargo,
                )
                wf_result = validator.evaluate(
                    returns_series=wf_matrix[:, 0],
                    dates=date_range,
                    purge_days=args.purge,
                    embargo_days=args.embargo,
                )
                print("\n" + wf_result.summary())
        
        # Save if requested
        if args.output:
            import json
            output_path = PROJECT_ROOT / args.output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"Results saved to {output_path}")
        
        # Return code based on overfitting
        if result.is_overfit:
            logger.warning("⚠️ STRATEGY APPEARS OVERFIT")
            return 2
        else:
            logger.info("✅ Overfitting risk appears acceptable")
            return 0
            
    except Exception as e:
        logger.error(f"CPCV analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
