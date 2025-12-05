"""Quick test script for QMA Level 2 changes."""
import pandas as pd
import sys
sys.path.insert(0, 'I:/Winter-Break-Research')

from src.pairs_trading_etf.backtests import load_config, run_walkforward_backtest

# Load data
prices = pd.read_csv('data/raw/etf_prices_fresh.csv', index_col=0, parse_dates=True)

# Load config 
cfg = load_config('configs/experiments/v16_optimized.yaml')
print('Config loaded:')
print(f'  use_fixed_exit_params: {cfg.use_fixed_exit_params}')
print(f'  pvalue_threshold: {cfg.pvalue_threshold}')

# Run walkforward backtest for 2023 only
trades, yearly_summary = run_walkforward_backtest(prices, cfg, start_year=2023, end_year=2023)
print(f'Trades in 2023: {len(trades)}')

if trades:
    # Check the trades
    for t in trades[:5]:
        print(f"Trade: {t.get('leg_x', 'N/A')}/{t.get('leg_y', 'N/A')} - exit_reason={t.get('exit_reason', 'N/A')}, pnl={t.get('pnl', 0):.2f}")
    
    # Check win rate
    wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
    print(f'\nWin rate: {wins/len(trades):.1%} ({wins}/{len(trades)})')
    print(f'Total PnL: ${sum(t.get("pnl", 0) for t in trades):,.2f}')
else:
    print('No trades!')
