"""Compare V11 vs V12 backtest."""
import pandas as pd

old = pd.read_csv('results/2025-12-03_01-38_v11_crisis_aware/trades.csv')
new = pd.read_csv('results/2025-12-03_02-05_v11_crisis_aware/trades.csv')

print('=== SUMMARY ===')
print('V11 Rolling:', len(old), 'trades, PnL=', old.pnl.sum())
print('V12 Fixed:  ', len(new), 'trades, PnL=', new.pnl.sum())

print()
print('=== Exit Reason Counts ===')
print('V11:', dict(old['exit_reason'].value_counts()))
print('V12:', dict(new['exit_reason'].value_counts()))

print()
print('=== Average Holding Days ===')
for r in ['convergence', 'stop_loss', 'max_holding']:
    o_hold = old[old.exit_reason == r]['holding_days'].mean()
    n_hold = new[new.exit_reason == r]['holding_days'].mean()
    print(f'{r}: V11={o_hold:.1f}d, V12={n_hold:.1f}d')

print()
print('=== Stop-Loss Stats ===')
old_sl = old[old.exit_reason == 'stop_loss']
new_sl = new[new.exit_reason == 'stop_loss']
print('V11 SL: count=', len(old_sl), 'avg_pnl=', old_sl.pnl.mean(), 'avg_hold=', old_sl.holding_days.mean())
print('V12 SL: count=', len(new_sl), 'avg_pnl=', new_sl.pnl.mean(), 'avg_hold=', new_sl.holding_days.mean())
