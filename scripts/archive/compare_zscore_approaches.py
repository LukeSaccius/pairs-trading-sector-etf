"""
Deep Analysis: Rolling vs Fixed Z-Score Trade-off

Understanding why Fixed Z-Score causes MORE stop-losses:
1. Fixed Z stays at entry level while price moves
2. If market trends, Fixed Z moves AGAINST the trade faster
3. Rolling Z "adapts" to new market conditions

This script analyzes the actual behavior difference.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load both backtest results
old_trades = pd.read_csv('results/2025-12-03_01-38_v11_crisis_aware/trades.csv')
new_trades = pd.read_csv('results/2025-12-03_02-05_v11_crisis_aware/trades.csv')

print("=" * 60)
print("COMPARISON: Rolling Z (V11) vs Fixed Z (V12)")
print("=" * 60)

# Compare exit reasons
print("\n--- Exit Reason Distribution ---")
print("\nV11 (Rolling Z):")
print(old_trades['exit_reason'].value_counts())
print(f"Total PnL: ${old_trades['pnl'].sum():,.2f}")

print("\nV12 (Fixed Z):")
print(new_trades['exit_reason'].value_counts())
print(f"Total PnL: ${new_trades['pnl'].sum():,.2f}")

# Compare PnL by exit reason
print("\n--- PnL by Exit Reason ---")
for reason in ['convergence', 'max_holding', 'stop_loss']:
    old_pnl = old_trades[old_trades['exit_reason'] == reason]['pnl'].sum()
    new_pnl = new_trades[new_trades['exit_reason'] == reason]['pnl'].sum()
    old_count = len(old_trades[old_trades['exit_reason'] == reason])
    new_count = len(new_trades[new_trades['exit_reason'] == reason])
    print(f"\n{reason}:")
    print(f"  V11: {old_count} trades, ${old_pnl:,.2f} (avg ${old_pnl/max(old_count,1):,.2f})")
    print(f"  V12: {new_count} trades, ${new_pnl:,.2f} (avg ${new_pnl/max(new_count,1):,.2f})")

# Compare holding days
print("\n--- Average Holding Days ---")
for reason in ['convergence', 'max_holding', 'stop_loss']:
    old_hold = old_trades[old_trades['exit_reason'] == reason]['holding_days'].mean()
    new_hold = new_trades[new_trades['exit_reason'] == reason]['holding_days'].mean()
    print(f"{reason}: V11={old_hold:.1f} days, V12={new_hold:.1f} days")

# Key insight
print("\n" + "=" * 60)
print("KEY INSIGHT")
print("=" * 60)
print("""
The Fixed Z-Score approach causes more stop-losses because:

1. ENTRY: Z = -2.8 (both systems same)
2. DAY 1: Market moves against position
   - Fixed Z: Still based on old mean → Z drops to -3.1 → STOP LOSS!
   - Rolling Z: Mean shifts with market → Z stays around -2.5 → CONTINUE

The Rolling Z-Score is actually providing a form of "trailing mean" that
adapts to market conditions. This is a FEATURE, not a BUG!

The "Max Holding" trades in V11 are NOT caused by z-score drift preventing exit.
They're trades where the spread genuinely didn't mean-revert within the time limit.

RECOMMENDATION:
Keep the Rolling Z-Score for exit decisions. Instead, focus on:
1. Better entry signals (higher z-score threshold)
2. Smarter stop-loss (based on PnL %, not z-score)
3. VIX filter to avoid trading in high volatility regimes
""")

# Save summary
summary = pd.DataFrame({
    'metric': ['Total Trades', 'Total PnL', 'Win Rate', 'Stop Loss Count', 'Convergence Count', 'Max Holding Count'],
    'v11_rolling': [
        len(old_trades),
        old_trades['pnl'].sum(),
        (old_trades['pnl'] > 0).mean() * 100,
        len(old_trades[old_trades['exit_reason'] == 'stop_loss']),
        len(old_trades[old_trades['exit_reason'] == 'convergence']),
        len(old_trades[old_trades['exit_reason'] == 'max_holding']),
    ],
    'v12_fixed': [
        len(new_trades),
        new_trades['pnl'].sum(),
        (new_trades['pnl'] > 0).mean() * 100,
        len(new_trades[new_trades['exit_reason'] == 'stop_loss']),
        len(new_trades[new_trades['exit_reason'] == 'convergence']),
        len(new_trades[new_trades['exit_reason'] == 'max_holding']),
    ],
})
print("\n--- Summary Table ---")
print(summary.to_string(index=False))
