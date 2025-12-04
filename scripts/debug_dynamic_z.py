#!/usr/bin/env python
"""Debug dynamic z exit logic."""

import pandas as pd

# Load trades từ V16b để phân tích
trades = pd.read_csv('results/2025-12-03_17-14_v16b_best/trades.csv')

# Tính các metrics
trades['hl_ratio'] = trades['holding_days'] / trades['half_life']
trades['entry_z_abs'] = abs(trades['entry_z'])
trades['exit_z_abs'] = abs(trades['exit_z'])
trades['z_diverged'] = trades['exit_z_abs'] >= trades['entry_z_abs']

print('='*70)
print('DYNAMIC Z EXIT ANALYSIS')
print('='*70)

# Check max_holding trades
max_hold = trades[trades['exit_reason'] == 'max_holding']
print(f'\nMax Holding Trades: {len(max_hold)}')

# Check detailed analysis
print(f'\nDetailed analysis of max_holding trades:')
print(f"{'Pair':<15} {'Hold':>5} {'HL':>6} {'Ratio':>6} {'MaxHold':>8} {'1.5xHL':>7} {'Entry|Z|':>8} {'Exit|Z|':>8} {'Diverged':>8}")
print('-'*85)

can_check_count = 0
would_trigger_count = 0

for _, t in max_hold.iterrows():
    pair = f"{t['leg_x']}/{t['leg_y']}"[:14]
    max_hold_days = min(int(2.5 * t['half_life']), 35)  # V16b formula
    threshold_days = 1.5 * t['half_life']
    diverged = 'YES' if t['z_diverged'] else 'NO'
    
    # Problem: if max_hold_days <= 1.5 * half_life, dynamic z exit never gets checked!
    can_check = threshold_days < max_hold_days
    if can_check:
        can_check_count += 1
        if t['z_diverged']:
            would_trigger_count += 1
    
    print(f"{pair:<15} {t['holding_days']:>5} {t['half_life']:>6.1f} {t['hl_ratio']:>6.2f} {max_hold_days:>8} {threshold_days:>7.1f} {t['entry_z_abs']:>8.2f} {t['exit_z_abs']:>8.2f} {diverged:>8}")

print(f'\n' + '='*70)
print('KEY INSIGHT')
print('='*70)
print(f'max_hold = min(2.5 * HL, 35)')
print(f'dynamic_z_exit checks at: 1.5 * HL')
print(f'\nFor dynamic z exit to work: 1.5 * HL must be LESS than max_hold')
print(f'  i.e., 1.5 * HL < 2.5 * HL (always true)')
print(f'  BUT also < 35 days cap')
print(f'\nTrades where dynamic z CAN be checked (1.5*HL < max_hold): {can_check_count}')
print(f'Of those, trades that would trigger (Z diverged at exit): {would_trigger_count}')

print(f'\n' + '='*70)
print('THE REAL PROBLEM')
print('='*70)
print('The dynamic z exit logic checks: if holding_days >= 1.5 * HL AND |current_z| >= |entry_z|')
print('then exit with reason "z_diverging"')
print('')
print('The check runs on EVERY day, so if Z diverges at any point after 1.5*HL, it exits.')
print('If no z_diverging exits occurred, it means Z was always below entry_z after 1.5*HL.')

# Check if z was diverging at exit for max_holding trades
print('\nZ-score divergence analysis for max_holding trades:')
z_diverged = max_hold[max_hold['z_diverged']]
z_converged = max_hold[~max_hold['z_diverged']]

print(f'  Z diverged at exit (|exit_z| >= |entry_z|): {len(z_diverged)} trades')
print(f'  Z converged at exit (|exit_z| < |entry_z|): {len(z_converged)} trades')

print(f'\nZ-diverged trades PnL: ${z_diverged["pnl"].sum():+,.0f}')
print(f'Z-converged trades PnL: ${z_converged["pnl"].sum():+,.0f}')

# The issue: dynamic z exit checks CONTINUOUSLY, not just at 1.5*HL
# So if z diverges at day 10 (1.5*HL) but then converges by day 15 (max_hold),
# we would have exited at day 10

print(f'\n' + '='*70)
print('SOLUTION: Check if logic is being reached')
print('='*70)

# Let's trace a specific trade
print('\nExample trade analysis:')
for _, t in z_diverged.head(3).iterrows():
    pair = f"{t['leg_x']}/{t['leg_y']}"
    max_hold_days = min(int(2.5 * t['half_life']), 35)
    threshold_days = 1.5 * t['half_life']
    
    print(f"\n{pair} ({t['entry_date'][:10]}):")
    print(f"  Half-life: {t['half_life']:.1f} days")
    print(f"  Holding days: {t['holding_days']} days")
    print(f"  Max hold (2.5*HL capped at 35): {max_hold_days} days")
    print(f"  Dynamic Z check threshold (1.5*HL): {threshold_days:.1f} days")
    print(f"  Entry |Z|: {abs(t['entry_z']):.2f}")
    print(f"  Exit |Z|: {abs(t['exit_z']):.2f}")
    print(f"  Z diverged: {abs(t['exit_z']) >= abs(t['entry_z'])}")
    
    # Key check: was holding_days > 1.5 * HL at some point before max_hold?
    if t['holding_days'] >= threshold_days:
        print(f"  -> Trade held past 1.5*HL threshold!")
        print(f"  -> If Z was diverging at day {threshold_days:.0f}, should have exited")
        print(f"  -> But we don't know Z at day {threshold_days:.0f}, only at exit")
