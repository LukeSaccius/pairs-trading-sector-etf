#!/usr/bin/env python
"""Analyze slow convergence patterns."""

import pandas as pd
import numpy as np

trades = pd.read_csv('results/2025-12-03_17-14_v16b_best/trades.csv')
max_hold = trades[trades['exit_reason'] == 'max_holding'].copy()

print('SLOW CONVERGENCE ANALYSIS')
print('='*70)

# Calculate convergence rate
max_hold['z_change'] = abs(max_hold['entry_z']) - abs(max_hold['exit_z'])
max_hold['z_change_per_day'] = max_hold['z_change'] / max_hold['holding_days']
max_hold['expected_days_to_exit'] = abs(max_hold['exit_z']) / max_hold['z_change_per_day']
max_hold['hl_ratio'] = max_hold['holding_days'] / max_hold['half_life']
max_hold['z_pct_remaining'] = abs(max_hold['exit_z']) / abs(max_hold['entry_z'])

print('Convergence analysis:')
print(f"  Avg Z change: {max_hold['z_change'].mean():.2f}")
print(f"  Avg Z change per day: {max_hold['z_change_per_day'].mean():.3f}")
print(f"  Avg expected additional days to reach 0: {max_hold['expected_days_to_exit'].mean():.1f}")

# Winners vs Losers
winners = max_hold[max_hold['pnl'] > 0]
losers = max_hold[max_hold['pnl'] <= 0]

print()
print('Winners vs Losers (max_holding only):')
print(f"  Winners ({len(winners)}): avg Z change/day = {winners['z_change_per_day'].mean():.4f}")
print(f"  Losers ({len(losers)}): avg Z change/day = {losers['z_change_per_day'].mean():.4f}")
print()
print(f"  Winners: avg exit |Z| = {abs(winners['exit_z']).mean():.2f}")
print(f"  Losers: avg exit |Z| = {abs(losers['exit_z']).mean():.2f}")

print()
print('='*70)
print('NEW RULE IDEA: Exit if |Z| still > X% of |entry_Z| after 1.5*HL')
print('='*70)

print()
print('Z remaining at exit (as % of entry):')
print(f"{'Pair':<15} {'Hold':>5} {'Entry|Z|':>8} {'Exit|Z|':>8} {'%Remain':>8} {'PnL':>10}")
print('-'*60)
for _, t in max_hold.head(15).iterrows():
    pct = t['z_pct_remaining'] * 100
    print(f"{t['leg_x']}/{t['leg_y']:<8} {t['holding_days']:>5} {abs(t['entry_z']):>8.2f} {abs(t['exit_z']):>8.2f} {pct:>7.0f}% ${t['pnl']:>+8,.0f}")

print()
print('Average Z remaining:')
print(f"  All max_holding: {max_hold['z_pct_remaining'].mean()*100:.0f}%")
print(f"  Winners: {winners['z_pct_remaining'].mean()*100:.0f}%")
print(f"  Losers: {losers['z_pct_remaining'].mean()*100:.0f}%")

# Test different thresholds
print()
print('='*70)
print('TESTING DIFFERENT SLOW CONVERGENCE THRESHOLDS')
print('='*70)
print()
print('Rule: Exit early if Z remaining > threshold% at 1.5*HL')
print()

for threshold in [30, 40, 50, 60, 70]:
    would_exit_early = max_hold[max_hold['z_pct_remaining'] > threshold/100]
    avoided_losses = would_exit_early[would_exit_early['pnl'] < 0]['pnl'].sum()
    lost_wins = would_exit_early[would_exit_early['pnl'] > 0]['pnl'].sum()
    net_impact = abs(avoided_losses) - lost_wins
    
    print(f"  Threshold {threshold}%: {len(would_exit_early)} trades would exit early")
    print(f"    Avoided losses: ${abs(avoided_losses):+,.0f}")
    print(f"    Lost wins: ${-lost_wins:,.0f}")
    print(f"    Net impact: ${net_impact:+,.0f}")
    print()
