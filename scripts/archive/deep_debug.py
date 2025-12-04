"""
DEEP DEBUG: Check PnL calculation logic step by step
"""

import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
DATA_PATH = project_root / "data/raw/etf_prices_fresh.csv"

# Load data
prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
trades = pd.read_csv(project_root / "results/2025-12-03_01-27_v9_compounding/trades.csv")

print("="*100)
print("DEEP DEBUG: PnL Calculation Logic Check")
print("="*100)

# Focus on 2018 - the worst year (-$1,486)
trades_2018 = trades[pd.to_datetime(trades['entry_date']).dt.year == 2018]
print(f"\n2018 Trades: {len(trades_2018)}")
print(trades_2018[['leg_x', 'leg_y', 'direction', 'entry_date', 'exit_date', 'hedge_ratio', 'pnl', 'exit_reason']])

print("\n" + "="*100)
print("TRADE-BY-TRADE ANALYSIS")
print("="*100)

for idx, trade in trades_2018.iterrows():
    print(f"\n{'='*80}")
    etf1 = trade['leg_x']
    etf2 = trade['leg_y']
    direction = trade['direction']
    entry_date = pd.to_datetime(trade['entry_date'])
    exit_date = pd.to_datetime(trade['exit_date'])
    hr = trade['hedge_ratio']
    reported_pnl = trade['pnl']
    exit_reason = trade['exit_reason']
    
    print(f"Pair: {etf1}/{etf2} | Direction: {direction}")
    print(f"Entry: {entry_date.date()} | Exit: {exit_date.date()} | Hedge Ratio: {hr:.4f}")
    print(f"Exit Reason: {exit_reason}")
    
    # Get prices
    entry_px = prices.loc[entry_date:, etf1].iloc[0]
    entry_py = prices.loc[entry_date:, etf2].iloc[0]
    exit_px = prices.loc[exit_date:, etf1].iloc[0]
    exit_py = prices.loc[exit_date:, etf2].iloc[0]
    
    print(f"\nPrices:")
    print(f"  Entry: {etf1}=${entry_px:.2f}, {etf2}=${entry_py:.2f}")
    print(f"  Exit:  {etf1}=${exit_px:.2f}, {etf2}=${exit_py:.2f}")
    
    # Price changes
    pct_x = (exit_px / entry_px - 1) * 100
    pct_y = (exit_py / entry_py - 1) * 100
    print(f"\nPrice Changes:")
    print(f"  {etf1}: {pct_x:+.2f}%")
    print(f"  {etf2}: {pct_y:+.2f}%")
    
    # Simulate position
    capital = 10000  # Assume $10k per trade for simplicity
    notional_x = capital / (1 + abs(hr))
    notional_y = abs(hr) * notional_x
    
    is_long = direction == 'LONG'
    
    if is_long:  # Long spread = Long X, Short Y
        qty_x = notional_x / entry_px   # Buy X
        qty_y = -notional_y / entry_py  # Short Y
    else:  # Short spread = Short X, Long Y
        qty_x = -notional_x / entry_px  # Short X
        qty_y = notional_y / entry_py   # Buy Y
    
    print(f"\nPosition (Capital=${capital:,.0f}, HR={hr:.4f}):")
    print(f"  {etf1}: qty={qty_x:+.4f} shares (notional ${notional_x:,.0f})")
    print(f"  {etf2}: qty={qty_y:+.4f} shares (notional ${notional_y:,.0f})")
    
    # PnL calculation
    pnl_x = qty_x * (exit_px - entry_px)
    pnl_y = qty_y * (exit_py - entry_py)
    total_pnl = pnl_x + pnl_y
    
    print(f"\nPnL Breakdown:")
    print(f"  {etf1}: {qty_x:+.4f} × ({exit_px:.2f} - {entry_px:.2f}) = ${pnl_x:+.2f}")
    print(f"  {etf2}: {qty_y:+.4f} × ({exit_py:.2f} - {entry_py:.2f}) = ${pnl_y:+.2f}")
    print(f"  Total: ${total_pnl:+.2f}")
    
    print(f"\nReported PnL: ${reported_pnl:+.2f}")
    
    # Logic check
    print(f"\n🔍 LOGIC CHECK:")
    if is_long:
        print(f"   LONG spread = Long {etf1}, Short {etf2}")
        print(f"   We profit if {etf1} outperforms {etf2}")
        print(f"   Did {etf1} outperform? {pct_x:.2f}% vs {pct_y:.2f}% → {'YES' if pct_x > pct_y else 'NO'}")
    else:
        print(f"   SHORT spread = Short {etf1}, Long {etf2}")
        print(f"   We profit if {etf2} outperforms {etf1}")
        print(f"   Did {etf2} outperform? {pct_y:.2f}% vs {pct_x:.2f}% → {'YES' if pct_y > pct_x else 'NO'}")
    
    expected_profit = (pct_x > pct_y) if is_long else (pct_y > pct_x)
    actual_profit = total_pnl > 0
    
    if expected_profit == actual_profit:
        print(f"   ✅ Logic CORRECT: Expected profit={expected_profit}, Actual profit={actual_profit}")
    else:
        print(f"   ⚠️ Logic MISMATCH: Expected profit={expected_profit}, Actual profit={actual_profit}")
        print(f"   This can happen when hedge ratio != 1.0!")

print("\n" + "="*100)
print("KEY INSIGHT:")
print("="*100)
print("""
When hedge_ratio > 1: We're buying/shorting MORE of Y relative to X
When hedge_ratio < 1: We're buying/shorting LESS of Y relative to X

The issue is with the SCALING of positions via hedge ratio!

For LONG spread with HR=1.6:
  - We buy $3846 of X (DIA)  
  - We short $6154 of Y (RSP)
  
If both go up 2%:
  - X gain: $3846 × 2% = +$77
  - Y loss (we're short): $6154 × 2% = -$123  
  - Net: -$46

So with HR>1 and market moving together, SHORT spread is safer!
""")

# Calculate what happens in different scenarios
print("\n" + "="*100)
print("SCENARIO ANALYSIS FOR DIA/RSP WITH HR=1.62")
print("="*100)

hr = 1.62
capital = 10000
notional_x = capital / (1 + abs(hr))
notional_y = abs(hr) * notional_x

print(f"Capital allocation with HR={hr}:")
print(f"  X (DIA) notional: ${notional_x:,.0f} ({notional_x/capital*100:.1f}%)")
print(f"  Y (RSP) notional: ${notional_y:,.0f} ({notional_y/capital*100:.1f}%)")

scenarios = [
    ("Both up 5%", 5, 5),
    ("Both down 5%", -5, -5),
    ("X up 3%, Y up 1%", 3, 1),  
    ("X down 3%, Y down 5%", -3, -5),
    ("X up 2%, Y down 2%", 2, -2),
]

print("\nLONG spread (Long X, Short Y):")
for name, pct_x, pct_y in scenarios:
    pnl_x = notional_x * pct_x / 100
    pnl_y = -notional_y * pct_y / 100  # Short Y
    total = pnl_x + pnl_y
    print(f"  {name}: X=${pnl_x:+.0f}, Y=${pnl_y:+.0f}, Total=${total:+.0f}")

print("\nSHORT spread (Short X, Long Y):")
for name, pct_x, pct_y in scenarios:
    pnl_x = -notional_x * pct_x / 100  # Short X
    pnl_y = notional_y * pct_y / 100   # Long Y
    total = pnl_x + pnl_y
    print(f"  {name}: X=${pnl_x:+.0f}, Y=${pnl_y:+.0f}, Total=${total:+.0f}")
