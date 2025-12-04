#!/usr/bin/env python3
"""Debug capital flow in backtest engine."""
import sys
sys.path.insert(0, '.')

from src.pairs_trading_etf.backtests.config import BacktestConfig

def analyze_capital_flow():
    """Analyze how position capital is calculated."""
    
    print("=" * 70)
    print("CAPITAL FLOW ANALYSIS")
    print("=" * 70)
    
    # V15b settings
    configs = {
        'V15b (current)': {
            'initial_capital': 50000,
            'leverage': 1.5,
            'max_positions': 8,
            'compounding': True,
            'capital_per_pair': 10000,
            'max_capital_per_trade': 15000,
            'use_vol_sizing': True,
        },
        'With capital_per_pair=20000': {
            'initial_capital': 50000,
            'leverage': 1.5,
            'max_positions': 8,
            'compounding': True,
            'capital_per_pair': 20000,  # Changed
            'max_capital_per_trade': 15000,
            'use_vol_sizing': True,
        },
        'compounding=false, capital_per_pair=20000': {
            'initial_capital': 50000,
            'leverage': 1.5,
            'max_positions': 8,
            'compounding': False,  # Changed
            'capital_per_pair': 20000,  # Changed
            'max_capital_per_trade': 15000,
            'use_vol_sizing': True,
        },
        'max_capital_per_trade=25000': {
            'initial_capital': 50000,
            'leverage': 1.5,
            'max_positions': 8,
            'compounding': True,
            'capital_per_pair': 10000,
            'max_capital_per_trade': 25000,  # Changed
            'use_vol_sizing': True,
        },
        'max_positions=5, max_capital=25000': {
            'initial_capital': 50000,
            'leverage': 1.5,
            'max_positions': 5,  # Changed
            'compounding': True,
            'capital_per_pair': 10000,
            'max_capital_per_trade': 25000,  # Changed
            'use_vol_sizing': True,
        },
    }
    
    print("\n" + "-" * 70)
    print(f"{'Config':<45} | {'Position Capital':>20}")
    print("-" * 70)
    
    for name, cfg in configs.items():
        current_capital = cfg['initial_capital']
        leverage = cfg['leverage']
        max_positions = cfg['max_positions']
        compounding = cfg['compounding']
        capital_per_pair = cfg['capital_per_pair']
        max_capital_per_trade = cfg['max_capital_per_trade']
        
        # Replicate engine.py logic
        if compounding:
            max_pos = max_positions if max_positions > 0 else 5
            position_capital = (current_capital * leverage) / max(1, max_pos)
            
            if max_capital_per_trade > 0:
                position_capital = min(position_capital, max_capital_per_trade)
        else:
            position_capital = capital_per_pair * leverage
        
        print(f"{name:<45} | ${position_capital:>18,.2f}")
    
    print("-" * 70)
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    print("""
KEY FINDINGS:

1. When compounding=true, capital_per_pair is COMPLETELY IGNORED
   - Position capital = (current_capital * leverage) / max_positions
   - Then capped by max_capital_per_trade
   
2. To increase position size to $20k, you have 3 options:

   OPTION A: Set compounding=false
   - capital_per_pair will be used directly
   - Pro: Simple, capital_per_pair works as expected
   - Con: No equity growth, fixed position sizes
   
   OPTION B: Increase max_capital_per_trade to $20,000+
   - Still uses dynamic capital based on equity
   - Pro: Maintains compounding
   - Con: Doesn't directly control position size
   
   OPTION C: Reduce max_positions to concentrate capital
   - Current: $50k * 1.5 / 8 = $9,375 per position
   - With 5: $50k * 1.5 / 5 = $15,000 per position
   - With 4: $50k * 1.5 / 4 = $18,750 per position
   - Pro: Automatically increases as equity grows
   - Con: Less diversification

RECOMMENDED: Combination of B + C
   - max_positions: 5 (from sensitivity analysis - best PnL)
   - max_capital_per_trade: 25000 (allows position size to grow)
   - compounding: true (maintain equity growth)
   - This gives: $50k * 1.5 / 5 = $15,000 initially, growing with equity
""")
    
    # Show the issue with vol_sizing
    print("\n" + "=" * 70)
    print("ADDITIONAL: VOL-SIZING EFFECT")
    print("=" * 70)
    
    print("""
With use_vol_sizing=true, position_capital is FURTHER adjusted:

position_capital = position_capital * vol_adjustment

Where vol_adjustment = min(max(target_vol/spread_vol, vol_size_min), vol_size_max)
                     = min(max(0.02/spread_vol, 0.25), 2.0)

If spread_vol = 0.01 → adjustment = min(max(2.0, 0.25), 2.0) = 2.0 (DOUBLE)
If spread_vol = 0.02 → adjustment = min(max(1.0, 0.25), 2.0) = 1.0 (SAME)
If spread_vol = 0.04 → adjustment = min(max(0.5, 0.25), 2.0) = 0.5 (HALF)
If spread_vol = 0.10 → adjustment = min(max(0.2, 0.25), 2.0) = 0.25 (MIN)

This means actual position sizes vary from 25% to 200% of base capital!
""")
    
    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION FOR V16")
    print("=" * 70)
    
    print("""
CONFIG CHANGES:

1. entry_zscore: 2.8           (optimal from sensitivity)
2. max_positions: 5            (optimal from sensitivity)  
3. max_capital_per_trade: 25000  (allows larger positions)
4. compounding: true           (keep equity growth)
5. capital_per_pair: 10000     (ignored but keep for reference)

EXPECTED INITIAL POSITION SIZE:
   Base: $50k * 1.5 / 5 = $15,000
   With vol_sizing: $3,750 - $30,000 (capped at $25,000)

NOTE: capital_per_pair parameter should be RENAMED or REMOVED 
      to avoid confusion. It only works when compounding=false.
""")


if __name__ == '__main__':
    analyze_capital_flow()
