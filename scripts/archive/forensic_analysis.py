"""
Forensic Analysis: Deep investigation of Max Holding trades.

This script investigates the "Rolling Beta Trap" hypothesis:
- Z-score used for EXIT is calculated with rolling mean/std
- But actual position is based on FIXED hedge ratio at entry
- This mismatch may cause trades to never exit properly

We compare:
1. Z-score (Rolling) - what the system sees
2. Z-score (Fixed) - what it SHOULD see based on entry parameters
3. Spread (Rolling) vs Spread (Fixed)
4. Actual PnL trajectory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Configuration
DATA_PATH = Path("data/raw/etf_prices_fresh.csv")
TRADES_PATH = Path("results/2025-12-03_01-38_v11_crisis_aware/trades.csv")
OUTPUT_DIR = Path("results/figures/forensic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load price data and trades."""
    prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    trades = pd.read_csv(TRADES_PATH)
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['exit_date'] = pd.to_datetime(trades['exit_date'])
    return prices, trades


def calculate_fixed_zscore(prices, leg_x, leg_y, entry_date, lookback=60):
    """
    Calculate z-score using FIXED parameters from entry date.
    
    This is what the system SHOULD use for exit decisions.
    """
    # Get prices around entry
    entry_idx = prices.index.get_loc(entry_date)
    
    # Use formation period before entry to estimate fixed parameters
    formation_prices = prices.iloc[entry_idx-lookback:entry_idx]
    
    log_x = np.log(formation_prices[leg_x])
    log_y = np.log(formation_prices[leg_y])
    
    # Fixed hedge ratio from formation
    slope, intercept, _, _, _ = stats.linregress(log_y, log_x)
    fixed_hr = slope
    
    # Fixed spread parameters
    spread_formation = log_x - fixed_hr * log_y
    fixed_mean = spread_formation.mean()
    fixed_std = spread_formation.std()
    
    return fixed_hr, fixed_mean, fixed_std


def calculate_rolling_zscore(prices, leg_x, leg_y, hedge_ratio, lookback=60):
    """
    Calculate z-score using ROLLING mean/std.
    
    This is what the current system uses.
    """
    log_x = np.log(prices[leg_x])
    log_y = np.log(prices[leg_y])
    
    spread = log_x - hedge_ratio * log_y
    
    rolling_mean = spread.rolling(window=lookback, min_periods=30).mean()
    rolling_std = spread.rolling(window=lookback, min_periods=30).std()
    
    zscore = (spread - rolling_mean) / rolling_std
    
    return spread, rolling_mean, rolling_std, zscore


def analyze_trade(prices, trade, lookback=60):
    """
    Deep forensic analysis of a single trade.
    
    Returns DataFrame with daily breakdown.
    """
    leg_x = trade['leg_x']
    leg_y = trade['leg_y']
    entry_date = trade['entry_date']
    exit_date = trade['exit_date']
    entry_hr = trade['hedge_ratio']
    direction = 1 if trade['direction'] == 'LONG' else -1
    
    # Get fixed parameters at entry
    fixed_hr, fixed_mean, fixed_std = calculate_fixed_zscore(
        prices, leg_x, leg_y, entry_date, lookback
    )
    
    # Get price data for trade period (with some buffer)
    entry_idx = prices.index.get_loc(entry_date)
    exit_idx = prices.index.get_loc(exit_date)
    
    # Include lookback period before entry for context
    start_idx = max(0, entry_idx - lookback)
    trade_prices = prices.iloc[start_idx:exit_idx+1][[leg_x, leg_y]].copy()
    
    # Calculate both spreads
    log_x = np.log(trade_prices[leg_x])
    log_y = np.log(trade_prices[leg_y])
    
    # Spread with entry hedge ratio
    spread_entry_hr = log_x - entry_hr * log_y
    
    # Spread with fixed (formation) hedge ratio
    spread_fixed_hr = log_x - fixed_hr * log_y
    
    # Rolling z-score (what system uses)
    rolling_mean = spread_entry_hr.rolling(window=lookback, min_periods=30).mean()
    rolling_std = spread_entry_hr.rolling(window=lookback, min_periods=30).std()
    zscore_rolling = (spread_entry_hr - rolling_mean) / rolling_std
    
    # Fixed z-score (what system SHOULD use)
    zscore_fixed = (spread_entry_hr - fixed_mean) / fixed_std
    
    # Calculate daily PnL if we were holding
    entry_px = trade_prices.loc[entry_date, leg_x]
    entry_py = trade_prices.loc[entry_date, leg_y]
    
    # Rough position sizing (normalized to $10k notional)
    notional = 10000
    notional_x = notional / (1 + abs(entry_hr))
    notional_y = abs(entry_hr) * notional_x
    
    if direction == 1:  # Long spread = Long X, Short Y
        qty_x = notional_x / entry_px
        qty_y = -notional_y / entry_py
    else:  # Short spread = Short X, Long Y
        qty_x = -notional_x / entry_px
        qty_y = notional_y / entry_py
    
    # Daily PnL
    pnl_x = qty_x * (trade_prices[leg_x] - entry_px)
    pnl_y = qty_y * (trade_prices[leg_y] - entry_py)
    cumulative_pnl = pnl_x + pnl_y
    
    # Build result DataFrame
    result = pd.DataFrame({
        'date': trade_prices.index,
        'price_x': trade_prices[leg_x].values,
        'price_y': trade_prices[leg_y].values,
        'spread_entry_hr': spread_entry_hr.values,
        'spread_fixed_hr': spread_fixed_hr.values,
        'zscore_rolling': zscore_rolling.values,
        'zscore_fixed': zscore_fixed.values,
        'rolling_mean': rolling_mean.values,
        'rolling_std': rolling_std.values,
        'cumulative_pnl': cumulative_pnl.values,
    })
    result.set_index('date', inplace=True)
    
    # Add metadata
    result.attrs['trade'] = trade.to_dict()
    result.attrs['fixed_hr'] = fixed_hr
    result.attrs['fixed_mean'] = fixed_mean
    result.attrs['fixed_std'] = fixed_std
    result.attrs['entry_hr'] = entry_hr
    result.attrs['direction'] = direction
    
    return result


def plot_forensic_analysis(analysis_df, trade, output_path):
    """
    Create comprehensive forensic plot for a trade.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
    entry_date = trade['entry_date']
    exit_date = trade['exit_date']
    
    # Mark trade period
    trade_mask = (analysis_df.index >= entry_date) & (analysis_df.index <= exit_date)
    
    # Plot 1: Prices
    ax1 = axes[0]
    ax1.plot(analysis_df.index, analysis_df['price_x'], label=trade['leg_x'], alpha=0.8)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(analysis_df.index, analysis_df['price_y'], label=trade['leg_y'], 
                  color='orange', alpha=0.8)
    ax1.axvline(entry_date, color='green', linestyle='--', label='Entry')
    ax1.axvline(exit_date, color='red', linestyle='--', label='Exit')
    ax1.set_ylabel(f'{trade["leg_x"]} Price')
    ax1_twin.set_ylabel(f'{trade["leg_y"]} Price')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.set_title(f'Trade: {trade["leg_x"]}/{trade["leg_y"]} | {trade["direction"]} | '
                  f'Exit: {trade["exit_reason"]} | PnL: ${trade["pnl"]:.2f}')
    ax1.axvspan(entry_date, exit_date, alpha=0.2, color='yellow')
    
    # Plot 2: Spreads comparison
    ax2 = axes[1]
    ax2.plot(analysis_df.index, analysis_df['spread_entry_hr'], 
             label=f'Spread (Entry HR={analysis_df.attrs["entry_hr"]:.3f})', alpha=0.8)
    ax2.plot(analysis_df.index, analysis_df['spread_fixed_hr'], 
             label=f'Spread (Fixed HR={analysis_df.attrs["fixed_hr"]:.3f})', 
             alpha=0.8, linestyle='--')
    ax2.axhline(analysis_df.attrs['fixed_mean'], color='gray', linestyle=':', 
                label=f'Fixed Mean={analysis_df.attrs["fixed_mean"]:.4f}')
    ax2.axvline(entry_date, color='green', linestyle='--')
    ax2.axvline(exit_date, color='red', linestyle='--')
    ax2.axvspan(entry_date, exit_date, alpha=0.2, color='yellow')
    ax2.set_ylabel('Spread (Log)')
    ax2.legend()
    ax2.set_title('Spread: Entry HR vs Fixed HR from Formation')
    
    # Plot 3: Z-Scores comparison - THE KEY PLOT!
    ax3 = axes[2]
    ax3.plot(analysis_df.index, analysis_df['zscore_rolling'], 
             label='Z-Score (Rolling) - SYSTEM USES THIS', linewidth=2, color='blue')
    ax3.plot(analysis_df.index, analysis_df['zscore_fixed'], 
             label='Z-Score (Fixed) - SHOULD USE THIS', linewidth=2, 
             color='green', linestyle='--')
    
    # Entry/Exit thresholds
    ax3.axhline(2.8, color='red', linestyle=':', alpha=0.5, label='Entry (±2.8)')
    ax3.axhline(-2.8, color='red', linestyle=':', alpha=0.5)
    ax3.axhline(0.3, color='green', linestyle=':', alpha=0.5, label='Exit (±0.3)')
    ax3.axhline(-0.3, color='green', linestyle=':', alpha=0.5)
    ax3.axhline(3.0, color='purple', linestyle=':', alpha=0.5, label='Stop Loss (±3.0)')
    ax3.axhline(-3.0, color='purple', linestyle=':', alpha=0.5)
    ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    ax3.axvline(entry_date, color='green', linestyle='--')
    ax3.axvline(exit_date, color='red', linestyle='--')
    ax3.axvspan(entry_date, exit_date, alpha=0.2, color='yellow')
    ax3.set_ylabel('Z-Score')
    ax3.legend(loc='best')
    ax3.set_title('🔍 KEY: Z-Score Rolling vs Fixed - Is Rolling Z preventing exit?')
    ax3.set_ylim(-5, 5)
    
    # Plot 4: PnL trajectory
    ax4 = axes[3]
    colors = ['green' if x >= 0 else 'red' for x in analysis_df.loc[trade_mask, 'cumulative_pnl']]
    ax4.fill_between(analysis_df.index[trade_mask], 
                     0, 
                     analysis_df.loc[trade_mask, 'cumulative_pnl'],
                     alpha=0.3, color='blue')
    ax4.plot(analysis_df.index, analysis_df['cumulative_pnl'], 
             label='Cumulative PnL', linewidth=2)
    ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax4.axvline(entry_date, color='green', linestyle='--', label='Entry')
    ax4.axvline(exit_date, color='red', linestyle='--', label='Exit')
    ax4.axvspan(entry_date, exit_date, alpha=0.2, color='yellow')
    ax4.set_ylabel('PnL ($)')
    ax4.set_xlabel('Date')
    ax4.legend()
    ax4.set_title('PnL Trajectory During Trade')
    
    # Add annotations for key observations
    trade_data = analysis_df.loc[trade_mask]
    if len(trade_data) > 0:
        # Find if Fixed Z crossed exit threshold while Rolling didn't
        direction = analysis_df.attrs['direction']
        
        if direction == 1:  # Long spread
            fixed_crossed = (trade_data['zscore_fixed'] >= -0.3).any()
            rolling_crossed = (trade_data['zscore_rolling'] >= -0.3).any()
        else:
            fixed_crossed = (trade_data['zscore_fixed'] <= 0.3).any()
            rolling_crossed = (trade_data['zscore_rolling'] <= 0.3).any()
        
        max_pnl = trade_data['cumulative_pnl'].max()
        min_pnl = trade_data['cumulative_pnl'].min()
        
        info_text = f"""
FORENSIC FINDINGS:
━━━━━━━━━━━━━━━━━━━━━━━
Holding Days: {trade['holding_days']}
Entry HR: {analysis_df.attrs['entry_hr']:.4f}
Fixed HR: {analysis_df.attrs['fixed_hr']:.4f}
HR Difference: {abs(analysis_df.attrs['entry_hr'] - analysis_df.attrs['fixed_hr']):.4f}

Entry Z (Rolling): {trade['entry_z']:.2f}
Exit Z (Rolling): {trade['exit_z']:.2f}

Fixed Z crossed exit? {fixed_crossed}
Rolling Z crossed exit? {rolling_crossed}
🚨 BUG INDICATOR: {fixed_crossed and not rolling_crossed}

Max PnL during trade: ${max_pnl:.2f}
Min PnL during trade: ${min_pnl:.2f}
Final PnL: ${trade['pnl']:.2f}
"""
        fig.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fixed_crossed, rolling_crossed


def main():
    print("=" * 60)
    print("FORENSIC ANALYSIS: The Rolling Beta Trap")
    print("=" * 60)
    
    prices, trades = load_data()
    
    # Filter for Max Holding trades with losses
    max_holding_trades = trades[
        (trades['exit_reason'] == 'max_holding') & 
        (trades['pnl'] < 0)
    ].copy()
    
    print(f"\nTotal trades: {len(trades)}")
    print(f"Max Holding trades: {len(trades[trades['exit_reason'] == 'max_holding'])}")
    print(f"Max Holding LOSSES: {len(max_holding_trades)}")
    
    # Sort by loss magnitude
    max_holding_trades = max_holding_trades.sort_values('pnl').head(10)
    
    print("\n" + "=" * 60)
    print("TOP 10 WORST MAX HOLDING TRADES:")
    print("=" * 60)
    
    bug_indicators = []
    
    for idx, (_, trade) in enumerate(max_holding_trades.iterrows()):
        print(f"\n[{idx+1}] {trade['leg_x']}/{trade['leg_y']}")
        print(f"    Date: {trade['entry_date'].date()} → {trade['exit_date'].date()}")
        print(f"    Direction: {trade['direction']}, PnL: ${trade['pnl']:.2f}")
        print(f"    Holding: {trade['holding_days']} days, Half-Life: {trade['half_life']:.1f}")
        
        try:
            analysis = analyze_trade(prices, trade)
            
            output_path = OUTPUT_DIR / f"forensic_{idx+1}_{trade['leg_x']}_{trade['leg_y']}.png"
            fixed_crossed, rolling_crossed = plot_forensic_analysis(analysis, trade, output_path)
            
            is_bug = fixed_crossed and not rolling_crossed
            bug_indicators.append({
                'trade_idx': idx + 1,
                'pair': f"{trade['leg_x']}/{trade['leg_y']}",
                'pnl': trade['pnl'],
                'fixed_crossed': fixed_crossed,
                'rolling_crossed': rolling_crossed,
                'is_bug': is_bug,
            })
            
            if is_bug:
                print(f"    🚨 BUG DETECTED: Fixed Z crossed exit, but Rolling Z didn't!")
            
            print(f"    Saved: {output_path}")
            
        except Exception as e:
            print(f"    Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("BUG SUMMARY")
    print("=" * 60)
    
    bug_df = pd.DataFrame(bug_indicators)
    print(bug_df.to_string(index=False))
    
    n_bugs = bug_df['is_bug'].sum()
    print(f"\n🚨 BUG INDICATORS FOUND: {n_bugs}/{len(bug_df)} trades")
    
    if n_bugs > 0:
        print("\nCONCLUSION: The Rolling Beta Trap hypothesis is CONFIRMED!")
        print("The system uses rolling z-score for exit decisions, causing trades")
        print("to miss exit opportunities when the FIXED z-score has already crossed.")
    else:
        print("\nCONCLUSION: Rolling Beta Trap not confirmed in these trades.")
        print("Need to investigate other causes.")
    
    # Save CSV
    bug_df.to_csv(OUTPUT_DIR / "forensic_summary.csv", index=False)
    print(f"\nSaved summary to {OUTPUT_DIR / 'forensic_summary.csv'}")


if __name__ == "__main__":
    main()
