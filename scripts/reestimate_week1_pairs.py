"""Re-estimate Week 1 pairs using 252-day rolling windows.

This script:
1. Loads the 14 tradeable pairs from Week 1
2. Re-estimates cointegration and OU parameters using recent 252-day windows
3. Identifies pairs that are still tradeable vs broken (regime change)
4. Checks if XLU-SPLV reappears with rolling window estimation
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pairs_trading_etf.pipelines.rolling_pair_scan import (
    run_rolling_cointegration,
    RollingScanConfig,
)
from pairs_trading_etf.ou_model.estimation import estimate_ou_from_prices
from pairs_trading_etf.cointegration.engle_granger import run_engle_granger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # Load prices
    prices_path = Path("data/raw/etf_prices.csv")
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded prices: {prices.shape[0]} days, {prices.shape[1]} ETFs")
    logger.info(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # Load Week 1 pairs
    week1_path = Path("results/week1_tradeable_pairs.csv")
    week1_pairs = pd.read_csv(week1_path)
    logger.info(f"\nWeek 1 tradeable pairs ({len(week1_pairs)}):")
    print(week1_pairs[['leg_x', 'leg_y', 'half_life', 'coint_pvalue']].to_string())
    
    # Also check XLU-SPLV specifically
    additional_pairs = [("XLU", "SPLV"), ("EWA", "EWC")]
    
    # Combine all pairs to analyze
    all_pairs = [(row['leg_x'], row['leg_y']) for _, row in week1_pairs.iterrows()]
    for pair in additional_pairs:
        if pair not in all_pairs:
            all_pairs.append(pair)
    
    print("\n" + "="*80)
    print("RE-ESTIMATION WITH 252-DAY ROLLING WINDOW")
    print("="*80)
    
    results = []
    
    for leg_x, leg_y in all_pairs:
        if leg_x not in prices.columns or leg_y not in prices.columns:
            logger.warning(f"Missing data for {leg_x}-{leg_y}")
            continue
        
        # Get aligned prices
        px = prices[leg_x].dropna()
        py = prices[leg_y].dropna()
        df = pd.concat([px, py], axis=1, join="inner").dropna()
        
        # Full history estimate
        try:
            full_result = run_engle_granger(df.iloc[:, 0], df.iloc[:, 1], use_log=True)
            full_hl = full_result.half_life
            full_pv = full_result.pvalue
        except Exception as e:
            logger.warning(f"Full history failed for {leg_x}-{leg_y}: {e}")
            full_hl = np.nan
            full_pv = np.nan
        
        # Recent 252-day estimate
        recent_px = df.iloc[-252:, 0] if len(df) >= 252 else df.iloc[:, 0]
        recent_py = df.iloc[-252:, 1] if len(df) >= 252 else df.iloc[:, 1]
        
        try:
            recent_result = run_engle_granger(recent_px, recent_py, use_log=True)
            recent_hl = recent_result.half_life
            recent_pv = recent_result.pvalue
            recent_hr = recent_result.hedge_ratio
        except Exception as e:
            logger.warning(f"Recent estimate failed for {leg_x}-{leg_y}: {e}")
            recent_hl = np.nan
            recent_pv = np.nan
            recent_hr = np.nan
        
        # Recent 126-day estimate (6 months)
        short_px = df.iloc[-126:, 0] if len(df) >= 126 else df.iloc[:, 0]
        short_py = df.iloc[-126:, 1] if len(df) >= 126 else df.iloc[:, 1]
        
        try:
            short_result = run_engle_granger(short_px, short_py, use_log=True)
            short_hl = short_result.half_life
            short_pv = short_result.pvalue
        except Exception as e:
            short_hl = np.nan
            short_pv = np.nan
        
        # Determine status
        is_tradeable_recent = (
            recent_pv is not None and recent_pv < 0.10 and
            recent_hl is not None and 15 <= recent_hl <= 120
        )
        
        status = "✓ TRADEABLE" if is_tradeable_recent else "✗ NOT TRADEABLE"
        
        # Check for regime break
        if full_hl is not None and recent_hl is not None:
            hl_change = (recent_hl - full_hl) / full_hl * 100 if full_hl > 0 else np.nan
            if abs(hl_change) > 100:  # More than 100% change
                status += " (REGIME BREAK)"
        else:
            hl_change = np.nan
        
        results.append({
            "leg_x": leg_x,
            "leg_y": leg_y,
            "full_hl": full_hl,
            "full_pv": full_pv,
            "recent_252d_hl": recent_hl,
            "recent_252d_pv": recent_pv,
            "recent_126d_hl": short_hl,
            "recent_126d_pv": short_pv,
            "hedge_ratio": recent_hr,
            "hl_change_pct": hl_change,
            "status": status,
        })
        
        print(f"\n{leg_x}-{leg_y}: {status}")
        print(f"  Full History:  HL={full_hl:>6.1f}d  p={full_pv:.4f}" if full_hl else f"  Full History:  HL=N/A")
        print(f"  Recent 252d:   HL={recent_hl:>6.1f}d  p={recent_pv:.4f}" if recent_hl else f"  Recent 252d:   HL=N/A")
        print(f"  Recent 126d:   HL={short_hl:>6.1f}d  p={short_pv:.4f}" if short_hl else f"  Recent 126d:   HL=N/A")
    
    # Summary
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    tradeable = results_df[results_df['status'].str.contains("TRADEABLE") & ~results_df['status'].str.contains("NOT")]
    not_tradeable = results_df[results_df['status'].str.contains("NOT TRADEABLE")]
    regime_break = results_df[results_df['status'].str.contains("REGIME BREAK")]
    
    print(f"\nTradeable pairs (recent 252d): {len(tradeable)}")
    if not tradeable.empty:
        print(tradeable[['leg_x', 'leg_y', 'recent_252d_hl', 'recent_252d_pv']].to_string())
    
    print(f"\nNot tradeable: {len(not_tradeable)}")
    print(f"Regime breaks detected: {len(regime_break)}")
    
    # Save results
    output_path = Path("results/week1_rolling_reestimation.csv")
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")
    
    # Check specifically for XLU-SPLV
    print("\n" + "="*80)
    print("XLU-SPLV SPECIAL ANALYSIS")
    print("="*80)
    
    if "XLU" in prices.columns and "SPLV" in prices.columns:
        xlu = prices["XLU"].dropna()
        splv = prices["SPLV"].dropna()
        
        # Rolling analysis
        result = run_rolling_cointegration(
            xlu, splv,
            formation_window=252,
            step_size=21,
            use_log=True,
        )
        
        if result is not None:
            print(f"\nRolling analysis ({len(result.pvalues)} windows):")
            print(f"  p-value range: {result.pvalues.min():.4f} - {result.pvalues.max():.4f}")
            print(f"  p-value mean: {result.pvalue_mean:.4f}")
            print(f"  % significant (p<0.10): {result.pvalue_pct_significant:.1%}")
            print(f"  Half-life range: {result.half_lives.min():.1f} - {result.half_lives.max():.1f}")
            print(f"  Half-life mean: {result.half_life_mean:.1f}")
            print(f"  Hedge ratio CV: {result.hedge_ratio_stability:.3f}")
            
            # Show recent windows
            print(f"\nRecent 5 windows:")
            for i in range(-5, 0):
                if i < -len(result.pvalues):
                    continue
                date = result.pvalues.index[i]
                pv = result.pvalues.iloc[i]
                hl = result.half_lives.iloc[i]
                hr = result.hedge_ratios.iloc[i]
                print(f"  {date.date()}: p={pv:.4f}, HL={hl:.1f}d, HR={hr:.3f}")
    
    return results_df


if __name__ == "__main__":
    main()
