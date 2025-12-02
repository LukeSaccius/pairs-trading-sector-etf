"""Find currently tradeable pairs using 252-day rolling windows.

Scans the full ETF universe for pairs that show:
1. Consistent cointegration in recent windows (p < 0.10)
2. Tradeable half-life (15-120 days)
3. Stable hedge ratio (low CV)
"""

import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pairs_trading_etf.pipelines.rolling_pair_scan import (
    run_rolling_cointegration,
    RollingPairResult,
)
from pairs_trading_etf.cointegration.engle_granger import run_engle_granger
from pairs_trading_etf.features.pair_generation import enumerate_pairs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    start_time = time.time()
    
    # Load prices
    prices_path = Path("data/raw/etf_prices.csv")
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded prices: {prices.shape[0]} days, {prices.shape[1]} ETFs")
    
    # Generate all pairs
    tickers = list(prices.columns)
    all_pairs = enumerate_pairs(tickers)
    logger.info(f"Total candidate pairs: {len(all_pairs)}")
    
    # First pass: Quick correlation filter on recent data
    logger.info("\n=== PHASE 1: Correlation Filter (recent 252d) ===")
    recent_prices = prices.iloc[-252:]
    
    corr_filtered = []
    for leg_x, leg_y in all_pairs:
        px = recent_prices[leg_x].dropna()
        py = recent_prices[leg_y].dropna()
        df = pd.concat([px, py], axis=1, join="inner").dropna()
        
        if len(df) < 200:  # Need sufficient recent data
            continue
        
        corr = df.iloc[:, 0].corr(df.iloc[:, 1])
        if 0.60 <= corr <= 0.98:  # Correlated but not identical
            corr_filtered.append((leg_x, leg_y, corr))
    
    logger.info(f"Pairs passing correlation filter: {len(corr_filtered)}")
    
    # Second pass: Quick cointegration test on recent 252 days
    logger.info("\n=== PHASE 2: Quick Cointegration Filter (recent 252d) ===")
    coint_candidates = []
    
    for i, (leg_x, leg_y, corr) in enumerate(corr_filtered):
        if (i + 1) % 500 == 0:
            logger.info(f"  Processing {i + 1}/{len(corr_filtered)}")
        
        try:
            px = recent_prices[leg_x]
            py = recent_prices[leg_y]
            result = run_engle_granger(px, py, use_log=True)
            
            if result.pvalue < 0.10 and result.half_life:
                if 15 <= result.half_life <= 120:
                    coint_candidates.append({
                        "leg_x": leg_x,
                        "leg_y": leg_y,
                        "corr": corr,
                        "pvalue": result.pvalue,
                        "half_life": result.half_life,
                        "hedge_ratio": result.hedge_ratio,
                    })
        except Exception:
            continue
    
    logger.info(f"Pairs passing quick cointegration filter: {len(coint_candidates)}")
    
    if not coint_candidates:
        logger.warning("No pairs found! Market regime may be challenging for pairs trading.")
        return
    
    # Sort by half-life and show candidates
    candidates_df = pd.DataFrame(coint_candidates)
    candidates_df = candidates_df.sort_values("half_life")
    
    print("\n" + "="*80)
    print("QUICK FILTER CANDIDATES (recent 252d)")
    print("="*80)
    print(candidates_df.to_string())
    
    # Third pass: Rolling window stability check on top candidates
    logger.info("\n=== PHASE 3: Rolling Window Stability Analysis ===")
    
    # Take top 50 by half-life for deeper analysis
    top_candidates = candidates_df.head(50)
    
    stable_pairs = []
    
    for _, row in top_candidates.iterrows():
        leg_x, leg_y = row["leg_x"], row["leg_y"]
        
        try:
            result = run_rolling_cointegration(
                prices[leg_x],
                prices[leg_y],
                formation_window=252,
                step_size=21,
                use_log=True,
            )
            
            if result is None:
                continue
            
            # Check stability criteria
            is_stable = (
                result.pvalue_pct_significant >= 0.50 and  # At least 50% of windows significant
                result.latest_pvalue < 0.10 and
                15 <= result.latest_half_life <= 120 and
                result.hedge_ratio_stability < 0.5  # CV < 50%
            )
            
            if is_stable:
                stable_pairs.append({
                    "leg_x": leg_x,
                    "leg_y": leg_y,
                    "latest_pvalue": result.latest_pvalue,
                    "latest_half_life": result.latest_half_life,
                    "latest_hedge_ratio": result.latest_hedge_ratio,
                    "pvalue_pct_sig": result.pvalue_pct_significant,
                    "hl_mean": result.half_life_mean,
                    "hl_std": result.half_life_std,
                    "hr_cv": result.hedge_ratio_stability,
                })
                logger.info(f"  ✓ {leg_x}-{leg_y}: HL={result.latest_half_life:.1f}d, p={result.latest_pvalue:.4f}, stability={result.pvalue_pct_significant:.0%}")
            
        except Exception as e:
            logger.debug(f"  Error analyzing {leg_x}-{leg_y}: {e}")
            continue
    
    # Results
    print("\n" + "="*80)
    print("CURRENTLY TRADEABLE PAIRS (Rolling 252d, Stable)")
    print("="*80)
    
    if stable_pairs:
        stable_df = pd.DataFrame(stable_pairs)
        stable_df = stable_df.sort_values("latest_half_life")
        print(stable_df.to_string())
        
        # Save results
        output_path = Path("results/rolling_tradeable_pairs.csv")
        stable_df.to_csv(output_path, index=False)
        logger.info(f"\nResults saved to {output_path}")
    else:
        logger.warning("No stable tradeable pairs found with rolling window analysis!")
        
        # Fall back to showing best quick-filter candidates
        print("\nFalling back to quick-filter candidates (less rigorous):")
        print(candidates_df.head(20).to_string())
        
        output_path = Path("results/rolling_candidates_quick.csv")
        candidates_df.to_csv(output_path, index=False)
        logger.info(f"\nQuick candidates saved to {output_path}")
    
    elapsed = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed:.1f}s")
    
    return stable_pairs if stable_pairs else coint_candidates


if __name__ == "__main__":
    main()
