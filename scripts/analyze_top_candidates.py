"""Detailed analysis of top rolling-window pair candidates.

For each candidate, analyzes:
1. Rolling cointegration history (when did it become cointegrated?)
2. OU parameters and regime stability
3. Spread behavior and z-score patterns
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pairs_trading_etf.pipelines.rolling_pair_scan import run_rolling_cointegration
from pairs_trading_etf.cointegration.engle_granger import run_engle_granger
from pairs_trading_etf.ou_model.estimation import estimate_ou_from_prices

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_pair_detail(prices: pd.DataFrame, leg_x: str, leg_y: str):
    """Detailed analysis of a single pair."""
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS: {leg_x}-{leg_y}")
    print("="*80)
    
    px = prices[leg_x].dropna()
    py = prices[leg_y].dropna()
    df = pd.concat([px, py], axis=1, join="inner").dropna()
    
    print(f"\nData: {len(df)} trading days, {df.index[0].date()} to {df.index[-1].date()}")
    
    # Rolling cointegration analysis
    result = run_rolling_cointegration(
        px, py,
        formation_window=252,
        step_size=21,
        use_log=True,
    )
    
    if result is None:
        print("Insufficient data for rolling analysis")
        return
    
    # Summary statistics
    print(f"\nRolling Window Summary ({len(result.pvalues)} windows):")
    print(f"  p-value: {result.pvalue_mean:.4f} ± {result.pvalue_std:.4f}")
    print(f"  % significant (p<0.10): {result.pvalue_pct_significant:.1%}")
    print(f"  Half-life: {result.half_life_mean:.1f} ± {result.half_life_std:.1f} days")
    print(f"  Hedge ratio CV: {result.hedge_ratio_stability:.3f}")
    print(f"  Latest: p={result.latest_pvalue:.4f}, HL={result.latest_half_life:.1f}d, HR={result.latest_hedge_ratio:.3f}")
    
    # Identify cointegration windows
    sig_windows = result.pvalues[result.pvalues < 0.10]
    if len(sig_windows) > 0:
        print(f"\nCointegration History:")
        print(f"  First significant: {sig_windows.index[0].date()}")
        print(f"  Last significant: {sig_windows.index[-1].date()}")
        
        # Recent trend
        recent_5 = result.pvalues.iloc[-5:]
        recent_sig = (recent_5 < 0.10).sum()
        print(f"  Recent 5 windows significant: {recent_sig}/5")
    
    # Period analysis
    print(f"\nPeriod Analysis:")
    years = [(2014, 2016), (2017, 2019), (2020, 2021), (2022, 2023), (2024, 2025)]
    
    for start_year, end_year in years:
        mask = (result.pvalues.index.year >= start_year) & (result.pvalues.index.year <= end_year)
        period_pv = result.pvalues[mask]
        period_hl = result.half_lives[mask]
        
        if len(period_pv) > 0:
            pct_sig = (period_pv < 0.10).mean()
            avg_hl = period_hl.mean()
            print(f"  {start_year}-{end_year}: {pct_sig:.0%} significant, avg HL={avg_hl:.1f}d")
    
    # OU estimation on recent data
    try:
        recent_hr = result.latest_hedge_ratio
        ou_params = estimate_ou_from_prices(
            df.iloc[-252:, 0], df.iloc[-252:, 1],
            hedge_ratio=recent_hr,
            use_log=True,
        )
        print(f"\nOU Parameters (recent 252d):")
        print(f"  θ (mean reversion): {ou_params.theta:.4f}")
        print(f"  μ (equilibrium): {ou_params.mu:.4f}")
        print(f"  σ (volatility): {ou_params.sigma:.4f}")
        print(f"  Half-life: {ou_params.half_life:.1f} days")
        print(f"  R²: {ou_params.r_squared:.4f}")
        print(f"  Mean-reverting: {'Yes' if ou_params.is_mean_reverting() else 'No'} (p={ou_params.theta_pvalue:.4f})")
    except Exception as e:
        print(f"\nOU estimation failed: {e}")
    
    # Current spread z-score
    try:
        log_px = np.log(df.iloc[-252:, 0])
        log_py = np.log(df.iloc[-252:, 1])
        spread = log_px - result.latest_hedge_ratio * log_py
        z_score = (spread - spread.mean()) / spread.std()
        
        print(f"\nCurrent Spread Status:")
        print(f"  Current z-score: {z_score.iloc[-1]:.2f}")
        print(f"  z-score range (252d): [{z_score.min():.2f}, {z_score.max():.2f}]")
        
        # Trading signals
        long_signals = (z_score <= -2.0).sum()
        short_signals = (z_score >= 2.0).sum()
        print(f"  Entry opportunities (252d): {long_signals} long, {short_signals} short")
        
        # Signal quality
        if abs(z_score.iloc[-1]) >= 2.0:
            signal = "LONG" if z_score.iloc[-1] <= -2.0 else "SHORT"
            print(f"  ⚡ ACTIVE SIGNAL: {signal} spread at z={z_score.iloc[-1]:.2f}")
        elif abs(z_score.iloc[-1]) >= 1.5:
            direction = "approaching long" if z_score.iloc[-1] < 0 else "approaching short"
            print(f"  ⏳ {direction} entry zone")
        else:
            print(f"  No active signal (z near mean)")
    except Exception as e:
        print(f"\nSpread analysis failed: {e}")
    
    return result


def main():
    # Load prices
    prices_path = Path("data/raw/etf_prices.csv")
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    
    # Load candidates
    candidates_path = Path("results/rolling_candidates_quick.csv")
    candidates = pd.read_csv(candidates_path)
    
    print(f"Analyzing top {min(10, len(candidates))} candidates...")
    
    # Analyze top candidates by half-life
    top_pairs = candidates.nsmallest(10, "half_life")
    
    results = {}
    for _, row in top_pairs.iterrows():
        leg_x, leg_y = row["leg_x"], row["leg_y"]
        result = analyze_pair_detail(prices, leg_x, leg_y)
        if result is not None:
            results[(leg_x, leg_y)] = result
    
    # Summary table
    print("\n" + "="*80)
    print("FINAL RANKING")
    print("="*80)
    
    summary_rows = []
    for (leg_x, leg_y), result in results.items():
        score = (
            (1 - result.latest_pvalue) * 0.3 +  # Lower p-value is better
            (1 / max(result.latest_half_life, 1)) * 100 * 0.3 +  # Faster mean reversion
            result.pvalue_pct_significant * 0.2 +  # Consistency
            (1 - min(result.hedge_ratio_stability, 1)) * 0.2  # Stability
        )
        summary_rows.append({
            "pair": f"{leg_x}-{leg_y}",
            "latest_hl": result.latest_half_life,
            "latest_pv": result.latest_pvalue,
            "pct_sig": result.pvalue_pct_significant,
            "hr_cv": result.hedge_ratio_stability,
            "score": score,
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("score", ascending=False)
    print(summary_df.to_string())
    
    # Save final ranking
    output_path = Path("results/top_pairs_detailed.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\nRanking saved to {output_path}")


if __name__ == "__main__":
    main()
