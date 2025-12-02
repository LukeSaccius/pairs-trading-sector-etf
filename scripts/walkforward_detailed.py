"""
Walk-Forward Detailed Analysis
Formation Year -> Trading Year persistence check
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'src')

from pairs_trading_etf.cointegration.engle_granger import run_engle_granger

# Load fresh data
prices = pd.read_csv('data/raw/etf_prices_fresh.csv', index_col=0, parse_dates=True)

print("="*80)
print("DETAILED WALK-FORWARD: 2008 -> 2009 (Crisis Period)")
print("="*80)

formation_data = prices.loc['2008-01-01':'2008-12-31']
trading_data = prices.loc['2009-01-01':'2009-12-31']

all_etfs = [col for col in prices.columns if prices[col].notna().sum() > 500]

print("Formation: 2008, Trading: 2009")
print()

pairs_found = []

for i, etf_a in enumerate(all_etfs):
    for etf_b in all_etfs[i+1:]:
        price_a = formation_data[etf_a].dropna()
        price_b = formation_data[etf_b].dropna()
        
        if len(price_a) < 200 or len(price_b) < 200:
            continue
        
        common_idx = price_a.index.intersection(price_b.index)
        if len(common_idx) < 200:
            continue
        
        corr = price_a.loc[common_idx].pct_change().corr(price_b.loc[common_idx].pct_change())
        if corr < 0.6 or corr > 0.95:
            continue
        
        try:
            eg = run_engle_granger(price_a.loc[common_idx], price_b.loc[common_idx], use_log=True)
            
            if eg.pvalue < 0.10 and eg.half_life and 15 <= eg.half_life <= 120:
                pairs_found.append((etf_a, etf_b, eg.pvalue, eg.half_life, eg.hedge_ratio, corr))
        except:
            pass

print(f"Pairs found in 2008: {len(pairs_found)}")
print()

validated = 0
for etf_a, etf_b, f_pval, f_hl, f_hr, corr in pairs_found:
    price_a = trading_data[etf_a].dropna()
    price_b = trading_data[etf_b].dropna()
    common_idx = price_a.index.intersection(price_b.index)
    
    if len(common_idx) < 100:
        print(f"{etf_a}-{etf_b}: No trading data")
        continue
    
    eg_t = run_engle_granger(price_a.loc[common_idx], price_b.loc[common_idx], use_log=True)
    
    t_hl = eg_t.half_life if eg_t.half_life else float('inf')
    t_pass = eg_t.pvalue < 0.10 and 15 <= t_hl <= 120
    
    status = "PASS" if t_pass else "FAIL"
    if t_pass:
        validated += 1
    
    print(f"{etf_a}-{etf_b}:")
    print(f"  Formation 2008: p={f_pval:.4f}, HL={f_hl:.1f}d")
    print(f"  Trading 2009:   p={eg_t.pvalue:.4f}, HL={t_hl:.1f}d [{status}]")
    print()

print("="*80)
print(f"SUMMARY: {validated}/{len(pairs_found)} pairs validated ({validated/len(pairs_found)*100:.1f}%)" if pairs_found else "No pairs found")


# Now check what happens if we RELAX the trading criteria
print()
print("="*80)
print("RELAXED CRITERIA: Only p-value < 0.10 (ignore HL in trading)")
print("="*80)

validated_relaxed = 0
for etf_a, etf_b, f_pval, f_hl, f_hr, corr in pairs_found:
    price_a = trading_data[etf_a].dropna()
    price_b = trading_data[etf_b].dropna()
    common_idx = price_a.index.intersection(price_b.index)
    
    if len(common_idx) < 100:
        continue
    
    eg_t = run_engle_granger(price_a.loc[common_idx], price_b.loc[common_idx], use_log=True)
    
    if eg_t.pvalue < 0.10:
        validated_relaxed += 1
        t_hl = eg_t.half_life if eg_t.half_life else float('inf')
        print(f"{etf_a}-{etf_b}: p={eg_t.pvalue:.4f}, HL={t_hl:.1f}d")

print()
print(f"Validated with relaxed criteria: {validated_relaxed}/{len(pairs_found)}")
