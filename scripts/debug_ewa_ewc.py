"""
Debug EWA-EWC - Known working pair
Check each step of our logic
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'src')

from pairs_trading_etf.cointegration.engle_granger import run_engle_granger
from pairs_trading_etf.ou_model.estimation import estimate_ou_parameters

# Load data
prices = pd.read_csv('data/raw/etf_prices_fresh.csv', index_col=0, parse_dates=True)

print("="*80)
print("DEBUG: EWA-EWC (Known Working Pair)")
print("="*80)

# Check data availability
print("Data check:")
print(f"  EWA: {prices['EWA'].notna().sum()} days")
print(f"  EWC: {prices['EWC'].notna().sum()} days")

# Get aligned prices
price_ewa = prices['EWA'].dropna()
price_ewc = prices['EWC'].dropna()
common_idx = price_ewa.index.intersection(price_ewc.index)
price_ewa = price_ewa.loc[common_idx]
price_ewc = price_ewc.loc[common_idx]

print(f"  Aligned: {len(common_idx)} days from {common_idx[0].date()} to {common_idx[-1].date()}")
print()

# Correlation
corr = price_ewa.pct_change().corr(price_ewc.pct_change())
print(f"Return Correlation: {corr:.4f}")

# ========================================
# FULL HISTORY TEST
# ========================================
print()
print("="*80)
print("FULL HISTORY TEST (All available data)")
print("="*80)
eg_full = run_engle_granger(price_ewa, price_ewc, use_log=True)
print(f"P-value: {eg_full.pvalue:.6f}")
print(f"Half-life (from EG): {eg_full.half_life}")
print(f"Hedge ratio: {eg_full.hedge_ratio:.4f}")
print(f"Test statistic: {eg_full.test_statistic:.4f}")

# Compute spread manually
log_ewa = np.log(price_ewa)
log_ewc = np.log(price_ewc)
spread_full = log_ewa - eg_full.hedge_ratio * log_ewc

print()
print("Spread statistics (full):")
print(f"  Mean: {spread_full.mean():.6f}")
print(f"  Std: {spread_full.std():.6f}")

# Manual half-life using our function
ou_params = estimate_ou_parameters(spread_full)
print(f"  HL via estimate_ou_parameters(): {ou_params.half_life:.1f} days")

# ========================================
# YEARLY ANALYSIS
# ========================================
print()
print("="*80)
print("YEARLY ANALYSIS (Formation -> Check Trading)")
print("="*80)

years = range(2007, 2024)

for year in years:
    # Formation period
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    
    mask = (price_ewa.index >= start) & (price_ewa.index <= end)
    ewa_year = price_ewa.loc[mask]
    ewc_year = price_ewc.loc[mask]
    
    if len(ewa_year) < 200:
        continue
    
    try:
        eg = run_engle_granger(ewa_year, ewc_year, use_log=True)
        hl = eg.half_life if eg.half_life else float('inf')
        
        status = ""
        if eg.pvalue < 0.10:
            if hl >= 15 and hl <= 120:
                status = "TRADEABLE"
            elif hl < 15:
                status = "HL too short"
            else:
                status = f"HL too long ({hl:.0f}d)"
        else:
            status = "Not cointegrated"
        
        print(f"{year}: p={eg.pvalue:.4f}, HL={hl:.1f}d, HR={eg.hedge_ratio:.3f} [{status}]")
    except Exception as e:
        print(f"{year}: Error - {e}")

# ========================================
# CHECK OUR ENGLE-GRANGER IMPLEMENTATION
# ========================================
print()
print("="*80)
print("VERIFY ENGLE-GRANGER IMPLEMENTATION")
print("="*80)

# Read the source
from pairs_trading_etf.cointegration.engle_granger import run_engle_granger
import inspect
print("run_engle_granger source location:")
print(inspect.getfile(run_engle_granger))

# Manual ADF test for comparison
from statsmodels.tsa.stattools import adfuller, coint

# Test 2010 data
ewa_2010 = price_ewa.loc["2010-01-01":"2010-12-31"]
ewc_2010 = price_ewc.loc["2010-01-01":"2010-12-31"]

print()
print("2010 Data - Comparing our EG vs statsmodels coint:")
print(f"  Observations: {len(ewa_2010)}")

# Our implementation
eg_2010 = run_engle_granger(ewa_2010, ewc_2010, use_log=True)
print(f"  Our EG p-value: {eg_2010.pvalue:.6f}")

# Statsmodels coint (Engle-Granger)
log_ewa_2010 = np.log(ewa_2010)
log_ewc_2010 = np.log(ewc_2010)
coint_stat, coint_pvalue, coint_crit = coint(log_ewa_2010, log_ewc_2010)
print(f"  statsmodels coint p-value: {coint_pvalue:.6f}")

# Check spread stationarity manually
spread_2010 = log_ewa_2010 - eg_2010.hedge_ratio * log_ewc_2010
adf_result = adfuller(spread_2010, maxlag=1)
print(f"  ADF on spread p-value: {adf_result[1]:.6f}")

# ========================================
# CHECK HALF-LIFE CALCULATION
# ========================================
print()
print("="*80)
print("VERIFY HALF-LIFE CALCULATION")
print("="*80)

# Our estimate_ou_parameters
ou_2010 = estimate_ou_parameters(spread_2010)
print(f"Our estimate_ou_parameters(spread_2010): {ou_2010.half_life:.1f} days")

# Manual calculation
spread = spread_2010
delta = spread.diff().dropna()
spread_lag = spread.shift(1).dropna()

# Align
delta = delta.iloc[1:]
spread_lag = spread_lag.iloc[:-1]

# OLS: delta = a + b * spread_lag
X = np.column_stack([np.ones(len(spread_lag)), spread_lag.values])
y = delta.values
beta = np.linalg.lstsq(X, y, rcond=None)[0]
a, b = beta[0], beta[1]

print(f"Manual regression: a={a:.6f}, b={b:.6f}")

if b < 0 and (1 + b) > 0:
    hl_manual = -np.log(2) / np.log(1 + b)
    print(f"Manual HL = -ln(2)/ln(1+b) = {hl_manual:.1f} days")
else:
    print(f"b={b:.6f} -> Not mean-reverting or invalid")

# Alternative: AR(1) on spread level
from statsmodels.tsa.ar_model import AutoReg

ar_model = AutoReg(spread_2010.values, lags=1).fit()
phi = ar_model.params[1]  # AR(1) coefficient
print(f"AR(1) phi coefficient: {phi:.6f}")
if phi < 1 and phi > 0:
    hl_ar = -np.log(2) / np.log(phi)
    print(f"HL from AR(1) phi: {hl_ar:.1f} days")
