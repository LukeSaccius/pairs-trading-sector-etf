"""Check rolling consistency with fresh data."""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'src')

from pairs_trading_etf.cointegration.engle_granger import run_engle_granger

print('='*70)
print('ROLLING CONSISTENCY CHECK (252d lookback, 252d window)')
print('='*70)

# Load fresh data
prices = pd.read_csv('data/raw/etf_prices_fresh.csv', index_col=0, parse_dates=True)
print(f'Data: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices.columns)} ETFs')

# Parameters
WINDOW = 252
STEP = 63  # Quarterly
PVALUE_THRESHOLD = 0.10
MIN_HL = 15
MAX_HL = 120

# Test pairs
test_pairs = [
    ('SPY', 'VOO'), ('SPY', 'IVV'), ('GLD', 'IAU'),  # Should be high consistency
    ('TLT', 'IEF'), ('EEM', 'VWO'),  # Should be decent
    ('XLF', 'VFH'), ('XLE', 'VDE'),  # Same sector
    ('XLU', 'SPLV'), ('XLY', 'USMV'),  # From previous analysis
    ('XLK', 'XLF'), ('SPY', 'TLT'),  # Different sectors (should fail)
]

print(f'Testing {len(test_pairs)} pairs...')
print()

results = []

for ticker_a, ticker_b in test_pairs:
    if ticker_a not in prices.columns or ticker_b not in prices.columns:
        print(f'Warning: {ticker_a}-{ticker_b}: Missing data')
        continue
    
    price_a = prices[ticker_a].dropna()
    price_b = prices[ticker_b].dropna()
    
    # Align
    common_idx = price_a.index.intersection(price_b.index)
    price_a = price_a.loc[common_idx]
    price_b = price_b.loc[common_idx]
    
    if len(price_a) < WINDOW * 2:
        print(f'Warning: {ticker_a}-{ticker_b}: Not enough data')
        continue
    
    n_windows = 0
    n_significant = 0
    
    for start in range(0, len(price_a) - WINDOW, STEP):
        end = start + WINDOW
        
        window_a = price_a.iloc[start:end]
        window_b = price_b.iloc[start:end]
        
        try:
            result = run_engle_granger(window_a, window_b, use_log=True)
            
            is_significant = (
                result.pvalue < PVALUE_THRESHOLD and 
                result.half_life is not None and
                MIN_HL <= result.half_life <= MAX_HL
            )
            
            n_windows += 1
            if is_significant:
                n_significant += 1
        except Exception:
            continue
    
    if n_windows > 0:
        consistency = n_significant / n_windows * 100
        results.append({
            'pair': f'{ticker_a}-{ticker_b}',
            'n_windows': n_windows,
            'n_significant': n_significant,
            'consistency_pct': consistency
        })

# Sort and display
results_df = pd.DataFrame(results).sort_values('consistency_pct', ascending=False)

print('Results by Consistency:')
print('-'*60)
for _, row in results_df.iterrows():
    status = 'OK' if row['consistency_pct'] >= 70 else ('MED' if row['consistency_pct'] >= 30 else 'LOW')
    pct = row['consistency_pct']
    sig = row['n_significant']
    win = row['n_windows']
    print(f"[{status:3s}] {row['pair']:12s}: {pct:5.1f}% ({sig}/{win} windows)")

print()
print('='*70)
print('SUMMARY')
print('='*70)
high = (results_df['consistency_pct'] >= 70).sum()
med = (results_df['consistency_pct'] >= 50).sum()
low = (results_df['consistency_pct'] >= 30).sum()
avg = results_df['consistency_pct'].mean()

print(f"Total pairs: {len(results_df)}")
print(f"Pairs >= 70% consistency: {high}")
print(f"Pairs >= 50% consistency: {med}")
print(f"Pairs >= 30% consistency: {low}")
print(f"Average consistency: {avg:.1f}%")

# Save
results_df.to_csv('results/rolling_consistency_fresh.csv', index=False)
print()
print('Saved: results/rolling_consistency_fresh.csv')
