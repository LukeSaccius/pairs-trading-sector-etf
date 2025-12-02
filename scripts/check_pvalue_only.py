"""Check rolling consistency with p-value only (no half-life filter)."""

import pandas as pd
import sys
sys.path.insert(0, 'src')

from pairs_trading_etf.cointegration.engle_granger import run_engle_granger

print('='*70)
print('ROLLING CONSISTENCY - P-VALUE ONLY (no half-life filter)')
print('='*70)

prices = pd.read_csv('data/raw/etf_prices_fresh.csv', index_col=0, parse_dates=True)

WINDOW = 252
STEP = 63
PVALUE_THRESHOLD = 0.10

test_pairs = [
    ('SPY', 'VOO'), ('SPY', 'IVV'), ('GLD', 'IAU'),
    ('TLT', 'IEF'), ('EEM', 'VWO'),
    ('XLF', 'VFH'), ('XLE', 'VDE'),
]

results = []

for ticker_a, ticker_b in test_pairs:
    if ticker_a not in prices.columns or ticker_b not in prices.columns:
        continue
    
    price_a = prices[ticker_a].dropna()
    price_b = prices[ticker_b].dropna()
    
    common_idx = price_a.index.intersection(price_b.index)
    price_a = price_a.loc[common_idx]
    price_b = price_b.loc[common_idx]
    
    n_windows = 0
    n_significant = 0
    half_lives = []
    
    for start in range(0, len(price_a) - WINDOW, STEP):
        end = start + WINDOW
        
        window_a = price_a.iloc[start:end]
        window_b = price_b.iloc[start:end]
        
        try:
            result = run_engle_granger(window_a, window_b, use_log=True)
            
            n_windows += 1
            if result.pvalue < PVALUE_THRESHOLD:
                n_significant += 1
                if result.half_life:
                    half_lives.append(result.half_life)
        except Exception:
            pass
    
    if n_windows > 0:
        consistency = n_significant / n_windows * 100
        avg_hl = sum(half_lives) / len(half_lives) if half_lives else None
        results.append({
            'pair': f'{ticker_a}-{ticker_b}',
            'consistency_pct': consistency,
            'n_sig': n_significant,
            'n_win': n_windows,
            'avg_hl': avg_hl
        })

print()
print('Results (p-value < 0.10 only, no HL filter):')
print('-'*70)
for r in sorted(results, key=lambda x: x['consistency_pct'], reverse=True):
    hl_str = f"{r['avg_hl']:.0f}d" if r['avg_hl'] else 'N/A'
    status = 'OK' if r['consistency_pct'] >= 70 else 'LOW'
    print(f"[{status}] {r['pair']:12s}: {r['consistency_pct']:5.1f}% ({r['n_sig']}/{r['n_win']}) | Avg HL: {hl_str}")

print()
print('KEY INSIGHT:')
print('  When including half-life 15-120d filter: ~0% consistency')
print('  Without half-life filter: Higher consistency BUT...')
print('  Half-lives are often very long (not tradeable)')
