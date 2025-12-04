#!/usr/bin/env python
"""Debug script to compare Kalman vs OLS spreads and understand why Kalman triggers regime_break."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pairs_trading_etf.backtests.engine import estimate_kalman_hedge_ratio

def compare_kalman_vs_ols(prices: pd.DataFrame, pair: tuple, formation_end: str):
    """Compare Kalman vs OLS spreads for a single pair."""
    leg_x, leg_y = pair
    
    # Get log prices
    log_x = np.log(prices[leg_x])
    log_y = np.log(prices[leg_y])
    
    # Formation period for OLS hedge ratio
    formation_data = log_x.loc[:formation_end].dropna()
    formation_y = log_y.loc[:formation_end].dropna()
    
    # Align indices
    common_idx = formation_data.index.intersection(formation_y.index)
    formation_data = formation_data.loc[common_idx]
    formation_y = formation_y.loc[common_idx]
    
    # OLS hedge ratio (static) using statsmodels for stability
    from statsmodels.regression.linear_model import OLS
    import statsmodels.api as sm
    
    X = sm.add_constant(formation_data.values)
    y = formation_y.values
    model = OLS(y, X).fit()
    ols_intercept = model.params[0]
    ols_hr = model.params[1]
    
    # OLS spread (static hedge ratio)
    ols_spread = log_y - ols_intercept - ols_hr * log_x
    
    # Kalman spread (time-varying hedge ratio)
    kalman_df = estimate_kalman_hedge_ratio(
        prices[leg_x], prices[leg_y],
        use_log=True, delta=0.00001, vw=0.001,
        use_momentum=True
    )
    
    kalman_hr = kalman_df['hedge_ratio']
    kalman_intercept = kalman_df['intercept']
    kalman_spread = log_y - kalman_intercept - kalman_hr * log_x
    
    return {
        'ols_spread': ols_spread,
        'ols_hr': ols_hr,
        'ols_intercept': ols_intercept,
        'kalman_spread': kalman_spread,
        'kalman_hr': kalman_hr,
        'kalman_intercept': kalman_intercept,
    }


def analyze_regime_breaks(ols_spread: pd.Series, kalman_spread: pd.Series, 
                          trading_start: str, zscore_lookback: int = 60):
    """Analyze how often each spread triggers regime break."""
    trading_data_ols = ols_spread.loc[trading_start:]
    trading_data_kalman = kalman_spread.loc[trading_start:]
    
    # Rolling z-score
    def rolling_zscore(spread, lookback):
        mean = spread.rolling(window=lookback, min_periods=30).mean()
        std = spread.rolling(window=lookback, min_periods=30).std()
        return (spread - mean) / std
    
    ols_z = rolling_zscore(trading_data_ols, zscore_lookback)
    kalman_z = rolling_zscore(trading_data_kalman, zscore_lookback)
    
    # Count sign changes (proxy for regime breaks)
    ols_sign_changes = (trading_data_ols * trading_data_ols.shift(1) < 0).sum()
    kalman_sign_changes = (trading_data_kalman * trading_data_kalman.shift(1) < 0).sum()
    
    # Count how many times z-score crosses 2 and then spread changes sign
    entry_threshold = 2.0
    
    print(f"\n{'='*60}")
    print(f"Regime Break Analysis")
    print(f"{'='*60}")
    print(f"OLS spread sign changes: {ols_sign_changes}")
    print(f"Kalman spread sign changes: {kalman_sign_changes}")
    print(f"Kalman has {kalman_sign_changes / max(ols_sign_changes, 1):.2f}x more sign changes")
    
    # Spread statistics
    print(f"\nOLS spread - mean: {trading_data_ols.mean():.6f}, std: {trading_data_ols.std():.6f}")
    print(f"Kalman spread - mean: {trading_data_kalman.mean():.6f}, std: {trading_data_kalman.std():.6f}")
    
    return ols_z, kalman_z


def main():
    # Load data
    prices = pd.read_csv("data/raw/etf_prices_fresh.csv", index_col=0, parse_dates=True)
    
    # Test on a few pairs that showed up in backtest
    test_pairs = [
        (("KBE", "IAI"), "2009-12-31", "2010-01-04"),
        (("XLY", "XRT"), "2009-12-31", "2010-01-04"),
        (("EWG", "EWU"), "2009-12-31", "2010-01-04"),
    ]
    
    fig, axes = plt.subplots(len(test_pairs), 3, figsize=(15, 4*len(test_pairs)))
    if len(test_pairs) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (pair, formation_end, trading_start) in enumerate(test_pairs):
        print(f"\n{'='*60}")
        print(f"Pair: {pair}")
        print(f"{'='*60}")
        
        results = compare_kalman_vs_ols(prices, pair, formation_end)
        
        # Compare spreads
        trading_ols = results['ols_spread'].loc[trading_start:"2010-12-31"]
        trading_kalman = results['kalman_spread'].loc[trading_start:"2010-12-31"]
        
        # Rolling z-scores
        def rolling_zscore(spread, lookback=60):
            mean = spread.rolling(window=lookback, min_periods=30).mean()
            std = spread.rolling(window=lookback, min_periods=30).std()
            return (spread - mean) / std
        
        ols_z = rolling_zscore(trading_ols)
        kalman_z = rolling_zscore(trading_kalman)
        
        # Plot spreads
        ax1 = axes[i, 0]
        ax1.plot(trading_ols.index, trading_ols.values, label='OLS spread', alpha=0.7)
        ax1.plot(trading_kalman.index, trading_kalman.values, label='Kalman spread', alpha=0.7)
        ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax1.set_title(f'{pair} - Spreads')
        ax1.legend()
        
        # Plot z-scores
        ax2 = axes[i, 1]
        ax2.plot(ols_z.index, ols_z.values, label='OLS z-score', alpha=0.7)
        ax2.plot(kalman_z.index, kalman_z.values, label='Kalman z-score', alpha=0.7)
        ax2.axhline(2, color='r', linestyle='--', alpha=0.3)
        ax2.axhline(-2, color='g', linestyle='--', alpha=0.3)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax2.set_title(f'{pair} - Z-scores')
        ax2.legend()
        
        # Plot hedge ratios
        ax3 = axes[i, 2]
        trading_kalman_hr = results['kalman_hr'].loc[trading_start:"2010-12-31"]
        ax3.plot(trading_kalman_hr.index, trading_kalman_hr.values, label='Kalman HR', alpha=0.7)
        ax3.axhline(results['ols_hr'], color='r', linestyle='--', label=f'OLS HR={results["ols_hr"]:.3f}')
        ax3.set_title(f'{pair} - Hedge Ratios')
        ax3.legend()
        
        # Print statistics
        print(f"\nOLS hedge ratio: {results['ols_hr']:.4f}")
        print(f"OLS intercept: {results['ols_intercept']:.4f}")
        print(f"Kalman HR range: [{trading_kalman_hr.min():.4f}, {trading_kalman_hr.max():.4f}]")
        
        # Analyze regime breaks
        analyze_regime_breaks(results['ols_spread'], results['kalman_spread'], trading_start)
    
    plt.tight_layout()
    plt.savefig("results/figures/kalman_vs_ols_debug.png", dpi=150)
    print(f"\nSaved figure to results/figures/kalman_vs_ols_debug.png")


if __name__ == "__main__":
    main()
