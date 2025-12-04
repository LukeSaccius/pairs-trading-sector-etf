# Kalman Filter Analysis Summary

## Investigation: Why Kalman Filter Doesn't Work for Pairs Trading

### Background
Based on Palomar (2025) Chapter 15.6 "Kalman Filtering for Pairs Trading", we implemented:
1. **Basic Kalman (Section 15.6.3)**: State = [intercept, hedge_ratio]
2. **Momentum Kalman (Section 15.6.4)**: State = [intercept, hedge_ratio, velocity]

### Implementation Details
- Process noise: `Q = delta * I` (small delta = 1e-5 for stability)
- Adaptive observation variance: `R = 0.99 * R + 0.01 * innovation²`
- Joseph form covariance update for numerical stability
- Both basic and momentum models tested

### Test Results

| Version | Kalman Type | PnL | Win Rate | Issue |
|---------|-------------|-----|----------|-------|
| V15b | None (OLS) | **$5,241** | **69.1%** | Best performer |
| V15c v1 | Momentum + Full Palomar spread | $7,787 | 45.5% | 100% period_end exits |
| V15c v2 | Momentum + Spread sign change fix | -$1,441 | 24.7% | Too many regime_break |
| V15c v3 | Momentum + Z-score regime break | -$1,305 | 24.1% | All convergence exits (losers) |
| V15c v4 | Momentum + OLS-style spread | -$8,686 | 29.4% | Spread instability |

### Root Cause Analysis

#### Debug Script Results (`scripts/debug_kalman_vs_ols.py`)
```
Pair: ('KBE', 'IAI')
OLS spread sign changes: 34
Kalman spread sign changes: 1,751 (51.5x more!)

Pair: ('XLY', 'XRT')  
OLS spread sign changes: 19
Kalman spread sign changes: 1,821 (95.8x more!)
```

#### Key Findings

1. **Time-varying hedge ratio creates unstable spread**
   - With Kalman, spread changes daily even if prices don't change
   - This generates false trading signals

2. **Kalman spread with intercept has very small variance**
   - OLS spread std: ~0.24
   - Kalman spread std: ~0.002 (100x smaller)
   - Z-score calculation becomes meaningless

3. **Rolling z-score incompatible with time-varying hedge ratio**
   - Rolling window looks at historical spreads
   - But those spreads were calculated with different hedge ratios
   - Creates inconsistent signal series

4. **Excessive zero-crossings**
   - Kalman spread crosses zero 50-100x more than OLS spread
   - Regime break triggers constantly
   - Even with z-score based regime break, trades exit prematurely

### Why Palomar's Approach Works Differently

Palomar (2025) uses Kalman spread differently:
1. **Direct trading on Kalman spread** - Uses the innovation (residual) as signal
2. **No rolling z-score** - The spread itself is already normalized
3. **Different entry/exit logic** - Based on prediction error, not historical mean

Our implementation tried to retrofit Kalman into existing rolling z-score framework, which is fundamentally incompatible.

### Recommendation

**Use V15b (OLS-based) for production:**
- Static hedge ratio per trading year
- Rolling z-score on consistent spread
- Proven performance: $5,241 PnL, 69.1% win rate, 2.47 profit factor

**If pursuing Kalman, need complete redesign:**
1. Use Kalman innovation (prediction error) as trading signal
2. No rolling z-score - spread is already mean-zero
3. Entry/exit based on prediction confidence intervals
4. This requires rewriting the entire trading engine

### Files Modified
- `src/pairs_trading_etf/backtests/engine.py`: Added `estimate_kalman_hedge_ratio()` function
- `src/pairs_trading_etf/backtests/config.py`: Added Kalman config parameters
- `configs/experiments/v15c_kalman_momentum.yaml`: Kalman config (not recommended for use)

### Reference
- Palomar, D. (2025). *Portfolio Optimization: Theory and Application*. Chapter 15.6
- Kalman model equations preserved in code for future reference
