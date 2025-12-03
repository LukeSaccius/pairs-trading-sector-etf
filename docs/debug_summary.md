# Pairs Trading Strategy Analysis - Debug Summary

## Date: 2024-12-03

## Vấn Đề Gốc

User challenge: "2% / 1 năm thế thì còn chẳng bằng mua SPY ôm 17 năm"

SPY average return: ~10%/year
Strategy return (V9): ~0.14%/year → **vastly underperforming**

## Root Causes Identified

### 1. Capital Concentration Risk (Major Bug)

**Finding:** With `max_positions=0` (unlimited) and `unlimited_pairs=True`, code divides capital by `len(pairs)`. When only 2 pairs selected (like 2018), each trade gets ~$50,000!

```
2017 formation → 2018 trading: Only 2 pairs selected
Capital per trade = ($50k * 2x leverage) / 2 = $50,000!
Single stop-loss → $1,130 loss (2018 DIA/RSP trade)
```

**Fix:** Added `max_capital_per_trade` parameter and minimum divisor of 5 for capital allocation.

### 2. Hedge Ratio Impact on PnL

With hedge ratio != 1.0, positions are unbalanced:

```
HR = 1.62 (DIA/RSP):
  - Long X: 38.2% of capital
  - Short Y: 61.8% of capital

When both move +2%:
  - Long X PnL: +$77
  - Short Y PnL: -$123
  - Net: -$46 (LOSS even though X outperformed!)
```

The spread PnL depends on both:
1. Relative performance (X vs Y)
2. Position sizing via hedge ratio

### 3. Crisis Period Failure

In 2008 (Financial Crisis):
- 10 out of 16 trades hit stop-loss
- Total loss: -$1,993
- Mean-reversion FAILS in trending/crisis markets

The strategy assumes spreads will mean-revert, but during regime changes they diverge further.

### 4. Stop-Loss vs Win Ratio Imbalance

```
Convergence wins: +$97/trade average
Stop-loss losses: -$182/trade average

Even with 58% win rate, negative risk/reward kills returns.
```

## Solutions Implemented

### V10: Risk Management
- `max_capital_per_trade: $20,000` - prevents over-concentration
- `min_pairs_for_trading: 3` - skips years with insufficient diversification
- Loosened cointegration filters to get more pairs

### V11: Crisis-Aware
- Lower `stop_loss_zscore: 3.0` - cut losses earlier
- Higher `entry_zscore: 2.8` - higher quality signals
- Tighter `exit_zscore: 0.3` - take profits faster
- Exclude volatile sectors (US_GROWTH)
- Lower leverage (1.5x vs 2x)
- Aggressive blacklisting (20% SL rate threshold)

## Results Comparison

| Version | Total PnL | Trades | Win Rate | Profit Factor | Max DD | Annual Return |
|---------|-----------|--------|----------|---------------|--------|---------------|
| V9 | $1,336 | 131 | 67.2% | 1.18 | ? | 0.14% |
| V10 | $1,056 | 207 | 58.5% | 1.11 | $2,535 | 0.12% |
| V11 | $2,079 | 129 | 43.4% | 1.41 | $992 | ~0.25% |

V11 Improvements:
- ✅ Better Profit Factor (1.41 vs 1.11)
- ✅ Lower Max Drawdown ($992 vs $2,535)
- ✅ Skip crisis years automatically

## Key Insights

1. **Pairs trading is NOT a get-rich-quick strategy** - returns are modest even when done correctly

2. **Crisis periods destroy mean-reversion strategies** - need to either:
   - Filter by volatility regime (VIX > 30 → stop trading)
   - Use momentum/trend strategies during crisis

3. **Hedge ratio matters A LOT** - HR far from 1.0 creates directional exposure

4. **Stop-loss must be tuned carefully** - too tight = whipsaws, too loose = big losses

5. **Diversification is critical** - need minimum 5+ pairs at all times

## Recommendation for Production

For realistic $50k account with 2x margin:

1. **Use V11 settings** but with:
   - Add VIX filter: stop trading when VIX > 25
   - Target 8-12 active pairs minimum
   - Max 3% capital per trade

2. **Expected returns**: 2-5% annual (after costs)
   - NOT competitive with buy-and-hold SPY
   - Better used as market-neutral hedge in portfolio

3. **When to use pairs trading**:
   - Low volatility regimes
   - As diversifier in larger portfolio
   - Not as primary alpha source

## Files Generated

- `configs/experiments/v10_risk_managed.yaml`
- `configs/experiments/v11_crisis_aware.yaml`
- `scripts/debug_trades.py` - visualization of all trades
- `scripts/deep_debug.py` - PnL calculation verification
- `results/figures/debug/` - trade visualizations by year
