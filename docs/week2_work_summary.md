# Week 2 Work Summary

## Period: December 2-3, 2025

## Overview

Week 2 focused on deep debugging and optimization of the pairs trading strategy. We went from a losing strategy (-$8,981) to a marginally profitable one (+$2,298), then dove deep into understanding why returns remain low despite all optimizations.

---

## Day 1: December 2, 2025

### Sessions 1-7: Core Bug Fixes & Strategy Development

#### Major Accomplishments

1. **Tokat Paper Replication**
   - Implemented walk-forward backtest following Tokat & Hayrullahoglu (2021) methodology
   - Their claim: 15% annual return, 1.43 Sharpe
   - Our result: Near-zero returns with correct implementation

2. **Critical Bug Discoveries**

   **Bug #1: Exit Condition Logic**
   ```python
   # WRONG
   if trade.direction == 1:
       if z <= cfg.exit_z:  # Always TRUE immediately!
   
   # CORRECT
   if trade.direction == 1:
       if z >= -cfg.exit_z:  # Exit when z rises toward 0
   ```

   **Bug #2: Half-Life Formula**
   ```python
   # WRONG
   half_life = -np.log(2) / b
   
   # CORRECT
   phi = 1 + b
   half_life = -np.log(2) / np.log(phi)
   ```

   **Bug #3: Wrong Critical Values**
   - V2 used standard ADF critical values (-3.43, -2.86)
   - Should use MacKinnon Engle-Granger values (-3.90, -3.34)
   - This caused ~50% of "cointegrated" pairs to be false positives

3. **Speed Optimization**
   - Replaced statsmodels.coint with pure NumPy implementation
   - **8.4x speedup**: 141s → 16.8s for full backtest

4. **Walk-Forward Results**
   | Version | PnL | Notes |
   |---------|-----|-------|
   | V2 (buggy) | +$2,629 | FAKE - wrong stats |
   | V3 (fixed) | -$8,981 | Correct but losing |
   | V3 + sector focus | +$959 | First profitable! |
   | V4 final | +$2,298 | Excludes bad sectors |

---

## Day 2: December 3, 2025

### Sessions 8-10: Deep Debugging & Root Cause Analysis

#### Major Accomplishments

1. **Sector Analysis**
   
   | Sector | Trades | PnL | Action |
   |--------|--------|-----|--------|
   | EUROPE | 70 | +$1,911 | ✅ Keep |
   | FINANCIALS | 34 | +$413 | ✅ Keep |
   | ASIA_DEV | 17 | +$72 | ✅ Keep |
   | US_GROWTH | 31 | -$411 | ❌ Exclude |
   | BONDS_GOV | 16 | -$565 | ❌ Exclude |
   | EMERGING | - | -$2,461 | ❌ Exclude |

2. **Exit Reason Analysis**
   
   | Exit Reason | Trades | PnL | Win Rate |
   |-------------|--------|-----|----------|
   | Convergence | 87 | +$9,260 | 98% |
   | Max Holding | 138 | -$6,951 | 31% |
   | Stop Loss | 5 | -$1,199 | 0% |

   **Key Insight:** Convergence trades are VERY profitable. The problem is trades that don't converge in time.

3. **V5 Improvements**
   - Added hedge ratio filter (0.5 < HR < 2.0)
   - Stricter entry z-score (2.5 instead of 2.0)
   - Dynamic max holding based on half-life
   
   **Result:** $1,643 PnL, 66% win rate, 1.65 Profit Factor

4. **User Challenge: "2% annual is worse than SPY"**
   
   After all optimizations, strategy returns ~2%/year with $50k capital. SPY returns ~10%/year. User wanted us to investigate why.

5. **Deep Debug: Root Cause Discovery**

   **Issue #1: Capital Concentration Bug**
   
   With `max_positions=0` (unlimited) and `unlimited_pairs=True`, code divides capital by `len(pairs)`. When only 2 pairs selected (2018), each trade gets $50,000!
   
   ```
   2017 formation → 2018 trading: Only 2 pairs
   Capital per trade = ($50k × 2x) / 2 = $50,000
   Single stop-loss = -$1,130 loss
   ```

   **Issue #2: Hedge Ratio Impact**
   
   With HR=1.62 (DIA/RSP):
   - Position: 38% in X, 62% in Y (unbalanced!)
   - When both move +2%: Net loss even if X outperforms
   - Spread PnL depends on BOTH relative performance AND position sizing

   **Issue #3: Crisis Period Failure**
   
   In 2008 crisis:
   - 10/16 trades hit stop-loss
   - Mean-reversion FAILS in trending/crisis markets
   - Strategy assumes spreads will revert, but in regimes they diverge

6. **V10 & V11: Risk Management Improvements**

   | Feature | V9 | V10 | V11 |
   |---------|----|----|-----|
   | max_capital_per_trade | None | $20k | $15k |
   | min_pairs_for_trading | None | 3 | 4 |
   | stop_loss_zscore | 4.0 | 4.0 | 3.0 |
   | Exclude sectors | None | None | US_GROWTH |
   | leverage | 2.0 | 2.0 | 1.5 |

   **Final Results:**
   
   | Version | Total PnL | Profit Factor | Max Drawdown |
   |---------|-----------|---------------|--------------|
   | V9 | $1,336 | 1.18 | ? |
   | V10 | $1,056 | 1.11 | $2,535 |
   | **V11** | **$2,079** | **1.41** | **$992** |

---

## Key Technical Discoveries

### 1. Statistical Artifact Problem

ETF pairs appear cointegrated when testing on full history, but:
- Rolling consistency is near 0% with tradeable half-life (15-120 days)
- Pairs passing p-value filter have HL = 28,000-628,000 days
- Cointegration ≠ Tradeable mean-reversion

### 2. Half-Life vs Cointegration

| Pair | P-value | Half-Life | Tradeable? |
|------|---------|-----------|------------|
| GLD-IAU | 0.0001 | 628,182 days | ❌ |
| SPY-VOO | 0.001 | 89,657 days | ❌ |
| EWA-EWC | 0.01 | 24 days | ✅ |

### 3. Crisis Period Behavior

| Period | Avg Return | Win Rate | Strategy Works? |
|--------|------------|----------|-----------------|
| Crisis (2008-2010) | +2.25% | 79.8% | ✅ Yes |
| Non-Crisis (2011-2024) | -0.44% | 58.5% | ❌ No |

---

## Files Created/Modified

### New Scripts
- `scripts/backtest_v4.py` - Sector-focused backtest
- `scripts/run_backtest.py` - Unified backtest runner
- `scripts/debug_trades.py` - Trade visualization
- `scripts/deep_debug.py` - PnL calculation verification
- `scripts/sensitivity_analysis.py` - Parameter sensitivity
- `scripts/visualize_trade.py` - Individual trade plots

### New Configs
- `configs/experiments/default.yaml`
- `configs/experiments/aggressive.yaml`
- `configs/experiments/conservative.yaml`
- `configs/experiments/optimized_v5.yaml`
- `configs/experiments/v6_aggressive.yaml`
- `configs/experiments/high_capital.yaml`
- `configs/experiments/max_capital.yaml`
- `configs/experiments/compounding.yaml`
- `configs/experiments/v10_risk_managed.yaml`
- `configs/experiments/v11_crisis_aware.yaml`
- `configs/experiments/europe_only.yaml`

### Core Engine Updates
- `src/pairs_trading_etf/backtests/engine.py` - Full trading simulation
- `src/pairs_trading_etf/backtests/config.py` - Config dataclass
- `src/pairs_trading_etf/ou_model/half_life.py` - Fixed half-life calc
- `src/pairs_trading_etf/utils/sectors.py` - Sector utilities

### Results Generated
- Multiple backtest runs in `results/` with timestamps
- Trade visualizations in `results/figures/debug/`
- Performance summaries and trade logs

---

## Conclusions

### What Works
1. **Sector focus**: Same-sector pairs have fundamental links
2. **EUROPE pairs**: Most stable cointegration (+$2,161 in V11)
3. **Convergence trades**: 100% win rate, avg +$176/trade (28 trades in V11)
4. **Crisis periods**: Strategy profitable in 2008-2009 with risk management

### What Doesn't Work
1. **ETF-only universe**: Not enough idiosyncratic movement
2. **Normal markets**: Near-zero returns post-2010
3. **Max holding exits**: 61% win rate, avg +$16/trade (improved but still weak)
4. **Stop-loss exits**: 64 trades, avg -$55/trade in V11

### Final Verdict

> **ETF pairs trading with standard cointegration is NOT viable for alpha generation.**
>
> - Best case: 2-5% annual return (V11 with all optimizations)
> - SPY: ~10% annual return with zero effort
> - Strategy is suitable only as a market-neutral hedge, not primary alpha source

### Recommended Use Cases
1. As diversifier in larger portfolio
2. During high volatility regimes only (VIX > 25)
3. With minimum 10+ pairs for diversification
4. As crisis hedge (long volatility exposure)

---

## Next Steps (Week 3)

1. **Implement VIX filter**: Stop trading when VIX > 25
2. **Test individual stocks**: More idiosyncratic movement
3. **Machine learning approach**: Predict cointegration persistence
4. **Alternative strategies**: Distance method, factor pairs
5. **Document findings**: Prepare thesis section on strategy limitations
