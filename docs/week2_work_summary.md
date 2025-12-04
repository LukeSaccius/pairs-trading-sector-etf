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

---

## Day 3: December 3, 2025 (continued)

### Sessions 11-13: Kalman Filter Investigation & Parameter Optimization

#### Major Accomplishments

1. **Kalman Filter Deep Dive**

   **Motivation:** V15 với Kalman filter thất bại hoàn toàn (-$8,686 PnL, 29.4% win rate). Mục tiêu: tìm hiểu tại sao.

   **Experiments Conducted:**
   
   | Version | Configuration | PnL | Win Rate | Issue |
   |---------|---------------|-----|----------|-------|
   | V15b | No Kalman (OLS only) | +$5,241 | 69.1% | ✅ Works |
   | V15c v1 | Basic Kalman | -$8,686 | 29.4% | All trades timeout |
   | V15c v2 | Kalman + Adaptive R | -$8,720 | 29.0% | Same issue |
   | V15c v3 | Momentum Model | -$8,686 | 29.4% | Same issue |

   **Root Cause Discovery:**
   
   Forensic analysis cho thấy Kalman spread có 50-100x nhiều lần đổi dấu hơn OLS spread:
   
   | Metric | OLS Spread | Kalman Spread |
   |--------|------------|---------------|
   | Sign Changes (GLD-GDX) | 11 | 1,162 |
   | Std Dev | 0.24 | 0.002 |
   
   **Lý do kỹ thuật:**
   - Kalman hedge ratio β_t thay đổi liên tục
   - Spread = y - β_t × x oscillates quanh 0 rất nhanh
   - Rolling z-score trên chuỗi không stationary → vô nghĩa
   - Z-score không bao giờ exit conditions → trades timeout sau 130 ngày

   **So sánh với Literature (Palomar Chapter 15):**
   - Palomar dùng Kalman cho price prediction
   - Momentum model dự đoán xu hướng, không phải mean-reversion
   - **Kết luận:** Kalman KHÔNG phù hợp cho z-score based pairs trading

   **Files Created:**
   - `docs/kalman_analysis_summary.md` - Chi tiết phân tích
   - `scripts/debug_kalman_vs_ols.py` - So sánh Kalman vs OLS spreads

2. **Sensitivity Analysis - Entry Threshold & Position Sizing**

   **Problem:** V15b chỉ đạt 0.70% annualized return vs SPY 13.44%
   
   **Experiment Setup:**
   - Entry z-score: [1.5, 2.0, 2.5, 2.8, 3.0]
   - Max positions: [5, 8, 10, 15]
   - Capital per pair: [10k, 15k, 20k]
   - Total: 60 combinations tested

   **Results - Top 5 Configurations:**
   
   | Entry Z | Max Pos | PnL | Win Rate | Profit Factor | Annualized |
   |---------|---------|-----|----------|---------------|------------|
   | 2.8 | 5 | $9,189 | 62.8% | 2.70 | 1.19% |
   | 2.5 | 5 | $8,969 | 56.4% | 1.99 | 1.16% |
   | 3.0 | 5 | $7,110 | 52.0% | 2.89 | 0.92% |
   | 2.5 | 8 | $5,606 | 52.7% | 1.81 | 0.72% |
   | 2.8 | 8 | $5,241 | 69.1% | 2.47 | 0.70% |

   **Key Insights:**
   
   - **Entry Z = 2.8 optimal**: Best balance between signal quality và trade frequency
   - **Entry Z = 1.5 loses money**: Too many false signals (-$3,431 avg PnL)
   - **Max Positions = 5 best**: Capital concentration on best opportunities
   - **Capital per pair không ảnh hưởng**: Do compounding + vol_sizing override

   **Files Created:**
   - `scripts/sensitivity_entry_position.py` - Grid search script
   - `results/sensitivity_entry_position.csv` - Full results

3. **Capital Utilization Problem Analysis**

   **Issue:** Dù tối ưu, strategy vẫn chỉ đạt 1.19% annualized
   
   **Root Cause:**
   - Chỉ có 74 trades trong 14 năm = 5.2 trades/năm
   - Entry z-score = 2.8 = signal rất hiếm trong ETF universe
   - Capital idle phần lớn thời gian
   
   **Comparison:**
   | Strategy | Annualized Return | Effort |
   |----------|-------------------|--------|
   | SPY Buy & Hold | 13.44% | None |
   | V15b Baseline | 0.70% | Full |
   | V15b Optimized | 1.19% | Full |

---

## Updated Conclusions

### What Works
1. **V15b (No Kalman)**: Best performer with $5,241 PnL, 69% win rate
2. **Entry z-score 2.8**: Optimal threshold, 62.8% win rate
3. **Max positions 5**: Concentrate capital on best opportunities
4. **Sector focus**: EUROPE pairs most stable
5. **Vol-sizing**: Dynamically adjusts position based on volatility

### What Doesn't Work
1. **Kalman Filter**: 50-100x more spread sign changes, breaks z-score signals
2. **ETF-only universe**: Not enough mean-reversion opportunities
3. **Low entry threshold (z=1.5)**: Too many false signals, loses money
4. **High max positions (15+)**: Over-diversification, dilutes returns

### Final Strategy Performance

| Metric | V15b Baseline | V16 Optimized |
|--------|---------------|---------------|
| Total PnL | $5,241 | **$8,602** |
| Win Rate | 69.1% | 69.1% |
| Profit Factor | 2.47 | 2.43 |
| Annualized | 0.70% | **1.10%** |
| vs SPY | -12.74% | -12.34% |

### Honest Assessment

> **ETF pairs trading with cointegration approach cannot beat SPY.**
>
> Best achievable: ~1.1% annualized return (after extensive optimization)
> SPY benchmark: ~13.4% annualized return
>
> The strategy may have value as:
> - Market-neutral component in portfolio
> - Crisis hedge (performs better in high volatility)
> - Academic exercise in statistical arbitrage
>
> But NOT as primary alpha source.

---

## Sessions 14-15: V16 Implementation & Cleanup

### 4. Project Cleanup

Removed unused/empty folders and archived debug scripts:

**Deleted (Empty Folders):**
```
src/backtests/
src/data/
src/features/
src/models/
src/pipelines/
src/utils/
```

**Archived Scripts:**
```
scripts/archive/
├── compare_zscore_approaches.py
├── debug_capital_flow.py
├── debug_kalman_vs_ols.py
├── debug_trades.py
├── deep_debug.py
├── forensic_analysis.py
└── quick_compare.py
```

**Archived Configs:**
```
configs/experiments/archive/
├── v10_risk_managed.yaml
├── v11_crisis_aware.yaml
├── v15_full_features.yaml
└── v15c_kalman_momentum.yaml
```

**Active Configs:**
- `default.yaml` - Base config
- `v14_vidyamurthy_full.yaml` - Vidyamurthy framework
- `v15b_vix_volsizing.yaml` - Previous best
- `v16_optimized.yaml` - **Current best**

### 5. V16 Implementation

**Config Changes:**
| Parameter | V15b | V16 | Reason |
|-----------|------|-----|--------|
| `max_positions` | 8 | **5** | Concentrate capital |
| `max_capital_per_trade` | 15000 | **25000** | Larger positions |
| `use_vix_filter` | false | **true** | Risk management |
| `vix_threshold` | N/A | **30** | Halt in high vol |

**VIX Data Added:**
- Downloaded ^VIX from Yahoo Finance
- Added to `data/raw/etf_prices_fresh.csv`
- VIX range: 9.14 - 82.69
- 435 days with VIX > 30

**V16 Results:**
| Metric | Value |
|--------|-------|
| Total PnL | $8,602 |
| Total Trades | 68 |
| Win Rate | 69.1% |
| Profit Factor | 2.43 |
| Annualized | ~1.10% |

### 6. Capital Flow Debug

**Issue:** `capital_per_pair` không ảnh hưởng PnL

**Root Cause:**
```python
if cfg.compounding:
    # capital_per_pair IGNORED!
    position_capital = (current_capital * leverage) / max_positions
else:
    position_capital = cfg.capital_per_pair * leverage
```

**Recommendation:** Rename or document that `capital_per_pair` only works when `compounding=false`

---

## Files Created/Modified (Day 3)

### New Scripts
- `scripts/debug_kalman_vs_ols.py` → archived
- `scripts/sensitivity_entry_position.py` - Parameter grid search
- `scripts/debug_capital_flow.py` → archived

### New Documentation
- `docs/kalman_analysis_summary.md` - Kalman failure analysis

### New Results
- `results/sensitivity_entry_position.csv` - 60 configuration results
- `results/2025-12-03_15-56_v16_optimized/` - V16 backtest results
- `results/2025-12-03_15-59_v16_optimized/` - V16 with VIX filter

### New Configs
- `configs/experiments/v16_optimized.yaml` - **Current best config**

### Data Updated
- `data/raw/etf_prices_fresh.csv` - Added VIX column (119 columns now)
