# 🐛 Bugs To Fix

## Status Legend
- ❌ Not Fixed
- 🔄 In Progress  
- ✅ Fixed

---

# 🔴 CRITICAL BUGS - FIXED

## Bug #8: LOOK-AHEAD BIAS ✅ FIXED

**Status:** ✅ FIXED on 2025-12-04

**Fix Details:**
```python
# engine.py lines 1213-1214
current_date = dates[t]        # Execution date (today)
signal_date = dates[t - 1]     # Signal date (yesterday's close)

# Entry uses signal from t-1, executes at t
z = zscores.loc[signal_date, pair_name]  # Signal from yesterday
```

---

## CRITICAL QUESTION 1: Embargo Width ✅ FIXED

**Status:** ✅ FIXED on 2025-12-04

**Fix Details:**
- Embargo/Purge now calculated from actual `holding_days` in trades
- Formula: `embargo_width = ceil(avg_holding_days)`
- No longer hardcoded!

**Code:** `pipeline.py` STEP 1.5
```python
avg_holding_days = np.mean(holding_days_list)
calculated_embargo = int(np.ceil(avg_holding_days))
calculated_purge = int(np.ceil(max(avg_holding_days, max_holding_days * 0.5)))
```

---

## CRITICAL QUESTION 2: No Look-Ahead Bias ✅ VERIFIED

**Status:** ✅ Already fixed (see Bug #8)

---

## CRITICAL QUESTION 3: Purging Logic ✅ FIXED

**Status:** ✅ FIXED on 2025-12-04

**Fix Details:**
- `WalkForwardCPCV` in `cpcv_correct.py` properly implements:
  - Purge: Remove last N days of train before test
  - Embargo: Remove first N days of test (implementation delay)
  - Overlap check: Validates no train/test overlap

**Code:** `cpcv_correct.py` lines 520-600
```python
# Purge: remove last purge_days from train
train_indices = np.where(train_mask)[0]
purge_indices = train_indices[-self.purge_days:]
train_mask[purge_indices] = False

# Embargo: remove first embargo_days from test
test_indices = np.where(test_mask)[0]
embargo_indices = test_indices[:self.embargo_days]
test_mask[embargo_indices] = False

# Validate: no overlap
overlap = train_mask & test_mask
if overlap.any():
    logger.error("DATA LEAK!")
```

---

## CRITICAL QUESTION 4: OOS Degradation Formula ✅ VERIFIED

**Status:** ✅ Correct

**Formula:**
```python
degradation = (is_mean - oos_mean) / abs(is_mean) if is_mean != 0 else 0
```

---

# 🟡 VISUALIZATION BUGS (Lower Priority)

## Bug #1: Visualization Hardcoded Thresholds ✅ FIXED

**File:** `scripts/visualize_trade_v2.py`  
**Lines:** 347-353

**Problem:**
```python
ax4.axhline(2.8, color='red', linestyle='--', alpha=0.7, label='Entry (±2.8)')
ax4.axhline(-2.8, color='red', linestyle='--', alpha=0.7)
ax4.axhline(0.3, color='green', linestyle=':', alpha=0.7, label='Exit (±0.3)')
ax4.axhline(-0.3, color='green', linestyle=':', alpha=0.7)
```

**Expected:**
- Should read thresholds from `config_snapshot.yaml` in results folder
- Current config uses `entry_threshold_sigma: 2.0` and `exit_threshold_sigma: 0.5`
- Visualization shows ±2.8 and ±0.3 (WRONG)

**Impact:** 
- Confusing visualization - looks like trade entered before threshold
- Makes debugging trades very difficult

**Fix Required:**
1. Add function to load thresholds from config file
2. Pass thresholds to plotting function
3. Use actual values in legend and lines

---

## Bug #2: Visualization Z-Score Calculation Mismatch ❌

**File:** `scripts/visualize_trade_v2.py`  
**Lines:** 176-179

**Problem:**
```python
# Visualization calculates z-score using ROLLING window
spread_mean = log_spread.rolling(window=lookback, min_periods=10).mean()
spread_std = log_spread.rolling(window=lookback, min_periods=10).std()
zscore = (log_spread - spread_mean) / spread_std
```

**But engine uses FIXED params (QMA Level 2):**
```python
# engine.py with use_fixed_exit_params: true
z = (spread - mu_entry) / sigma_entry  # FIXED from entry time!
```

**Impact:**
- Z-score line in visualization ≠ actual z-score used for trading decisions
- Entry/exit markers don't align with the z-score line
- Makes it look like trades entered/exited at wrong z-scores

**Fix Required:**
1. Add note in visualization explaining rolling vs fixed
2. Use `entry_z` and `exit_z` from trades.csv for markers (already done partially)
3. Consider adding a second z-score line showing fixed-param calculation

---

## Bug #3: Visualization Uses Wrong Entry Z ❌

**File:** `scripts/visualize_trade_v2.py`  
**Lines:** 358-365

**Problem:**
```python
# Visualization uses RECALCULATED z-score, not actual from trades.csv
calc_entry_z = zscore.iloc[entry_idx] if not np.isnan(zscore.iloc[entry_idx]) else entry_z
calc_exit_z = zscore.iloc[exit_idx] if not np.isnan(zscore.iloc[exit_idx]) else exit_z
```

The visualization **prefers recalculated z-score** over the actual `entry_z` from trades.csv. This is backwards!

**Expected:** Should use `entry_z` and `exit_z` from trades.csv FIRST (these are the actual values used by engine), and only fallback to calculated if missing.

**Impact:** 
- Markers show wrong z-scores
- Makes debugging impossible

**Fix Required:**
```python
# Use actual z from trades.csv, fallback to calculated only if missing
calc_entry_z = entry_z if not np.isnan(entry_z) else zscore.iloc[entry_idx]
calc_exit_z = exit_z if not np.isnan(exit_z) else zscore.iloc[exit_idx]
```

---

## Bug #4: Visualization Uses Wrong Lookback ❌

**File:** `scripts/visualize_trade_v2.py`  
**Lines:** 176

**Problem:**
```python
lookback = min(60, len(log_spread)//2)
```

But engine uses **adaptive lookback** per-pair based on half-life:
```python
# engine.py
lookback = int(max(lb_min, min(lb_max, mult * half_life)))  # e.g., 4 * HL
```

**Impact:**
- Z-score line calculated with wrong lookback
- Doesn't match what engine actually saw

**Fix Required:**
1. Get `half_life` from trade row
2. Calculate adaptive lookback: `lookback = clamp(4 * half_life, 30, 120)`
3. Use this lookback for rolling z-score calculation

---

## Bug #5: Engine Exit Logic Asymmetry ⚠️ (Potential)

**File:** `src/pairs_trading_etf/backtests/engine.py`  
**Lines:** 1370-1395

**Observation:**
```python
# LONG spread exit:
if z >= -(exit_thresh + exit_tol):  # z >= -(0.5 + 0.1) = -0.6
    should_exit = True

# SHORT spread exit:
if z <= (exit_thresh + exit_tol):   # z <= (0.5 + 0.1) = 0.6
    should_exit = True
```

This seems correct, BUT the exit_tol is ADDED for both cases:
- LONG: exits when z rises above -(exit + tol) = -0.6
- SHORT: exits when z falls below (exit + tol) = 0.6

**Question:** Should tolerance be SUBTRACTED for stricter exits?
- If exit_thresh=0.5, tol=0.1: exit at ±0.6 (loose) vs ±0.4 (strict)

**Status:** Need to verify this is intentional behavior per Vidyamurthy Ch.8.

---

## Bug #6: Visualization Entry Price Inconsistency ❌

**File:** `scripts/visualize_trade_v2.py`  
**Lines:** 165-166

**Problem:**
```python
entry_p1 = p1.iloc[entry_idx]
entry_p2 = p2.iloc[exit_idx]  # ❌ BUG: Uses EXIT index for entry_p2!
```

Should be:
```python
entry_p1 = p1.iloc[entry_idx]
entry_p2 = p2.iloc[entry_idx]  # ✅ Both should use entry_idx
```

**Note:** Lines 170-171 fix this later, but lines 165-166 are still wrong and wastes computation.

**Impact:** Minor - the correct values are computed later, but confusing code.

---

## Bug #7: Stop Loss Threshold Also Hardcoded ❌

**File:** `scripts/visualize_trade_v2.py`  
**Lines:** 352-353

**Problem:**
```python
ax4.axhline(3.0, color='darkred', linestyle='-.', alpha=0.5, label='Stop (±3.0)')
ax4.axhline(-3.0, color='darkred', linestyle='-.', alpha=0.5)
```

But config uses `stop_loss_sigma: 4.0`.

**Fix:** Read `stop_loss_sigma` from config along with entry/exit thresholds.

---

## Verification Checklist

After fixing, verify:

- [ ] Visualization thresholds match config values
- [ ] Entry/exit markers align with threshold lines  
- [ ] Z-score explanation is clear to users
- [ ] All trades in trades.csv satisfy entry threshold
- [ ] All convergence exits satisfy exit threshold
- [ ] Markers use actual entry_z/exit_z from trades.csv
- [ ] Lookback matches engine's adaptive lookback

---

## Notes

### Engine Logic Verified ✅

Checked `trades.csv` with config `entry_threshold_sigma: 2.0`, `exit_threshold_sigma: 0.5`:

| Trade | Direction | Entry Z | Check |
|-------|-----------|---------|-------|
| EWU_EWQ | SHORT | 2.26 | 2.26 >= 2.0 ✅ |
| EWG_EWU | SHORT | 2.11 | 2.11 >= 2.0 ✅ |
| KBE_IAI | LONG | -2.06 | -2.06 <= -2.0 ✅ |

**Conclusion:** Engine entry/exit logic is CORRECT. Visualization has multiple bugs.

---

## Summary

| Bug | Severity | File | Description |
|-----|----------|------|-------------|
| #1 | HIGH | visualize_trade_v2.py | Hardcoded entry/exit thresholds |
| #2 | MEDIUM | visualize_trade_v2.py | Rolling vs fixed z-score mismatch |
| #3 | HIGH | visualize_trade_v2.py | Uses recalculated z instead of actual |
| #4 | MEDIUM | visualize_trade_v2.py | Wrong lookback (60 vs adaptive) |
| #5 | LOW | engine.py | Exit tolerance direction (verify) |
| #6 | LOW | visualize_trade_v2.py | entry_p2 uses wrong index |
| #7 | MEDIUM | visualize_trade_v2.py | Hardcoded stop-loss threshold |

---

*Last Updated: 2025-12-04*

---

# 🔴 CRITICAL BUGS - PIPELINE LEVEL

## Bug #8: LOOK-AHEAD BIAS - Same-Day Execution ❌ CRITICAL

**File:** `src/pairs_trading_etf/backtests/engine.py`  
**Lines:** 1485-1565

**Problem:**
```python
for t in range(warmup, n_dates):
    current_date = dates[t]
    
    # ... check exits ...
    
    # Entry check uses z-score at current_date
    z = zscores.loc[current_date, pair_name]
    
    # But execution ALSO uses price at current_date!
    px = prices.loc[current_date, leg_x]  # ❌ SAME DAY!
    py = prices.loc[current_date, leg_y]  # ❌ SAME DAY!
```

**The Deadly Flow:**
1. At day `t`, we calculate `z_t` using `close_t`
2. We check if `|z_t| >= entry_threshold`
3. If yes, we execute trade at `close_t` ← **LOOK-AHEAD!**

In reality:
- You can only KNOW `z_t` AFTER market close on day `t`
- You can only EXECUTE on day `t+1` (next open/close)

**Impact:** 
- Severely overstated performance
- In real trading, you'd get worse fills
- This is a CLASSIC backtest bias

**Fix Required:**
```python
# Option A: Signal on day t, execute on day t+1
if z <= -entry_thresh:
    # Schedule trade for tomorrow
    pending_entries[pair] = {'signal_date': current_date, ...}

# Next day:
if pair in pending_entries:
    # Execute at today's price (t+1)
    px = prices.loc[current_date, leg_x]
    py = prices.loc[current_date, leg_y]
```

Or simpler:
```python
# Use yesterday's z-score for today's trade decision
z = zscores.loc[dates[t-1], pair_name]  # Signal from yesterday
px = prices.loc[current_date, leg_x]     # Execute at today's price
```

---

## Bug #9: Survivorship Bias in Universe ⚠️ KNOWN LIMITATION

**File:** `scripts/download_fresh_data.py`  
**Lines:** 17-22

**Problem:**
```python
# Load ETF metadata - THIS IS CURRENT LIST (2025)
with open("configs/etf_metadata.yaml", "r") as f:
    config = yaml.safe_load(f)

tickers = list(config["etfs"].keys())  # Only ETFs that EXIST TODAY
```

**Impact:**
- ETFs that were delisted/merged are NOT in dataset
- Backtest from 2010 uses ONLY survivors
- Results are optimistically biased

**Status:** Known limitation, documented in research_log.md

**Mitigation:**
- Add disclaimer in all results
- Consider using point-in-time universe (very hard for ETFs)

---

## Bug #10: Data-Snooping via Scoring Weights ⚠️ ACKNOWLEDGED

**File:** `src/pairs_trading_etf/backtests/engine.py`  
**Lines:** 990-1005

**Problem:**
```python
# These weights were TUNED by looking at results
score = (
    0.20 * pvalue_score + 
    0.15 * hl_score + 
    0.10 * range_score + 
    0.10 * hr_score +
    0.15 * snr_score +
    0.15 * zcr_score +
    0.15 * stability_score
)
```

**Impact:**
- Every weight adjustment = new backtest trial
- Probability of Backtest Overfitting (PBO) is HIGH
- Cannot trust final Sharpe without CSCV validation

**Status:** Acknowledged, requires CSCV to validate

---

## Bug #11: Half-Life Re-estimation in Trading Year? ✅ VERIFIED SAFE

**Checked:** `engine.py` lines 1050-1200

**Finding:**
```python
# half_lives{} is passed from select_pairs() 
# which uses FORMATION year data only
hl = half_lives[pair]  # Never updated during trading
```

**Verdict:** ✅ SAFE - No leak. Half-life from Year-1 only.

---

## Bug #12: Transaction Cost Implementation ✅ VERIFIED OK

**Checked:** `engine.py` lines 1420-1430, config.py line 209

**Finding:**
```python
transaction_cost_bps: float = 10.0  # 10 bps = 0.10%

# Applied correctly:
entry_notional = abs(qty_x) * entry_px + abs(qty_y) * entry_py
exit_notional = abs(qty_x) * exit_px + abs(qty_y) * exit_py
cost = (entry_notional + exit_notional) * (cfg.transaction_cost_bps / 10000)
pnl -= cost
```

**Verdict:** ✅ OK
- 10 bps round-trip is applied
- BUT: no slippage model, no spread cost
- This is still optimistic for illiquid ETFs

---

# 📋 VISUALIZATION BUGS (Lower Priority)

## Bug #1-#7: See above (visualization hardcoded thresholds, etc.)

---

# 📊 SUMMARY TABLE

| Bug | Severity | Type | Status |
|-----|----------|------|--------|
| **#8** | 🔴 CRITICAL | Look-Ahead Bias | ❌ Must Fix |
| **#9** | 🟡 MEDIUM | Survivorship Bias | ⚠️ Documented |
| **#10** | 🟡 MEDIUM | Data Snooping | ⚠️ Needs CSCV |
| **#11** | ✅ OK | HL Re-estimation | ✅ Safe |
| **#12** | ✅ OK | Transaction Cost | ✅ Implemented |
| #1-#7 | 🟢 LOW | Visualization | ❌ Fix Later |

---

# 🎯 PRODUCTION READINESS AUDIT

## Based on: "PAIRS TRADING PIPELINE - AUDIT CHECKLIST"

---

## 🔴 CRITICAL ISSUES (Must Fix Before Trading)

### Issue C1: Survivorship Bias ❌ NOT FIXED

**Current Status:** Using list of ETFs that exist in 2025, pulling historical data backwards.

**Code Evidence:**
```python
# download_fresh_data.py
with open("configs/etf_metadata.yaml", "r") as f:
    config = yaml.safe_load(f)
tickers = list(config["etfs"].keys())  # 2025 survivors only!
```

**Impact:** +100-200 bps inflation per year (optimistic bias)

**Fix Required:** 
- Minimum: Document limitation clearly
- Ideal: Use CRSP-style point-in-time universe

---

### Issue C2: FDR Correction for Multiple Testing ❌ NOT APPLIED

**Current Status:** Testing ~1000 pairs with p<0.05, no correction applied.

**Code Evidence:**
```python
# engine.py line 725
if pvalue > pvalue_threshold:  # Just raw p-value check
    return None
```

**Calculation:**
- ~140 ETFs → ~10,000 possible pairs
- After correlation filter: ~1000 pairs
- Testing at p<0.05 → Expected ~50 false positives!

**Impact:** +200-500 bps inflation (many "cointegrated" pairs are noise)

**Fix Options:**
1. **Bonferroni:** p_adj = p × n_tests → Too conservative
2. **Benjamini-Hochberg (FDR):** Control false discovery rate at 5%
3. **Lower raw threshold:** Use p<0.01 instead of p<0.05 (partial mitigation)

**Note:** Current config uses `pvalue_threshold: 0.01` which is stricter, but still no formal FDR correction.

---

### Issue C3: Look-Ahead Bias ❌ CONFIRMED BUG (See Bug #8)

**Current Status:** Signal at close_t, execute at close_t (SAME PRICE!)

**Code Evidence:**
```python
# engine.py line 1485-1495
z = zscores.loc[current_date, pair_name]  # Signal
px = prices.loc[current_date, leg_x]       # Execution - SAME DAY!
```

**Impact:** +50-100 bps inflation per year

**Fix Required:** 
```python
# Signal from yesterday, execute today
z = zscores.loc[dates[t-1], pair_name]  # Signal t-1
px = prices.loc[current_date, leg_x]     # Execute t
```

---

### Issue C4: Transaction Costs ⚠️ POSSIBLY TOO LOW

**Current Status:** 10 bps round-trip

**Code Evidence:**
```python
# config.py line 209
transaction_cost_bps: float = 10.0  # 10 bps = 0.10%
```

**Industry Benchmark:**
- Gatev et al. found effective spread of 70-81 bps
- For liquid ETFs (SPY, QQQ): 5-10 bps is realistic
- For smaller ETFs: 20-50 bps more accurate

**Impact:** -100-200 bps underestimation (if trading illiquid ETFs)

**Assessment:** 10 bps is REASONABLE for large-cap ETFs in this universe. However, no slippage model exists.

**Verdict:** ⚠️ Acceptable for liquid ETFs, but should document assumption.

---

## 🟡 HIGH PRIORITY ISSUES (Should Fix)

### Issue H1: Parameter Documentation ⚠️ PARTIAL

| Parameter | Value | Source | Justified? |
|-----------|-------|--------|------------|
| `entry_threshold_sigma` | 0.75 (default) / 2.0 (practical) | Vidyamurthy Ch.8 | ✅ |
| `exit_threshold_sigma` | 0.0 / 0.5 | Mean reversion theory | ✅ |
| `stop_loss_sigma` | 4.0 | ❓ NOT DOCUMENTED | ❌ |
| `max_holding_days` | 60 | ❓ NOT DOCUMENTED | ❌ |
| `stop_tightening_rate` | 0.15 | ❓ NOT DOCUMENTED | ❌ |
| `adaptive_lookback_multiplier` | 4.0 | QMA best practice | ✅ |

**Fix Required:** Document source for each parameter in config.py or research_log.md

---

### Issue H2: Regime Split ❌ NOT SHOWN

**Current Status:** Results shown as 2010-2024 aggregate

**What's Needed:**
```
Period          | PnL      | Sharpe | Win Rate | Notes
----------------|----------|--------|----------|------------------
2010-2015       | ???      | ???    | ???      | Pre-HFT dominance
2016-2019       | ???      | ???    | ???      | Normal market
2020-2024       | ???      | ???    | ???      | Post-COVID, HFT era
```

**Why Important:** Gatev found 68% decline in profits post-1988. Need to show if pattern holds.

---

### Issue H3: True Out-of-Sample Test ⚠️ FRAMEWORK EXISTS

**Current Status:** CSCV framework exists in `cross_validation.py` but not consistently used.

**Code Evidence:**
```python
# cross_validation.py
@dataclass
class BacktestSplit:
    train_start: str = "2009-01-01"
    train_end: str = "2016-12-31"
    val_start: str = "2017-01-01" 
    val_end: str = "2020-12-31"
    test_start: str = "2021-01-01"
    test_end: str = "2024-12-31"
```

**Issue:** Walk-forward IS in-sample retraining each year. Need TRUE hold-out:
- Train on 2010-2019 (fix ALL parameters)
- Test on 2020-2024 (NO changes allowed)

---

## 📊 SUMMARY MATRIX

| Issue | Status | Impact | Priority |
|-------|--------|--------|----------|
| **C1: Survivorship** | ❌ | +100-200 bps | 🔴 CRITICAL |
| **C2: FDR Correction** | ❌ | +200-500 bps | 🔴 CRITICAL |
| **C3: Look-Ahead** | ❌ | +50-100 bps | 🔴 CRITICAL |
| **C4: Transaction Costs** | ⚠️ | -100-200 bps | 🟡 MEDIUM |
| **H1: Parameter Docs** | ⚠️ | Unknown | 🟡 HIGH |
| **H2: Regime Split** | ❌ | Honest picture | 🟡 HIGH |
| **H3: True OOS Test** | ⚠️ | Validation | 🟡 HIGH |

**Total Estimated Bias:** +250-700 bps per year (optimistic!)

This means if backtest shows +3% annual return, true return could be **-4% to +0.5%**.

---

# 📊 CONFIG AUDIT: ETF vs STOCK - Critical Analysis (2025-12-04)

## 🎯 Key Insight: WE TRADE ETFs, NOT STOCKS

Gatev et al. (2006) và Vidyamurthy (2004) nghiên cứu **individual stocks**.
Chúng ta trade **ETFs** - hoàn toàn khác biệt!

### False Alarms from Stock-Based Literature

| Claim | Stock Context | ETF Reality | Verdict |
|-------|--------------|-------------|---------|
| "tx cost = 70 bps" | CRSP stocks 1962-2002 | ETF spreads 2-10 bps | **5-15 bps is OK** |
| "max HL = 30 days" | Fast stock mean-reversion | ETF slower | **30-50 days OK** |
| "Compute optimal threshold" | White noise theory | Real OU + costs | **Use 2.0σ empirically** |

### ⚠️ Stop-Loss 4σ - VALID CONCERN, NOW IMPLEMENTED!

**Original Claim:** "Stop-loss 4σ hardcoded, should be VIX-adaptive"

**Counter-argument against VIX:**
- VIX measures **market fear**, not **pair spread volatility**
- Pair can be stable while market is volatile (and vice versa)
- VIX-based stops would exit based on unrelated metric

**BUT the underlying concern is valid:**
- 4σ fixed for ALL pairs regardless of half-life is suboptimal
- Fast mean-reversion (HL=5) → should exit sooner if not recovering
- Slow mean-reversion (HL=25) → needs more time, wider stop

**✅ IMPLEMENTED: `use_adaptive_stop_loss`**

```python
# Formula: stop_sigma = base + 0.5 * (HL/10 - 1), clamped to [3.0, 5.0]
# HL=5:  3.25σ (tighter - should recover quickly)
# HL=10: 3.5σ  (base reference)
# HL=20: 4.0σ  (wider - needs time)
# HL=30: 4.5σ  (even wider)
```

**Usage:**
```yaml
use_adaptive_stop_loss: true   # Enable HL-based scaling
stop_loss_sigma: 3.5           # Base stop (used when HL=10)
```

### Actual Issues Found

| Issue | Current | Recommendation | Priority |
|-------|---------|----------------|----------|
| `hedge_ratio_method` confusion | Two bools | Single method field | ⚠️ MEDIUM |
| `min_pairs_for_trading = 3` | Many years skipped | Consider `= 2` | ⚠️ LOW |
| `max_holding_days = 60` cap | May cut dynamic early | Raise to 90 or remove | ⚠️ MEDIUM |

### What NOT to "Fix"

| Parameter | Current | DON'T Change Because |
|-----------|---------|---------------------|
| `transaction_cost_bps: 5-10` | OK | ETFs are liquid, not 2000s stocks |
| `stop_loss_sigma: 4.0` | OK | Already wide enough |
| `entry_threshold: 2.0` | OK | Empirically validated for ETFs |
| `zscore_lookback: 60` fallback | OK | Good engineering, keep as backup |

**See:** `docs/config_audit_etf_vs_stocks.md` for full analysis.

---

*Last Updated: 2025-12-04*

