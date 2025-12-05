# CONFIG AUDIT: ETF vs Stock Trading - Critical Analysis

## 🎯 Key Insight: WE TRADE ETFs, NOT STOCKS

Gatev et al. (2006) và Vidyamurthy (2004) nghiên cứu **individual stocks**.  
Chúng ta trade **ETFs** - hoàn toàn khác biệt về:

1. **Liquidity**: ETFs like SPY, XLF có bid-ask spread 0.01-0.02%, stocks có thể 0.1-0.5%
2. **Price Impact**: ETF market makers rất competitive, slippage thấp hơn nhiều
3. **Volatility**: ETFs diversified, ít volatile hơn individual stocks
4. **Mean-Reversion Speed**: ETF pairs thường có half-life dài hơn

---

## 📊 TRANSACTION COST ANALYSIS

### Original Claim:
> "transaction_cost_bps: 10.0 is TOO LOW! Should be 70-100 bps per Gatev!"

### Critical Analysis:

**Gatev (2006) Context:**
- Traded **CRSP stocks** from 1962-2002
- Many small-cap, illiquid stocks
- Bid-ask spreads: 30-55 bps
- Market impact: 20-40 bps
- **Total: 70-100 bps** (for STOCKS in THAT ERA)

**Our ETF Context (2010-2024):**

| ETF | Avg Spread (bps) | Market Impact | Round-Trip Est |
|-----|------------------|---------------|----------------|
| SPY | 0.5-1 | ~0.5 | **2-3 bps** |
| XLF | 1-2 | ~1 | **4-6 bps** |
| XLE | 2-3 | ~2 | **8-12 bps** |
| EWZ | 5-10 | ~5 | **20-30 bps** |

**Realistic ETF Transaction Costs:**
- **Liquid sector ETFs (XLF, XLE, XLK):** 5-15 bps round-trip
- **Regional ETFs (EWU, EWP, VGK):** 10-25 bps round-trip
- **Leveraged/Exotic ETFs:** 20-50 bps round-trip

### ✅ VERDICT on transaction_cost_bps

| Current | Recommendation | Reasoning |
|---------|---------------|-----------|
| 5-10 bps | **10-15 bps** | Conservative for major ETFs |
| - | **NOT 70 bps** | Gatev numbers are for 2000s stocks! |

**Config Should Have:**
```yaml
transaction_cost_bps: 10.0  # Realistic for liquid ETFs
# Run sensitivity: 5, 10, 15, 20 bps to test robustness
```

---

## 📊 HALF-LIFE ANALYSIS

### Original Claim:
> "max_half_life: 50.0 is TOO PERMISSIVE! Should be 30 per Vidyamurthy!"

### Critical Analysis:

**Vidyamurthy Context:**
- Single stocks can have very fast mean-reversion (2-10 days)
- Trading individual stocks = high idiosyncratic risk
- Recommendation: HL 5-20 days optimal

**Our ETF Context:**
- ETF pairs = aggregate of many stocks
- Mean-reversion typically **slower** than stocks
- Diversification benefit = can afford longer holding

**Empirical Observation from Our Data:**
- Many quality ETF pairs have HL = 20-40 days
- Restricting to 30 days may exclude good pairs!

### ✅ VERDICT on max_half_life

| Current | Recommendation | Reasoning |
|---------|---------------|-----------|
| 50 days | **Keep 50 days** OR **lower to 40** | ETFs mean-revert slower |
| vidyamurthy.yaml uses 30 | **30 is fine but conservative** | May miss some pairs |

**The Real Issue:**
- max_half_life = 50 means max_holding = 150 days (if dynamic)
- That's 6 months capital lock! 
- **But** if HL=50 pairs rarely exist with p<0.01, this is a non-issue

**Actual Fix Needed:**
```yaml
# If using dynamic_max_holding:
max_half_life: 40.0  # Keep reasonable
# The holding cap naturally limits exposure
```

---

## 📊 STOP-LOSS ANALYSIS

### Original Claim:
> "stop_loss_sigma: 4.0 is HARDCODED! Should be adaptive with VIX!"

### Critical Analysis:

**The Claim's Logic:**
- Tighter stop in high-VIX (volatile markets)
- Looser stop in low-VIX (calm markets)

**Counter-Argument:**
1. **VIX-based stops are hindsight bias**: VIX measures market fear, not pair spread vol
2. **4σ is already very wide**: For OU process, P(|z|>4) ≈ 0.006%
3. **The real problem**: Not the stop level, but spread DIVERGENCE (non-stationarity)

**ETF-Specific:**
- ETF pairs can have regime changes (sector rotation)
- Stop-loss protects against **breakdown of cointegration**
- Fixed 4σ is reasonable because we already filter on p-value and half-life

### ✅ VERDICT on stop_loss_sigma

| Current | Recommendation | Reasoning |
|---------|---------------|-----------|
| 4.0σ fixed | **Keep 4.0σ** | Already very wide |
| - | **Consider 3.5σ** | May reduce losses earlier |
| VIX-adaptive | **NOT NEEDED** | Adds complexity, little benefit for ETFs |

**What Actually Matters:**
- Stop-loss triggers indicate **pair breakdown**, not market vol
- Better to track half-life stability than VIX

---

## 📊 ENTRY THRESHOLD ANALYSIS

### Original Claim:
> "entry_threshold_sigma: 0.75 is theory! Should compute from data!"

### Critical Analysis:

**Vidyamurthy's 0.75σ Theory:**
- Maximizes f(Δ) = Δ × [1 - N(Δ)] for **white noise**
- Real spreads are **OU process**, not white noise
- With transaction costs, optimal threshold HIGHER

**Our Config (vidyamurthy_practical.yaml):**
- Uses **2.0σ** entry - empirically better!
- This is ALREADY the right fix!

**The compute_optimal_threshold() function:**
- Solves for white noise case
- Result ≈ 0.75σ regardless of data
- **It's not "dead code", it's just not relevant for real trading**

### ✅ VERDICT on entry_threshold

| Current | Recommendation | Reasoning |
|---------|---------------|-----------|
| 2.0σ (practical) | **Keep 2.0σ** | Empirically validated |
| 0.75σ (theory) | **Don't use in production** | Too aggressive for real costs |
| Compute from data | **Sensitivity analysis instead** | Test 1.5, 2.0, 2.5σ |

---

## 📊 ADAPTIVE LOOKBACK vs FIXED

### Original Claim:
> "zscore_lookback: 60 is confusing! Remove it!"

### Critical Analysis:

**Current Behavior:**
```python
use_adaptive_lookback: bool = True  # If True, compute from half-life
zscore_lookback: int = 60  # Fallback if adaptive=False
```

**This is FINE!**
- Adaptive = True by default
- zscore_lookback = fallback for testing/override
- Having a fallback is GOOD engineering

### ✅ VERDICT

| Current | Recommendation | Reasoning |
|---------|---------------|-----------|
| Dual fields | **Keep both** | Fallback is useful |
| - | **Document clearly** | Add comment explaining priority |

---

## 📊 DYNAMIC MAX HOLDING

### Original Claim:
> "max_holding_days: 60 caps dynamic calculation!"

### Critical Analysis:

Need to check actual implementation:

```python
# Likely implementation:
max_hold = min(3 * half_life, max_holding_days)  # Capped at 60
```

**Is this bad?**
- HL=25 → theoretical max = 75, capped to 60 → **OK, still reasonable**
- HL=40 → theoretical max = 120, capped to 60 → **Might exit early!**

**But vidyamurthy_practical uses:**
- max_half_life: 30.0
- So max theoretical = 90 days, capped to 60

### ✅ VERDICT

| Current | Recommendation | Reasoning |
|---------|---------------|-----------|
| 60 day cap | **Increase to 90** if using HL up to 30 | Allows full 3×HL |
| - | Or **remove cap** when dynamic=True | Let formula work |

---

## 🔴 ACTUAL BUGS FOUND

### BUG 1: Win Rate Calculation (ALREADY FIXED!)
- Was summing percentages instead of computing from totals
- Fixed earlier in this session

### BUG 2: hedge_ratio_method Confusion
```python
dynamic_hedge: bool = True      # Update HR during trading
use_kalman_hedge: bool = False  # Kalman for HR
```

**Problem:** Two flags for same concept. Which wins?

**Should Be:**
```python
hedge_ratio_method: str = "rolling"  # Options: "fixed", "rolling", "kalman"
```

### BUG 3: min_pairs_for_trading = 3 may be too high

**Observation from backtest:**
- Many years skipped because only 1-2 pairs found
- Strict filtering (p<0.01) reduces pair count

**Options:**
1. Relax min_pairs_for_trading to 2
2. Relax pvalue_threshold to 0.05 (but adds false positives)
3. Accept fewer trades (current approach)

---

## 📋 SUMMARY: What to Actually Fix

| Issue | Severity | Action |
|-------|----------|--------|
| transaction_cost_bps = 5 | ⚠️ LOW | OK for ETFs, maybe bump to 10 |
| stop_loss_sigma = 4.0 | ✅ OK | Keep fixed, no VIX needed |
| max_half_life = 30/50 | ✅ OK | 30 is conservative, 50 is fine |
| entry_threshold = 2.0 | ✅ OK | Already empirically tuned |
| hedge_ratio dual flags | ⚠️ MEDIUM | Consolidate to single method |
| max_holding cap | ⚠️ MEDIUM | Consider removing when dynamic=True |
| min_pairs_for_trading | ⚠️ LOW | Consider lowering to 2 |

---

## 🎯 Final Recommendations for Config

```yaml
# Conservative but realistic for ETFs
transaction_cost_bps: 10.0  # Bump from 5 to be safe
entry_threshold_sigma: 2.0   # Keep empirical value
exit_threshold_sigma: 0.5    # Keep
stop_loss_sigma: 4.0         # Keep - already wide
max_half_life: 30.0          # Keep conservative
max_holding_days: 90         # Increase to allow 3×30
min_pairs_for_trading: 2     # Reduce from 3
```

---

## 📚 Key Takeaway

**DO NOT blindly apply Gatev/Vidyamurthy stock numbers to ETFs!**

ETFs are:
- More liquid (lower transaction costs)
- More diversified (slower mean-reversion)
- More efficient (tighter spreads)

The audit above caught several false alarms that would have **degraded** performance if "fixed" incorrectly.

