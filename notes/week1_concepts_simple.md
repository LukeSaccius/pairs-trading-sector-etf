# Week 1 – Stationarity & Cointegration (Plain-Language Notes)

## 1. Two Personalities of a Time Series

### I(0): "Wigglers"
- Hang out near a stable long-run average.
- Shocks fade; the series drifts back toward the mean.
- Examples: daily returns, an AR(1) with |phi| < 1.

### I(1): "Wanderers"
- No fixed average in levels; they can drift forever.
- First differences (today − yesterday) behave more stably.
- Classic case: random walk `x_t = x_{t-1} + shock_t`.

### Why finance cares
- Prices/log-prices ≈ I(1); they wander.
- Returns/price changes ≈ I(0); they wiggle around zero.
- Cointegration = combine two wanderers so their spread wiggles (mean reverts).

---

## 2. Engle–Granger in Three Plain Steps

1. **Check each series is I(1)**
   - Run a unit-root test (e.g., ADF) on each price series.
   - We expect levels to look non-stationary but first differences to pass.

2. **Estimate long-run relationship**
   - Regress `log P_A,t = alpha + gamma * log P_B,t + e_t`.
   - `gamma` is the hedge ratio; `e_t` are residuals (the spread).

3. **Test the residuals**
   - Run a stationarity test on `spread_t = e_t`.
   - If the spread is I(0), the pair is cointegrated; otherwise, the nice regression was spurious.

Takeaway: *Regress first, test the residual second.* Cointegration means two wanderers create a wiggler.

---

## 3. Correlation vs. Cointegration

| Concept | What it says | Why it can mislead |
|---------|--------------|--------------------|
| Correlation | “They moved together in this sample window.” | Two unrelated random walks can show high correlation. |
| Cointegration | “Despite wandering, their spread snaps back to equilibrium.” | Requires each series to be I(1) and a stationary spread. |

Mental picture:
- Correlation = two dancers moving in sync.
- Cointegration = two hikers tied by a rubber band; they can separate, but the band pulls them back together.

For pairs trading we need the rubber band, not just synchronized dancing.

---

## 4. Applying This to ETF Pairs

1. **Collect data**
   - Grab ~10 years of daily prices for ETFs in the same sector.
   - Work with log prices for stability.

2. **Pre-screen**
   - Compute correlations; keep pairs with Corr > 0.8.
   - This is only a filter to avoid testing every combination.

3. **Run Engle–Granger**
   - Confirm both series are I(1).
   - Regress one on the other to estimate the hedge ratio.
   - Test the residual spread for stationarity.
   - If stationary ⇒ store hedge ratio + spread for trading.

4. **Deliverables from Week 1**
   - Shortlist of ETFs with economically linked exposures.
   - Hedge ratios and spreads that appear mean-reverting.
   - Correlation plots and summary tables in `results/`.

5. **Where it leads**
   - Week 2: fit mean-reversion models (e.g., OU) to the spreads.
   - Later: build z-score signals and trading rules.

---

## Quick Reference Cheat Sheet
- **I(0)** = stationary = wiggles around a mean.
- **I(1)** = unit root = random walk wanderer.
- **Cointegration** = linear combo of I(1) series that is I(0).
- **Engle–Granger** = regress in levels → test residuals.
- **Correlation** ≠ **Cointegration**; we need the spread to mean revert.
- **Pairs trading recipe**
  1. Filter by correlation.
  2. Test for cointegration.
  3. Trade the stationary spread.

---

## 5. The Hard Truth: Cointegration ≠ Tradeable (Added 2025-12-03)

### What We Learned the Hard Way

**The Setup:**
- We found ETF pairs that ARE cointegrated (p < 0.10)
- Example: GLD-IAU (gold ETFs), SPY-VOO (S&P 500 ETFs)
- Statistically, their spread is "stationary" (I(0))

**The Problem:**
- Half-life = how long for spread to revert halfway to mean
- GLD-IAU half-life: **628,182 days = 1,721 YEARS**
- SPY-VOO half-life: **89,657 days = 246 YEARS**

**Translation:**
- Yes, the spread will eventually mean-revert
- No, you won't live long enough to profit from it

### Why ETFs Have This Problem

Think of two S&P 500 ETFs (SPY vs VOO):
- They hold the same 500 stocks
- Any price difference = tiny tracking error + expense ratio
- These differences are essentially random noise
- The "cointegration" is mathematical, not economic

### What This Means for Trading

| Filter | Pairs Found |
|--------|-------------|
| After correlation filter | ~4,500 pairs |
| After cointegration p-value | ~100 pairs |
| After half-life 15-120 days | 16 pairs |
| After 30% rolling consistency | **0 pairs** |

**Reality Check:**
- The alpha has been arbitraged away
- Only crisis periods (2008-2009) showed profits (+2.3%)
- Normal markets: -0.4% annual return

### Updated Mental Model

Old thinking:
> "Cointegration = rubber band → spread snaps back → profit!"

New thinking:
> "Cointegration = rubber band, BUT if the band takes 200 years to snap back, there's no trade."

### Key Numbers to Remember

**Tradeable half-life range:**
- Min: 15 days (need time to execute, avoid noise)
- Max: 120 days (complete trade in reasonable time)
- Sweet spot: 30-60 days

**What we actually found:**
- ETF half-lives: 28,000 - 628,000 days
- That's 77 to 1,721 YEARS
- Conclusion: ETF pairs trading is dead (for standard methods)
