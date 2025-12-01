# Week 1 – Stationarity & Cointegration

## 1. I(0) vs I(1) and random walk
    I(0): stationary time series
        - Constant mean: E[x_t] = μ (does not depend on t)
        - Constant, finite variance: Var(x_t) < ∞, does not grow over time
        - Autocovariances depend only on the lag (h), not on calendar time t
        - Shocks are temporary: after a shock, the series tends to revert toward μ
        - Examples: white noise, AR(1) with |φ| < 1

    I(1): integrated of order 1 (unit root / random-walk-type series)
        - The level x_t is nonstationary; its variance typically grows with t
        - The first difference Δx_t = x_t − x_{t−1} is I(0)
        - Shocks are permanent: one innovation shifts the path for all future t
        - Canonical example: random walk
              x_t = x_{t−1} + ε_t,   ε_t ~ I(0)

    Why this matters for cointegration
        - In finance, log prices of stocks/ETFs are usually modeled as I(1)
        - Cointegration: each series is I(1) but some linear combination a'x_t is I(0)
        - That stationary linear combination is the “spread” we want to trade
          (mean-reverting relationship suitable for pairs trading).


## 2. Engle–Granger 2-step method

(Residual-based procedure to test for cointegration between two I(1) series)

1. Check integration order
   - For each series (e.g. log P_A,t and log P_B,t), run a unit root test (ADF, etc.).
   - Goal: confirm both series are I(1) (nonstationary in levels, stationary in first differences).

2. Estimate the long-run equilibrium relationship
   - Run an OLS regression in levels, for example:
         log P_A,t = α + γ log P_B,t + e_t
   - The fitted residuals ê_t define the candidate spread:
         spread_t = ê_t = log P_A,t − α̂ − γ̂ log P_B,t
   - Interpretation: this regression extracts the long-run equilibrium relation
     between the two nonstationary price series.

3. Test residuals for stationarity
   - Apply a unit root test (ADF with Engle–Granger critical values) to ê_t.
   - If ê_t is stationary (I(0)):
        • the spread is mean-reverting
        • the two series are cointegrated with vector (1, −γ̂)
   - If ê_t is still I(1), reject cointegration: the apparent relationship is
     likely a spurious regression between unrelated random walks.


## 3. Correlation vs Cointegration

- Correlation
  - Measures strength of linear co-movement over a sample.
  - High correlation between two nonstationary series does NOT imply any stable
    long-run equilibrium.
  - Two independent random walks can show very high correlation and R²
    (classic “spurious regression” problem).

- Cointegration
  - A long-run equilibrium concept for nonstationary series.
  - Each series (e.g. y_t, x_t) is I(1), but there exists γ such that
        y_t − γ x_t is I(0).
  - Economically: they share a common stochastic trend; deviations from the
    equilibrium (the spread) are temporary and mean-reverting (error correction).

- Why cointegration > correlation for pairs trading
  - Correlation only says “they tend to move together” over the sample.
  - Cointegration guarantees the existence of a stationary spread series, which
    gives a statistically grounded target for mean reversion.
  - Therefore, we always test for cointegration (Engle–Granger, Johansen) instead
    of relying on raw correlation.


## 4. Application to ETF pairs

- Economic intuition
  - Sector / industry ETFs (e.g. two tech ETFs, two financials ETFs) track
    very similar underlying indices.
  - We expect their log prices to share a common trend; a suitable linear
    combination of the two prices should be stationary if they are cointegrated.

- Week 1 workflow for ETF pairs
  - Collect ~10 years of daily prices for several candidate ETF pairs.
  - Pre-screen using:
      • High correlation in log prices (e.g. > 0.8) as a rough filter.
  - For each candidate pair:
      1. Verify both log price series are I(1) (unit root tests on levels and differences).
      2. Apply Engle–Granger 2-step:
           - Regress log P_A,t on log P_B,t, obtain γ̂ and residuals ê_t.
           - Test ê_t for stationarity; if stationary → cointegrated pair.
      3. Keep only pairs with statistically significant cointegration.

- Bridge to later weeks
  - The chosen ETF pairs provide:
      • A stationary spread process to model (e.g. as an OU process),
      • Cointegration coefficients (hedge ratios) for constructing the tradable spread.
  - These spreads and parameters become inputs for:
      • Mean-reversion modeling (Week 2),
      • Signal generation and backtesting (Weeks 3+).
