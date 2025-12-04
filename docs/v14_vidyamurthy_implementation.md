# Vidyamurthy Framework Implementation - V14

## Overview

Implementation of Ganapathy Vidyamurthy's pairs trading framework from 
"Pairs Trading: Quantitative Methods and Analysis" (Chapters 6-7).

## Key Concepts Implemented

### 1. Signal-to-Noise Ratio (SNR) Filter

**Formula:** `SNR = σ_stationary / σ_nonstationary`

- σ_stationary = standard deviation of the spread
- σ_nonstationary = standard deviation of spread changes (innovations)

**Interpretation:**
- Higher SNR indicates stronger cointegration
- The spread is more "signal" (mean-reverting) vs "noise" (random walk)
- Config: `min_snr: 1.5` (pairs with SNR < 1.5 are filtered out)

### 2. Zero-Crossing Rate (ZCR) Filter

**Formula:** `ZCR = number of times spread crosses mean per year`

**Interpretation:**
- Higher ZCR = more tradeable (more mean-reversion opportunities)
- Also estimates expected holding period: `E[holding] ≈ days / (2 × crossings)`
- Config: `min_zero_crossing_rate: 5.0` (minimum 5 crossings per year)

### 3. Time-Based Stop Tightening

**Insight from Vidyamurthy:** "The mere passage of time represents an increase in risk"

**Implementation:**
- Stop loss starts at `base_stop_zscore` (e.g., 3.0)
- After 1 half-life: stop begins tightening
- After 2 half-lives: stop tightens by `tightening_rate × base_stop`
- Floor at z=1.5 to avoid premature exits

**Config:**
```yaml
time_based_stops: true
stop_tightening_rate: 0.1  # 10% tightening per half-life
```

### 4. Updated Pair Scoring

Scoring now includes Vidyamurthy metrics:
- 25% p-value score
- 20% half-life score
- 15% spread range score
- 10% hedge ratio score
- **15% SNR score** (new)
- **15% ZCR score** (new)

## Results Comparison

| Metric | V11 (baseline) | V14 (Vidyamurthy) | Change |
|--------|----------------|-------------------|--------|
| Total PnL | $2,079 | $3,783 | **+82%** |
| Win Rate | 43.4% | 69.1% | **+26%** |
| Profit Factor | 1.41 | 2.54 | **+80%** |
| Total Trades | 129 | 68 | -47% |
| Stop-losses | 64 | 2 | **-97%** |
| Max Drawdown | ~$1,500 | $747 | **-50%** |
| Avg Holding | 12.5 days | 16.6 days | +33% |

## Key Insights

1. **Quality over Quantity**: V14 takes fewer trades (68 vs 129) but with 
   much higher quality. The SNR and ZCR filters remove marginal pairs.

2. **Dramatic Stop-Loss Reduction**: From 64 stop-losses to only 2! 
   The filters ensure we only trade pairs with strong mean-reversion.

3. **Higher Win Rate**: 69% vs 43% - the Vidyamurthy metrics select 
   pairs that are more likely to converge.

4. **Better Risk-Adjusted Returns**: Profit Factor of 2.54 means 
   winners are 2.5x larger than losers on average.

## Configuration (v14_vidyamurthy_full.yaml)

```yaml
# Vidyamurthy Framework Parameters
min_snr: 1.5                    # Minimum Signal-to-Noise Ratio
min_zero_crossing_rate: 5.0     # Minimum crossings per year
time_based_stops: true          # Enable time-based stop tightening
stop_tightening_rate: 0.1       # 10% tightening per half-life
```

## Files Modified

1. `src/pairs_trading_etf/backtests/engine.py`
   - Added `calculate_snr()` function
   - Added `calculate_zero_crossing_rate()` function
   - Added `bootstrap_holding_period()` function
   - Added `calculate_factor_correlation()` function
   - Added `calculate_time_based_stop()` function
   - Updated `run_engle_granger_test()` to return SNR/ZCR
   - Updated `select_pairs()` with SNR/ZCR filters and scoring
   - Updated exit logic with time-based stops

2. `src/pairs_trading_etf/backtests/config.py`
   - Added `min_snr` parameter
   - Added `min_zero_crossing_rate` parameter
   - Added `time_based_stops` parameter
   - Added `stop_tightening_rate` parameter

3. `configs/experiments/v14_vidyamurthy_full.yaml`
   - New config with all Vidyamurthy parameters

## Future Improvements

1. **Bootstrap Holding Period Estimation**: Already implemented but not 
   yet used for filtering. Could add `max_expected_holding` filter.

2. **VWAP Regression**: Vidyamurthy suggests volume-weighted regression 
   for more reliable equilibrium estimation.

3. **Factor Correlation**: Already implemented but not yet used. Could 
   add `min_factor_correlation` filter.

4. **Adaptive SNR Thresholds**: Could adjust min_snr based on market regime.

## References

- Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis"
  - Chapter 6: Common Trends Model, APT, Distance Measure, SNR
  - Chapter 7: Testing for Tradability, Zero-Crossing Rate, Bootstrap
