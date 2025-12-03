# Backtest v2 vs v3 Comparison

## Critical Discovery: v2 Uses Wrong Critical Values

### The Bug
v2 uses **ADF critical values** for single time series, not **Engle-Granger critical values** for cointegration residuals:

| Confidence | v2 (ADF, wrong) | E-G (correct) | Difference |
|------------|-----------------|---------------|------------|
| **1%**     | -3.43          | **-3.94**     | 0.51 looser |
| **5%**     | -2.86          | **-3.36**     | 0.50 looser |
| **10%**    | -2.57          | **-3.06**     | 0.49 looser |

### Impact
v2 accepts pairs that are **not truly cointegrated**:
- A test statistic of -3.50 would:
  - **v2**: Pass at 1% significance (wrong!)
  - **v3**: Fail at 1%, pass at 5% (correct)

### Full Results Comparison (2010-2024)

| Metric | v2 (buggy) | v3 (no rolling) | v3 (rolling 2/4) |
|--------|-----------|-----------------|------------------|
| Correlation | 0.60-0.95 | 0.75-0.95 | 0.75-0.95 |
| P-value | 0.01 | 0.05 | 0.05 |
| Rolling Check | 2/4 | None | 2/4 |
| **Trades** | 222 | 699 | 37 |
| **Total PnL** | **+$2,629** | **-$8,981** | **-$452** |
| **Win Rate** | 60.6% | 57.8% | 62.2% |
| Profitable Years | 9/15 | 2/15 | 1/5 |

### Yearly Breakdown v3 (no rolling)

| Year | Pairs | Trades | PnL |
|------|-------|--------|-----|
| 2010 | 20 | 69 | -$98 |
| 2011 | 16 | 72 | -$87 |
| 2012 | 20 | 57 | **-$1,846** |
| 2013 | 4 | 12 | **-$1,669** |
| 2014 | 12 | 53 | -$390 |
| 2015 | 9 | 41 | -$102 |
| 2016 | 13 | 52 | +$172 ✓ |
| 2017 | 14 | 63 | **-$1,060** |
| 2018 | 10 | 41 | -$531 |
| 2019 | 9 | 30 | -$519 |
| 2020 | 14 | 33 | **-$1,183** |
| 2021 | 17 | 62 | **-$1,082** |
| 2022 | 11 | 39 | +$788 ✓ |
| 2023 | 13 | 49 | -$778 |
| 2024 | 7 | 26 | -$598 |

### Key Conclusions

1. **v2's profits are FAKE** - caused by using wrong critical values
   - ADF critical values are ~0.5 units LESS strict than E-G critical values
   - This accepts many pairs that are NOT truly cointegrated

2. **Pairs trading ETF is UNPROFITABLE** when using correct statistics
   - v3 without rolling: -$8,981 (699 trades, 57.8% win)
   - v3 with rolling: -$452 (37 trades, 62.2% win)
   - Only 2 out of 15 years were profitable

3. **Rolling consistency filter is essential** but still not enough
   - Without it: 699 trades, massive losses
   - With it: 37 trades, smaller losses
   - Filter removes bad pairs but not all

4. **High win rate means nothing**
   - 57-62% win rate but overall loss
   - Losing trades are bigger than winning trades
   - Classic pairs trading problem: unbounded loss on divergence

5. **Regime breaks are common**
   - Even cointegrated pairs break during market stress
   - 2012-2013: Large losses during European debt crisis aftermath
   - 2020-2021: Large losses during COVID regime change

### Implications for Research

Many academic papers on pairs trading may have similar bugs:
- Using ADF instead of E-G critical values
- Not properly accounting for multiple testing
- Not doing out-of-sample testing
- Survivorship bias in ETF selection

### Recommendations

1. **Always use statsmodels.tsa.stattools.coint()** for Engle-Granger test
2. **Do NOT use ADF test directly on residuals** with ADF critical values
3. **Include transaction costs** - we use 10 bps which is realistic
4. **Use rolling consistency checks** - pairs must be stable across regimes
5. **Accept the reality**: Pairs trading on broad ETFs is very difficult
