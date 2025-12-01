# Week 1 Summary Report

## Overview
- **Total Pairs Tested**: 102
- **Average Correlation**: 0.912
- **Cointegrated Pairs (p < 0.05)**: 10

## Methodology
1. **Universe**: 50 cross-sector ETFs (equity, international, and core bonds) downloaded from Yahoo Finance.
2. **Correlation**: Calculated rolling correlations; filtered for pairs > 0.80.
3. **Cointegration**: Engle-Granger test applied to high-correlation pairs.

## Top Findings
The top 5 cointegrated pairs were analyzed for price divergence.
See `results/figures/` for overlay plots.

## Next Steps
- Refine entry/exit thresholds based on spread z-scores.
- Implement backtesting engine for the top cointegrated pairs.
