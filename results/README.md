# Results Directory Structure

This directory contains all backtest results, experiments, and visualizations for the ETF Pairs Trading project.

## 📁 Directory Structure

```
results/
├── experiments/           # Organized by version/phase
│   ├── v5_v8_early/       # Early experiments (basic setup)
│   ├── v9_v11_risk/       # Risk management experiments
│   ├── v14_vidyamurthy/   # Vidyamurthy framework implementation
│   ├── v15_kalman/        # Kalman filter experiments
│   ├── v16_optimized/     # Optimization experiments
│   └── v17_final/         # Final optimization (BEST RESULTS)
│       └── 2025-12-03_20-34_v17a_vol_filter/  # ⭐ Best config: $9,608 PnL
│
├── figures/               # All visualizations
│   ├── trades/            # Individual trade charts (WIN/LOSS)
│   ├── debug/             # Debug visualizations by year
│   ├── forensic/          # Forensic analysis charts
│   └── analysis/          # General analysis charts
│
├── archive/               # Old/duplicate runs
│   └── duplicates/
│       ├── v15c_kalman/   # Failed Kalman experiments
│       ├── v16b_runs/     # Duplicate v16b runs
│       └── v17_early/     # Early v17 attempts
│
├── legacy/                # Old CSV files from early development
│   ├── backtest_v4_*.csv  # V4 backtest results
│   ├── week1_*.csv        # Week 1 scanning results
│   └── *.csv              # Other legacy files
│
├── backtests/             # Empty (legacy folder)
│
└── README.md              # This file
```

## 🏆 Best Configuration

**V17a (Vol Filter)** - Located in `experiments/v17_final/2025-12-03_20-34_v17a_vol_filter/`

| Metric | Value |
|--------|-------|
| Total PnL | **$9,608** |
| Total Trades | 74 |
| Win Rate | 68.9% |
| Profit Factor | 2.76 |
| Annualized Return | ~1.2% |

## 📊 Version History

| Version | Description | PnL | Key Change |
|---------|-------------|-----|------------|
| V5-V8 | Early experiments | Variable | Basic setup |
| V9 | Compounding | $1,336 | Capital growth |
| V10 | Risk managed | $1,056 | Position limits |
| V11 | Crisis aware | $2,079 | Sector exclusions |
| V14 | Vidyamurthy | $3,783 | SNR/ZCR filters |
| V15 | Kalman tests | Negative | Failed approach |
| V16 | Optimized | $8,602 | Parameter tuning |
| V16b | Best before V17 | $9,189 | Entry z=2.8 |
| **V17a** | **Vol filter** | **$9,608** | vol_size_min=0.50 |

## 📈 Key Findings

1. **Vol sizing filter works** - Higher minimum position filters out high-vol pairs
2. **Convergence exits are profitable** - 100% win rate, avg +$311/trade
3. **Max holding exits are marginal** - 47.5% win rate, avg +$4/trade
4. **EUROPE sector performs best** - Most stable cointegration

## 📁 Figure Categories

### trades/
Individual trade visualizations showing:
- Price movements for both legs
- Entry/exit points
- Z-score evolution
- PnL calculation

Naming: `trade_{WIN|LOSS}_{ETF1}_{ETF2}_{YYYYMMDD}.png`

### debug/
Yearly trade summaries:
- `all_trades_YYYY.png` - All trades for a specific year
- `all_trades_all.png` - Combined view of all years
- `kalman_vs_ols_debug.png` - Kalman vs OLS comparison

### forensic/
Detailed analysis of problematic trades:
- `forensic_N_{ETF1}_{ETF2}.png` - Individual forensic analysis
- `forensic_summary.csv` - Summary statistics

---

*Last Updated: 2025-12-03*
