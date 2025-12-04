# Mini-Thesis Pairs Trading Pipeline

Research repo for a cointegration-based pairs trading strategy on U.S. sector ETFs. Includes data fetchers, OU mean-reversion modeling, backtesting with realistic costs, cross-validation framework, and reproducible Jupyter notebooks plus final paper.

## 🚨 Critical Finding: Cross-Validation Results

**Original V17a backtest showed $9,608 profit — but this was OVERFIT!**

After implementing proper train/validation/test splits:

| Period | Date Range | PnL | Win Rate |
|--------|------------|-----|----------|
| Train | 2009-2016 | +$2,530 | 90.0% |
| Validation | 2017-2020 | +$1,488 | 72.7% |
| **Test (Unseen)** | 2021-2024 | **-$3** | 36.4% |

**Key Discovery:** Stop-loss was triggering on 100% of trades before mean-reversion could complete. Removing stop-loss and using time-based exits achieves near-breakeven on truly out-of-sample data.

See `docs/cross_validation_findings.md` for full analysis.

## 📊 Current Status

**Latest Cross-Validated Results (Robust Config):**
| Metric | Value |
|--------|-------|
| Test Period PnL | -$3 (breakeven) |
| Test Win Rate | 36.4% |
| Test Trades | 11 |
| Entry Z-Score | 3.0 |
| Stop-Loss | Disabled |
| Max Holding | 5× half-life |

**Key Findings:**
- Original $9,608 backtest was severely overfit to historical data
- Stop-loss is harmful for mean-reversion strategies (cuts winners early)
- Higher entry threshold (z=3.0) reduces false signals
- Pairs DO mean-revert, but need time (no premature stops)
- True out-of-sample performance is near breakeven, not profitable

## Project Structure *(Core)*
- `src/pairs_trading_etf/` – research package (data, cointegration, OU model, signals, backtests, utils)
- `configs/` – YAML configs for universe, experiments, and backtests
  - `experiments/` – version-specific backtest configs (v5-v11)
- `data/raw|processed/` – downloaded CSVs and cleaned artifacts (git-ignored)
- `notebooks/` – week-by-week research narratives (Week 1 scaffold included)
- `results/` – backtest outputs organized by timestamp and version
- `reports/drafts|final/` – thesis write-ups
- `tests/` – unit and smoke tests
- `notes/`, `advisor_logs/`, `docs/` – qualitative planning materials, research log

## Quickstart *(Core)*
```bash
python -m venv .venv
source .venv/Scripts/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install --upgrade pip
pip install -r requirements.txt

pytest tests/  # optional smoke checks
jupyter notebook notebooks/week1_data_cointegration.ipynb
```

### Run Backtest
```bash
# Run with default config
python scripts/run_backtest.py

# Run with specific experiment config
python scripts/run_backtest.py configs/experiments/v11_crisis_aware.yaml

# Debug trade analysis
python scripts/debug_trades.py
```

### Pair Scan CLI
- Run `python -m pairs_trading_etf.pipelines.pair_scan --max-pairs none` to rank every valid pair.
- Any integer passed to `--max-pairs` keeps just the top N; omitting or using `none`/`all` leaves results untrimmed.

## Backtest Configurations

### Available Experiment Configs
| Config | Description |
|--------|-------------|
| `default.yaml` | Baseline conservative settings |
| `v5_optimized.yaml` | First optimized version |
| `v6_aggressive.yaml` | Higher leverage, looser filters |
| `v10_risk_managed.yaml` | Capital limits, min pairs requirement |
| `v11_crisis_aware.yaml` | **Recommended** - Crisis detection, tight stops |

### Key Configuration Options
```yaml
trading:
  initial_capital: 50000
  leverage: 1.5          # Capital multiplier
  max_capital_per_trade: 15000  # Prevents over-concentration (V11)
  min_pairs_for_trading: 4      # Minimum pairs to start trading
  
signals:
  entry_zscore: 2.8      # Higher = fewer but higher-quality trades
  exit_zscore: 0.3       # Mean-reversion target
  stop_loss_zscore: 3.0  # Cut losses early
  
risk:
  blacklist_stop_loss_rate: 0.20  # Exclude pairs with high SL rate
```

## 1. Overview *(Core)*
- Purpose of the six-week research sprint
- High-level description of the pairs trading methodology and success metrics
- Diagram of the end-to-end workflow (data -> signals -> backtests -> reporting)

### Research Findings Summary

**What Works:**
- ✅ Cointegration-based pair selection filters noise effectively
- ✅ Half-life filtering removes slow mean-reverting pairs (keep only <25 days)
- ✅ Z-score entry/exit signals capture short-term mean-reversion
- ✅ Pairs DO eventually mean-revert when given enough time
- ✅ Higher entry threshold (z=3.0) improves signal quality

**What Doesn't Work:**
- ❌ **Stop-loss kills mean-reversion trades** — exits before convergence
- ❌ Lower entry thresholds (z<2.5) generate too many false signals
- ❌ Full-period backtests hide overfitting — always use train/val/test splits
- ❌ Crisis periods (2008, 2020, 2022) stress cointegration relationships

**Critical Lesson:**
The $9,608 backtest profit was an illusion of overfitting. Proper cross-validation reveals near-breakeven performance on unseen data.

**Recommended Use Case:**
Pairs trading as portfolio diversifier/hedge, NOT primary alpha strategy.

## 2. Project Layout *(Core)*
- Table describing each top-level folder (data, src, notebooks, results, reports, notes, advisor_logs, configs, tests, docs)
- Conventions for naming files, notebooks, and experiment outputs

## 3. Environment Setup *(Core)*
- Python version and virtual environment instructions (venv or conda)
- Dependency installation steps using `requirements.txt` or `pip-tools`
- Optional Docker/DevContainer guidance for reproducibility

## 4. Data Workflow *(Core)*
- Data sources (exchanges, APIs, research feeds) and access requirements
- Ingestion scripts location (`src/data`) and configuration references (`configs/`)
- Data provenance logging, checksum strategy, and storage rules (raw vs processed)

## 5. Pipeline Stages *(Core)*
- Feature engineering modules (`src/features`) and factors used
- Model calibration / signal generation logic (`src/models`, `src/pipelines`)
- Backtesting framework (`src/backtests`) with assumptions on fees, slippage, execution

## 6. Notebooks & Experiments *(Core)*
- Naming standard (`YYYY-MM-DD_topic.ipynb`) and storage (`notebooks/exploratory`, `notebooks/production`)
- Instructions for converting notebooks to reports (e.g., Jupytext, nbconvert)
- Mapping between notebooks and generated artifacts in `results/` and `reports/`

## 7. Testing & Validation *(Core)*
- Running unit/integration tests (`pytest`, coverage, data integrity checks)
- Static analysis (`ruff`, `black`, `mypy`) and pre-commit hooks

## 8. Reporting & Documentation *(Core)*
- How to regenerate figures and tables in `results/`
- Draft vs final report workflow (`reports/drafts`, `reports/final`)
- Advisor touchpoints logged in `advisor_logs/` and supporting notes in `notes/`

## 9. Roadmap *(Optional)*
- Planned enhancements, stretch goals, and future datasets
- Open questions for advisors/mentors

## 10. License & Acknowledgments *(Core)*
- Licensing for project code and any third-party dependencies
- Data usage terms and citations for academic references

## 11. Documentation

### Research Logs
- `docs/research_log.md` - Detailed session-by-session research narrative
- `docs/cross_validation_findings.md` - **Critical overfitting discovery**
- `docs/kalman_analysis_summary.md` - Kalman filter analysis
- `docs/week2_work_summary.md` - Week 2 deep debugging summary

### Key Scripts
| Script | Purpose |
|--------|---------|
| `scripts/run_backtest.py` | Main backtest runner with config support |
| `scripts/run_cv_backtest.py` | **Cross-validated backtest with train/val/test** |
| `scripts/visualize_trade.py` | Individual trade visualization |
| `scripts/download_fresh_data.py` | Fetch latest ETF price data |

### Core Modules
| Module | Purpose |
|--------|---------|
| `src/pairs_trading_etf/backtests/engine.py` | Backtest engine |
| `src/pairs_trading_etf/backtests/validation.py` | Pair stability validation |
| `src/pairs_trading_etf/backtests/cross_validation.py` | Train/val/test framework |

### Version History
| Version | Date | Key Changes |
|---------|------|-------------|
| V4 | 2025-12-02 | First complete backtest, $2,297 PnL |
| V11 | 2025-12-03 | Crisis-aware config |
| V17a | 2025-12-03 | Best full-period results ($9,608) — OVERFIT |
| **CV** | 2025-12-03 | **Cross-validation reveals true performance: ~$0** |

---
*Last Updated: 2025-12-03 (Cross-Validation Update)*
