# Mini-Thesis Pairs Trading Pipeline

Research repo for a cointegration-based pairs trading strategy on U.S. sector ETFs. Includes data fetchers, OU mean-reversion modeling, backtesting with realistic costs, compounding/leverage features, and reproducible Jupyter notebooks plus final paper.

## 📊 Current Status

**Latest Backtest Results (V11 Crisis-Aware):**
| Metric | Value |
|--------|-------|
| Total PnL | $2,079 |
| Trades | 129 |
| Win Rate | 43.4% |
| Profit Factor | 1.41 |
| Max Drawdown | $992 |
| Period | 2007-2024 (18 years) |

**Key Findings:**
- ETF pairs trading works as a market-neutral hedge, NOT as primary alpha source
- Strategy underperforms buy-and-hold SPY (~10%/year) due to:
  - Few valid pairs remain after strict filtering (only 2-7 pairs/year)
  - Crisis period failures (2008, 2020) break cointegration
  - ETF homogeneity limits profit opportunities

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
- ✅ Half-life filtering removes slow mean-reverting pairs (keep only <60 days)
- ✅ Z-score entry/exit signals capture short-term mean-reversion
- ✅ Convergence trades have 100% win rate with avg +$176 profit (28 trades)
- ✅ Strategy provides market-neutral exposure (low beta)

**What Doesn't Work:**
- ❌ Stop-loss trades dominate losses (-$55 avg, 64 trades in V11)
- ❌ Crisis periods (2008, 2020) break cointegration relationships
- ❌ Few valid pairs remain after strict filtering (2-4 pairs/year)
- ❌ Annual returns (~0.23%) below SPY (~10%) - better as hedge than primary strategy

**Recommended Use Case:**
Pairs trading as portfolio diversifier/hedge, NOT primary strategy.

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
- `docs/week2_work_summary.md` - Week 2 deep debugging summary
- `docs/debug_summary.md` - Technical debugging findings

### Key Scripts
| Script | Purpose |
|--------|---------|
| `scripts/run_backtest.py` | Main backtest runner with config support |
| `scripts/debug_trades.py` | Analyze and visualize all trades |
| `scripts/deep_debug.py` | PnL calculation verification |
| `scripts/visualize_trade.py` | Individual trade visualization |
| `scripts/download_fresh_data.py` | Fetch latest ETF price data |

### Version History
| Version | Date | Key Changes |
|---------|------|-------------|
| V4 | 2025-12-02 | First complete backtest, $2,297 PnL |
| V5 | 2025-12-03 | Optimized parameters |
| V9 | 2025-12-03 | Compounding & leverage implemented |
| V10 | 2025-12-03 | Risk management (capital limits) |
| V11 | 2025-12-03 | Crisis-aware config, best results |

---
*Last Updated: 2025-12-03*
