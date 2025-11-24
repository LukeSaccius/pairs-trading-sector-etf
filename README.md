# Mini-Thesis Pairs Trading Pipeline

Research repo for a cointegration-based pairs trading strategy on U.S. sector ETFs. Includes data fetchers, OU mean-reversion modeling, backtesting with realistic costs, and reproducible Jupyter notebooks plus final paper.

## Project Structure *(Core)*
- `src/pairs_trading_etf/` – research package (data, cointegration, OU model, signals, backtests, utils)
- `configs/` – YAML configs (currently `data.yaml` for ticker universe and dates)
- `data/raw|processed/` – downloaded CSVs and cleaned artifacts (git-ignored)
- `notebooks/` – week-by-week research narratives (Week 1 scaffold included)
- `results/figures|backtests/` – generated plots and evaluation exports
- `reports/drafts|final/` – thesis write-ups
- `tests/` – unit and smoke tests
- `notes/`, `advisor_logs/`, `docs/` – qualitative planning materials

## Quickstart *(Core)*
```bash
python -m venv .venv
source .venv/Scripts/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install --upgrade pip
pip install -r requirements.txt

pytest tests/  # optional smoke checks
jupyter notebook notebooks/week1_data_cointegration.ipynb
```

## 1. Overview *(Core)*
- Purpose of the six-week research sprint
- High-level description of the pairs trading methodology and success metrics
- Diagram of the end-to-end workflow (data -> signals -> backtests -> reporting)

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
