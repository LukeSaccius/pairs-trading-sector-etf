"""
Quick Backtest Runner
---------------------

Edit CONFIG_PATH and OVERRIDES below, then run:

    python scripts/quick_backtest_runner.py

to execute a validated backtest without typing long CLI commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import sys

import pandas as pd

try:
    from pairs_trading_etf.backtests import (
        load_config,
        PipelineConfig,
        run_validated_backtest,
    )
except ModuleNotFoundError:  # pragma: no cover
    project_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root))
    sys.path.append(str(project_root / "src"))
    from pairs_trading_etf.backtests import (  # type: ignore[no-redef]
        load_config,
        PipelineConfig,
        run_validated_backtest,
    )

# ============================================================================
# User Settings
# ============================================================================

# Base YAML config to start from
CONFIG_PATH = Path("configs/experiments/v17a_vol_filter.yaml")

# Override any BacktestConfig attribute here (optional)
OVERRIDES: Dict[str, Any] = {
    # Example:
    # "max_capital_per_trade": 20000,
    # "min_pairs_for_trading": 2,
}

# Validation / output options
RUN_CPCV = True          # Set False for quick (non-validated) run
SAVE_RESULTS = True
OUTPUT_DIR = "results"   # Folder for timestamped runs
START_YEAR = 2010
END_YEAR = 2024


def apply_overrides(config, overrides: Dict[str, Any]) -> None:
    """Apply dictionary overrides to a BacktestConfig instance."""
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise AttributeError(f"Config has no attribute '{key}' to override.")


def main() -> None:
    config_path = CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(str(config_path))
    if OVERRIDES:
        apply_overrides(cfg, OVERRIDES)

    pipeline_cfg = PipelineConfig(
        run_cpcv=RUN_CPCV,
        save_results=SAVE_RESULTS,
        output_dir=OUTPUT_DIR,
    )

    prices = pd.read_csv(cfg.price_data_path, index_col=0, parse_dates=True)

    result = run_validated_backtest(
        prices=prices,
        config=cfg,
        pipeline_config=pipeline_cfg,
        start_year=START_YEAR,
        end_year=END_YEAR,
        verbose=True,
    )

    print(result.summary())


if __name__ == "__main__":
    main()
