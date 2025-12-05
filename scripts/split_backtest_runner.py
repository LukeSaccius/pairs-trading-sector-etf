"""
Split Backtest Runner with Visualizations
-----------------------------------------

Runs walk-forward backtests for train/validation/test splits and
generates trade visualizations for every trade automatically.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

os.environ.setdefault("MPLBACKEND", "Agg")

try:
    from pairs_trading_etf.backtests import (
        load_config,
        run_walkforward_backtest,
    )
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(SRC_PATH))
    from pairs_trading_etf.backtests import (  # type: ignore[no-redef]
        load_config,
        run_walkforward_backtest,
    )

# ============================================================================
# User Settings
# ============================================================================

CONFIG_PATH = Path("configs/experiments/v17a_vol_filter.yaml")
OVERRIDES: Dict[str, Any] = {
    # Example overrides:
    # "pvalue_threshold": 0.05,
    # "min_pairs_for_trading": 2,
}

SPLITS: List[Tuple[str, int, int]] = [
    ("train", 2005, 2012),
    ("validation", 2013, 2020),
    ("test", 2021, 2024),
]

SHOW_ALL_TRADES = True          # Print every trade to console
GENERATE_VISUALS = True         # Call visualize_trade_v2.py --all
RESULTS_ROOT = Path("results")


def apply_overrides(config, overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise AttributeError(f"Config has no attribute '{key}' to override.")


def run_split(label: str, start_year: int, end_year: int, prices: pd.DataFrame, cfg) -> Path:
    trades, summary_df = run_walkforward_backtest(
        prices=prices,
        cfg=cfg,
        start_year=start_year,
        end_year=end_year,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    split_dir = RESULTS_ROOT / f"{timestamp}_{cfg.experiment_name}" / label
    split_dir.mkdir(parents=True, exist_ok=True)

    trades_path = split_dir / f"{label}_trades.csv"
    summary_path = split_dir / f"{label}_summary.csv"

    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(trades_path, index=False)
    else:
        trades_df = pd.DataFrame()
        trades_path.write_text("No trades\n", encoding="utf-8")

    summary_df.to_csv(summary_path, index=False)

    print(f"\n==== {label.upper()} ({start_year}-{end_year}) ====")
    print(f"Trades: {len(trades_df)} | Summary rows: {len(summary_df)}")
    if SHOW_ALL_TRADES and not trades_df.empty:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(trades_df)

    print(f"Saved trades to {trades_path}")
    print(f"Saved summary to {summary_path}")

    if GENERATE_VISUALS and not trades_df.empty:
        visualize_all_trades(trades_path)

    return trades_path


def visualize_all_trades(trades_csv: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "visualize_trade_v2.py"
    cmd = [
        sys.executable,
        str(script),
        "--trades",
        str(trades_csv),
        "--all",
    ]
    print(f"Generating visualizations for {trades_csv} ...")
    subprocess.run(cmd, check=False)


def main() -> None:
    config_path = CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(str(config_path))
    if OVERRIDES:
        apply_overrides(cfg, OVERRIDES)

    prices = pd.read_csv(cfg.price_data_path, index_col=0, parse_dates=True)

    for label, start_year, end_year in SPLITS:
        run_split(label, start_year, end_year, prices, cfg)


if __name__ == "__main__":
    main()
