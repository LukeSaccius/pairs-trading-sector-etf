"""Utility script to refresh the Week 1 pair-scan CSV using repo defaults."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pairs_trading_etf.pipelines.pair_scan import PairScanConfig, run_pair_scan
from pairs_trading_etf.utils.config import load_yaml_config


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "data.yaml"
    price_path = PROJECT_ROOT / "data" / "raw" / "etf_prices.csv"
    metadata_path = PROJECT_ROOT / "configs" / "etf_metadata.yaml"
    results_path = PROJECT_ROOT / "results" / "week1_pair_scores.csv"

    config = load_yaml_config(config_path)
    pair_defaults = config.get("pair_scan", {})

    cfg = PairScanConfig(
        config_path=config_path,
        price_path=price_path,
        output_path=results_path,
        list_name=pair_defaults.get("list_name"),
        metadata_path=metadata_path,
        lookback_days=pair_defaults.get("lookback_days", 252),
        min_obs=pair_defaults.get("min_obs", 180),
        min_corr=pair_defaults.get("min_corr", 0.85),
        max_pairs=pair_defaults.get("max_pairs", 75),
        engle_granger_maxlag=pair_defaults.get("engle_granger_maxlag", 1),
        return_method=pair_defaults.get("return_method", "log"),
        allow_cross_sector=pair_defaults.get("allow_cross_sector", True),
    )

    df = run_pair_scan(cfg)
    print(f"Generated {len(df)} pairs. Results saved to {results_path}.")
    if not df.empty:
        print(df.head())


if __name__ == "__main__":
    main()
