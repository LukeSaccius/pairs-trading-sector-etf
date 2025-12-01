"""Pipeline for scanning ETF baskets using Johansen cointegration test."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pairs_trading_etf.analysis.cointegration.johansen import (
    JohansenResult,
    scan_johansen_baskets,
)
from pairs_trading_etf.data.loader import build_price_frame
from pairs_trading_etf.data.universe import load_configured_universe

DEFAULT_CONFIG_PATH = Path("configs/data.yaml")
DEFAULT_PRICE_PATH = Path("data/raw/etf_prices.csv")
DEFAULT_OUTPUT_PATH = Path("results/johansen_baskets.csv")


@dataclass(slots=True)
class JohansenScanConfig:
    """Typed configuration for the Johansen basket scan pipeline."""

    config_path: Path = DEFAULT_CONFIG_PATH
    price_path: Path = DEFAULT_PRICE_PATH
    output_path: Path | None = DEFAULT_OUTPUT_PATH
    metadata_path: Path | None = None
    min_basket_size: int = 3
    max_basket_size: int = 4
    det_order: int = 0
    k_ar_diff: int = 1
    min_corr_prefilter: float | None = 0.70
    max_baskets: int | None = 50
    min_obs: int = 180


def johansen_results_to_frame(results: list[JohansenResult]) -> pd.DataFrame:
    """Convert JohansenResult objects into a DataFrame."""
    rows = []
    for r in results:
        row = {
            "tickers": ",".join(r.tickers),
            "basket_size": r.basket_size,
            "cointegration_rank": r.cointegration_rank,
            "score": r.score,
            "trace_stat_r0": r.trace_stats[0] if r.trace_stats else None,
            "crit_95_r0": r.crit_95[0] if r.crit_95 else None,
            "eigenvalue_0": r.eigenvalues[0] if r.eigenvalues else None,
            "hedge_ratios": json.dumps([round(v, 4) for v in r.eigenvectors[0]]) if r.eigenvectors else "[]",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def johansen_results_to_json(results: list[JohansenResult]) -> list[dict]:
    """Convert JohansenResult objects to list of dicts for JSON export."""
    return [r.as_dict() for r in results]


def run_johansen_scan(cfg: JohansenScanConfig) -> pd.DataFrame:
    """Execute the Johansen basket scanning pipeline."""
    # Load universe
    universe = load_configured_universe(cfg.config_path, metadata_path=cfg.metadata_path)

    # Load prices
    price_frame = build_price_frame(
        cfg.price_path,
        tickers=universe.tickers,
        min_non_na=cfg.min_obs,
        allow_missing=True,
    )

    # Run Johansen scan
    results = scan_johansen_baskets(
        prices=price_frame.prices,
        min_basket_size=cfg.min_basket_size,
        max_basket_size=cfg.max_basket_size,
        det_order=cfg.det_order,
        k_ar_diff=cfg.k_ar_diff,
        min_corr_prefilter=cfg.min_corr_prefilter,
        max_baskets=cfg.max_baskets,
    )

    # Convert to DataFrame
    result_df = johansen_results_to_frame(results)

    # Save outputs
    if cfg.output_path is not None:
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(cfg.output_path, index=False)

        # Also save JSON for richer data
        json_path = cfg.output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(johansen_results_to_json(results), f, indent=2)

    return result_df


def main() -> None:
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Scan ETF baskets for Johansen cointegration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to YAML config")
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICE_PATH, help="Path to price CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output CSV path")
    parser.add_argument("--min-size", type=int, default=3, help="Minimum basket size")
    parser.add_argument("--max-size", type=int, default=4, help="Maximum basket size")
    parser.add_argument("--min-corr", type=float, default=0.70, help="Minimum avg correlation prefilter")
    parser.add_argument("--max-baskets", type=int, default=50, help="Maximum baskets to return")
    args = parser.parse_args()

    cfg = JohansenScanConfig(
        config_path=args.config,
        price_path=args.prices,
        output_path=args.output,
        min_basket_size=args.min_size,
        max_basket_size=args.max_size,
        min_corr_prefilter=args.min_corr,
        max_baskets=args.max_baskets,
    )

    result_df = run_johansen_scan(cfg)
    print(f"Found {len(result_df)} cointegrated baskets.")
    if not result_df.empty:
        print(result_df.head(10).to_string())


if __name__ == "__main__":
    main()
