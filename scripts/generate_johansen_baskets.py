#!/usr/bin/env python
"""Generate Johansen cointegrated baskets and summary report.

Usage:
    python scripts/generate_johansen_baskets.py
    python scripts/generate_johansen_baskets.py --max-size 5 --max-baskets 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pairs_trading_etf.pipelines.johansen_scan import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PRICE_PATH,
    JohansenScanConfig,
    run_johansen_scan,
)


def main() -> None:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Generate Johansen cointegrated baskets")
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

    print("=" * 60)
    print("JOHANSEN BASKET SCANNER")
    print("=" * 60)
    print(f"Config:        {cfg.config_path}")
    print(f"Prices:        {cfg.price_path}")
    print(f"Basket sizes:  {cfg.min_basket_size} - {cfg.max_basket_size}")
    print(f"Min corr:      {cfg.min_corr_prefilter}")
    print(f"Max baskets:   {cfg.max_baskets}")
    print("-" * 60)

    result_df = run_johansen_scan(cfg)

    print("-" * 60)
    print(f"RESULTS: {len(result_df)} cointegrated baskets found")
    print("-" * 60)

    if not result_df.empty:
        print("\nTop 10 baskets by score:")
        print(result_df.head(10).to_string(index=False))

        print(f"\nOutput saved to: {cfg.output_path}")
        print(f"JSON saved to:   {cfg.output_path.with_suffix('.json')}")
    else:
        print("No cointegrated baskets found with current parameters.")
        print("Try lowering --min-corr or increasing --max-size.")


if __name__ == "__main__":
    main()
