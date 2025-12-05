#!/usr/bin/env python
"""Download global ETF data with currency conversion to USD.

This script downloads daily prices for 400+ global ETFs from Yahoo Finance,
converts all prices to USD using FX rates, and saves to disk.

Usage:
    python scripts/download_global_data.py
    python scripts/download_global_data.py --list us_only
    python scripts/download_global_data.py --start 2010-01-01 --end 2024-12-31

Author: Research Team
Date: 2025-12-03
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pairs_trading_etf.data.global_downloader import (
    GlobalDownloaderConfig,
    download_global_etfs,
    save_global_data,
)
from pairs_trading_etf.data.global_universe import (
    GlobalETFUniverse,
    get_ticker_regions,
    load_global_universe,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Download global ETF data with USD conversion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data config
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/global_data.yaml"),
        help="Path to global data config",
    )
    parser.add_argument(
        "--list",
        dest="list_name",
        type=str,
        default=None,
        help="Universe list to download (default: from config)",
    )
    
    # Date range
    parser.add_argument(
        "--start",
        type=str,
        default="2006-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-12-01",
        help="End date (YYYY-MM-DD)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/global"),
        help="Output directory for data files",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="global_etf",
        help="Prefix for output files",
    )
    
    # Download settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Tickers per download batch",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds between batches",
    )
    parser.add_argument(
        "--no-convert-usd",
        action="store_true",
        help="Skip USD conversion (keep local currency)",
    )
    
    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Load universe
    logger.info(f"Loading universe from {args.config}...")
    try:
        universe = load_global_universe(args.config, args.list_name)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        logger.info("Creating default global_data.yaml...")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print(universe.summary())
    print("=" * 60 + "\n")
    
    # Get ticker -> region mapping
    ticker_regions = get_ticker_regions(universe)
    
    # Configure download
    config = GlobalDownloaderConfig(
        batch_size=args.batch_size,
        sleep_between_batches=args.sleep,
        convert_to_usd=not args.no_convert_usd,
        output_dir=args.output_dir,
    )
    
    # Confirm before downloading
    n_tickers = len(universe.tickers)
    n_batches = (n_tickers + config.batch_size - 1) // config.batch_size
    est_time = n_batches * (config.sleep_between_batches + 2)  # ~2s per batch download
    
    print(f"Download plan:")
    print(f"  Tickers: {n_tickers}")
    print(f"  Batches: {n_batches}")
    print(f"  Estimated time: {est_time/60:.1f} minutes")
    print(f"  USD conversion: {config.convert_to_usd}")
    print(f"  Output: {args.output_dir}")
    print()
    
    response = input("Proceed with download? [y/N]: ").strip().lower()
    if response != "y":
        print("Cancelled.")
        sys.exit(0)
    
    # Download
    print("\nStarting download...")
    result = download_global_etfs(
        tickers=universe.tickers,
        ticker_regions=ticker_regions,
        start=args.start,
        end=args.end,
        config=config,
    )
    
    # Report
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(result.summary())
    
    if result.failed_tickers:
        print(f"\nFailed tickers ({len(result.failed_tickers)}):")
        for t in sorted(result.failed_tickers)[:20]:
            print(f"  - {t}")
        if len(result.failed_tickers) > 20:
            print(f"  ... and {len(result.failed_tickers) - 20} more")
    
    # Save
    print(f"\nSaving to {args.output_dir}...")
    paths = save_global_data(result, args.output_dir, args.prefix)
    
    print("\nFiles created:")
    for name, path in paths.items():
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  {name}: {path.name} ({size_mb:.1f} MB)")
    
    # Summary stats
    if not result.prices_usd.empty:
        print("\nData summary:")
        print(f"  Date range: {result.prices_usd.index.min()} to {result.prices_usd.index.max()}")
        print(f"  Trading days: {len(result.prices_usd)}")
        print(f"  Tickers: {result.prices_usd.shape[1]}")
        
        # Missing data stats
        missing_pct = result.prices_usd.isna().mean().mean() * 100
        print(f"  Missing data: {missing_pct:.1f}%")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
