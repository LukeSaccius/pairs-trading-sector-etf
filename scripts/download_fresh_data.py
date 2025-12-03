"""Download fresh ETF price data from Yahoo Finance."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import yaml
from pairs_trading_etf.data.ingestion import download_etf_data, save_raw_data, validate_price_data

def main():
    # Load ETF metadata
    with open("configs/etf_metadata.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    tickers = list(config["etfs"].keys())
    print(f"Found {len(tickers)} ETFs in metadata")
    
    # Download data từ 2005 đến nay
    print("Downloading data from 2005-01-01 to 2024-12-01...")
    prices = download_etf_data(tickers, "2005-01-01", "2024-12-01")
    
    # Validate
    stats = validate_price_data(prices)
    print(f"Downloaded: {stats['n_rows']} days, {stats['n_cols']} ETFs")
    
    # Lọc bỏ ETF có quá nhiều missing data
    missing_pct = prices.isna().mean()
    good_tickers = missing_pct[missing_pct < 0.3].index.tolist()
    prices_clean = prices[good_tickers]
    print(f"After filtering (missing < 30%): {len(good_tickers)} ETFs")
    
    # Save
    output_path = Path("data/raw/etf_prices_fresh.csv")
    save_raw_data(prices_clean, output_path)
    print(f"Saved to {output_path}")
    
    # Summary
    print(f"\nDate range: {prices_clean.index.min()} to {prices_clean.index.max()}")
    print(f"Total trading days: {len(prices_clean)}")

if __name__ == "__main__":
    main()
