"""Global ETF data downloader with batching, rate limiting, and currency conversion.

Designed to handle 500+ global ETFs from Yahoo Finance with:
- Batched downloads to avoid API limits
- Currency conversion to USD (using FX rates)
- Regional metadata tracking
- Progress reporting and error recovery

Author: Research Team
Date: 2025-12-03
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# Major currency ETFs/tickers for FX conversion
FX_TICKERS = {
    "EUR": "EURUSD=X",  # Euro to USD
    "GBP": "GBPUSD=X",  # British Pound to USD
    "JPY": "JPYUSD=X",  # Actually need USDJPY=X inverted
    "CHF": "CHFUSD=X",
    "AUD": "AUDUSD=X",
    "CAD": "CADUSD=X",
    "HKD": "HKDUSD=X",
    "SGD": "SGDUSD=X",
    "KRW": "KRWUSD=X",
    "CNY": "CNYUSD=X",
    "INR": "INRUSD=X",
    "BRL": "BRLUSD=X",
    "MXN": "MXNUSD=X",
    "ZAR": "ZARUSD=X",
    "TRY": "TRYUSD=X",
    "TWD": "TWDUSD=X",
}

# Currency per region (for ETFs traded in local currency)
# NOTE: Most ETFs in our universe are US-listed (NYSE/NASDAQ) and trade in USD,
# even if they track foreign markets. We only need to convert ETFs that
# trade on foreign exchanges in local currency.
#
# For our global ETF universe, ALL ETFs are USD-denominated because:
# - iShares country ETFs (EWG, EWJ, etc.) trade on NYSE in USD
# - Vanguard international ETFs (VGK, VPL, etc.) trade on NYSE in USD
# - SPDR international ETFs trade in USD
#
# Currency conversion would only be needed if we added ETFs from:
# - London Stock Exchange (GBP)
# - Tokyo Stock Exchange (JPY)
# - Xetra (EUR)
# - etc.
REGION_CURRENCY = {
    "US": "USD",
    "EUROPE": "USD",      # US-listed ETFs tracking Europe
    "UK": "USD",          # US-listed ETFs tracking UK
    "JAPAN": "USD",       # US-listed ETFs tracking Japan
    "ASIA_PACIFIC": "USD",  # US-listed ETFs
    "EMERGING": "USD",    # US-listed ETFs
    "LATAM": "USD",       # US-listed ETFs
    "GLOBAL": "USD",      # US-listed ETFs
    # For future: if adding foreign-exchange-listed ETFs
    # "EUROPE_LOCAL": "EUR",
    # "UK_LOCAL": "GBP",
    # "JAPAN_LOCAL": "JPY",
}


@dataclass
class DownloadResult:
    """Container for download operation results."""
    
    prices_usd: pd.DataFrame
    prices_local: pd.DataFrame
    fx_rates: pd.DataFrame
    failed_tickers: List[str]
    metadata: Dict[str, Dict]
    
    @property
    def success_rate(self) -> float:
        """Percentage of tickers successfully downloaded."""
        total = self.prices_usd.shape[1] + len(self.failed_tickers)
        if total == 0:
            return 0.0
        return self.prices_usd.shape[1] / total
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Downloaded: {self.prices_usd.shape[1]} tickers, "
            f"Failed: {len(self.failed_tickers)}, "
            f"Date range: {self.prices_usd.index.min()} to {self.prices_usd.index.max()}, "
            f"Success rate: {self.success_rate:.1%}"
        )


@dataclass
class GlobalDownloaderConfig:
    """Configuration for global ETF downloader."""
    
    # Download settings
    batch_size: int = 50  # Tickers per batch
    sleep_between_batches: float = 1.0  # Seconds between batches
    max_retries: int = 3
    retry_delay: float = 5.0
    
    # Currency settings
    convert_to_usd: bool = True
    base_currency: str = "USD"
    
    # Data quality
    min_history_days: int = 252  # Minimum 1 year of history
    max_missing_pct: float = 0.20  # Max 20% missing data
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("data/raw/global"))
    save_local_prices: bool = True  # Also save non-USD prices


def download_fx_rates(
    currencies: Sequence[str],
    start: str,
    end: str,
    config: GlobalDownloaderConfig | None = None,
) -> pd.DataFrame:
    """Download FX rates for currency conversion to USD.
    
    Args:
        currencies: List of currency codes (e.g., ["EUR", "GBP", "JPY"])
        start: Start date YYYY-MM-DD
        end: End date YYYY-MM-DD
        config: Optional config for retries
        
    Returns:
        DataFrame with columns = currency codes, values = rate to USD
    """
    config = config or GlobalDownloaderConfig()
    
    # Build ticker list
    fx_tickers = []
    currency_map = {}
    for ccy in currencies:
        if ccy == "USD":
            continue
        ticker = FX_TICKERS.get(ccy)
        if ticker:
            fx_tickers.append(ticker)
            currency_map[ticker] = ccy
    
    if not fx_tickers:
        logger.info("No FX rates to download (all USD)")
        return pd.DataFrame()
    
    logger.info(f"Downloading FX rates for {len(fx_tickers)} currencies...")
    
    end_plus_one = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    
    for attempt in range(config.max_retries):
        try:
            raw = yf.download(
                tickers=fx_tickers,
                start=start,
                end=end_plus_one,
                interval="1d",
                progress=False,
                group_by="column",
            )
            
            if raw.empty:
                raise ValueError("Empty FX data returned")
            
            # Extract adjusted close
            if isinstance(raw.columns, pd.MultiIndex):
                if "Adj Close" in raw.columns.get_level_values(0):
                    fx_prices = raw["Adj Close"]
                else:
                    fx_prices = raw["Close"]
            else:
                fx_prices = raw
            
            # Rename columns to currency codes
            fx_prices = fx_prices.rename(columns=currency_map)
            
            # Handle JPY (need to invert USDJPY)
            if "JPY" not in fx_prices.columns and "USDJPY=X" not in raw.columns:
                # Try to get USDJPY and invert
                try:
                    usdjpy = yf.download("USDJPY=X", start=start, end=end_plus_one, progress=False)
                    if not usdjpy.empty:
                        if isinstance(usdjpy.columns, pd.MultiIndex):
                            jpy_rate = 1.0 / usdjpy["Adj Close"]["USDJPY=X"]
                        else:
                            jpy_rate = 1.0 / usdjpy["Adj Close"]
                        fx_prices["JPY"] = jpy_rate
                except Exception as e:
                    logger.warning(f"Could not get JPY rate: {e}")
            
            # Forward fill missing FX rates (weekends/holidays)
            fx_prices = fx_prices.ffill().bfill()
            
            # Add USD column (always 1.0)
            fx_prices["USD"] = 1.0
            
            logger.info(f"Downloaded FX rates: {list(fx_prices.columns)}")
            return fx_prices
            
        except Exception as e:
            logger.warning(f"FX download attempt {attempt + 1} failed: {e}")
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay)
    
    logger.error("Failed to download FX rates after retries")
    return pd.DataFrame()


def download_batch(
    tickers: Sequence[str],
    start: str,
    end: str,
    attempt: int = 0,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> tuple[pd.DataFrame, List[str]]:
    """Download a batch of tickers with retry logic.
    
    Returns:
        Tuple of (prices DataFrame, list of failed tickers)
    """
    end_plus_one = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    
    try:
        raw = yf.download(
            tickers=list(tickers),
            start=start,
            end=end_plus_one,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        
        if raw.empty:
            return pd.DataFrame(), list(tickers)
        
        # Extract close prices
        # NOTE: With auto_adjust=True, yfinance returns adjusted prices in "Close"
        # The "Adj Close" column only contains failed/delisted tickers (as NaN)
        # So we ALWAYS use "Close" when auto_adjust=True
        if isinstance(raw.columns, pd.MultiIndex):
            # Always prefer "Close" when auto_adjust=True (it contains adjusted prices)
            if "Close" in raw.columns.get_level_values(0):
                prices = raw["Close"]
            elif "Adj Close" in raw.columns.get_level_values(0):
                prices = raw["Adj Close"]
            else:
                logger.warning("No Close or Adj Close found, using first level")
                prices = raw.iloc[:, raw.columns.get_level_values(0) == raw.columns.get_level_values(0)[0]]
        else:
            prices = raw
        
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        
        # Identify failed tickers (all NaN)
        failed = [col for col in prices.columns if prices[col].isna().all()]
        
        return prices, failed
        
    except Exception as e:
        logger.warning(f"Batch download failed (attempt {attempt + 1}): {e}")
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
            return download_batch(tickers, start, end, attempt + 1, max_retries, retry_delay)
        return pd.DataFrame(), list(tickers)


def download_global_etfs(
    tickers: Sequence[str],
    ticker_regions: Dict[str, str],
    start: str,
    end: str,
    config: GlobalDownloaderConfig | None = None,
) -> DownloadResult:
    """Download global ETFs with currency conversion to USD.
    
    Args:
        tickers: List of ETF tickers to download
        ticker_regions: Mapping of ticker -> region (e.g., {"EWG": "EUROPE"})
        start: Start date YYYY-MM-DD
        end: End date YYYY-MM-DD
        config: Download configuration
        
    Returns:
        DownloadResult with USD-converted prices and metadata
    """
    config = config or GlobalDownloaderConfig()
    tickers = list(tickers)
    
    logger.info(f"Starting global ETF download: {len(tickers)} tickers")
    
    # Step 1: Identify currencies needed
    currencies_needed = set()
    ticker_currency = {}
    for ticker in tickers:
        region = ticker_regions.get(ticker, "US")
        currency = REGION_CURRENCY.get(region, "USD")
        currencies_needed.add(currency)
        ticker_currency[ticker] = currency
    
    logger.info(f"Currencies needed: {currencies_needed}")
    
    # Step 2: Download FX rates if converting
    fx_rates = pd.DataFrame()
    if config.convert_to_usd and len(currencies_needed - {"USD"}) > 0:
        fx_rates = download_fx_rates(list(currencies_needed), start, end, config)
    
    # Step 3: Download ETF prices in batches
    all_prices = []
    all_failed = []
    n_batches = (len(tickers) + config.batch_size - 1) // config.batch_size
    
    for i in range(0, len(tickers), config.batch_size):
        batch = tickers[i:i + config.batch_size]
        batch_num = i // config.batch_size + 1
        
        logger.info(f"Downloading batch {batch_num}/{n_batches}: {len(batch)} tickers")
        
        prices, failed = download_batch(
            batch, start, end,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay
        )
        
        if not prices.empty:
            all_prices.append(prices)
        all_failed.extend(failed)
        
        # Rate limiting
        if i + config.batch_size < len(tickers):
            time.sleep(config.sleep_between_batches)
    
    if not all_prices:
        logger.error("No prices downloaded!")
        return DownloadResult(
            prices_usd=pd.DataFrame(),
            prices_local=pd.DataFrame(),
            fx_rates=fx_rates,
            failed_tickers=all_failed,
            metadata={},
        )
    
    # Step 4: Combine all batches
    prices_local = pd.concat(all_prices, axis=1)
    prices_local.index = pd.to_datetime(prices_local.index).tz_localize(None)
    prices_local = prices_local.sort_index()
    prices_local = prices_local[~prices_local.index.duplicated(keep="first")]
    
    # Remove completely failed tickers
    valid_cols = [c for c in prices_local.columns if not prices_local[c].isna().all()]
    prices_local = prices_local[valid_cols]
    
    logger.info(f"Downloaded {len(valid_cols)} tickers, {len(all_failed)} failed")
    
    # Step 5: Convert to USD
    if config.convert_to_usd and not fx_rates.empty:
        prices_usd = _convert_to_usd(prices_local, fx_rates, ticker_currency)
    else:
        prices_usd = prices_local.copy()
    
    # Step 6: Apply data quality filters
    prices_usd, dropped = _filter_data_quality(prices_usd, config)
    all_failed.extend(dropped)
    
    # Step 7: Build metadata
    metadata = {}
    for ticker in prices_usd.columns:
        region = ticker_regions.get(ticker, "US")
        currency = ticker_currency.get(ticker, "USD")
        metadata[ticker] = {
            "region": region,
            "original_currency": currency,
            "converted_to_usd": config.convert_to_usd and currency != "USD",
            "first_date": str(prices_usd[ticker].first_valid_index()),
            "last_date": str(prices_usd[ticker].last_valid_index()),
            "observations": int(prices_usd[ticker].count()),
        }
    
    result = DownloadResult(
        prices_usd=prices_usd,
        prices_local=prices_local if config.save_local_prices else pd.DataFrame(),
        fx_rates=fx_rates,
        failed_tickers=list(set(all_failed)),
        metadata=metadata,
    )
    
    logger.info(result.summary())
    return result


def _convert_to_usd(
    prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
    ticker_currency: Dict[str, str],
) -> pd.DataFrame:
    """Convert local currency prices to USD using FX rates."""
    
    prices_usd = prices.copy()
    
    # Align dates
    common_dates = prices_usd.index.intersection(fx_rates.index)
    if len(common_dates) == 0:
        logger.warning("No overlapping dates between prices and FX rates!")
        return prices_usd
    
    for ticker in prices_usd.columns:
        currency = ticker_currency.get(ticker, "USD")
        if currency == "USD":
            continue
        
        if currency not in fx_rates.columns:
            logger.warning(f"No FX rate for {currency}, keeping {ticker} as-is")
            continue
        
        # Convert: USD_price = local_price * fx_rate
        # where fx_rate = how many USD per 1 unit of local currency
        fx = fx_rates[currency].reindex(prices_usd.index).ffill().bfill()
        prices_usd[ticker] = prices_usd[ticker] * fx
    
    return prices_usd


def _filter_data_quality(
    prices: pd.DataFrame,
    config: GlobalDownloaderConfig,
) -> tuple[pd.DataFrame, List[str]]:
    """Filter out tickers with poor data quality."""
    
    dropped = []
    
    # Check minimum history
    for col in prices.columns:
        n_obs = prices[col].count()
        if n_obs < config.min_history_days:
            dropped.append(col)
            logger.debug(f"Dropping {col}: only {n_obs} observations")
            continue
        
        missing_pct = prices[col].isna().mean()
        if missing_pct > config.max_missing_pct:
            dropped.append(col)
            logger.debug(f"Dropping {col}: {missing_pct:.1%} missing data")
    
    valid_cols = [c for c in prices.columns if c not in dropped]
    
    if dropped:
        logger.info(f"Dropped {len(dropped)} tickers due to data quality")
    
    return prices[valid_cols], dropped


def save_global_data(
    result: DownloadResult,
    output_dir: Path | str,
    prefix: str = "global_etf",
) -> Dict[str, Path]:
    """Save download results to disk.
    
    Creates:
        - {prefix}_prices_usd.csv: USD-converted prices
        - {prefix}_prices_local.csv: Local currency prices (if available)
        - {prefix}_fx_rates.csv: FX rates used
        - {prefix}_metadata.csv: Ticker metadata
        - {prefix}_failed.txt: Failed tickers
        
    Returns:
        Dict of file paths created
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # USD prices (main output)
    usd_path = output_dir / f"{prefix}_prices_usd.csv"
    result.prices_usd.to_csv(usd_path)
    paths["prices_usd"] = usd_path
    logger.info(f"Saved USD prices: {usd_path}")
    
    # Local prices (optional)
    if not result.prices_local.empty:
        local_path = output_dir / f"{prefix}_prices_local.csv"
        result.prices_local.to_csv(local_path)
        paths["prices_local"] = local_path
    
    # FX rates
    if not result.fx_rates.empty:
        fx_path = output_dir / f"{prefix}_fx_rates.csv"
        result.fx_rates.to_csv(fx_path)
        paths["fx_rates"] = fx_path
    
    # Metadata
    if result.metadata:
        meta_df = pd.DataFrame(result.metadata).T
        meta_path = output_dir / f"{prefix}_metadata.csv"
        meta_df.to_csv(meta_path)
        paths["metadata"] = meta_path
    
    # Failed tickers
    if result.failed_tickers:
        failed_path = output_dir / f"{prefix}_failed.txt"
        failed_path.write_text("\n".join(sorted(result.failed_tickers)))
        paths["failed"] = failed_path
    
    return paths


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    from pairs_trading_etf.data.global_universe import load_global_universe
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Download global ETF data")
    parser.add_argument("--config", type=Path, default=Path("configs/global_data.yaml"))
    parser.add_argument("--start", type=str, default="2006-01-01")
    parser.add_argument("--end", type=str, default="2025-12-01")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/global"))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--no-convert-usd", action="store_true")
    
    args = parser.parse_args()
    
    # Load universe
    universe = load_global_universe(args.config)
    tickers = universe.tickers
    ticker_regions = {t: universe.metadata[t].region for t in tickers if t in universe.metadata}
    
    # Configure download
    config = GlobalDownloaderConfig(
        batch_size=args.batch_size,
        convert_to_usd=not args.no_convert_usd,
        output_dir=args.output_dir,
    )
    
    # Download
    result = download_global_etfs(tickers, ticker_regions, args.start, args.end, config)
    
    # Save
    paths = save_global_data(result, args.output_dir)
    
    print(f"\n{result.summary()}")
    print(f"Files saved: {list(paths.keys())}")
