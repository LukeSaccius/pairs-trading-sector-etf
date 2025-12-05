"""Global ETF universe management with region-aware loading.

Extends the base universe module to support:
- Region-based ETF grouping
- Cross-region pair filtering
- Hierarchical category resolution
- USD currency normalization metadata

Author: Research Team
Date: 2025-12-03
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from pairs_trading_etf.utils.config import ConfigError, load_yaml_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalETFMetadata:
    """Metadata for a global ETF with region and currency info."""
    
    ticker: str
    name: str
    region: str
    sector: str | None = None
    category: str | None = None
    currency: str = "USD"
    exchange: str | None = None
    issuer: str | None = None
    expense_ratio: float | None = None
    benchmark: str | None = None
    description: str | None = None
    inception: str | None = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "region": self.region,
            "sector": self.sector,
            "category": self.category,
            "currency": self.currency,
            "exchange": self.exchange,
            "issuer": self.issuer,
            "expense_ratio": self.expense_ratio,
            "benchmark": self.benchmark,
            "description": self.description,
            "inception": self.inception,
        }


@dataclass(frozen=True)
class GlobalETFUniverse:
    """Global ETF universe with region awareness."""
    
    name: str
    tickers: tuple[str, ...]
    regions: Dict[str, tuple[str, ...]]  # region -> tickers
    metadata: Dict[str, GlobalETFMetadata]
    description: str | None = None
    allow_cross_region: bool = False
    
    @property
    def n_tickers(self) -> int:
        """Total number of tickers."""
        return len(self.tickers)
    
    @property
    def n_regions(self) -> int:
        """Number of regions."""
        return len(self.regions)
    
    def tickers_by_region(self, region: str) -> tuple[str, ...]:
        """Get tickers for a specific region."""
        return self.regions.get(region, ())
    
    def region_for_ticker(self, ticker: str) -> str | None:
        """Get region for a ticker."""
        meta = self.metadata.get(ticker)
        return meta.region if meta else None
    
    def pairs_within_region(self, region: str) -> list[tuple[str, str]]:
        """Enumerate all pairs within a region."""
        tickers = list(self.tickers_by_region(region))
        pairs = []
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                pairs.append((tickers[i], tickers[j]))
        return pairs
    
    def all_pairs(self, cross_region: bool | None = None) -> list[tuple[str, str]]:
        """Enumerate all valid pairs based on cross-region setting.
        
        Args:
            cross_region: Override allow_cross_region setting
            
        Returns:
            List of (ticker1, ticker2) tuples
        """
        allow_cross = cross_region if cross_region is not None else self.allow_cross_region
        
        if allow_cross:
            # All pairs globally
            tickers = list(self.tickers)
            pairs = []
            for i in range(len(tickers)):
                for j in range(i + 1, len(tickers)):
                    pairs.append((tickers[i], tickers[j]))
            return pairs
        else:
            # Only within-region pairs
            all_pairs = []
            for region in self.regions:
                all_pairs.extend(self.pairs_within_region(region))
            return all_pairs
    
    def pair_count_estimate(self, cross_region: bool | None = None) -> int:
        """Estimate number of pairs without enumeration."""
        allow_cross = cross_region if cross_region is not None else self.allow_cross_region
        
        if allow_cross:
            n = self.n_tickers
            return n * (n - 1) // 2
        else:
            total = 0
            for region, tickers in self.regions.items():
                n = len(tickers)
                total += n * (n - 1) // 2
            return total
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"GlobalETFUniverse: {self.name}",
            f"  Total tickers: {self.n_tickers}",
            f"  Regions: {self.n_regions}",
            f"  Cross-region pairs: {self.allow_cross_region}",
            f"  Estimated pairs: {self.pair_count_estimate():,}",
            "",
            "  Tickers per region:"
        ]
        for region, tickers in sorted(self.regions.items()):
            n_pairs = len(tickers) * (len(tickers) - 1) // 2
            lines.append(f"    {region}: {len(tickers)} tickers, {n_pairs:,} pairs")
        return "\n".join(lines)


def _normalize_tickers(tickers: Iterable[str]) -> list[str]:
    """Normalize ticker list: uppercase, strip, unique."""
    seen = set()
    result = []
    for t in tickers:
        ticker = str(t).strip().upper()
        if ticker and ticker not in seen:
            result.append(ticker)
            seen.add(ticker)
    return result


def _resolve_categories(
    category_names: Sequence[str],
    categories_cfg: Mapping[str, Any],
) -> tuple[list[str], Dict[str, str]]:
    """Resolve category names to tickers and build ticker->region mapping.
    
    Returns:
        (list of tickers, dict of ticker->region)
    """
    all_tickers = []
    ticker_regions = {}
    ticker_categories = {}
    
    for cat_name in category_names:
        cat = categories_cfg.get(cat_name)
        if not isinstance(cat, Mapping):
            logger.warning(f"Category '{cat_name}' not found, skipping")
            continue
        
        region = cat.get("region", "GLOBAL")
        etfs = cat.get("etfs", [])
        
        for ticker in _normalize_tickers(etfs):
            if ticker not in ticker_regions:
                all_tickers.append(ticker)
                ticker_regions[ticker] = region
                ticker_categories[ticker] = cat_name
    
    return all_tickers, ticker_regions, ticker_categories


def load_global_universe(
    config_path: str | Path,
    list_name: str | None = None,
) -> GlobalETFUniverse:
    """Load a global ETF universe from config file.
    
    Args:
        config_path: Path to global_data.yaml
        list_name: Optional list name (default: uses default_list)
        
    Returns:
        GlobalETFUniverse with region mappings
    """
    config = load_yaml_config(config_path)
    universe_cfg = config.get("universe", {})
    
    if not universe_cfg:
        raise ConfigError("Config missing 'universe' section")
    
    # Determine which list to load
    list_name = list_name or universe_cfg.get("default_list", "global_full")
    lists_cfg = universe_cfg.get("lists", {})
    
    if list_name not in lists_cfg:
        available = list(lists_cfg.keys())
        raise ConfigError(f"List '{list_name}' not found. Available: {available}")
    
    list_entry = lists_cfg[list_name]
    categories_cfg = universe_cfg.get("categories", {})
    
    # Resolve categories to tickers
    category_names = list_entry.get("categories", [])
    tickers, ticker_regions, ticker_categories = _resolve_categories(
        category_names, categories_cfg
    )
    
    # Add explicit tickers if any
    explicit_tickers = list_entry.get("etfs", [])
    for ticker in _normalize_tickers(explicit_tickers):
        if ticker not in ticker_regions:
            tickers.append(ticker)
            ticker_regions[ticker] = "GLOBAL"
            ticker_categories[ticker] = "explicit"
    
    if not tickers:
        raise ConfigError(f"List '{list_name}' resolved to zero tickers")
    
    # Build region -> tickers mapping
    regions: Dict[str, list[str]] = {}
    for ticker, region in ticker_regions.items():
        if region not in regions:
            regions[region] = []
        regions[region].append(ticker)
    
    # Convert to tuples
    regions_tuple = {r: tuple(t) for r, t in regions.items()}
    
    # Build metadata
    metadata = {}
    for ticker in tickers:
        region = ticker_regions.get(ticker, "GLOBAL")
        category = ticker_categories.get(ticker)
        metadata[ticker] = GlobalETFMetadata(
            ticker=ticker,
            name=ticker,  # Will be enriched from etf_metadata.yaml if available
            region=region,
            category=category,
            currency="USD",  # All converted to USD
        )
    
    allow_cross_region = list_entry.get("allow_cross_region", False)
    description = list_entry.get("description")
    
    universe = GlobalETFUniverse(
        name=list_name,
        tickers=tuple(tickers),
        regions=regions_tuple,
        metadata=metadata,
        description=description,
        allow_cross_region=allow_cross_region,
    )
    
    logger.info(f"Loaded global universe: {list_name}")
    logger.info(f"  {universe.n_tickers} tickers across {universe.n_regions} regions")
    logger.info(f"  Estimated {universe.pair_count_estimate():,} pairs")
    
    return universe


def get_ticker_regions(universe: GlobalETFUniverse) -> Dict[str, str]:
    """Extract ticker -> region mapping from universe."""
    return {t: universe.metadata[t].region for t in universe.tickers if t in universe.metadata}


def filter_universe_by_regions(
    universe: GlobalETFUniverse,
    include_regions: Sequence[str] | None = None,
    exclude_regions: Sequence[str] | None = None,
) -> GlobalETFUniverse:
    """Create a filtered universe with only specified regions.
    
    Args:
        universe: Source universe
        include_regions: If provided, only include these regions
        exclude_regions: If provided, exclude these regions
        
    Returns:
        New GlobalETFUniverse with filtered tickers
    """
    include_set = set(include_regions) if include_regions else None
    exclude_set = set(exclude_regions) if exclude_regions else set()
    
    new_tickers = []
    new_regions: Dict[str, list[str]] = {}
    new_metadata = {}
    
    for region, tickers in universe.regions.items():
        # Check inclusion/exclusion
        if include_set is not None and region not in include_set:
            continue
        if region in exclude_set:
            continue
        
        new_regions[region] = list(tickers)
        for ticker in tickers:
            new_tickers.append(ticker)
            if ticker in universe.metadata:
                new_metadata[ticker] = universe.metadata[ticker]
    
    return GlobalETFUniverse(
        name=f"{universe.name}_filtered",
        tickers=tuple(new_tickers),
        regions={r: tuple(t) for r, t in new_regions.items()},
        metadata=new_metadata,
        description=f"Filtered: {universe.description}",
        allow_cross_region=universe.allow_cross_region,
    )


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="Inspect global ETF universe")
    parser.add_argument("--config", type=Path, default=Path("configs/global_data.yaml"))
    parser.add_argument("--list", dest="list_name", type=str, default=None)
    parser.add_argument("--regions", nargs="+", help="Filter to specific regions")
    
    args = parser.parse_args()
    
    universe = load_global_universe(args.config, args.list_name)
    
    if args.regions:
        universe = filter_universe_by_regions(universe, include_regions=args.regions)
    
    print(universe.summary())
    print()
    
    # Show cross-region vs within-region pair counts
    within_pairs = universe.pair_count_estimate(cross_region=False)
    cross_pairs = universe.pair_count_estimate(cross_region=True)
    print(f"Within-region pairs: {within_pairs:,}")
    print(f"Cross-region pairs:  {cross_pairs:,}")
    print(f"Reduction: {(1 - within_pairs/cross_pairs)*100:.1f}% fewer pairs with region filtering")
