"""Utility helpers (config, logging, sectors, etc.)."""

from pairs_trading_etf.utils.validation import (
    WalkForwardWindow,
    walk_forward_split,
    simple_train_test_split,
    count_walk_forward_windows,
)

from pairs_trading_etf.utils.sectors import (
    SECTOR_GROUPS,
    DEFAULT_EXCLUDED_SECTORS,
    RECOMMENDED_SECTORS,
    get_sector,
    are_same_sector,
    get_sector_tickers,
    get_all_tickers,
    filter_by_sectors,
)

__all__ = [
    # Validation
    "WalkForwardWindow",
    "walk_forward_split",
    "simple_train_test_split",
    "count_walk_forward_windows",
    # Sectors
    "SECTOR_GROUPS",
    "DEFAULT_EXCLUDED_SECTORS",
    "RECOMMENDED_SECTORS",
    "get_sector",
    "are_same_sector",
    "get_sector_tickers",
    "get_all_tickers",
    "filter_by_sectors",
]
