"""
Sector definitions and utilities for ETF pairs trading.

This module contains sector groupings for ETFs and helper functions
to determine sector membership and same-sector pair validation.
"""

from typing import Dict, Set, Tuple

# =============================================================================
# SECTOR DEFINITIONS
# =============================================================================

# Same-sector pairs that have fundamental links
SECTOR_GROUPS: Dict[str, Set[str]] = {
    # US Equity - Broad Market
    'US_BROAD': {'SPY', 'VOO', 'IVV', 'VTI', 'IWB', 'RSP', 'OEF', 'DIA'},
    'US_GROWTH': {'QQQ', 'VUG', 'IWF', 'SPYG', 'SCHG', 'VGT', 'XLK', 'IYW'},
    'US_VALUE': {'VTV', 'IWD', 'SPYV', 'DVY', 'VYM', 'SCHV', 'SDY'},
    'US_SMALL': {'IWM', 'VB', 'IJR', 'VBK', 'VBR', 'SCHA'},
    'US_MID': {'IJH', 'MDY', 'VO', 'SCHM'},
    
    # US Equity - Sectors
    'TECH': {'XLK', 'VGT', 'IYW', 'SMH', 'SOXX', 'IGV'},
    'FINANCIALS': {'XLF', 'VFH', 'IYF', 'KRE', 'KBE', 'IAI'},
    'HEALTHCARE': {'XLV', 'VHT', 'IBB', 'XBI', 'IHI'},
    'INDUSTRIALS': {'XLI', 'VIS', 'IYT', 'ITA', 'XAR'},
    'CONSUMER_DISC': {'XLY', 'VCR', 'XRT', 'XHB', 'ITB'},
    'CONSUMER_STAPLES': {'XLP', 'VDC', 'IYK'},
    'ENERGY': {'XLE', 'VDE', 'OIH', 'XOP', 'IEO'},
    'MATERIALS': {'XLB', 'VAW', 'XME', 'GDX', 'GDXJ'},
    'UTILITIES': {'XLU', 'VPU', 'IDU'},
    'REITS': {'VNQ', 'IYR', 'XLRE', 'RWR'},
    
    # International - Developed
    'EUROPE': {'VGK', 'EZU', 'FEZ', 'EWU', 'EWG', 'EWQ', 'EWI', 'EWP', 'EWN', 'EWL'},
    'ASIA_DEV': {'EWJ', 'EWA', 'EWS', 'EWH', 'EWT', 'EWY', 'DXJ'},
    
    # International - Emerging
    'EMERGING': {'EEM', 'VWO', 'EWZ', 'FXI', 'GXC', 'EWW', 'EWY', 'IEMG', 'ILF', 'EPP'},
    
    # Fixed Income
    'BONDS_CORP': {'LQD', 'VCIT', 'IGIB', 'HYG', 'JNK'},
    'BONDS_GOV': {'TLT', 'IEF', 'SHY', 'IEI', 'BND', 'AGG', 'TIP', 'MUB'},
    
    # Alternatives
    'COMMODITIES': {'GLD', 'IAU', 'SLV', 'DBC', 'USO', 'UNG'},
}

# Sectors that historically perform poorly in pairs trading
DEFAULT_EXCLUDED_SECTORS: Tuple[str, ...] = (
    'EMERGING',      # High volatility, regime changes
    'BONDS_GOV',     # Dominated by interest rate moves
    'US_GROWTH',     # Tech-heavy, regime changes
    'INDUSTRIALS',   # Cyclical, macro-driven
    'HEALTHCARE',    # Regulatory/binary events
)

# Sectors that historically perform well in pairs trading
RECOMMENDED_SECTORS: Tuple[str, ...] = (
    'EUROPE',        # Strong cointegration
    'FINANCIALS',    # Similar business models
    'US_BROAD',      # Index tracking
    'ASIA_DEV',      # Regional correlation
    'CONSUMER_DISC', # Similar drivers
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_sector(ticker: str) -> str:
    """
    Get the sector for a given ticker.
    
    Parameters
    ----------
    ticker : str
        ETF ticker symbol
        
    Returns
    -------
    str
        Sector name, or 'OTHER' if not found in any sector group
    """
    for sector, tickers in SECTOR_GROUPS.items():
        if ticker in tickers:
            return sector
    return 'OTHER'


def are_same_sector(ticker_a: str, ticker_b: str) -> bool:
    """
    Check if two tickers belong to the same sector.
    
    Parameters
    ----------
    ticker_a : str
        First ETF ticker
    ticker_b : str
        Second ETF ticker
        
    Returns
    -------
    bool
        True if both tickers are in the same sector (and not 'OTHER')
    """
    sector_a = get_sector(ticker_a)
    sector_b = get_sector(ticker_b)
    return sector_a == sector_b and sector_a != 'OTHER'


def get_sector_tickers(sector: str) -> Set[str]:
    """
    Get all tickers in a given sector.
    
    Parameters
    ----------
    sector : str
        Sector name
        
    Returns
    -------
    Set[str]
        Set of ticker symbols in the sector
    """
    return SECTOR_GROUPS.get(sector, set())


def get_all_tickers() -> Set[str]:
    """
    Get all tickers across all sectors.
    
    Returns
    -------
    Set[str]
        Set of all ticker symbols
    """
    all_tickers = set()
    for tickers in SECTOR_GROUPS.values():
        all_tickers.update(tickers)
    return all_tickers


def filter_by_sectors(
    pairs: list,
    include_sectors: Tuple[str, ...] = None,
    exclude_sectors: Tuple[str, ...] = None,
) -> list:
    """
    Filter pairs by sector inclusion/exclusion.
    
    Parameters
    ----------
    pairs : list
        List of (ticker_a, ticker_b) tuples
    include_sectors : tuple, optional
        If provided, only include pairs where both tickers are in these sectors
    exclude_sectors : tuple, optional
        Exclude pairs where either ticker is in these sectors
        
    Returns
    -------
    list
        Filtered list of pairs
    """
    filtered = []
    for pair in pairs:
        ticker_a, ticker_b = pair
        sector_a = get_sector(ticker_a)
        sector_b = get_sector(ticker_b)
        
        # Check exclusion first
        if exclude_sectors:
            if sector_a in exclude_sectors or sector_b in exclude_sectors:
                continue
        
        # Check inclusion
        if include_sectors:
            if sector_a not in include_sectors or sector_b not in include_sectors:
                continue
        
        filtered.append(pair)
    
    return filtered
