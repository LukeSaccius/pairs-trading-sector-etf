"""
ETF price data fetching utilities using yfinance
"""

import yfinance as yf
import pandas as pd
from typing import List, Optional


def download_etf_prices(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Download historical price data for specified ETF tickers using yfinance.
    
    Parameters
    ----------
    tickers : List[str]
        List of ETF ticker symbols (e.g., ['XLF', 'XLK', 'XLE'])
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : Optional[str], default None
        End date in format 'YYYY-MM-DD'. If None, uses current date.
    interval : str, default '1d'
        Data interval ('1d', '1wk', '1mo', etc.)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and ticker symbols as columns,
        containing adjusted close prices.
    
    Examples
    --------
    >>> # Download daily prices for financial and technology sector ETFs
    >>> prices = download_etf_prices(['XLF', 'XLK'], '2020-01-01', '2023-12-31')
    >>> print(prices.head())
    """
    # Download data for all tickers
    # Using auto_adjust=True to get split/dividend-adjusted prices automatically
    # This is appropriate for research purposes to ensure price series are comparable
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False,
        auto_adjust=True
    )
    
    # Handle single ticker vs multiple tickers
    if len(tickers) == 1:
        # For single ticker, yfinance returns a simple DataFrame
        prices = data[['Close']].copy()
        prices.columns = tickers
    else:
        # For multiple tickers, extract Close prices
        prices = data['Close'].copy()
    
    # Ensure we have a DataFrame with proper structure
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    
    # Remove any NaN rows
    prices = prices.dropna()
    
    return prices
