"""Rolling window pair scanning pipeline.

Implements time-varying cointegration analysis using rolling windows,
addressing regime changes and parameter instability in pairs trading.

Key features:
- Rolling cointegration tests with configurable window sizes
- Walk-forward validation structure
- Regime-aware filtering

References:
- Krauss, C. (2017). "Statistical Arbitrage Pairs Trading Strategies"
- Clegg, M., Krauss, C. (2018). "Pairs trading with partial cointegration"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from pairs_trading_etf.cointegration.engle_granger import run_engle_granger
from pairs_trading_etf.ou_model.estimation import estimate_ou_parameters
from pairs_trading_etf.data.loader import PriceFrame, build_price_frame
from pairs_trading_etf.data.universe import ETFUniverse, load_configured_universe
from pairs_trading_etf.features.pair_generation import enumerate_pairs

logger = logging.getLogger(__name__)


@dataclass
class RollingPairResult:
    """Results from rolling window cointegration analysis for a single pair."""
    
    leg_x: str
    leg_y: str
    
    # Time-varying statistics
    pvalues: pd.Series          # Cointegration p-values over time
    hedge_ratios: pd.Series     # Hedge ratios over time
    half_lives: pd.Series       # Half-lives over time
    spreads: pd.DataFrame       # Spread series for each window
    
    # Summary statistics
    pvalue_mean: float
    pvalue_std: float
    pvalue_pct_significant: float  # % of windows with p < threshold
    half_life_mean: float
    half_life_std: float
    hedge_ratio_stability: float   # CV of hedge ratio (lower = more stable)
    
    # Latest estimates
    latest_pvalue: float
    latest_hedge_ratio: float
    latest_half_life: float
    
    def is_consistently_cointegrated(
        self,
        pvalue_threshold: float = 0.10,
        min_significant_pct: float = 0.70,
    ) -> bool:
        """Check if pair shows consistent cointegration across windows."""
        return (
            self.pvalue_pct_significant >= min_significant_pct
            and self.latest_pvalue < pvalue_threshold
        )


@dataclass
class RollingScanConfig:
    """Configuration for rolling pair scan pipeline."""
    
    # Data paths
    config_path: Path = Path("configs/data.yaml")
    price_path: Path = Path("data/raw/etf_prices.csv")
    output_path: Path = Path("results/rolling_scan_results.csv")
    
    # Rolling window parameters
    formation_window: int = 252      # Window for cointegration estimation (1 year)
    trading_window: int = 126        # Out-of-sample trading period (6 months)
    step_size: int = 21              # Step between windows (1 month)
    min_windows: int = 3             # Minimum windows for stability assessment
    
    # Cointegration thresholds
    pvalue_threshold: float = 0.10
    min_half_life: float = 15
    max_half_life: float = 120
    min_significant_pct: float = 0.70  # 70% of windows must be significant
    
    # Correlation filters
    min_corr: float = 0.60
    max_corr: float = 0.99
    
    # Other settings
    use_log: bool = True
    exclude_same_index: bool = True
    max_pairs: int | None = None


@dataclass
class RollingScanResults:
    """Container for complete rolling scan results."""
    
    pair_results: list[RollingPairResult]
    scan_dates: list[pd.Timestamp]
    config: RollingScanConfig
    
    def to_summary_dataframe(self) -> pd.DataFrame:
        """Convert to summary DataFrame of all pairs."""
        rows = []
        for pr in self.pair_results:
            rows.append({
                "leg_x": pr.leg_x,
                "leg_y": pr.leg_y,
                "pvalue_mean": pr.pvalue_mean,
                "pvalue_std": pr.pvalue_std,
                "pvalue_pct_significant": pr.pvalue_pct_significant,
                "half_life_mean": pr.half_life_mean,
                "half_life_std": pr.half_life_std,
                "hedge_ratio_stability": pr.hedge_ratio_stability,
                "latest_pvalue": pr.latest_pvalue,
                "latest_hedge_ratio": pr.latest_hedge_ratio,
                "latest_half_life": pr.latest_half_life,
                "is_tradeable": pr.is_consistently_cointegrated(
                    self.config.pvalue_threshold,
                    self.config.min_significant_pct,
                ),
            })
        return pd.DataFrame(rows)
    
    def get_tradeable_pairs(self) -> list[RollingPairResult]:
        """Filter to consistently cointegrated pairs."""
        return [
            pr for pr in self.pair_results
            if pr.is_consistently_cointegrated(
                self.config.pvalue_threshold,
                self.config.min_significant_pct,
            )
            and self.config.min_half_life <= pr.latest_half_life <= self.config.max_half_life
        ]


def run_rolling_cointegration(
    price_x: pd.Series,
    price_y: pd.Series,
    formation_window: int = 252,
    step_size: int = 21,
    min_observations: int = 126,
    use_log: bool = True,
) -> RollingPairResult | None:
    """Run cointegration tests over rolling windows for a single pair.
    
    Parameters
    ----------
    price_x : pd.Series
        Price series for first leg.
    price_y : pd.Series  
        Price series for second leg.
    formation_window : int
        Size of each estimation window.
    step_size : int
        Step between windows.
    min_observations : int
        Minimum observations required per window.
    use_log : bool
        Whether to use log prices.
        
    Returns
    -------
    RollingPairResult | None
        Results container or None if insufficient data.
    """
    # Align prices
    df = pd.concat([price_x, price_y], axis=1, join="inner").dropna()
    if df.shape[0] < formation_window + min_observations:
        logger.debug(f"Insufficient data for rolling analysis: {df.shape[0]} obs")
        return None
    
    px = df.iloc[:, 0]
    py = df.iloc[:, 1]
    leg_x = price_x.name or "X"
    leg_y = price_y.name or "Y"
    
    # Rolling estimation
    pvalues = []
    hedge_ratios = []
    half_lives = []
    dates = []
    spreads_dict = {}
    
    start_idx = formation_window
    
    while start_idx <= len(df):
        window_end = start_idx
        window_start = start_idx - formation_window
        
        px_window = px.iloc[window_start:window_end]
        py_window = py.iloc[window_start:window_end]
        window_date = px_window.index[-1]
        
        try:
            result = run_engle_granger(
                px_window, py_window,
                use_log=use_log,
            )
            
            pvalues.append(result.pvalue)
            hedge_ratios.append(result.hedge_ratio)
            half_lives.append(result.half_life if result.half_life else np.nan)
            dates.append(window_date)
            
            # Store spread for this window
            if use_log:
                log_px = np.log(px_window)
                log_py = np.log(py_window)
                spread = log_px - result.hedge_ratio * log_py
            else:
                spread = px_window - result.hedge_ratio * py_window
            spreads_dict[window_date] = spread
            
        except Exception as e:
            logger.debug(f"Error in window ending {window_date}: {e}")
            pvalues.append(np.nan)
            hedge_ratios.append(np.nan)
            half_lives.append(np.nan)
            dates.append(window_date)
        
        start_idx += step_size
    
    if len(dates) < 3:
        return None
    
    # Build series
    pv_series = pd.Series(pvalues, index=dates, name="pvalue")
    hr_series = pd.Series(hedge_ratios, index=dates, name="hedge_ratio")
    hl_series = pd.Series(half_lives, index=dates, name="half_life")
    
    # Calculate summary statistics
    valid_pv = pv_series.dropna()
    valid_hr = hr_series.dropna()
    valid_hl = hl_series.dropna()
    
    pvalue_mean = float(valid_pv.mean()) if len(valid_pv) > 0 else np.nan
    pvalue_std = float(valid_pv.std()) if len(valid_pv) > 1 else np.nan
    pvalue_pct_sig = float((valid_pv < 0.10).mean()) if len(valid_pv) > 0 else 0.0
    
    hl_mean = float(valid_hl.mean()) if len(valid_hl) > 0 else np.nan
    hl_std = float(valid_hl.std()) if len(valid_hl) > 1 else np.nan
    
    # Hedge ratio stability (coefficient of variation)
    hr_cv = float(valid_hr.std() / abs(valid_hr.mean())) if len(valid_hr) > 1 and valid_hr.mean() != 0 else np.nan
    
    return RollingPairResult(
        leg_x=str(leg_x),
        leg_y=str(leg_y),
        pvalues=pv_series,
        hedge_ratios=hr_series,
        half_lives=hl_series,
        spreads=pd.DataFrame(spreads_dict),
        pvalue_mean=pvalue_mean,
        pvalue_std=pvalue_std,
        pvalue_pct_significant=pvalue_pct_sig,
        half_life_mean=hl_mean,
        half_life_std=hl_std,
        hedge_ratio_stability=hr_cv,
        latest_pvalue=float(pv_series.iloc[-1]) if not pv_series.empty else np.nan,
        latest_hedge_ratio=float(hr_series.iloc[-1]) if not hr_series.empty else np.nan,
        latest_half_life=float(hl_series.iloc[-1]) if not hl_series.empty else np.nan,
    )


def run_rolling_pair_scan(
    config: RollingScanConfig | None = None,
    prices_df: pd.DataFrame | None = None,
    candidate_pairs: Sequence[tuple[str, str]] | None = None,
) -> RollingScanResults:
    """Execute rolling window pair scan on ETF universe.
    
    Parameters
    ----------
    config : RollingScanConfig | None
        Configuration object. If None, uses defaults.
    prices_df : pd.DataFrame | None
        Pre-loaded price DataFrame. If None, loads from config.price_path.
    candidate_pairs : Sequence[tuple[str, str]] | None
        Pre-filtered list of pairs to analyze. If None, generates all pairs.
        
    Returns
    -------
    RollingScanResults
        Container with all pair analysis results.
    """
    if config is None:
        config = RollingScanConfig()
    
    # Load prices
    if prices_df is None:
        logger.info(f"Loading prices from {config.price_path}")
        prices_df = pd.read_csv(config.price_path, index_col=0, parse_dates=True)
    
    # Generate candidate pairs if not provided
    if candidate_pairs is None:
        logger.info("Generating candidate pairs from universe")
        universe = load_configured_universe(config.config_path)
        tickers = [t for t in universe.tickers if t in prices_df.columns]
        candidate_pairs = list(enumerate_pairs(tickers))
        logger.info(f"Generated {len(candidate_pairs)} candidate pairs from {len(tickers)} tickers")
    
    # Run rolling analysis on each pair
    pair_results = []
    total_pairs = len(candidate_pairs)
    
    for i, (leg_x, leg_y) in enumerate(candidate_pairs):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing pair {i + 1}/{total_pairs}")
        
        if leg_x not in prices_df.columns or leg_y not in prices_df.columns:
            continue
        
        # Quick correlation filter
        px = prices_df[leg_x].dropna()
        py = prices_df[leg_y].dropna()
        df = pd.concat([px, py], axis=1, join="inner").dropna()
        
        if df.shape[0] < config.formation_window:
            continue
        
        corr = df.iloc[:, 0].corr(df.iloc[:, 1])
        if corr < config.min_corr or corr > config.max_corr:
            continue
        
        # Run rolling cointegration
        result = run_rolling_cointegration(
            prices_df[leg_x],
            prices_df[leg_y],
            formation_window=config.formation_window,
            step_size=config.step_size,
            use_log=config.use_log,
        )
        
        if result is not None:
            pair_results.append(result)
    
    logger.info(f"Completed rolling analysis on {len(pair_results)} pairs")
    
    # Get scan dates from first valid result
    scan_dates = []
    if pair_results and len(pair_results[0].pvalues) > 0:
        scan_dates = list(pair_results[0].pvalues.index)
    
    return RollingScanResults(
        pair_results=pair_results,
        scan_dates=scan_dates,
        config=config,
    )


def get_current_tradeable_pairs(
    results: RollingScanResults,
    as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Extract currently tradeable pairs from rolling scan results.
    
    Parameters
    ----------
    results : RollingScanResults
        Output from run_rolling_pair_scan.
    as_of_date : pd.Timestamp | None
        Date for filtering. If None, uses latest available.
        
    Returns
    -------
    pd.DataFrame
        DataFrame of tradeable pairs with current statistics.
    """
    tradeable = results.get_tradeable_pairs()
    
    rows = []
    for pr in tradeable:
        # Get values as of specified date
        if as_of_date is not None:
            # Find nearest date <= as_of_date
            valid_dates = pr.pvalues.index[pr.pvalues.index <= as_of_date]
            if len(valid_dates) == 0:
                continue
            date_idx = valid_dates[-1]
            pvalue = pr.pvalues.loc[date_idx]
            hedge_ratio = pr.hedge_ratios.loc[date_idx]
            half_life = pr.half_lives.loc[date_idx]
        else:
            pvalue = pr.latest_pvalue
            hedge_ratio = pr.latest_hedge_ratio
            half_life = pr.latest_half_life
        
        rows.append({
            "leg_x": pr.leg_x,
            "leg_y": pr.leg_y,
            "pvalue": pvalue,
            "hedge_ratio": hedge_ratio,
            "half_life": half_life,
            "pvalue_stability": pr.pvalue_std,
            "hedge_ratio_cv": pr.hedge_ratio_stability,
            "pct_significant": pr.pvalue_pct_significant,
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("half_life")
    
    return df


def compare_windows(
    result: RollingPairResult,
    window_dates: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    """Compare cointegration metrics across specific time windows.
    
    Useful for identifying regime changes and parameter drift.
    """
    rows = []
    for date in window_dates:
        if date not in result.pvalues.index:
            continue
        rows.append({
            "date": date,
            "pvalue": result.pvalues.loc[date],
            "hedge_ratio": result.hedge_ratios.loc[date],
            "half_life": result.half_lives.loc[date],
        })
    return pd.DataFrame(rows)
