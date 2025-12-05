"""Pipeline for ranking ETF pairs by correlation and cointegration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from pairs_trading_etf.data.loader import PriceFrame, build_price_frame
from pairs_trading_etf.data.universe import ETFUniverse, load_configured_universe
from pairs_trading_etf.features.pair_generation import PairScore, score_pairs
from pairs_trading_etf.pipelines.rolling_pair_scan import run_rolling_cointegration

DEFAULT_CONFIG_PATH = Path("configs/data.yaml")
DEFAULT_PRICE_PATH = Path("data/raw/etf_prices.csv")
DEFAULT_OUTPUT_PATH = Path("results/pair_scan_candidates.csv")


def _parse_max_pairs(value: str) -> int | None:
    """Allow CLI users to pass ``none`` to keep every scored pair."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"none", "all"}:
        return None
    return int(value)


@dataclass(slots=True)
class PairScanConfig:
    """Typed configuration for the pair-scan pipeline.
    
    Note on lookback_days:
        Academic research (Gatev et al. 2006, Krauss 2017) recommends 
        1-2 year lookback periods for pairs trading. Using full history
        (11+ years) leads to regime-averaged statistics that may not
        reflect current market conditions. The 252-day default aligns
        with the industry standard 12-month formation period.
        
        Set lookback_days=None for research/analysis of historical behavior.
    """
    config_path: Path = DEFAULT_CONFIG_PATH
    price_path: Path = DEFAULT_PRICE_PATH
    output_path: Path | None = DEFAULT_OUTPUT_PATH
    excluded_output_path: Path | None = None
    list_name: str | None = None
    metadata_path: Path | None = None
    lookback_days: int | None = 252  # 1-year rolling window (production default)
    min_obs: int = 126
    min_corr: float = 0.60
    max_pairs: int | None = None
    return_method: str = "log"
    engle_granger_maxlag: int = 1
    allow_cross_sector: bool = True
    use_log: bool = True
    exclude_same_index: bool = True
    max_corr: float = 0.99
    keep_excluded_pairs: bool = False
    # Literature-based filtering thresholds
    pvalue_threshold: float = 0.10
    min_spread_range_pct: float = 8.0
    min_half_life: float = 15.0
    max_half_life: float = 120.0  # Tighter bound for production trading
    # Rolling consistency check (recommended for robust pair selection)
    require_rolling_consistency: bool = False  # Enable for production
    min_rolling_pct_significant: float = 0.70  # 70% of windows must be significant


def _load_universe(cfg: PairScanConfig) -> ETFUniverse:
    """Load the ETF universe (+metadata) defined in ``cfg``."""
    return load_configured_universe(
        cfg.config_path,
        list_name=cfg.list_name,
        metadata_path=cfg.metadata_path,
    )


def _load_prices(cfg: PairScanConfig, tickers: Sequence[str]) -> PriceFrame:
    """Fetch and optionally truncate the price frame used for scoring."""
    frame = build_price_frame(
        cfg.price_path,
        tickers=tickers,
        min_non_na=cfg.min_obs,
        return_method=cfg.return_method,
        allow_missing=True,
    )
    if cfg.lookback_days is not None and cfg.lookback_days > 0:
        frame = frame.slice_last(cfg.lookback_days)
    return frame


def pair_scores_to_frame(
    scores: Sequence[PairScore], 
    universe_name: str,
    excluded_reason: str | None = None
) -> pd.DataFrame:
    """Convert ``PairScore`` objects into a dataframe annotated with universe name."""
    rows = []
    for score in scores:
        record = {"universe": universe_name, **score.as_dict(), "excluded_reason": excluded_reason}
        rows.append(record)
    return pd.DataFrame(rows)


def _filter_sector_pairs(
    scores: Sequence[PairScore], universe: ETFUniverse, allow_cross_sector: bool
) -> list[PairScore]:
    """Optionally restrict scores to same-sector pairs when metadata allows."""
    if allow_cross_sector:
        return list(scores)

    metadata = universe.metadata or {}
    if not metadata:
        return list(scores)

    sector_lookup = {ticker.upper(): meta.sector for ticker, meta in metadata.items()}
    filtered: list[PairScore] = []
    for score in scores:
        sector_x = sector_lookup.get(score.leg_x)
        sector_y = sector_lookup.get(score.leg_y)
        if sector_x is None or sector_y is None:
            continue
        if sector_x == sector_y:
            filtered.append(score)
    return filtered


def _filter_same_index_pairs(
    scores: Sequence[PairScore], universe: ETFUniverse
) -> tuple[list[PairScore], list[tuple[PairScore, str]]]:
    """Remove pairs where both ETFs track the same underlying index.
    
    Returns:
        Tuple of (kept_scores, excluded_pairs_with_reason)
    """
    metadata = universe.metadata or {}
    if not metadata:
        return list(scores), []

    index_lookup = {
        ticker.upper(): meta.tracks_index 
        for ticker, meta in metadata.items() 
        if meta.tracks_index is not None
    }
    
    kept: list[PairScore] = []
    excluded: list[tuple[PairScore, str]] = []
    
    for score in scores:
        idx_x = index_lookup.get(score.leg_x)
        idx_y = index_lookup.get(score.leg_y)
        
        if idx_x is not None and idx_y is not None and idx_x == idx_y:
            excluded.append((score, f"same_index:{idx_x}"))
        else:
            kept.append(score)
    
    return kept, excluded


def _filter_high_correlation_pairs(
    scores: Sequence[PairScore], max_corr: float
) -> tuple[list[PairScore], list[tuple[PairScore, str]]]:
    """Remove pairs with correlation above threshold (likely duplicates).
    
    Returns:
        Tuple of (kept_scores, excluded_pairs_with_reason)
    """
    kept: list[PairScore] = []
    excluded: list[tuple[PairScore, str]] = []
    
    for score in scores:
        if score.correlation > max_corr:
            excluded.append((score, f"high_corr:{score.correlation:.4f}"))
        else:
            kept.append(score)
    
    return kept, excluded


def _filter_cointegration_metrics(
    scores: Sequence[PairScore],
    pvalue_threshold: float,
    min_half_life: float,
    max_half_life: float,
    min_spread_range_pct: float,
) -> tuple[list[PairScore], list[tuple[PairScore, str]]]:
    """Filter pairs by cointegration quality metrics.
    
    Applies literature-based filters:
    - p-value threshold for cointegration significance
    - Half-life bounds for practical mean-reversion speed
    - Spread range minimum for tradeable signal generation
    
    Returns:
        Tuple of (kept_scores, excluded_pairs_with_reason)
    """
    kept: list[PairScore] = []
    excluded: list[tuple[PairScore, str]] = []
    
    for score in scores:
        # Check p-value threshold
        if score.coint_pvalue is None or score.coint_pvalue > pvalue_threshold:
            pv = score.coint_pvalue if score.coint_pvalue is not None else "None"
            excluded.append((score, f"pvalue:{pv}"))
            continue
        
        # Check half-life bounds
        if score.half_life is None:
            excluded.append((score, "half_life:None"))
            continue
        if score.half_life < min_half_life:
            excluded.append((score, f"half_life_low:{score.half_life:.1f}"))
            continue
        if score.half_life > max_half_life:
            excluded.append((score, f"half_life_high:{score.half_life:.1f}"))
            continue
        
        # Check spread range (only if score has it computed)
        if score.spread_range_pct is not None:
            if score.spread_range_pct < min_spread_range_pct:
                excluded.append((score, f"spread_range_low:{score.spread_range_pct:.1f}"))
                continue
        
        kept.append(score)
    
    return kept, excluded


def _filter_rolling_consistency(
    scores: Sequence[PairScore],
    prices: pd.DataFrame,
    window_days: int,
    step_days: int,
    pvalue_threshold: float,
    min_pct_significant: float,
    max_half_life: float,
) -> tuple[list[PairScore], list[tuple[PairScore, str]]]:
    """Filter pairs by rolling window cointegration consistency.
    
    This is the KEY production filter - ensures pairs are consistently 
    cointegrated across multiple time windows, not just in aggregate.
    
    A pair passes if >= min_pct_significant of rolling windows show:
    - Significant cointegration (p-value < pvalue_threshold)
    - Tradeable half-life (< max_half_life)
    
    This catches regime breaks that aggregate testing misses.
    
    Args:
        scores: Pair scores that passed initial filters
        prices: Full price DataFrame (will be sliced by rolling_pair_scan)
        window_days: Rolling window size (e.g., 252 for 1 year)
        step_days: Step between windows (e.g., 63 for quarterly)
        pvalue_threshold: Max p-value for significance
        min_pct_significant: Min fraction of windows that must pass (e.g., 0.70)
        max_half_life: Max half-life for tradeable windows
        
    Returns:
        Tuple of (kept_scores, excluded_pairs_with_reason)
    """
    kept: list[PairScore] = []
    excluded: list[tuple[PairScore, str]] = []
    
    for score in scores:
        ticker_a, ticker_b = score.leg_x, score.leg_y
        
        # Get price series for the pair
        if ticker_a not in prices.columns or ticker_b not in prices.columns:
            excluded.append((score, "rolling_error:ticker_not_found"))
            continue
            
        price_x = prices[ticker_a]
        price_y = prices[ticker_b]
        
        # Run rolling cointegration analysis
        try:
            rolling_result = run_rolling_cointegration(
                price_x=price_x,
                price_y=price_y,
                formation_window=window_days,
                step_size=step_days,
                use_log=True,  # Consistent with main pipeline
            )
        except Exception as e:
            # If rolling analysis fails, exclude the pair
            excluded.append((score, f"rolling_error:{str(e)[:30]}"))
            continue
        
        if rolling_result is None:
            excluded.append((score, "rolling_no_windows"))
            continue
        
        # Count windows that pass BOTH cointegration AND half-life filters
        pvalues = rolling_result.pvalues.dropna()
        half_lives = rolling_result.half_lives.dropna()
        
        n_total = len(pvalues)
        if n_total == 0:
            excluded.append((score, "rolling_no_windows"))
            continue
            
        # Count windows that meet both criteria
        significant_mask = pvalues < pvalue_threshold
        tradeable_hl_mask = (half_lives < max_half_life) & (half_lives > 0)
        
        # Align indices and count intersection
        common_idx = pvalues.index.intersection(half_lives.index)
        n_significant = (
            significant_mask.loc[common_idx] & tradeable_hl_mask.loc[common_idx]
        ).sum()
        
        pct_significant = n_significant / n_total if n_total > 0 else 0.0
        
        if pct_significant >= min_pct_significant:
            kept.append(score)
        else:
            excluded.append((
                score,
                f"rolling_consistency:{pct_significant:.0%}<{min_pct_significant:.0%}"
            ))
    
    return kept, excluded


def run_pair_scan(cfg: PairScanConfig) -> pd.DataFrame:
    """Execute the full pair scoring pipeline and return a dataframe of candidates."""
    universe = _load_universe(cfg)
    frame = _load_prices(cfg, tickers=universe.tickers)

    scores = score_pairs(
        frame.prices,
        min_obs=cfg.min_obs,
        min_corr=cfg.min_corr,
        lookback=None,
        max_pairs=cfg.max_pairs,
        run_cointegration=True,
        engle_granger_kwargs={"maxlag": cfg.engle_granger_maxlag, "use_log": cfg.use_log},
    )

    scores = _filter_sector_pairs(scores, universe, allow_cross_sector=cfg.allow_cross_sector)

    # Apply duplicate filters and track excluded pairs
    all_excluded: list[tuple[PairScore, str]] = []
    
    # Filter by same index tracking
    if cfg.exclude_same_index:
        scores, excluded_index = _filter_same_index_pairs(scores, universe)
        all_excluded.extend(excluded_index)
    
    # Filter by high correlation (likely same holdings)
    if cfg.max_corr < 1.0:
        scores, excluded_corr = _filter_high_correlation_pairs(scores, cfg.max_corr)
        all_excluded.extend(excluded_corr)
    
    # Apply literature-based cointegration quality filters
    scores, excluded_coint = _filter_cointegration_metrics(
        scores,
        pvalue_threshold=cfg.pvalue_threshold,
        min_half_life=cfg.min_half_life,
        max_half_life=cfg.max_half_life,
        min_spread_range_pct=cfg.min_spread_range_pct,
    )
    all_excluded.extend(excluded_coint)

    # Apply rolling consistency filter (PRODUCTION CRITICAL)
    # This catches regime breaks that aggregate testing misses
    if cfg.require_rolling_consistency:
        scores, excluded_rolling = _filter_rolling_consistency(
            scores,
            prices=frame.prices,
            window_days=cfg.lookback_days or 252,  # Use config window or default
            step_days=63,  # Quarterly steps for efficiency
            pvalue_threshold=cfg.pvalue_threshold,
            min_pct_significant=cfg.min_rolling_pct_significant,
            max_half_life=cfg.max_half_life,
        )
        all_excluded.extend(excluded_rolling)

    # Build result DataFrame with kept pairs (excluded_reason = None)
    result_df = pair_scores_to_frame(scores, universe_name=universe.name, excluded_reason=None)

    excluded_df = pd.DataFrame()
    if all_excluded:
        excluded_rows = []
        for score, reason in all_excluded:
            record = {"universe": universe.name, **score.as_dict(), "excluded_reason": reason}
            excluded_rows.append(record)
        excluded_df = pd.DataFrame(excluded_rows)

        if cfg.keep_excluded_pairs:
            excluded_path = cfg.excluded_output_path
            if excluded_path is None and cfg.output_path is not None:
                excluded_path = cfg.output_path.with_name(
                    f"{cfg.output_path.stem}_excluded{cfg.output_path.suffix}"
                )
            if excluded_path is not None:
                excluded_path.parent.mkdir(parents=True, exist_ok=True)
                excluded_df.to_csv(excluded_path, index=False)

    if cfg.output_path is not None:
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(cfg.output_path, index=False)

    return result_df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Scan ETF universes for correlated cointegrated pairs")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to YAML config")
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICE_PATH, help="Path to price CSV")
    parser.add_argument("--list", dest="list_name", default=None, help="Universe list identifier override")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional metadata YAML override")
    parser.add_argument("--lookback", type=int, default=None, help="Rows of history used for scoring (None=full history)")
    parser.add_argument("--min-obs", type=int, default=126, help="Minimum overlapping observations per pair")
    parser.add_argument("--min-corr", type=float, default=0.60, help="Minimum return correlation threshold")
    parser.add_argument(
        "--max-pairs",
        type=_parse_max_pairs,
        default=None,
        help="Maximum number of pairs to keep (use 'none' to keep all)",
    )
    parser.add_argument("--maxlag", type=int, default=1, help="Max lag for Engle-Granger test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to persist CSV results")
    parser.add_argument(
        "--same-sector-only",
        action="store_true",
        help="Restrict output to pairs where both legs share the same metadata sector",
    )
    parser.add_argument(
        "--no-exclude-same-index",
        action="store_true",
        help="Disable filtering of pairs that track the same underlying index",
    )
    parser.add_argument(
        "--max-corr",
        type=float,
        default=0.99,
        help="Maximum correlation threshold; pairs above this are flagged as duplicates (default: 0.99)",
    )
    parser.add_argument(
        "--keep-excluded",
        action="store_true",
        help="Persist excluded pairs to a companion CSV instead of discarding them",
    )
    parser.add_argument(
        "--excluded-output",
        type=Path,
        default=None,
        help="Optional override path for the excluded pairs CSV",
    )
    parser.add_argument(
        "--require-rolling-consistency",
        action="store_true",
        help="Enable rolling window consistency check (PRODUCTION recommended)",
    )
    parser.add_argument(
        "--min-rolling-pct",
        type=float,
        default=0.70,
        help="Minimum pct of rolling windows that must be significant (default: 0.70)",
    )

    args = parser.parse_args()

    cfg = PairScanConfig(
        config_path=args.config,
        price_path=args.prices,
        output_path=args.output,
        list_name=args.list_name,
        metadata_path=args.metadata,
        lookback_days=args.lookback,
        min_obs=args.min_obs,
        min_corr=args.min_corr,
        max_pairs=args.max_pairs,
        engle_granger_maxlag=args.maxlag,
        allow_cross_sector=not args.same_sector_only,
        exclude_same_index=not args.no_exclude_same_index,
        max_corr=args.max_corr,
        keep_excluded_pairs=args.keep_excluded,
        excluded_output_path=args.excluded_output,
        require_rolling_consistency=args.require_rolling_consistency,
        min_rolling_pct_significant=args.min_rolling_pct,
    )

    df = run_pair_scan(cfg)
    if df.empty:
        print("No qualifying pairs found")
    else:
        print(df)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    main()
