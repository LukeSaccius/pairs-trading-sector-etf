"""Pipeline for ranking ETF pairs by correlation and cointegration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from pairs_trading_etf.data.loader import PriceFrame, build_price_frame
from pairs_trading_etf.data.universe import ETFUniverse, load_configured_universe
from pairs_trading_etf.features.pair_generation import PairScore, score_pairs

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
    """Typed configuration for the pair-scan pipeline."""
    config_path: Path = DEFAULT_CONFIG_PATH
    price_path: Path = DEFAULT_PRICE_PATH
    output_path: Path | None = DEFAULT_OUTPUT_PATH
    list_name: str | None = None
    metadata_path: Path | None = None
    lookback_days: int | None = 252
    min_obs: int = 126
    min_corr: float = 0.80
    max_pairs: int | None = None
    return_method: str = "log"
    engle_granger_maxlag: int = 1
    allow_cross_sector: bool = True
    use_log: bool = True
    exclude_same_index: bool = True
    max_corr: float = 0.95


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

    # Build result DataFrame with kept pairs (excluded_reason = None)
    result_df = pair_scores_to_frame(scores, universe_name=universe.name, excluded_reason=None)
    
    # Add excluded pairs with their reasons
    if all_excluded:
        excluded_rows = []
        for score, reason in all_excluded:
            record = {"universe": universe.name, **score.as_dict(), "excluded_reason": reason}
            excluded_rows.append(record)
        excluded_df = pd.DataFrame(excluded_rows)
        result_df = pd.concat([result_df, excluded_df], ignore_index=True)
        
        # Re-sort: non-excluded first (by pvalue), then excluded
        result_df["_sort_key"] = result_df["excluded_reason"].notna().astype(int)
        result_df = result_df.sort_values(
            ["_sort_key", "coint_pvalue", "correlation"], 
            ascending=[True, True, False]
        ).drop(columns=["_sort_key"]).reset_index(drop=True)

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
    parser.add_argument("--lookback", type=int, default=252, help="Rows of history used for scoring")
    parser.add_argument("--min-obs", type=int, default=126, help="Minimum overlapping observations per pair")
    parser.add_argument("--min-corr", type=float, default=0.80, help="Minimum return correlation threshold")
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
        default=0.95,
        help="Maximum correlation threshold; pairs above this are flagged as duplicates (default: 0.95)",
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
    )

    df = run_pair_scan(cfg)
    if df.empty:
        print("No qualifying pairs found")
    else:
        print(df)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    main()
