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


@dataclass(slots=True)
class PairScanConfig:
    config_path: Path = DEFAULT_CONFIG_PATH
    price_path: Path = DEFAULT_PRICE_PATH
    output_path: Path | None = DEFAULT_OUTPUT_PATH
    list_name: str | None = None
    metadata_path: Path | None = None
    lookback_days: int | None = 252
    min_obs: int = 126
    min_corr: float = 0.85
    max_pairs: int | None = 10
    return_method: str = "log"
    engle_granger_maxlag: int = 1
    allow_cross_sector: bool = True


def _load_universe(cfg: PairScanConfig) -> ETFUniverse:
    return load_configured_universe(
        cfg.config_path,
        list_name=cfg.list_name,
        metadata_path=cfg.metadata_path,
    )


def _load_prices(cfg: PairScanConfig, tickers: Sequence[str]) -> PriceFrame:
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


def pair_scores_to_frame(scores: Sequence[PairScore], universe_name: str) -> pd.DataFrame:
    rows = []
    for score in scores:
        record = {"universe": universe_name, **score.as_dict()}
        rows.append(record)
    return pd.DataFrame(rows)


def _filter_sector_pairs(
    scores: Sequence[PairScore], universe: ETFUniverse, allow_cross_sector: bool
) -> list[PairScore]:
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


def run_pair_scan(cfg: PairScanConfig) -> pd.DataFrame:
    universe = _load_universe(cfg)
    frame = _load_prices(cfg, tickers=universe.tickers)

    scores = score_pairs(
        frame.prices,
        min_obs=cfg.min_obs,
        min_corr=cfg.min_corr,
        lookback=None,
        max_pairs=cfg.max_pairs,
        run_cointegration=True,
        engle_granger_kwargs={"maxlag": cfg.engle_granger_maxlag},
    )

    scores = _filter_sector_pairs(scores, universe, allow_cross_sector=cfg.allow_cross_sector)

    result_df = pair_scores_to_frame(scores, universe_name=universe.name)

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
    parser.add_argument("--min-corr", type=float, default=0.85, help="Minimum return correlation threshold")
    parser.add_argument("--max-pairs", type=int, default=10, help="Maximum number of pairs to keep")
    parser.add_argument("--maxlag", type=int, default=1, help="Max lag for Engle-Granger test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to persist CSV results")
    parser.add_argument(
        "--same-sector-only",
        action="store_true",
        help="Restrict output to pairs where both legs share the same metadata sector",
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
    )

    df = run_pair_scan(cfg)
    if df.empty:
        print("No qualifying pairs found")
    else:
        print(df)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    main()
