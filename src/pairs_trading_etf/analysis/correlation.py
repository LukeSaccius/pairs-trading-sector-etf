"""
Correlation analysis module.

This module handles the computation of return correlations and the identification
of highly correlated pairs. It is purely analytical and contains no visualization code.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


def compute_return_correlations(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the correlation matrix of daily returns.

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame of daily returns (tickers as columns).

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    return returns_df.corr()


def find_high_corr_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.8,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Identify pairs with correlation above a specified threshold.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix of returns.
    threshold : float, optional
        Minimum absolute correlation to consider a pair "highly correlated", by default 0.8.
    metadata : pd.DataFrame | None, optional
        DataFrame containing 'ticker' and 'sector' columns. If provided,
        adds sector information and 'pair_bucket' (Same Sector vs Different Sector).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['leg_x', 'leg_y', 'correlation', 'pair_bucket', ...].
    """
    tickers = corr_matrix.columns
    pairs_data = []

    # Iterate over the upper triangle of the correlation matrix
    for t1, t2 in itertools.combinations(tickers, 2):
        corr_val = corr_matrix.loc[t1, t2]
        if abs(corr_val) >= threshold:
            pairs_data.append({
                "leg_x": t1,
                "leg_y": t2,
                "correlation": corr_val,
            })

    pairs_df = pd.DataFrame(pairs_data)
    if pairs_df.empty:
        return pd.DataFrame(columns=["leg_x", "leg_y", "correlation", "pair_bucket"])

    # Sort by correlation descending
    pairs_df = pairs_df.sort_values("correlation", ascending=False).reset_index(drop=True)

    # If metadata is provided, enrich with sector info
    if metadata is not None and not metadata.empty:
        pairs_df = attach_sector_labels(pairs_df, metadata)
    else:
        pairs_df["pair_bucket"] = "Unknown"

    return pairs_df


@dataclass(slots=True)
class SectorMetadata:
    ticker: str
    sector: str | None


def attach_sector_labels(
    pairs_df: pd.DataFrame,
    metadata: Mapping[str, SectorMetadata] | Mapping[str, object] | pd.DataFrame,
) -> pd.DataFrame:
    """Attach sector_x, sector_y, and pair_bucket columns based on metadata."""

    if pairs_df.empty:
        return pairs_df.copy()

    if isinstance(metadata, pd.DataFrame):
        sector_map = metadata["sector"].to_dict()
    else:
        sector_map = {}
        for ticker, meta in metadata.items():
            sector = getattr(meta, "sector", None)
            if sector is None and isinstance(meta, Mapping):
                sector = meta.get("sector")
            sector_map[str(ticker).upper()] = sector

    enriched = pairs_df.copy()
    enriched["leg_x"] = enriched["leg_x"].str.upper()
    enriched["leg_y"] = enriched["leg_y"].str.upper()
    enriched["sector_x"] = enriched["leg_x"].map(sector_map)
    enriched["sector_y"] = enriched["leg_y"].map(sector_map)
    enriched["pair_bucket"] = np.where(
        enriched["sector_x"] == enriched["sector_y"], "Same Sector", "Cross Sector"
    )
    return enriched


def summarise_pairs_by_bucket(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Return counts and correlation stats per pair bucket."""

    if pairs_df.empty:
        return pd.DataFrame(columns=["n_pairs", "mean", "min", "max"])

    summary = (
        pairs_df.groupby("pair_bucket")["correlation"]
        .agg([("n_pairs", "count"), ("mean", "mean"), ("min", "min"), ("max", "max")])
        .round(3)
    )
    return summary
