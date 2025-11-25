"""Correlation and sector-analysis helpers for ETF universes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def compute_return_correlations(
    prices: pd.DataFrame,
    *,
    method: str = "simple",
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Return Pearson correlations of daily returns for the provided price frame."""

    if prices.empty:
        raise ValueError("Price frame is empty; cannot compute correlations")

    normalized = prices.sort_index()
    if method == "log":
        safe = normalized.where(normalized > 0)
        returns = np.log(safe).diff()
    elif method == "simple":
        returns = normalized.pct_change()
    else:
        raise ValueError("method must be 'simple' or 'log'")

    returns = returns.dropna(how="all")
    if returns.empty:
        raise ValueError("Return frame is empty after differencing")

    corr = returns.corr(min_periods=min_periods)
    return corr


def _mask_upper_triangle(corr: pd.DataFrame) -> np.ndarray:
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    return mask


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    *,
    lower_triangle: bool = True,
    mask_diagonal: bool = True,
    annot: bool = False,
    figsize: tuple[int, int] = (14, 10),
    output_path: Path | None = None,
    cmap: str = "RdBu_r",
    metadata: pd.DataFrame | None = None,
) -> plt.Figure:
    """Plot a report-ready correlation heatmap with optional sector ordering."""

    if corr.empty:
        raise ValueError("corr is empty; nothing to plot")

    corr_to_plot = corr.copy()
    sector_boundaries: list[int] = []

    if metadata is not None and not metadata.empty and "sector" in metadata.columns:
        # Re-order tickers by sector to make cross-sector blocks visually distinct.
        meta_df = metadata.copy()
        if "ticker" in meta_df.columns:
            meta_df = meta_df.set_index("ticker")
        meta_df = meta_df[~meta_df.index.duplicated(keep="first")]
        meta_df.index = meta_df.index.map(lambda x: str(x).upper())

        ticker_info = pd.DataFrame({
            "ticker": corr.columns,
            "sector": [meta_df["sector"].get(str(t).upper()) for t in corr.columns],
        })
        ticker_info["sector"] = ticker_info["sector"].fillna("No Metadata")
        ticker_info["sector_sort"] = ticker_info["sector"].fillna("zzz")
        ticker_info.sort_values(["sector_sort", "ticker"], inplace=True)

        ordered_tickers = ticker_info["ticker"].tolist()
        corr_to_plot = corr.loc[ordered_tickers, ordered_tickers]

        last_sector: str | None = None
        for idx, sector in enumerate(ticker_info["sector"]):
            if idx == 0:
                last_sector = sector
                continue
            if sector != last_sector:
                sector_boundaries.append(idx)
                last_sector = sector

    mask = None
    if lower_triangle:
        mask = np.zeros_like(corr_to_plot, dtype=bool)
        # k=0 masks the diagonal, k=1 keeps it
        k_val = 0 if mask_diagonal else 1
        mask[np.triu_indices_from(mask, k=k_val)] = True

    fig, ax = plt.subplots(figsize=figsize)

    # Respect user's annotation request regardless of matrix size
    sns.heatmap(
        corr_to_plot,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        annot=annot,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        linewidths=0.2,
        linecolor="white",
        square=False,
        ax=ax,
        cbar_kws={"label": "Return correlation", "shrink": 0.8},
    )

    ax.set_title("ETF Daily Return Correlations", pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(axis="both", labelsize=9)

    for boundary in sector_boundaries:
        ax.axhline(boundary, color="gray", linewidth=0.6, linestyle="--", alpha=0.7)
        ax.axvline(boundary, color="gray", linewidth=0.6, linestyle="--", alpha=0.7)

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)

    return fig


def plot_correlation_clustermap(
    corr: pd.DataFrame,
    *,
    figsize: tuple[int, int] = (13, 13),
    output_path: Path | None = None,
    cmap: str = "RdBu_r",
) -> sns.matrix.ClusterGrid:
    """Render a seaborn clustermap to highlight correlated clusters."""

    # Use a slightly larger figure and adjust dendrogram ratio for clarity
    grid = sns.clustermap(
        corr,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        annot=False,  # Keep false to avoid clutter in clustermap
        figsize=figsize,
        linewidths=0.5,
        linecolor="white",
        dendrogram_ratio=(0.15, 0.15),
        cbar_pos=(0.02, 0.82, 0.03, 0.12),
        cbar_kws={"label": "Correlation", "ticks": [-1, -0.5, 0, 0.5, 1]},
        tree_kws={"linewidths": 1.2},
    )

    # Improve title and label readability
    grid.ax_heatmap.set_title(
        "Clustered Correlation Matrix of ETF Returns",
        pad=20,
        fontsize=16,
        fontweight="bold",
    )
    
    # Rotate x-axis labels for better readability
    grid.ax_heatmap.set_xticklabels(
        grid.ax_heatmap.get_xticklabels(),
        rotation=45,
        ha="right",
        fontsize=9,
        fontweight="medium",
    )
    
    # Ensure y-axis labels are horizontal and readable
    grid.ax_heatmap.set_yticklabels(
        grid.ax_heatmap.get_yticklabels(),
        rotation=0,
        fontsize=9,
        fontweight="medium",
    )

    # Remove axis labels (ticker names are self-explanatory)
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid.savefig(output_path, dpi=300, bbox_inches="tight")

    return grid


def find_high_corr_pairs(
    corr: pd.DataFrame,
    *,
    threshold: float = 0.9,
) -> pd.DataFrame:
    """Enumerate unordered ticker pairs with correlation >= threshold."""

    tickers = list(corr.columns)
    records: list[dict[str, object]] = []
    for i, left in enumerate(tickers):
        for j in range(i + 1, len(tickers)):
            right = tickers[j]
            value = corr.iloc[i, j]
            if pd.isna(value) or value < threshold:
                continue
            records.append({"leg_x": left, "leg_y": right, "correlation": float(value)})
    return pd.DataFrame(records)


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


def plot_pair_bucket_counts(
    pairs_df: pd.DataFrame,
    *,
    output_path: Path | None = None,
) -> plt.Figure:
    """Bar chart showing how many high-corr pairs fall in each bucket."""

    count_df = (
        pairs_df["pair_bucket"].value_counts().rename_axis("pair_bucket").reset_index(name="pairs")
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=count_df, x="pair_bucket", y="pairs", palette="viridis", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("# of pairs")
    ax.set_title("High-correlation pairs by sector bucket")
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig


def plot_pair_bucket_boxplot(
    pairs_df: pd.DataFrame,
    *,
    output_path: Path | None = None,
) -> plt.Figure:
    """Boxplot comparing correlation distributions per bucket."""

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=pairs_df, x="pair_bucket", y="correlation", palette="pastel", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Correlation")
    ax.set_title("Correlation distribution by sector bucket")
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig


def plot_corr_vs_pvalue(
    pairs_df: pd.DataFrame,
    *,
    output_path: Path | None = None,
    alpha_line: float = 0.05,
) -> plt.Figure:
    """Scatter plot of correlation vs Engle–Granger p-value for enriched pairs."""

    if "coint_pvalue" not in pairs_df.columns:
        raise ValueError("pairs_df must include 'coint_pvalue' to plot correlation vs p-value")

    scatter_df = pairs_df.dropna(subset=["coint_pvalue"])
    if scatter_df.empty:
        raise ValueError("No Engle–Granger p-values available for plotting")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(
        data=scatter_df,
        x="correlation",
        y="coint_pvalue",
        hue="pair_bucket",
        style="pair_bucket",
        s=80,
        ax=ax,
    )
    ax.axhline(alpha_line, color="red", linestyle="--", label=f"alpha={alpha_line}")
    ax.set_xlim(0.75, 1.0)
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Engle–Granger p-value")
    ax.set_title("Correlation vs Engle–Granger p-value")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig
