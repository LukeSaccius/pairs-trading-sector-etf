"""Visualization helpers for ETF pairs trading analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
    """Plot a correlation heatmap optionally grouped by sector metadata.

    Parameters
    ----------
    corr
        Correlation matrix (square DataFrame).
    lower_triangle / mask_diagonal
        Control masking so that only the lower half is displayed for readability.
    metadata
        Optional ticker metadata with a ``sector`` column used to cluster labels.
    """

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
    """Render a seaborn clustermap to highlight correlated ETF clusters."""

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


def plot_pair_bucket_counts(
    pairs_df: pd.DataFrame,
    *,
    output_path: Path | None = None,
) -> plt.Figure:
    """Bar chart showing how many high-correlation pairs fall in each bucket."""

    count_df = (
        pairs_df["pair_bucket"].value_counts().rename_axis("pair_bucket").reset_index(name="pairs")
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=count_df,
        x="pair_bucket",
        y="pairs",
        hue="pair_bucket",
        palette="viridis",
        dodge=False,
        legend=False,
        ax=ax,
    )
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
    """Boxplot comparing the correlation distributions for each bucket."""

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(
        data=pairs_df,
        x="pair_bucket",
        y="correlation",
        hue="pair_bucket",
        palette="pastel",
        legend=False,
        ax=ax,
    )
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
