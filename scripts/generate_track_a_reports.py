"""Generate Track A reports and figures."""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from pairs_trading_etf.data.loader import build_price_frame
from pairs_trading_etf.analysis.cointegration.johansen import johansen_cointegration_test

RESULTS_DIR = project_root / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def generate_correlation_json():
    """Generate results/correlation_analysis.json from existing CSVs."""
    scores_path = RESULTS_DIR / "week1_pair_scores.csv"
    summary_path = RESULTS_DIR / "week1_pair_corr_summary.csv"
    
    if not scores_path.exists() or not summary_path.exists():
        print("Missing score/summary CSVs. Run the pair scan notebook first.")
        return

    scores = pd.read_csv(scores_path)
    summary = pd.read_csv(summary_path)
    
    # Convert to dictionary structure
    output = {
        "summary_by_bucket": summary.to_dict(orient="records"),
        "top_10_pairs": scores.head(10).to_dict(orient="records"),
        "total_pairs_scanned": len(scores),
        "correlation_distribution": {
            "mean": scores["correlation"].mean(),
            "median": scores["correlation"].median(),
            "std": scores["correlation"].std()
        }
    }
    
    out_path = RESULTS_DIR / "correlation_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Saved {out_path}")

def generate_cointegration_json():
    """Generate results/cointegration_results.json."""
    scores_path = RESULTS_DIR / "week1_pair_scores.csv"
    if not scores_path.exists():
        return

    scores = pd.read_csv(scores_path)
    
    # Filter for cointegrated pairs (p-value < 0.05)
    coint_pairs = scores[scores["coint_pvalue"] < 0.05].copy()
    
    output = {
        "cointegrated_pairs_count": len(coint_pairs),
        "cointegrated_pairs": coint_pairs.to_dict(orient="records"),
        "test_method": "Engle-Granger",
        "significance_level": 0.05
    }
    
    out_path = RESULTS_DIR / "cointegration_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Saved {out_path}")

def plot_overlays():
    """Generate overlay plots for top 5 cointegrated pairs."""
    scores_path = RESULTS_DIR / "week1_pair_scores.csv"
    price_path = project_root / "data" / "raw" / "etf_prices.csv"
    
    if not scores_path.exists() or not price_path.exists():
        return

    scores = pd.read_csv(scores_path)
    # Filter for cointegrated pairs (p <= 0.05), sorted by p-value
    coint_pairs = scores[scores["coint_pvalue"] <= 0.05].sort_values("coint_pvalue")
    top_5 = coint_pairs.head(5)
    
    if top_5.empty:
        print("No cointegrated pairs found for overlay plots.")
        return
    
    # Load prices
    tickers = set(top_5["leg_x"]).union(set(top_5["leg_y"]))
    pf = build_price_frame(price_path, tickers=list(tickers))
    prices = pf.prices
    
    # Normalize to 100
    normalized = prices / prices.iloc[0] * 100
    
    for i, row in top_5.iterrows():
        leg_x = row["leg_x"]
        leg_y = row["leg_y"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        normalized[[leg_x, leg_y]].plot(ax=ax)
        ax.set_title(f"Price Overlay: {leg_x} vs {leg_y} (Corr: {row['correlation']:.3f})")
        ax.set_ylabel("Normalized Price (Base=100)")
        
        out_path = FIGURES_DIR / f"overlay_{leg_x}_{leg_y}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

def generate_summary_md():
    """Create results/week1_summary.md."""
    scores_path = RESULTS_DIR / "week1_pair_scores.csv"
    if not scores_path.exists():
        return

    scores = pd.read_csv(scores_path)
    n_pairs = len(scores)
    n_coint = len(scores[scores["coint_pvalue"] < 0.05])
    avg_corr = scores["correlation"].mean()
    
    content = f"""# Week 1 Summary Report

## Overview
- **Total Pairs Tested**: {n_pairs}
- **Average Correlation**: {avg_corr:.3f}
- **Cointegrated Pairs (p < 0.05)**: {n_coint}

## Methodology
1. **Universe**: 50 cross-sector ETFs (equity, international, and core bonds) downloaded from Yahoo Finance.
2. **Correlation**: Calculated rolling correlations; filtered for pairs > 0.80.
3. **Cointegration**: Engle-Granger test applied to high-correlation pairs.

## Top Findings
The top 5 cointegrated pairs were analyzed for price divergence.
See `results/figures/` for overlay plots.

## Next Steps
- Refine entry/exit thresholds based on spread z-scores.
- Implement backtesting engine for the top cointegrated pairs.
"""
    
    out_path = RESULTS_DIR / "week1_summary.md"
    with open(out_path, "w") as f:
        f.write(content)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    print("Generating Track A reports...")
    generate_correlation_json()
    generate_cointegration_json()
    plot_overlays()
    generate_summary_md()
    print("Done.")
