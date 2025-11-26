"""Analysis module for pairs trading."""

from .correlation import (
    attach_sector_labels,
    compute_return_correlations,
    find_high_corr_pairs,
    summarise_pairs_by_bucket,
)

__all__ = [
    "attach_sector_labels",
    "compute_return_correlations",
    "find_high_corr_pairs",
    "summarise_pairs_by_bucket",
]
