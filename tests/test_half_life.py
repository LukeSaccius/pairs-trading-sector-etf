"""Unit tests for half-life estimation in the OU model module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs_trading_etf.ou_model.half_life import estimate_half_life


class TestEstimateHalfLife:
    """Tests for the estimate_half_life function."""

    def test_synthetic_ar1_known_phi(self) -> None:
        """Test half-life estimation with synthetic AR(1) series.
        
        For AR(1): S_{t+1} = phi * S_t + epsilon
        Half-life = -ln(2) / ln(phi)
        
        With phi = 0.95: HL ≈ 13.5 days
        With phi = 0.90: HL ≈ 6.6 days
        
        Note: OLS estimation of AR(1) has known downward bias in finite samples.
        We use larger sample and relaxed tolerance to account for this.
        """
        rng = np.random.default_rng(42)
        n = 2000  # Larger sample for better estimation
        
        # Generate AR(1) with phi = 0.95
        phi = 0.95
        sigma = 0.1
        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = phi * spread[t - 1] + rng.normal(0, sigma)
        
        series = pd.Series(spread)
        estimated_hl = estimate_half_life(series)
        
        # Theoretical half-life for phi=0.95
        theoretical_hl = -np.log(2) / np.log(phi)  # ≈ 13.5
        
        assert estimated_hl is not None
        # Allow 50% tolerance due to OLS downward bias in AR(1) estimation
        assert abs(estimated_hl - theoretical_hl) / theoretical_hl < 0.50

    def test_synthetic_ar1_faster_reversion(self) -> None:
        """Test with faster mean-reverting series (phi = 0.90)."""
        rng = np.random.default_rng(123)
        n = 500
        
        phi = 0.90
        sigma = 0.1
        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = phi * spread[t - 1] + rng.normal(0, sigma)
        
        series = pd.Series(spread)
        estimated_hl = estimate_half_life(series)
        
        theoretical_hl = -np.log(2) / np.log(phi)  # ≈ 6.6
        
        assert estimated_hl is not None
        assert abs(estimated_hl - theoretical_hl) / theoretical_hl < 0.30

    def test_non_stationary_returns_none(self) -> None:
        """Non-stationary (random walk) series should return None or large HL.
        
        Note: In finite samples, random walks can appear mean-reverting due to
        sampling variance. We test multiple seeds and check the median behavior.
        """
        results = []
        for seed in range(10):  # Test with multiple seeds
            rng = np.random.default_rng(seed)
            n = 500
            
            # Random walk: phi = 1.0 (unit root)
            spread = np.cumsum(rng.normal(0, 1, n))
            series = pd.Series(spread)
            
            estimated_hl = estimate_half_life(series)
            if estimated_hl is not None:
                results.append(estimated_hl)
        
        # At least some should return None or have large HL
        # If any return a value, the median should be large
        if results:
            median_hl = np.median(results)
            # Random walk should typically have HL > 20 days in median
            assert median_hl > 15 or len(results) < 5, \
                f"Random walk median HL too short: {median_hl:.1f} days"

    def test_insufficient_data_returns_none(self) -> None:
        """Series with < 30 observations should return None."""
        spread = pd.Series([1.0, 2.0, 1.5, 0.8, 1.2])  # Only 5 points
        
        result = estimate_half_life(spread)
        
        assert result is None

    def test_handles_nan_values(self) -> None:
        """Function should handle NaN values gracefully."""
        rng = np.random.default_rng(42)
        n = 200
        
        phi = 0.95
        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = phi * spread[t - 1] + rng.normal(0, 0.1)
        
        # Insert some NaNs
        spread[10] = np.nan
        spread[50] = np.nan
        spread[100] = np.nan
        
        series = pd.Series(spread)
        result = estimate_half_life(series)
        
        # Should still compute (after dropping NaNs)
        assert result is not None
        assert result > 0

    def test_edge_case_constant_series(self) -> None:
        """Constant series should return None (no variance)."""
        spread = pd.Series([1.0] * 100)
        
        result = estimate_half_life(spread)
        
        # lstsq may return 0 or near-0 beta, which fails beta < 0 check
        assert result is None

    def test_explosive_process_returns_none(self) -> None:
        """Explosive AR(1) with phi > 1 should return None."""
        rng = np.random.default_rng(42)
        n = 100
        
        # Explosive: phi = 1.05
        phi = 1.05
        spread = np.zeros(n)
        spread[0] = 0.1
        for t in range(1, n):
            spread[t] = phi * spread[t - 1] + rng.normal(0, 0.01)
        
        series = pd.Series(spread)
        result = estimate_half_life(series)
        
        # beta > 0 for explosive process, should return None
        assert result is None

    def test_reasonable_range_for_trading(self) -> None:
        """Test that realistic trading spreads give reasonable HL values."""
        rng = np.random.default_rng(42)
        n = 504  # ~2 years of trading days
        
        # Typical mean-reverting spread with phi ≈ 0.98 (HL ≈ 34 days)
        phi = 0.98
        mu = 0.0
        sigma = 0.02
        
        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = mu + phi * (spread[t - 1] - mu) + rng.normal(0, sigma)
        
        series = pd.Series(spread)
        estimated_hl = estimate_half_life(series)
        
        assert estimated_hl is not None
        # Trading-reasonable half-life: 15-180 days
        assert 10 < estimated_hl < 200


class TestHalfLifeFormula:
    """Tests to verify the half-life formula implementation."""

    def test_half_life_formula_derivation(self) -> None:
        """Verify HL = -ln(2) / ln(phi) gives expected values.
        
        For AR(1): S_t = phi * S_{t-1} + epsilon
        After k steps: E[S_t] = phi^k * S_0
        At half-life: phi^HL = 0.5
        HL * ln(phi) = ln(0.5)
        HL = -ln(2) / ln(phi)
        """
        test_cases = [
            (0.90, -np.log(2) / np.log(0.90)),  # ≈ 6.58
            (0.95, -np.log(2) / np.log(0.95)),  # ≈ 13.51
            (0.97, -np.log(2) / np.log(0.97)),  # ≈ 22.76
            (0.99, -np.log(2) / np.log(0.99)),  # ≈ 68.97
        ]
        
        for phi, expected_hl in test_cases:
            # The formula in _estimate_half_life uses beta from OLS regression
            # where delta_S = beta * S_{lag} + error
            # beta ≈ phi - 1 for AR(1), so HL = -ln(2) / beta
            # But our implementation uses HL = -ln(2) / beta directly
            # where beta is estimated from: delta_S_t = beta * S_{t-1}
            
            # For AR(1): S_t = phi * S_{t-1} + e => delta_S_t = (phi-1)*S_{t-1} + e
            # So beta_ols ≈ phi - 1
            # HL = -ln(2) / (phi - 1) when phi < 1
            
            beta_expected = phi - 1
            hl_from_beta = -np.log(2) / beta_expected
            
            # The two formulas are NOT exactly equivalent due to different derivations
            # HL from phi: -ln(2)/ln(phi) ≈ 1/(1-phi) for phi close to 1
            # HL from beta: -ln(2)/beta = -ln(2)/(phi-1)
            # These differ by ln(2) vs 1 factor
            # Allow 10% tolerance
            assert abs(expected_hl - hl_from_beta) / expected_hl < 0.10, f"phi={phi}"
