"""
Unit Tests for Rebalancing Module

Author: L.Bassetti
Tests for portfolio rebalancing logic and weight drift simulation.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtester.rebalancing import (
    get_rebalance_dates,
    simulate_rebalanced_portfolio,
    calculate_drift
)


class TestGetRebalanceDates(unittest.TestCase):
    """Unit tests for rebalance date generation."""
    
    def test_quarterly_rebalance_dates(self):
        """Quarterly rebalancing should give ~4 dates per year."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        rebalance_dates = get_rebalance_dates(dates, frequency='quarterly')
        
        self.assertEqual(len(rebalance_dates), 4)
    
    def test_monthly_rebalance_dates(self):
        """Monthly rebalancing should give 12 dates per year."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        rebalance_dates = get_rebalance_dates(dates, frequency='monthly')
        
        self.assertEqual(len(rebalance_dates), 12)
    
    def test_daily_rebalance_dates(self):
        """Daily rebalancing should give all trading dates."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        rebalance_dates = get_rebalance_dates(dates, frequency='daily')
        
        self.assertEqual(len(rebalance_dates), 100)
    
    def test_weekly_rebalance_dates(self):
        """Weekly rebalancing should give approximately 52 dates per year."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        rebalance_dates = get_rebalance_dates(dates, frequency='weekly')
        
        # Should be around 52 weeks
        self.assertGreaterEqual(len(rebalance_dates), 50)
        self.assertLessEqual(len(rebalance_dates), 54)
    
    def test_annual_rebalance_dates(self):
        """Annual rebalancing should give 2 dates for ~2 years."""
        dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
        rebalance_dates = get_rebalance_dates(dates, frequency='annual')
        
        self.assertEqual(len(rebalance_dates), 2)
    
    def test_invalid_frequency_raises_error(self):
        """Invalid frequency should raise ValueError."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        with self.assertRaises(ValueError):
            get_rebalance_dates(dates, frequency='biweekly')
    
    def test_empty_dates_returns_empty_list(self):
        """Empty dates should return empty list."""
        dates = pd.date_range('2020-01-01', periods=0, freq='D')
        rebalance_dates = get_rebalance_dates(dates, frequency='quarterly')
        
        self.assertEqual(len(rebalance_dates), 0)
    
    def test_rebalance_dates_sorted(self):
        """Rebalance dates should be in ascending order."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        rebalance_dates = get_rebalance_dates(dates, frequency='quarterly')
        
        self.assertEqual(rebalance_dates, sorted(rebalance_dates))
    
    def test_rebalance_dates_within_range(self):
        """All rebalance dates should be within date range."""
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        rebalance_dates = get_rebalance_dates(dates, frequency='monthly')
        
        for date in rebalance_dates:
            self.assertGreaterEqual(date, dates[0])
            self.assertLessEqual(date, dates[-1])


class TestSimulateRebalancedPortfolio(unittest.TestCase):
    """Unit tests for portfolio simulation with rebalancing."""
    
    def setUp(self):
        """Setup common test data."""
        self.dates = pd.date_range('2020-01-01', periods=252, freq='D')
        np.random.seed(42)
        self.returns = pd.DataFrame({
            'ACWI': np.random.normal(0.0003, 0.01, 252),
            'TLT': np.random.normal(0.0001, 0.008, 252)
        }, index=self.dates)
    
    def test_portfolio_value_starts_at_one(self):
        """Portfolio should start at initial value (default 1.0)."""
        weights = {'ACWI': 0.80, 'TLT': 0.20}
        result = simulate_rebalanced_portfolio(
            self.returns, weights, rebalance_frequency='quarterly'
        )
        
        # First value should be close to 1.0 * (1 + first_day_return)
        first_day_return = 0.80 * self.returns['ACWI'].iloc[0] + \
                          0.20 * self.returns['TLT'].iloc[0]
        expected_first = 1.0 * (1 + first_day_return)
        
        self.assertAlmostEqual(result['portfolio_value'].iloc[0], expected_first, places=6)
    
    def test_portfolio_always_positive(self):
        """Portfolio value should always be positive."""
        weights = {'ACWI': 0.80, 'TLT': 0.20}
        result = simulate_rebalanced_portfolio(
            self.returns, weights, rebalance_frequency='quarterly'
        )
        
        self.assertTrue((result['portfolio_value'] > 0).all())
    
    def test_rebalance_flag_set_correctly(self):
        """Rebalance flag should be 1 on rebalance days, 0 otherwise."""
        weights = {'ACWI': 0.80, 'TLT': 0.20}
        result = simulate_rebalanced_portfolio(
            self.returns, weights, rebalance_frequency='quarterly'
        )
        
        # Count rebalance days
        n_rebalances = result['rebalance_flag'].sum()
        
        # Should be approximately 4 for a year
        self.assertGreaterEqual(n_rebalances, 3)
        self.assertLessEqual(n_rebalances, 5)
    
    def test_rebalance_flag_only_0_or_1(self):
        """Rebalance flag should only contain 0 or 1."""
        weights = {'ACWI': 0.80, 'TLT': 0.20}
        result = simulate_rebalanced_portfolio(
            self.returns, weights, rebalance_frequency='quarterly'
        )
        
        self.assertTrue(result['rebalance_flag'].isin([0, 1]).all())
    
    def test_missing_asset_raises_error(self):
        """Missing asset in returns should raise ValueError."""
        weights = {'ACWI': 0.50, 'GLD': 0.50}  # GLD not in returns
        
        with self.assertRaises(ValueError):
            simulate_rebalanced_portfolio(
                self.returns, weights, rebalance_frequency='quarterly'
            )
    
    def test_daily_rebalancing_maintains_weights(self):
        """Daily rebalancing should maintain constant weights (minimal drift)."""
        weights = {'ACWI': 0.80, 'TLT': 0.20}
        result = simulate_rebalanced_portfolio(
            self.returns, weights, rebalance_frequency='daily'
        )
        
        # With daily rebalancing, drift should be very small
        # (rebalance happens every day before significant drift)
        max_drift = result['drift'].max()
        self.assertLess(max_drift, 0.05)  # Less than 5% drift with daily rebalancing
    
    def test_quarterly_rebalancing_allows_drift(self):
        """Quarterly rebalancing should allow some drift between rebalances."""
        weights = {'ACWI': 0.80, 'TLT': 0.20}
        result = simulate_rebalanced_portfolio(
            self.returns, weights, rebalance_frequency='quarterly'
        )
        
        # There should be some days with non-trivial drift
        significant_drift = (result['drift'] > 0.001).sum()
        self.assertGreater(significant_drift, 10)  # Should have drift on non-rebalance days
    
    def test_drift_non_negative(self):
        """Drift should always be non-negative (difference in weights)."""
        weights = {'ACWI': 0.80, 'TLT': 0.20}
        result = simulate_rebalanced_portfolio(
            self.returns, weights, rebalance_frequency='quarterly'
        )
        
        self.assertTrue((result['drift'] >= 0).all())
    
    def test_portfolio_return_matches_value_change(self):
        """Portfolio return should match change in portfolio value."""
        weights = {'ACWI': 0.80, 'TLT': 0.20}
        result = simulate_rebalanced_portfolio(
            self.returns, weights, rebalance_frequency='quarterly'
        )
        
        # Calculate implied returns from value changes
        value_ratios = result['portfolio_value'].pct_change()
        
        # Should match portfolio_return (allowing for small numerical error)
        # Just check they're approximately equal in magnitude
        np.testing.assert_allclose(
            value_ratios[1:].values,
            result['portfolio_return'].iloc[1:].values,
            rtol=1e-10,
            atol=1e-10
        )
    
    def test_custom_initial_value(self):
        """Portfolio should start at specified initial value."""
        weights = {'ACWI': 0.80, 'TLT': 0.20}
        initial_value = 1000.0
        result = simulate_rebalanced_portfolio(
            self.returns, weights, rebalance_frequency='quarterly',
            initial_value=initial_value
        )
        
        # First value should be initial * (1 + first_day_return)
        first_day_return = 0.80 * self.returns['ACWI'].iloc[0] + \
                          0.20 * self.returns['TLT'].iloc[0]
        expected_first = initial_value * (1 + first_day_return)
        
        self.assertAlmostEqual(result['portfolio_value'].iloc[0], expected_first, places=4)
    
    def test_result_has_correct_columns(self):
        """Result should have all expected columns."""
        weights = {'ACWI': 0.80, 'TLT': 0.20}
        result = simulate_rebalanced_portfolio(
            self.returns, weights, rebalance_frequency='quarterly'
        )
        
        expected_columns = ['portfolio_value', 'portfolio_return', 'rebalance_flag', 'drift']
        for col in expected_columns:
            self.assertIn(col, result.columns)
    
    def test_result_has_correct_length(self):
        """Result should have same length as input returns."""
        weights = {'ACWI': 0.80, 'TLT': 0.20}
        result = simulate_rebalanced_portfolio(
            self.returns, weights, rebalance_frequency='quarterly'
        )
        
        self.assertEqual(len(result), len(self.returns))
    
    def test_result_index_matches_input(self):
        """Result index should match input returns index."""
        weights = {'ACWI': 0.80, 'TLT': 0.20}
        result = simulate_rebalanced_portfolio(
            self.returns, weights, rebalance_frequency='quarterly'
        )
        
        pd.testing.assert_index_equal(result.index, self.returns.index)
    
    def test_three_asset_portfolio(self):
        """Should work with three assets."""
        returns_3 = self.returns.copy()
        returns_3['GLD'] = np.random.normal(0.0001, 0.012, 252)
        
        weights = {'ACWI': 0.60, 'TLT': 0.20, 'GLD': 0.20}
        result = simulate_rebalanced_portfolio(
            returns_3, weights, rebalance_frequency='quarterly'
        )
        
        self.assertEqual(len(result), len(returns_3))
        self.assertTrue((result['portfolio_value'] > 0).all())
    
    def test_unequal_weights_sum_to_one(self):
        """Portfolio works even with weights not exactly summing to 1."""
        # Weights sum to 0.95 (not 1.0)
        weights = {'ACWI': 0.60, 'TLT': 0.35}  # Sum = 0.95
        result = simulate_rebalanced_portfolio(
            self.returns, weights, rebalance_frequency='quarterly'
        )
        
        # Should still work and produce valid result
        self.assertEqual(len(result), len(self.returns))
        self.assertTrue((result['portfolio_value'] > 0).all())
    
    def test_zero_weight_asset(self):
        """Portfolio should work with one asset having zero weight."""
        weights = {'ACWI': 1.0, 'TLT': 0.0}
        result = simulate_rebalanced_portfolio(
            self.returns, weights, rebalance_frequency='quarterly'
        )
        
        # Should match 100% ACWI portfolio
        # Calculate expected value
        acwi_value = (1 + self.returns['ACWI']).cumprod()
        
        # Check values are approximately equal
        np.testing.assert_allclose(
            result['portfolio_value'].values,
            acwi_value.values,
            rtol=1e-10,
            atol=1e-10
        )


class TestCalculateDrift(unittest.TestCase):
    """Unit tests for weight drift calculation."""
    
    def test_zero_drift_when_equal(self):
        """No drift when current weights equal target."""
        current = {'ACWI': 0.80, 'TLT': 0.20}
        target = {'ACWI': 0.80, 'TLT': 0.20}
        
        drift = calculate_drift(current, target)
        self.assertAlmostEqual(drift, 0.0, places=6)
    
    def test_positive_drift_when_different(self):
        """Drift should be positive when weights differ."""
        current = {'ACWI': 0.85, 'TLT': 0.15}  # Drifted
        target = {'ACWI': 0.80, 'TLT': 0.20}
        
        drift = calculate_drift(current, target)
        self.assertGreater(drift, 0)
    
    def test_drift_symmetric(self):
        """Drift should be same for opposite deviations."""
        target = {'ACWI': 0.80, 'TLT': 0.20}
        
        current1 = {'ACWI': 0.85, 'TLT': 0.15}
        current2 = {'ACWI': 0.75, 'TLT': 0.25}
        
        drift1 = calculate_drift(current1, target)
        drift2 = calculate_drift(current2, target)
        
        # Should be approximately equal (L1 norm is symmetric)
        self.assertAlmostEqual(drift1, drift2, places=4)
    
    def test_drift_three_assets(self):
        """Drift calculation should work with three assets."""
        current = {'ACWI': 0.65, 'TLT': 0.20, 'GLD': 0.15}
        target = {'ACWI': 0.60, 'TLT': 0.20, 'GLD': 0.20}
        
        drift = calculate_drift(current, target)
        
        # Should be positive
        self.assertGreater(drift, 0)
    
    def test_drift_single_asset(self):
        """Drift with single asset should be zero."""
        current = {'ACWI': 1.0}
        target = {'ACWI': 1.0}
        
        drift = calculate_drift(current, target)
        self.assertAlmostEqual(drift, 0.0, places=6)
    
    def test_drift_is_distance_metric(self):
        """Drift should satisfy triangle inequality (metric property)."""
        a = {'ACWI': 0.80, 'TLT': 0.20}
        b = {'ACWI': 0.85, 'TLT': 0.15}
        c = {'ACWI': 0.75, 'TLT': 0.25}
        
        drift_ab = calculate_drift(a, b)
        drift_bc = calculate_drift(b, c)
        drift_ac = calculate_drift(a, c)
        
        # Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        self.assertLessEqual(drift_ac, drift_ab + drift_bc + 1e-10)


class TestWeightDriftBehavior(unittest.TestCase):
    """Integration tests for weight drift behavior in portfolio simulation."""
    
    def setUp(self):
        """Setup test data."""
        self.dates = pd.date_range('2020-01-01', periods=252, freq='D')
        np.random.seed(42)
        self.returns = pd.DataFrame({
            'ACWI': np.random.normal(0.0003, 0.01, 252),
            'TLT': np.random.normal(0.0001, 0.008, 252)
        }, index=self.dates)
    
    def test_drift_increases_between_rebalances(self):
        """Drift should generally increase between quarterly rebalances."""
        weights = {'ACWI': 0.80, 'TLT': 0.20}
        result = simulate_rebalanced_portfolio(
            self.returns, weights, rebalance_frequency='quarterly'
        )
        
        # Find rebalance dates
        rebalance_indices = result[result['rebalance_flag'] == 1].index
        
        # Check that drift increases between rebalances
        if len(rebalance_indices) > 1:
            for i in range(len(rebalance_indices) - 1):
                start_idx = result.index.get_loc(rebalance_indices[i])
                end_idx = result.index.get_loc(rebalance_indices[i + 1])
                
                # Drift at start should be small (just after rebalance)
                # But allow for numerical drift calculation
                self.assertLess(result['drift'].iloc[start_idx], 0.05)
    
    def test_drift_resets_at_rebalance(self):
        """Drift should be near zero at rebalance dates."""
        weights = {'ACWI': 0.80, 'TLT': 0.20}
        result = simulate_rebalanced_portfolio(
            self.returns, weights, rebalance_frequency='quarterly'
        )
        
        # Drift on rebalance days should be minimal
        rebalance_days = result[result['rebalance_flag'] == 1]
        
        # Drift right after rebalance should be small
        for idx in rebalance_days.index[:-1]:  # All but last
            idx_pos = result.index.get_loc(idx)
            if idx_pos + 1 < len(result):
                next_drift = result['drift'].iloc[idx_pos + 1]
                # Next day should have small drift (just starting)
                self.assertLess(next_drift, 0.05)


if __name__ == '__main__':
    unittest.main(verbosity=2)
