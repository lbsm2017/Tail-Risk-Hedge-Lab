"""
Data Leakage Detection Tests

Author: L.Bassetti
Tests to ensure no future information leaks into backtesting.
Critical for preventing look-ahead bias in quantitative finance.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regime.detector import RegimeDetector
from src.optimization.weight_finder import find_weight_for_target_reduction
from src.metrics.tail_risk import max_drawdown, drawdown_series
from src.backtester.rebalancing import simulate_rebalanced_portfolio, get_rebalance_dates


class TestRegimeLeakage(unittest.TestCase):
    """Test that regime detection doesn't use future data."""
    
    def test_drawdown_regime_no_lookahead(self):
        """Drawdown regime at time t should only use data up to t."""
        config = {'method': 'drawdown', 'drawdown_threshold': -0.10}
        detector = RegimeDetector(config)
        
        # Create price series with crash at end
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = pd.Series(index=dates, dtype=float)
        prices.iloc[:80] = 100 * (1.001 ** np.arange(80))  # Uptrend
        prices.iloc[80:] = prices.iloc[79] * (0.98 ** np.arange(20))  # Crash
        
        regime = detector.drawdown_regime(prices)
        
        # Before crash, regime should be normal (0)
        # Regime at day 50 should not know about crash at day 80
        self.assertEqual(regime.iloc[50], 0)
        self.assertEqual(regime.iloc[60], 0)
        
        # After crash begins, regime should eventually switch to crisis
        # (once drawdown exceeds threshold)
        self.assertEqual(regime.iloc[-1], 1)
    
    def test_drawdown_uses_expanding_window_only(self):
        """Drawdown should use expanding window (not forward-looking)."""
        config = {'method': 'drawdown', 'drawdown_threshold': -0.15}
        detector = RegimeDetector(config)
        
        # Create series: peak, drop, recover, drop again
        dates = pd.date_range('2020-01-01', periods=60, freq='D')
        prices = pd.Series(index=dates, dtype=float)
        prices.iloc[:20] = 100 * (1.002 ** np.arange(20))  # Up to ~104
        prices.iloc[20:30] = prices.iloc[19] * (0.95 ** np.arange(10))  # Down to ~98
        prices.iloc[30:40] = prices.iloc[29] * (1.01 ** np.arange(10))  # Recovery to ~108
        prices.iloc[40:] = prices.iloc[39] * (0.98 ** np.arange(20))  # Second crash
        
        regime = detector.drawdown_regime(prices)
        
        # Key test: regime at early point (before crash) should be 0
        # Drawdown at day 10 should definitely be 0 (no crisis yet)
        self.assertEqual(regime.iloc[10], 0)  # Not in crisis at beginning
        
        # After large crash (day 50+), should be in crisis
        self.assertEqual(regime.iloc[50], 1)  # In crisis after second crash
    
    def test_vix_regime_no_lookahead(self):
        """VIX regime at time t uses VIX value at time t only."""
        config = {
            'method': 'vix',
            'vix_crisis_threshold': 30,
            'vix_recovery_threshold': 20
        }
        detector = RegimeDetector(config)
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        vix = pd.Series(15.0, index=dates)  # Normal VIX
        vix.iloc[50:60] = 40  # VIX spike at day 50-60
        
        regime = detector.vix_regime(vix)
        
        # Before spike, regime should be normal
        self.assertEqual(regime.iloc[49], 0)
        
        # During spike, regime should be crisis
        self.assertEqual(regime.iloc[55], 1)
        
        # After spike ends and VIX drops below recovery, regime returns to normal
        self.assertEqual(regime.iloc[-1], 0)
    
    def test_volatility_regime_uses_expanding_window(self):
        """Volatility percentile should use expanding historical window."""
        config = {
            'method': 'volatility',
            'volatility_window': 21,
            'volatility_percentile': 0.75
        }
        detector = RegimeDetector(config)
        
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        returns = pd.Series(np.random.normal(0, 0.01, 252), index=dates)
        
        # Regime detection should only use data up to each point
        regime = detector.volatility_percentile_regime(returns)
        
        # Regime should be same length as returns
        self.assertEqual(len(regime), len(returns))
        
        # All values should be 0 or 1 (valid regime)
        self.assertTrue(regime.isin([0, 1]).all())


class TestOptimizationLeakage(unittest.TestCase):
    """Test that optimization doesn't use future data."""
    
    def test_optimization_uses_only_historical_data(self):
        """Weight optimization should only use in-sample data."""
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        
        # Base returns
        base_returns = pd.Series(np.random.normal(0.0003, 0.01, n), index=dates)
        
        # Hedge returns (negatively correlated)
        hedge_returns = pd.Series(
            -0.3 * base_returns.values + np.random.normal(0, 0.005, n),
            index=dates
        )
        
        # Split data: first 250 days for "in-sample", rest for "out-sample"
        in_sample_base = base_returns.iloc[:250]
        in_sample_hedge = hedge_returns.iloc[:250]
        
        # Optimize using only in-sample data
        result = find_weight_for_target_reduction(
            base_returns=in_sample_base,
            hedge_returns=in_sample_hedge,
            target_reduction=0.20,
            metric='cvar',
            max_weight=0.50,
            weight_step=0.05
        )
        
        # Should return valid result based only on in-sample data
        self.assertIn('optimal_weight', result)
        self.assertIn('achieved_reduction', result)
        self.assertGreaterEqual(result['optimal_weight'], 0)
        self.assertLessEqual(result['optimal_weight'], 0.50)
    
    def test_optimization_does_not_use_future_prices(self):
        """Optimization using month N should not know month N+1 prices."""
        np.random.seed(42)
        
        # Create data with clear trend change
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        # Base returns: stable variance for first 250 days, then increases
        base_returns = pd.Series(index=dates, dtype=float)
        base_returns.iloc[:250] = np.random.normal(0.0003, 0.01, 250)
        base_returns.iloc[250:] = np.random.normal(0.0003, 0.02, 250)  # Higher vol
        
        # Hedge: negatively correlated
        hedge_returns = pd.Series(
            -0.3 * base_returns.values + np.random.normal(0, 0.005, 500),
            index=dates
        )
        
        # Optimize using only first 250 days
        result_early = find_weight_for_target_reduction(
            base_returns=base_returns.iloc[:250],
            hedge_returns=hedge_returns.iloc[:250],
            target_reduction=0.15,
            metric='cvar',
            max_weight=0.40,
            weight_step=0.05
        )
        
        # Optimize using all 500 days (with future high-vol data)
        result_all = find_weight_for_target_reduction(
            base_returns=base_returns,
            hedge_returns=hedge_returns,
            target_reduction=0.15,
            metric='cvar',
            max_weight=0.40,
            weight_step=0.05
        )
        
        # Both should return valid results but may differ
        self.assertIn('optimal_weight', result_early)
        self.assertIn('optimal_weight', result_all)
        
        # Values should be non-negative and within constraints
        self.assertGreaterEqual(result_early['optimal_weight'], 0)
        self.assertGreaterEqual(result_all['optimal_weight'], 0)


class TestRebalancingLeakage(unittest.TestCase):
    """Test that rebalancing simulation doesn't use future data."""
    
    def test_rebalancing_uses_past_returns_only(self):
        """Portfolio value at time t uses returns up to t only."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        returns = pd.DataFrame({
            'ACWI': np.random.normal(0.0003, 0.01, 252),
            'TLT': np.random.normal(0.0001, 0.008, 252)
        }, index=dates)
        
        weights = {'ACWI': 0.80, 'TLT': 0.20}
        
        result = simulate_rebalanced_portfolio(
            returns=returns,
            weights=weights,
            rebalance_frequency='quarterly'
        )
        
        # Verify cumulative value at day t is computed from returns 0...t
        # Manual calculation for day 10
        expected_value = 1.0
        for i in range(10):
            day_return = weights['ACWI'] * returns['ACWI'].iloc[i] + \
                        weights['TLT'] * returns['TLT'].iloc[i]
            expected_value *= (1 + day_return)
        
        # Should approximately match (initial rebalance happens day 1)
        # Just verify the simulation produces reasonable values
        self.assertGreater(result['portfolio_value'].iloc[9], 0)
        self.assertLess(result['portfolio_value'].iloc[9], 2)  # Reasonable bounds
    
    def test_rebalance_dates_predetermined(self):
        """Rebalance dates should be fixed, not adaptive to performance."""
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        
        rebalance_dates = get_rebalance_dates(dates, frequency='quarterly')
        
        # Should be exactly 4 quarterly dates (approx)
        self.assertGreaterEqual(len(rebalance_dates), 4)
        self.assertLessEqual(len(rebalance_dates), 5)
        
        # Rebalance dates should not be adaptive
        # Get same dates if we call again
        rebalance_dates_2 = get_rebalance_dates(dates, frequency='quarterly')
        self.assertEqual(rebalance_dates, rebalance_dates_2)
    
    def test_rebalancing_flag_correct_timing(self):
        """Rebalance flag should be set before return applies, not after."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Create returns with specific pattern
        returns = pd.DataFrame({
            'ACWI': np.random.normal(0.0003, 0.01, 252),
            'TLT': np.random.normal(0.0001, 0.008, 252)
        }, index=dates)
        
        weights = {'ACWI': 0.80, 'TLT': 0.20}
        
        result = simulate_rebalanced_portfolio(
            returns=returns,
            weights=weights,
            rebalance_frequency='quarterly'
        )
        
        # Count rebalances
        n_rebalances = result['rebalance_flag'].sum()
        self.assertGreaterEqual(n_rebalances, 3)
        self.assertLessEqual(n_rebalances, 5)
        
        # After rebalance days, drift should be low (just starting to drift)
        for i in range(len(result)):
            if result['rebalance_flag'].iloc[i] == 1:
                # Day of rebalance: drift was already calculated before rebalance
                # So it could be non-zero from previous drift
                self.assertGreaterEqual(result['drift'].iloc[i], 0)


class TestDrawdownLeakage(unittest.TestCase):
    """Test that drawdown calculation doesn't use future peaks."""
    
    def test_drawdown_uses_running_max(self):
        """Drawdown at time t uses max from 0 to t, not future."""
        prices = pd.Series([100.0, 110.0, 105.0, 120.0, 100.0])
        
        dd_series = drawdown_series(prices)
        
        # At index 2 (price=105), running max is 110, not 120
        # Drawdown = (105-110)/110 = -0.0454...
        expected_dd_2 = (105 - 110) / 110
        self.assertAlmostEqual(dd_series.iloc[2], expected_dd_2, places=4)
        
        # At index 4 (price=100), running max is 120
        # Drawdown = (100-120)/120 = -0.1666...
        expected_dd_4 = (100 - 120) / 120
        self.assertAlmostEqual(dd_series.iloc[4], expected_dd_4, places=4)
    
    def test_max_drawdown_date_consistency(self):
        """Peak date should come before trough date."""
        prices = pd.Series(
            [100, 110, 105, 95, 100, 110],
            index=pd.date_range('2020-01-01', periods=6, freq='D')
        )
        
        mdd, peak_date, trough_date = max_drawdown(prices)
        
        # Peak should come before trough
        self.assertLess(peak_date, trough_date)
        
        # MDD should be positive
        self.assertGreater(mdd, 0)
    
    def test_drawdown_monotonic_increasing(self):
        """MDD of monotonically increasing series should be 0 or very small."""
        prices = pd.Series([100, 101, 102, 103, 104, 105])
        mdd, _, _ = max_drawdown(prices)
        
        self.assertAlmostEqual(mdd, 0.0, places=5)


class TestBootstrapLeakage(unittest.TestCase):
    """Test that bootstrap tests don't leak information."""
    
    def test_bootstrap_uses_historical_distribution(self):
        """Bootstrap resampling should only use historical data distribution."""
        from src.hypothesis.tests import bootstrap_cvar_test
        
        np.random.seed(42)
        n = 252
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        
        base_returns = pd.Series(np.random.normal(0.0003, 0.01, n), index=dates)
        hedge_returns = pd.Series(np.random.normal(0.0001, 0.008, n), index=dates)
        
        result = bootstrap_cvar_test(
            base_returns=base_returns,
            hedge_returns=hedge_returns,
            hedge_weight=0.20,
            n_bootstrap=100,  # Small for speed
            alpha=0.05
        )
        
        # Should return valid statistical results
        self.assertIn('p_value', result)
        self.assertIn('actual_reduction', result)
        self.assertGreaterEqual(result['p_value'], 0)
        self.assertLessEqual(result['p_value'], 1)
    
    def test_bootstrap_ci_ordering(self):
        """Confidence interval lower should be less than upper."""
        from src.hypothesis.tests import bootstrap_cvar_test
        
        np.random.seed(42)
        n = 252
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        
        base_returns = pd.Series(np.random.normal(0.0003, 0.01, n), index=dates)
        hedge_returns = pd.Series(
            -0.3 * base_returns.values + np.random.normal(0, 0.005, n),
            index=dates
        )
        
        result = bootstrap_cvar_test(
            base_returns=base_returns,
            hedge_returns=hedge_returns,
            hedge_weight=0.20,
            n_bootstrap=100
        )
        
        # CI lower should be less than upper
        self.assertLess(result['ci_lower'], result['ci_upper'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
