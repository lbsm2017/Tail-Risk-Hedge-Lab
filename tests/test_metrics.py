"""
Unit Tests for Tail-Risk Metrics

Author: L.Bassetti
Tests for CVaR, VaR, MDD, Sharpe, Sortino, and correlation calculations.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics.tail_risk import (
    cvar, var, max_drawdown, drawdown_series,
    downside_deviation, sharpe_ratio, sortino_ratio,
    calmar_ratio, omega_ratio, resample_returns, cagr
)
from src.metrics.correlations import (
    conditional_correlation, downside_beta, rolling_correlation,
    quantile_correlation, rolling_correlation_stats
)
from src.optimization.multi_asset import align_mixed_frequency_returns


class TestCVaR(unittest.TestCase):
    """Unit tests for CVaR calculation."""
    
    def test_cvar_returns_positive_value(self):
        """CVaR should return positive value representing loss."""
        returns = np.array([-0.05, -0.03, -0.01, 0.01, 0.02, 0.03])
        result = cvar(returns, alpha=0.95, frequency='daily')
        self.assertGreater(result, 0)
    
    def test_cvar_empty_array(self):
        """CVaR of empty array should be NaN."""
        returns = np.array([])
        result = cvar(returns, alpha=0.95, frequency='daily')
        self.assertTrue(np.isnan(result))
    
    def test_cvar_higher_alpha_equals_higher_risk(self):
        """Higher alpha should give higher (or equal) CVaR."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 1000)
        cvar_90 = cvar(returns, alpha=0.90, frequency='daily')
        cvar_95 = cvar(returns, alpha=0.95, frequency='daily')
        cvar_99 = cvar(returns, alpha=0.99, frequency='daily')
        
        self.assertLessEqual(cvar_90, cvar_95)
        self.assertLessEqual(cvar_95, cvar_99)
    
    def test_cvar_with_monthly_resampling(self):
        """CVaR with monthly frequency should resample correctly."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        returns = pd.Series(np.random.normal(0, 0.01, 252), index=dates)
        
        result = cvar(returns, alpha=0.95, frequency='monthly')
        self.assertIsNotNone(result)
        self.assertGreater(result, 0)
    
    def test_cvar_with_daily_frequency_matches_direct(self):
        """Daily CVaR should match direct calculation without resampling."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = pd.Series(np.random.normal(0, 0.01, 100), index=dates)
        
        result_daily = cvar(returns, alpha=0.95, frequency='daily')
        result_direct = cvar(returns.values, alpha=0.95, frequency='daily')
        
        self.assertAlmostEqual(result_daily, result_direct, places=6)
    
    def test_cvar_all_positive_returns(self):
        """CVaR of all positive returns should be very small."""
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = cvar(returns, alpha=0.95, frequency='daily')
        
        # Should be small or zero
        self.assertLess(result, 0.01)
    
    def test_cvar_symmetric_distribution(self):
        """CVaR should be approximately symmetric for normal distribution."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 5000)
        result = cvar(returns, alpha=0.95, frequency='daily')
        
        # For standard normal scaled by 0.01, CVaR(95%) should be ~2.06 * 0.01 = 0.0206
        self.assertGreater(result, 0.01)
        self.assertLess(result, 0.03)


class TestVaR(unittest.TestCase):
    """Unit tests for VaR calculation."""
    
    def test_var_returns_positive_value(self):
        """VaR should return positive value."""
        returns = np.array([-0.05, -0.03, -0.01, 0.01, 0.02, 0.03])
        result = var(returns, alpha=0.95)
        self.assertGreater(result, 0)
    
    def test_var_less_than_cvar(self):
        """VaR should be less than or equal to CVaR."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 1000)
        var_val = var(returns, alpha=0.95)
        cvar_val = cvar(returns, alpha=0.95, frequency='daily')
        
        self.assertLessEqual(var_val, cvar_val)
    
    def test_var_empty_array(self):
        """VaR of empty array should be NaN."""
        returns = np.array([])
        result = var(returns, alpha=0.95)
        self.assertTrue(np.isnan(result))
    
    def test_var_higher_alpha_equals_higher_risk(self):
        """Higher alpha should give higher (or equal) VaR."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 1000)
        var_90 = var(returns, alpha=0.90)
        var_95 = var(returns, alpha=0.95)
        var_99 = var(returns, alpha=0.99)
        
        self.assertLessEqual(var_90, var_95)
        self.assertLessEqual(var_95, var_99)


class TestMaxDrawdown(unittest.TestCase):
    """Unit tests for Maximum Drawdown calculation."""
    
    def test_mdd_returns_positive_value(self):
        """MDD should return positive value."""
        prices = pd.Series([100, 110, 105, 95, 100, 110])
        mdd, peak, trough = max_drawdown(prices)
        
        self.assertGreater(mdd, 0)
        self.assertIsNotNone(peak)
        self.assertIsNotNone(trough)
    
    def test_mdd_correct_calculation(self):
        """MDD should be calculated correctly."""
        prices = pd.Series([100, 110, 100, 110])
        mdd, _, _ = max_drawdown(prices)
        
        # 110 -> 100 is approximately 9.09% drawdown
        expected_mdd = (110 - 100) / 110
        self.assertAlmostEqual(mdd, expected_mdd, places=4)
    
    def test_mdd_monotonic_increasing(self):
        """MDD of monotonically increasing series should be 0 or very small."""
        prices = pd.Series([100, 101, 102, 103, 104, 105])
        mdd, _, _ = max_drawdown(prices)
        
        self.assertAlmostEqual(mdd, 0.0, places=5)
    
    def test_mdd_empty_series(self):
        """MDD of empty series should be NaN."""
        prices = pd.Series([])
        mdd, peak, trough = max_drawdown(prices)
        
        self.assertTrue(np.isnan(mdd))
        self.assertIsNone(peak)
        self.assertIsNone(trough)
    
    def test_mdd_peak_before_trough(self):
        """Peak date should always come before trough date."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = pd.Series(np.random.uniform(90, 110, 100), index=dates)
        
        mdd, peak, trough = max_drawdown(prices)
        
        if not np.isnan(mdd):
            self.assertLess(peak, trough)
    
    def test_mdd_single_value(self):
        """MDD of single value should be 0."""
        prices = pd.Series([100])
        mdd, _, _ = max_drawdown(prices)
        
        self.assertAlmostEqual(mdd, 0.0, places=5)
    
    def test_mdd_two_values(self):
        """MDD of two values should work correctly."""
        prices = pd.Series([100, 90])
        mdd, _, _ = max_drawdown(prices)
        
        expected_mdd = (100 - 90) / 100
        self.assertAlmostEqual(mdd, expected_mdd, places=4)


class TestDownsideDeviation(unittest.TestCase):
    """Unit tests for downside deviation."""
    
    def test_downside_dev_zero_for_all_positive(self):
        """Downside deviation should be 0 for all positive returns."""
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = downside_deviation(returns, mar=0.0)
        
        self.assertAlmostEqual(result, 0.0, places=6)
    
    def test_downside_dev_less_than_std(self):
        """Downside deviation should be less than or equal to std dev."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 500)
        
        downside = downside_deviation(returns, mar=0.0)
        std = np.std(returns)
        
        self.assertLessEqual(downside, std)
    
    def test_downside_dev_empty_array(self):
        """Downside deviation of empty array should be NaN."""
        returns = np.array([])
        result = downside_deviation(returns, mar=0.0)
        
        self.assertTrue(np.isnan(result))


class TestSharpeRatio(unittest.TestCase):
    """Unit tests for Sharpe Ratio calculation."""
    
    def test_sharpe_zero_volatility(self):
        """Sharpe with zero volatility should be NaN."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        result = sharpe_ratio(returns, rf_rate=0.01)
        self.assertTrue(np.isnan(result))
    
    def test_sharpe_positive_for_positive_excess(self):
        """Sharpe should be positive when excess return is clearly positive."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)  # Positive mean
        result = sharpe_ratio(returns, rf_rate=0.0)
        
        # With positive mean and 0 rf rate, Sharpe should be positive
        self.assertGreater(result, 0)
    
    def test_sharpe_empty_array(self):
        """Sharpe of empty array should be NaN."""
        returns = np.array([])
        result = sharpe_ratio(returns, rf_rate=0.0)
        self.assertTrue(np.isnan(result))


class TestSortinoRatio(unittest.TestCase):
    """Unit tests for Sortino Ratio calculation."""
    
    def test_sortino_positive_for_positive_excess(self):
        """Sortino should be positive when excess return is clearly positive."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        result = sortino_ratio(returns, mar=0.0)
        
        self.assertGreater(result, 0)
    
    def test_sortino_greater_than_sharpe(self):
        """Sortino should typically be >= Sharpe (uses downside vol only)."""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, 500)
        
        sharpe = sharpe_ratio(returns, rf_rate=0.0)
        sortino = sortino_ratio(returns, mar=0.0)
        
        # Sortino penalizes downside only, should be >= Sharpe
        self.assertGreaterEqual(sortino, sharpe - 0.01)  # Small tolerance for numerical


class TestCalmarRatio(unittest.TestCase):
    """Unit tests for Calmar Ratio calculation."""
    
    def test_calmar_positive_returns(self):
        """Calmar should be defined for positive returns with drawdown."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
        prices = (1 + returns).cumprod()
        
        result = calmar_ratio(returns, prices)
        
        # Should return a valid number
        self.assertIsNotNone(result)
    
    def test_calmar_zero_drawdown_nan(self):
        """Calmar should be NaN or inf when MDD is zero."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        returns = pd.Series([0.01] * 10, index=dates)
        prices = pd.Series([100] * 10, index=dates)  # No drawdown
        
        result = calmar_ratio(returns, prices)
        
        # MDD is 0, so result should be NaN or inf
        self.assertTrue(np.isnan(result) or np.isinf(result))


class TestOmegaRatio(unittest.TestCase):
    """Unit tests for Omega Ratio calculation."""
    
    def test_omega_positive_returns(self):
        """Omega should be > 1 for positive excess returns."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        
        result = omega_ratio(returns, threshold=0.0)
        
        # With positive mean, Omega should be > 1
        self.assertGreater(result, 1)
    
    def test_omega_less_than_one_for_negative(self):
        """Omega should be < 1 for negative excess returns."""
        np.random.seed(42)
        returns = np.random.normal(-0.001, 0.01, 252)
        
        result = omega_ratio(returns, threshold=0.0)
        
        # With negative mean, Omega should be < 1
        self.assertLess(result, 1)


class TestResampleReturns(unittest.TestCase):
    """Unit tests for return resampling."""
    
    def test_resample_daily_no_change(self):
        """Daily resampling should not change data."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        returns = pd.Series(np.random.normal(0, 0.01, 10), index=dates)
        
        result = resample_returns(returns, frequency='daily')
        pd.testing.assert_series_equal(result, returns)
    
    def test_resample_monthly_has_fewer_points(self):
        """Monthly resampling should have fewer points than daily."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        returns = pd.Series(np.random.normal(0, 0.01, 252), index=dates)
        
        result = resample_returns(returns, frequency='monthly')
        
        self.assertLess(len(result), len(returns))
        # 252 trading days ~ 8 months, so expect 8-9 month-end points
        self.assertGreaterEqual(len(result), 8)
        self.assertLessEqual(len(result), 10)
    
    def test_resample_weekly_has_fewer_than_daily(self):
        """Weekly resampling should have fewer points than daily."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        returns = pd.Series(np.random.normal(0, 0.01, 252), index=dates)
        
        result = resample_returns(returns, frequency='weekly')
        
        self.assertLess(len(result), len(returns))
        # 252 trading days ~ 50-52 weeks, so expect 36-52 week-end points
        self.assertGreaterEqual(len(result), 35)
        self.assertLessEqual(len(result), 55)
    
    def test_resample_invalid_frequency(self):
        """Invalid frequency should raise ValueError."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        returns = pd.Series(np.random.normal(0, 0.01, 10), index=dates)
        
        with self.assertRaises(ValueError):
            resample_returns(returns, frequency='invalid')
    
    def test_resample_array_input_raises_error(self):
        """Array input should raise error (requires DatetimeIndex)."""
        returns = np.array([0.01, 0.02, 0.03])
        
        with self.assertRaises(ValueError):
            resample_returns(returns, frequency='monthly')


class TestConditionalCorrelation(unittest.TestCase):
    """Unit tests for conditional correlation."""
    
    def test_conditional_corr_insufficient_data(self):
        """Conditional correlation with < 2 points should be NaN."""
        equity = pd.Series([0.01, 0.02, 0.03])
        hedge = pd.Series([0.01, 0.02, 0.03])
        regime = pd.Series([0, 0, 0])  # No crisis periods
        
        result = conditional_correlation(equity, hedge, regime, regime_value=1)
        self.assertTrue(np.isnan(result))
    
    def test_conditional_corr_perfectly_correlated(self):
        """Conditional correlation of perfectly correlated data should be 1."""
        equity = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        hedge = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        regime = pd.Series([0, 0, 1, 1, 1])
        
        result = conditional_correlation(equity, hedge, regime, regime_value=1)
        
        # Should be very close to 1 (same values)
        self.assertAlmostEqual(result, 1.0, places=5)


class TestDownsideBeta(unittest.TestCase):
    """Unit tests for downside beta."""
    
    def test_downside_beta_calculation(self):
        """Downside beta should only use negative equity returns."""
        equity = pd.Series([0.05, -0.03, -0.05, 0.02, -0.01])
        hedge = pd.Series([-0.03, 0.02, 0.04, -0.01, 0.01])
        
        result = downside_beta(equity, hedge, threshold=0.0)
        
        # Should be a valid number
        self.assertIsNotNone(result)
    
    def test_downside_beta_insufficient_data(self):
        """Downside beta with insufficient data should be NaN."""
        equity = pd.Series([0.05, 0.02])  # No negative returns
        hedge = pd.Series([0.03, 0.01])
        
        result = downside_beta(equity, hedge, threshold=0.0)
        
        self.assertTrue(np.isnan(result))
    
    def test_downside_beta_negatively_correlated(self):
        """Downside beta should be negative for negatively correlated assets."""
        np.random.seed(42)
        equity = np.random.normal(0, 0.01, 500)
        hedge = -0.8 * equity + np.random.normal(0, 0.002, 500)
        
        equity_series = pd.Series(equity)
        hedge_series = pd.Series(hedge)
        
        result = downside_beta(equity_series, hedge_series, threshold=0.0)
        
        # Negatively correlated, downside beta should be negative
        self.assertLess(result, 0)


class TestRollingCorrelation(unittest.TestCase):
    """Unit tests for rolling correlation."""
    
    def test_rolling_corr_returns_series(self):
        """Rolling correlation should return Series."""
        equity = pd.Series(np.random.normal(0, 0.01, 100))
        hedge = pd.Series(np.random.normal(0, 0.01, 100))
        
        result = rolling_correlation(equity, hedge, window=21)
        
        self.assertIsInstance(result, pd.Series)
    
    def test_rolling_corr_shorter_than_input(self):
        """Rolling correlation should have fewer non-NaN values than input."""
        equity = pd.Series(np.random.normal(0, 0.01, 100))
        hedge = pd.Series(np.random.normal(0, 0.01, 100))
        
        result = rolling_correlation(equity, hedge, window=21)
        
        # First 20 values should be NaN (window-1)
        self.assertTrue(result.iloc[0:20].isna().all())
        
        # Rest should be valid
        self.assertGreater(result.iloc[20:].notna().sum(), 0)


class TestRollingCorrelationStats(unittest.TestCase):
    """Unit tests for rolling correlation statistics."""
    
    def test_rolling_stats_returns_dict(self):
        """Should return dict with required keys."""
        equity = pd.Series(np.random.normal(0, 0.01, 100))
        hedge = pd.Series(np.random.normal(0, 0.01, 100))
        
        result = rolling_correlation_stats(equity, hedge, window=21)
        
        required_keys = ['rolling_mean', 'rolling_min', 'rolling_max', 'rolling_std']
        for key in required_keys:
            self.assertIn(key, result)
    
    def test_rolling_stats_ordering(self):
        """Min should be <= mean <= max."""
        np.random.seed(42)
        equity = pd.Series(np.random.normal(0, 0.01, 100))
        hedge = pd.Series(np.random.normal(0, 0.01, 100))
        
        result = rolling_correlation_stats(equity, hedge, window=21)
        
        if not np.isnan(result['rolling_min']):
            self.assertLessEqual(result['rolling_min'], result['rolling_mean'])
            self.assertLessEqual(result['rolling_mean'], result['rolling_max'])


class TestQuantileCorrelation(unittest.TestCase):
    """Unit tests for quantile correlation."""
    
    def test_quantile_corr_tail_focus(self):
        """Quantile correlation should focus on tail."""
        np.random.seed(42)
        equity = pd.Series(np.random.normal(0, 0.01, 500))
        
        # Create hedge that's negatively correlated especially in tail
        hedge = pd.Series(index=equity.index, dtype=float)
        hedge[equity > -0.02] = np.random.normal(0.005, 0.005, (equity > -0.02).sum())
        hedge[equity <= -0.02] = -0.8 * equity[equity <= -0.02] + np.random.normal(0, 0.002, (equity <= -0.02).sum())
        
        result = quantile_correlation(equity, hedge, quantile=0.05)
        
        # Should be a valid number
        self.assertIsNotNone(result)


class TestCAGR(unittest.TestCase):
    """Unit tests for CAGR (Compound Annual Growth Rate) calculation."""
    
    def test_cagr_positive_returns(self):
        """CAGR should be positive for generally positive returns."""
        # Simulate 5 years of daily returns with positive drift
        np.random.seed(42)
        returns = np.random.normal(0.0003, 0.01, 252 * 5)  # ~7.5% annual
        
        result = cagr(returns, periods_per_year=252)
        
        # Should be positive
        self.assertGreater(result, 0)
        # Should be roughly in expected range (5-15%)
        self.assertGreater(result, 0.03)
        self.assertLess(result, 0.20)
    
    def test_cagr_negative_returns(self):
        """CAGR should be negative for generally negative returns."""
        np.random.seed(42)
        returns = np.random.normal(-0.0003, 0.01, 252 * 3)  # Negative drift
        
        result = cagr(returns, periods_per_year=252)
        
        # Should be negative
        self.assertLess(result, 0)
    
    def test_cagr_empty_array(self):
        """CAGR of empty array should be NaN."""
        returns = np.array([])
        result = cagr(returns, periods_per_year=252)
        self.assertTrue(np.isnan(result))
    
    def test_cagr_monthly_frequency(self):
        """CAGR with monthly data should work correctly."""
        np.random.seed(42)
        # 10 years of monthly returns
        monthly_returns = np.random.normal(0.007, 0.04, 120)  # ~8% annual
        
        result = cagr(monthly_returns, periods_per_year=12)
        
        # Should be in reasonable range
        self.assertGreater(result, 0.02)
        self.assertLess(result, 0.20)
    
    def test_cagr_geometric_compounding(self):
        """CAGR should use geometric compounding correctly."""
        # Known returns: 10% then -10% should NOT equal 0%
        returns = np.array([0.10, -0.10])
        
        result = cagr(returns, periods_per_year=2)
        
        # (1.10 * 0.90) = 0.99, so CAGR should be slightly negative
        self.assertLess(result, 0)
        self.assertAlmostEqual(result, -0.01, places=2)
    
    def test_cagr_matches_price_cagr(self):
        """CAGR from returns should match CAGR calculated from prices."""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, 252 * 3)
        
        # Calculate CAGR from returns
        result_returns = cagr(returns, periods_per_year=252)
        
        # Calculate CAGR from prices
        prices = 100 * np.cumprod(1 + returns)
        start_price = 100
        end_price = prices[-1]
        n_years = len(returns) / 252
        result_prices = (end_price / start_price) ** (1 / n_years) - 1
        
        # Should match closely
        self.assertAlmostEqual(result_returns, result_prices, places=4)
    
    def test_cagr_total_loss(self):
        """CAGR with total loss should return -1."""
        returns = np.array([-0.5, -0.5, -0.5])  # Cumulative loss > 100%
        
        result = cagr(returns, periods_per_year=3)
        
        # After 3 returns of -50%, cumulative return is (0.5^3 - 1) = -87.5%
        # This shouldn't be -1 (total loss), but significant loss
        self.assertLess(result, -0.5)


class TestMixedFrequencyAlignment(unittest.TestCase):
    """Unit tests for mixed-frequency return alignment."""
    
    def test_daily_only_preserves_frequency(self):
        """Aligning daily-only data should preserve daily frequency."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        base = pd.Series(np.random.normal(0, 0.01, 252), index=dates)
        hedge = pd.DataFrame({
            'HEDGE1': np.random.normal(0, 0.01, 252),
            'HEDGE2': np.random.normal(0, 0.01, 252)
        }, index=dates)
        
        result = align_mixed_frequency_returns(base, hedge)
        
        # Should preserve most of the data (minus dropna alignment)
        self.assertGreater(len(result), 200)
    
    def test_mixed_frequency_converts_to_monthly(self):
        """Mixing daily with monthly should convert to monthly frequency."""
        # Daily data for 3 years
        daily_dates = pd.date_range('2020-01-01', periods=252*3, freq='D')
        base = pd.Series(np.random.normal(0, 0.01, 252*3), index=daily_dates)
        
        # Monthly hedge (month-end dates with NaN elsewhere)
        monthly_dates = pd.date_range('2020-01-31', periods=36, freq='ME')
        monthly_returns = np.random.normal(0, 0.04, 36)
        
        # Create hedge DataFrame with monthly data embedded in daily index
        hedge_data = pd.Series(index=daily_dates, dtype=float)
        for date, ret in zip(monthly_dates, monthly_returns):
            if date in hedge_data.index:
                hedge_data.loc[date] = ret
        hedge = pd.DataFrame({'MONTHLY_HEDGE': hedge_data})
        
        result = align_mixed_frequency_returns(base, hedge)
        
        # Should have ~36 monthly observations (or less due to alignment)
        self.assertLess(len(result), 100)
        self.assertGreater(len(result), 20)
    
    def test_alignment_aggregates_daily_returns_correctly(self):
        """Daily returns should be compounded to monthly correctly."""
        # Create simple daily data: 1% every day for ~21 trading days
        daily_dates = pd.date_range('2020-01-01', periods=63, freq='D')  # ~3 months
        daily_returns = pd.Series(0.001, index=daily_dates)  # 0.1% daily
        
        # Monthly hedge
        monthly_dates = pd.date_range('2020-01-31', periods=3, freq='ME')
        monthly_hedge = pd.Series(index=daily_dates, dtype=float)
        for date in monthly_dates:
            if date in monthly_hedge.index:
                monthly_hedge.loc[date] = 0.02  # 2% monthly return
        hedge = pd.DataFrame({'MONTHLY': monthly_hedge})
        
        result = align_mixed_frequency_returns(daily_returns, hedge)
        
        # Check that base returns are compounded (not just month-end daily returns)
        if len(result) > 0:
            # Monthly compounded return should be much larger than single daily return
            first_monthly_return = result['base'].iloc[0]
            # ~21 trading days * 0.1% ≈ 2.1% monthly (compounded)
            self.assertGreater(abs(first_monthly_return), 0.005)
    
    def test_alignment_preserves_date_index(self):
        """Aligned data should have DatetimeIndex."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        base = pd.Series(np.random.normal(0, 0.01, 252), index=dates)
        hedge = pd.DataFrame({
            'HEDGE': np.random.normal(0, 0.01, 252)
        }, index=dates)
        
        result = align_mixed_frequency_returns(base, hedge)
        
        self.assertIsInstance(result.index, pd.DatetimeIndex)
    
    def test_alignment_handles_nan_in_monthly_data(self):
        """Alignment should handle NaN in monthly data correctly."""
        daily_dates = pd.date_range('2020-01-01', periods=100, freq='D')
        base = pd.Series(np.random.normal(0, 0.01, 100), index=daily_dates)
        
        # Sparse monthly data (NaN except month-ends)
        hedge_data = pd.Series(index=daily_dates, dtype=float)
        hedge_data.iloc[30] = 0.02  # One month-end
        hedge_data.iloc[60] = -0.01  # Another month-end
        hedge = pd.DataFrame({'SPARSE': hedge_data})
        
        result = align_mixed_frequency_returns(base, hedge)
        
        # Should have few aligned points
        self.assertLessEqual(len(result), 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
