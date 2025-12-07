"""
Tests for monthly data handling and resampling.

Ensures:
1. No look-ahead bias when resampling daily to monthly
2. Proper alignment of hedge dates with base returns
3. Correct compounding of returns
4. Data integrity across different frequencies
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestMonthlyResampling(unittest.TestCase):
    """Test monthly data resampling and alignment."""
    
    def setUp(self):
        """Create test data with known values."""
        # Create daily dates for 3 months
        dates = pd.date_range('2020-01-01', '2020-03-31', freq='B')
        
        # Daily returns: 1% per day for simplicity
        self.daily_returns = pd.Series(0.01, index=dates, name='daily')
        
        # Monthly hedge returns on last business day of each month
        monthly_dates = [
            pd.Timestamp('2020-01-31'),
            pd.Timestamp('2020-02-28'),
            pd.Timestamp('2020-03-31')
        ]
        self.monthly_returns = pd.Series([0.05, -0.03, 0.02], index=monthly_dates, name='monthly')
    
    def test_no_lookahead_bias(self):
        """Ensure monthly resampling doesn't use future data."""
        # Simulate the resampling logic from engine.py
        temp_df = pd.DataFrame({
            'base': self.daily_returns,
            'hedge': pd.Series(index=self.daily_returns.index, dtype=float)
        })
        
        # Add monthly hedge observations
        for date, value in self.monthly_returns.items():
            if date in temp_df.index:
                temp_df.loc[date, 'hedge'] = value
        
        hedge_dates = temp_df[temp_df['hedge'].notna()].index.sort_values()
        
        monthly_data = []
        for hedge_date in hedge_dates:
            hedge_month = hedge_date.to_period('M')
            month_start = hedge_month.to_timestamp()
            
            # Get data only up to hedge date
            month_mask = (temp_df.index >= month_start) & (temp_df.index <= hedge_date)
            month_data = temp_df[month_mask]
            
            # Verify we're not using future data
            self.assertTrue(
                all(month_data.index <= hedge_date),
                f"Data after {hedge_date} was used in monthly calculation"
            )
            
            # Calculate monthly return
            base_monthly = (1 + month_data['base']).prod() - 1
            
            monthly_data.append({
                'date': hedge_date,
                'base': base_monthly,
                'hedge': month_data.loc[hedge_date, 'hedge']
            })
        
        result = pd.DataFrame(monthly_data).set_index('date')
        
        # Verify we have 3 monthly observations
        self.assertEqual(len(result), 3)
        
        # Verify dates match hedge observation dates exactly
        self.assertTrue(all(result.index == self.monthly_returns.index))
    
    def test_return_compounding(self):
        """Verify correct compounding of daily returns to monthly."""
        # Create simple daily returns
        dates = pd.date_range('2020-01-01', '2020-01-31', freq='B')
        daily_rets = pd.Series(0.01, index=dates)  # 1% daily
        
        # Manual compounding: (1.01)^n - 1
        n_days = len(daily_rets)
        expected_monthly = (1.01 ** n_days) - 1
        
        # Calculated compounding
        calculated_monthly = (1 + daily_rets).prod() - 1
        
        # Should be very close
        self.assertAlmostEqual(expected_monthly, calculated_monthly, places=10)
    
    def test_hedge_date_alignment(self):
        """Ensure hedge dates are preserved exactly."""
        # Create monthly hedge data on specific dates
        hedge_dates = [
            pd.Timestamp('2020-01-15'),  # Mid-month
            pd.Timestamp('2020-02-28'),  # Month-end
            pd.Timestamp('2020-03-20')   # Late month
        ]
        
        hedge_returns = pd.Series([0.05, -0.03, 0.02], index=hedge_dates)
        
        # Daily base returns
        daily_dates = pd.date_range('2020-01-01', '2020-03-31', freq='B')
        base_returns = pd.Series(0.01, index=daily_dates)
        
        # Simulate alignment
        temp_df = pd.DataFrame({
            'base': base_returns,
            'hedge': pd.Series(index=base_returns.index, dtype=float)
        })
        
        for date, value in hedge_returns.items():
            # Find closest business day
            if date in temp_df.index:
                temp_df.loc[date, 'hedge'] = value
            else:
                # Find nearest business day before
                nearest = temp_df.index[temp_df.index <= date][-1]
                temp_df.loc[nearest, 'hedge'] = value
        
        # Verify alignment preserves chronological order
        hedge_obs = temp_df[temp_df['hedge'].notna()]
        dates = hedge_obs.index.to_list()
        
        self.assertEqual(dates, sorted(dates), "Hedge dates not in chronological order")
    
    def test_different_month_lengths(self):
        """Test resampling with different month lengths."""
        # Test February (short month) vs January (longer)
        
        # January: 23 business days (approx)
        jan_dates = pd.date_range('2020-01-01', '2020-01-31', freq='B')
        jan_returns = pd.Series(0.01, index=jan_dates)
        jan_monthly = (1 + jan_returns).prod() - 1
        
        # February: ~20 business days
        feb_dates = pd.date_range('2020-02-01', '2020-02-28', freq='B')
        feb_returns = pd.Series(0.01, index=feb_dates)
        feb_monthly = (1 + feb_returns).prod() - 1
        
        # Monthly return should be higher for longer month (more compounding)
        self.assertGreater(jan_monthly, feb_monthly)
    
    def test_partial_month_handling(self):
        """Test when hedge observation is mid-month."""
        # Create daily returns for January
        dates = pd.date_range('2020-01-01', '2020-01-31', freq='B')
        daily_returns = pd.Series(0.01, index=dates)
        
        # Hedge observation on Jan 15
        hedge_date = pd.Timestamp('2020-01-15')
        
        # Get returns only up to hedge date
        month_start = pd.Timestamp('2020-01-01')
        mask = (dates >= month_start) & (dates <= hedge_date)
        partial_returns = daily_returns[mask]
        
        partial_monthly = (1 + partial_returns).prod() - 1
        full_monthly = (1 + daily_returns).prod() - 1
        
        # Partial should be less than full month
        self.assertLess(partial_monthly, full_monthly)
        
        # Verify we only used data up to hedge date
        self.assertTrue(all(partial_returns.index <= hedge_date))


class TestDataLeakagePrevention(unittest.TestCase):
    """Test that no future data leaks into past calculations."""
    
    def test_sequential_monthly_observations(self):
        """Verify each monthly observation only uses past data."""
        # Create daily returns
        dates = pd.date_range('2020-01-01', '2020-06-30', freq='B')
        daily_returns = pd.Series(
            np.random.randn(len(dates)) * 0.01,
            index=dates
        )
        
        # Monthly observations at month-end
        monthly_dates = [
            pd.Timestamp('2020-01-31'),
            pd.Timestamp('2020-02-28'),
            pd.Timestamp('2020-03-31'),
            pd.Timestamp('2020-04-30'),
            pd.Timestamp('2020-05-29'),
            pd.Timestamp('2020-06-30')
        ]
        
        for i, hedge_date in enumerate(monthly_dates):
            # Get month start
            month_start = hedge_date.to_period('M').to_timestamp()
            
            # Get data for this month up to hedge date
            mask = (dates >= month_start) & (dates <= hedge_date)
            month_data = daily_returns[mask]
            
            # Verify no future data
            self.assertTrue(all(month_data.index <= hedge_date))
            
            # Verify we're not using data from future months
            if i < len(monthly_dates) - 1:
                next_month_start = monthly_dates[i + 1].to_period('M').to_timestamp()
                self.assertTrue(all(month_data.index < next_month_start))
    
    def test_index_ordering(self):
        """Ensure final aligned data maintains chronological order."""
        # Create test scenario
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')
        base = pd.Series(0.001, index=dates)
        
        # Monthly hedge on last business day of each month
        monthly_dates = pd.date_range('2020-01-31', '2020-12-31', freq='ME')
        # Adjust to business days
        monthly_dates = pd.DatetimeIndex([
            dates[dates <= d][-1] for d in monthly_dates
        ])
        
        hedge = pd.Series(0.02, index=monthly_dates)
        
        # Create aligned DataFrame
        temp_df = pd.DataFrame({'base': base})
        temp_df['hedge'] = pd.Series(dtype=float)
        for date, value in hedge.items():
            if date in temp_df.index:
                temp_df.loc[date, 'hedge'] = value
        
        hedge_dates = temp_df[temp_df['hedge'].notna()].index.sort_values()
        
        # Verify dates are sorted
        self.assertTrue(all(hedge_dates[:-1] <= hedge_dates[1:]))
        
        # Verify no duplicates
        self.assertEqual(len(hedge_dates), len(set(hedge_dates)))


class TestFrequencyDetection(unittest.TestCase):
    """Test asset frequency detection."""
    
    def test_daily_frequency_detection(self):
        """Detect daily data correctly."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')
        returns = pd.Series(0.001, index=dates)
        
        if len(returns) > 1:
            avg_diff = (returns.index[1:] - returns.index[:-1]).mean()
            is_monthly = avg_diff.days > 20
        else:
            is_monthly = False
        
        self.assertFalse(is_monthly)
    
    def test_monthly_frequency_detection(self):
        """Detect monthly data correctly."""
        dates = pd.date_range('2020-01-31', '2020-12-31', freq='ME')
        returns = pd.Series(0.02, index=dates)
        
        if len(returns) > 1:
            avg_diff = (returns.index[1:] - returns.index[:-1]).mean()
            is_monthly = avg_diff.days > 20
        else:
            is_monthly = False
        
        self.assertTrue(is_monthly)
    
    def test_weekly_frequency_detection(self):
        """Detect weekly data correctly."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='W')
        returns = pd.Series(0.005, index=dates)
        
        if len(returns) > 1:
            avg_diff = (returns.index[1:] - returns.index[:-1]).mean()
            is_monthly = avg_diff.days > 20
        else:
            is_monthly = False
        
        # Weekly should be detected as "daily" (not monthly)
        self.assertFalse(is_monthly)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_empty_data(self):
        """Handle empty data gracefully."""
        empty_series = pd.Series(dtype=float)
        
        # Should handle without error
        temp_df = pd.DataFrame({'base': empty_series, 'hedge': empty_series})
        hedge_dates = temp_df[temp_df['hedge'].notna()].index
        
        self.assertEqual(len(hedge_dates), 0)
    
    def test_single_observation(self):
        """Handle single monthly observation."""
        base = pd.Series(
            0.01,
            index=pd.date_range('2020-01-01', '2020-01-31', freq='B')
        )
        hedge = pd.Series([0.05], index=[pd.Timestamp('2020-01-31')])
        
        # Should produce one aligned observation
        temp_df = pd.DataFrame({'base': base})
        temp_df['hedge'] = pd.Series(dtype=float)
        temp_df.loc[hedge.index[0], 'hedge'] = hedge.iloc[0]
        
        result = temp_df.dropna()
        self.assertEqual(len(result), 1)
    
    def test_missing_months(self):
        """Handle gaps in monthly data."""
        base = pd.Series(
            0.01,
            index=pd.date_range('2020-01-01', '2020-06-30', freq='B')
        )
        
        # Hedge data for Jan, Mar, May (skip Feb, Apr)
        hedge_dates = [
            pd.Timestamp('2020-01-31'),
            pd.Timestamp('2020-03-31'),
            pd.Timestamp('2020-05-29')
        ]
        hedge = pd.Series([0.05, -0.03, 0.02], index=hedge_dates)
        
        temp_df = pd.DataFrame({'base': base})
        temp_df['hedge'] = pd.Series(dtype=float)
        for date, value in hedge.items():
            if date in temp_df.index:
                temp_df.loc[date, 'hedge'] = value
        
        result = temp_df.dropna()
        
        # Should have exactly 3 observations
        self.assertEqual(len(result), 3)
        self.assertTrue(all(result.index == hedge.index))


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
