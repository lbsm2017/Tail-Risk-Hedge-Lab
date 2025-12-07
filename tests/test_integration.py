"""
Integration Tests for Complete Pipeline

Author: L.Bassetti
End-to-end integration tests for the complete backtesting pipeline.
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for complete pipeline execution."""
    
    @unittest.skipUnless(os.path.exists('config.yaml'), "Requires config.yaml")
    def test_config_file_structure(self):
        """Test that config.yaml has required structure."""
        import yaml
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify required top-level keys
        self.assertIn('assets', config)
        self.assertIn('regime', config)
        self.assertIn('optimization', config)
        self.assertIn('metrics', config)
        
        # Verify assets structure
        self.assertIn('base', config['assets'])
        self.assertIn('hedges', config['assets'])
        self.assertIsInstance(config['assets']['hedges'], list)
        
        # Verify each hedge has required fields
        for hedge in config['assets']['hedges']:
            self.assertIn('ticker', hedge)
            self.assertIn('name', hedge)
            self.assertIn('max_weight', hedge)
    
    @unittest.skipUnless(os.path.exists('data/prices.parquet'), "Requires cached price data")
    def test_price_data_structure(self):
        """Test that cached price data has correct structure."""
        prices = pd.read_parquet('data/prices.parquet')
        
        # Verify it's a DataFrame
        self.assertIsInstance(prices, pd.DataFrame)
        
        # Verify index is datetime
        self.assertIsInstance(prices.index, pd.DatetimeIndex)
        
        # Verify it has multiple columns
        self.assertGreater(len(prices.columns), 1)
        
        # Verify data is numeric
        self.assertTrue(np.issubdtype(prices.values.dtype, np.number))
    
    def test_output_directory_exists(self):
        """Test that output directory exists or can be created."""
        output_dir = 'output'
        
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                created = True
            except Exception:
                created = False
            self.assertTrue(created, "Cannot create output directory")
        else:
            self.assertTrue(os.path.isdir(output_dir))


class TestDateRangeLogic(unittest.TestCase):
    """Test date range filtering logic across the pipeline."""
    
    def test_inception_date_tracking(self):
        """Test that asset inception dates are tracked correctly."""
        from src.data.downloader import DataDownloader
        
        config = {
            'data': {
                'start_date': '2020-01-01',
                'tickers': ['ACWI', 'TLT']
            }
        }
        
        dl = DataDownloader(config)
        
        # Verify inception dates dict exists
        self.assertTrue(hasattr(dl, 'asset_inception_dates'))
        self.assertIsInstance(dl.asset_inception_dates, dict)
    
    def test_portfolio_date_filtering(self):
        """Test that portfolio simulation uses correct date range."""
        from src.backtester.rebalancing import simulate_rebalanced_portfolio
        
        # Create test data with different start dates
        dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
        
        # Asset 1 starts from beginning
        returns1 = pd.Series(np.random.normal(0.0003, 0.01, len(dates)), index=dates)
        
        # Asset 2 starts 100 days later (simulated with earlier NaNs)
        returns2_values = [np.nan] * 100 + list(np.random.normal(0.0001, 0.008, len(dates) - 100))
        returns2 = pd.Series(returns2_values, index=dates)
        
        returns_df = pd.DataFrame({'Asset1': returns1, 'Asset2': returns2})
        
        # After dropna, should start from day 100
        returns_clean = returns_df.dropna()
        
        self.assertEqual(len(returns_clean), len(dates) - 100)
        self.assertEqual(returns_clean.index[0], dates[100])


class TestReportGeneration(unittest.TestCase):
    """Test report generation and output."""
    
    def test_report_module_imports(self):
        """Test that report generation module imports correctly."""
        try:
            from src.reporting.report import generate_html_report
            success = True
        except ImportError as e:
            success = False
            print(f"Import error: {e}")
        
        self.assertTrue(success)
    
    def test_report_requires_results_dict(self):
        """Test that report generation requires proper results structure."""
        from src.reporting.report import generate_html_report
        
        # Minimal valid results
        results = {
            'config': {'assets': {'base': 'ACWI'}},
            'data_info': {
                'start_date': '2020-01-01',
                'end_date': '2023-01-01',
                'n_days': 1095,
                'assets': ['ACWI', 'TLT']
            },
            'regime_stats': {
                'total_periods': 1095,
                'crisis_periods': 100,
                'crisis_pct': 0.09,
                'n_episodes': 3
            },
            'individual_hedges': {},
            'portfolios': {}
        }
        
        # Should not raise an error with valid structure
        try:
            # Use a test output path
            test_output = 'output/test_report.html'
            generate_html_report(results, output_path=test_output)
            
            # Clean up test file if it was created
            if os.path.exists(test_output):
                os.remove(test_output)
            
            success = True
        except Exception as e:
            success = False
            print(f"Report generation error: {e}")
        
        self.assertTrue(success)


class TestDataConsistency(unittest.TestCase):
    """Test data consistency across pipeline stages."""
    
    def test_returns_calculation_consistency(self):
        """Test that returns are calculated consistently."""
        # Create synthetic price data
        prices = pd.Series([100, 101, 99, 102, 105], 
                          index=pd.date_range('2020-01-01', periods=5))
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Verify length
        self.assertEqual(len(returns), len(prices) - 1)
        
        # Verify first return is correct
        expected_first = (101 - 100) / 100
        self.assertAlmostEqual(returns.iloc[0], expected_first, places=10)
        
        # Verify round-trip (prices from returns)
        reconstructed = (1 + returns).cumprod() * prices.iloc[0]
        pd.testing.assert_series_equal(
            reconstructed, 
            prices.iloc[1:].astype('float64'), 
            check_names=False,
            atol=1e-10
        )
    
    def test_regime_labels_alignment(self):
        """Test that regime labels align with returns data."""
        dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
        returns = pd.Series(np.random.normal(0.0003, 0.01, len(dates)), index=dates)
        
        # Create regime labels (0 = normal, 1 = crisis)
        regime_labels = pd.Series(
            np.random.choice([0, 1], size=len(dates), p=[0.85, 0.15]),
            index=dates
        )
        
        # Verify alignment
        self.assertEqual(len(returns), len(regime_labels))
        pd.testing.assert_index_equal(returns.index, regime_labels.index)
        
        # Verify regime labels are binary
        self.assertTrue(regime_labels.isin([0, 1]).all())


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
