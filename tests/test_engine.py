"""
Unit Tests for Backtester Engine

Author: L.Bassetti
Tests for the backtester engine and portfolio construction.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtester.engine import Backtester


class TestBacktesterInitialization(unittest.TestCase):
    """Test Backtester initialization and setup."""
    
    def setUp(self):
        """Create minimal test configuration."""
        self.test_config = {
            'data': {
                'start_date': '2020-01-01',
                'end_date': '2023-01-01'
            },
            'assets': {
                'base': 'ACWI',
                'hedges': [
                    {'ticker': 'TLT', 'name': 'Treasury', 'max_weight': 0.5}
                ]
            },
            'regime': {'method': 'drawdown'},
            'optimization': {
                'max_total_hedge_weight': 0.5,
                'weight_step': 0.01
            },
            'metrics': {'cvar_confidence': 0.95}
        }
    
    def test_initialization_with_config(self):
        """Test that Backtester initializes with valid config."""
        bt = Backtester(self.test_config)
        
        self.assertEqual(bt.config, self.test_config)
        self.assertIsNotNone(bt.downloader)
        self.assertIsNotNone(bt.regime_detector)
    
    def test_hedge_weights_parsing(self):
        """Test that hedge weights are correctly parsed from config."""
        bt = Backtester(self.test_config)
        
        self.assertIn('TLT', bt.hedge_weights)
        self.assertEqual(bt.hedge_weights['TLT'], 0.5)
    
    def test_hedge_names_parsing(self):
        """Test that hedge names are correctly parsed from config."""
        bt = Backtester(self.test_config)
        
        self.assertIn('TLT', bt.hedge_names)
        self.assertEqual(bt.hedge_names['TLT'], 'Treasury')


class TestBuildOptimalPortfolio(unittest.TestCase):
    """Test portfolio construction logic."""
    
    def setUp(self):
        """Setup test data and configuration."""
        self.test_config = {
            'data': {
                'start_date': '2020-01-01',
                'end_date': '2023-01-01'
            },
            'assets': {
                'base': 'ACWI',
                'hedges': [
                    {'ticker': 'TLT', 'name': 'Treasury', 'max_weight': 0.5},
                    {'ticker': 'GLD', 'name': 'Gold', 'max_weight': 0.4}
                ]
            },
            'regime': {'method': 'drawdown'},
            'optimization': {
                'max_total_hedge_weight': 0.5,
                'weight_step': 0.01
            },
            'metrics': {'cvar_confidence': 0.95},
            'rebalancing': {'frequency': 'quarterly'}
        }
        
        # Create synthetic returns data
        dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
        np.random.seed(42)
        self.returns = pd.DataFrame({
            'ACWI': np.random.normal(0.0003, 0.01, len(dates)),
            'TLT': np.random.normal(0.0001, 0.008, len(dates)),
            'GLD': np.random.normal(0.0002, 0.012, len(dates))
        }, index=dates)
    
    def test_portfolio_assets_filtering(self):
        """Test that only assets with non-zero weights are included."""
        bt = Backtester(self.test_config)
        bt.returns = self.returns
        bt.prices = (1 + self.returns).cumprod()
        bt.regime_labels = pd.Series(0, index=self.returns.index)
        
        # Mock the downloader asset_inception_dates
        bt.downloader = Mock()
        bt.downloader.asset_inception_dates = {
            'ACWI': self.returns.index[0],
            'TLT': self.returns.index[0],
            'GLD': self.returns.index[0]
        }
        
        # Mock optimization to return only TLT with weight
        with patch('src.backtester.engine.greedy_sequential_allocation') as mock_opt:
            mock_opt.return_value = {'TLT': 0.15, 'GLD': 0.0}
            
            with patch('src.backtester.engine.simulate_rebalanced_portfolio') as mock_sim:
                # Return DataFrame matching expected structure
                sim_df = pd.DataFrame({
                    'portfolio_return': self.returns['ACWI'] * 0.85 + self.returns['TLT'] * 0.15,
                    'portfolio_value': 1.0,
                    'rebalance_flag': 0,
                    'drift': 0.0
                }, index=self.returns.index)
                mock_sim.return_value = sim_df
                
                result = bt.build_optimal_portfolio(target_cvar_reduction=0.1)
                
                # Check that portfolio_metadata exists
                self.assertIn('portfolio_metadata', result)
                metadata = result['portfolio_metadata']
                
                # Only ACWI and TLT should be in asset_inceptions (not GLD with 0 weight)
                self.assertIn('ACWI', metadata['asset_inception_dates'])
                self.assertIn('TLT', metadata['asset_inception_dates'])
                self.assertNotIn('GLD', metadata['asset_inception_dates'])
    
    def test_inception_date_filtering(self):
        """Test that portfolio data is filtered to latest inception date."""
        bt = Backtester(self.test_config)
        
        # Create returns with different start dates (simulated by NaN padding)
        dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
        returns_with_gaps = pd.DataFrame({
            'ACWI': np.random.normal(0.0003, 0.01, len(dates)),
            'TLT': np.random.normal(0.0001, 0.008, len(dates)),
            'GLD': np.random.normal(0.0002, 0.012, len(dates))
        }, index=dates)
        
        bt.returns = returns_with_gaps
        bt.prices = (1 + returns_with_gaps).cumprod()
        bt.regime_labels = pd.Series(0, index=returns_with_gaps.index)
        
        # Mock downloader with different inception dates
        bt.downloader = Mock()
        late_start = dates[100]  # GLD starts 100 days later
        bt.downloader.asset_inception_dates = {
            'ACWI': dates[0],
            'TLT': dates[0],
            'GLD': late_start
        }
        
        # Mock optimization to include GLD
        with patch('src.backtester.engine.greedy_sequential_allocation') as mock_opt:
            mock_opt.return_value = {'TLT': 0.10, 'GLD': 0.05}
            
            with patch('src.backtester.engine.simulate_rebalanced_portfolio') as mock_sim:
                # Capture the returns data passed to simulation
                def capture_returns(returns, weights, rebalance_frequency, trading_fee_bps=0.0):
                    # The returns should start from late_start (GLD inception)
                    self.assertGreaterEqual(returns.index[0], late_start)
                    # Return DataFrame matching expected structure
                    return pd.DataFrame({
                        'portfolio_return': returns.iloc[:, 0] * 0.85,
                        'portfolio_value': 1.0,
                        'rebalance_flag': 0,
                        'drift': 0.0
                    }, index=returns.index)
                
                mock_sim.side_effect = capture_returns
                
                result = bt.build_optimal_portfolio(target_cvar_reduction=0.1)
                
                # Verify simulation was called
                self.assertTrue(mock_sim.called)
    
    def test_minimum_data_requirement(self):
        """Test that insufficient data raises an error."""
        bt = Backtester(self.test_config)
        
        # Create very short returns series (less than 252 days)
        short_dates = pd.date_range('2020-01-01', periods=100, freq='D')
        short_returns = pd.DataFrame({
            'ACWI': np.random.normal(0.0003, 0.01, len(short_dates)),
            'TLT': np.random.normal(0.0001, 0.008, len(short_dates))
        }, index=short_dates)
        
        bt.returns = short_returns
        bt.prices = (1 + short_returns).cumprod()
        bt.regime_labels = pd.Series(0, index=short_returns.index)
        
        bt.downloader = Mock()
        bt.downloader.asset_inception_dates = {
            'ACWI': short_dates[0],
            'TLT': short_dates[0]
        }
        
        with patch('src.backtester.engine.greedy_sequential_allocation') as mock_opt:
            mock_opt.return_value = {'TLT': 0.15}
            
            # Should raise ValueError for insufficient data
            with self.assertRaises(ValueError) as context:
                bt.build_optimal_portfolio(target_cvar_reduction=0.1)
            
            self.assertIn("Insufficient data", str(context.exception))


class TestQuickBacktest(unittest.TestCase):
    """Test quick_backtest function."""
    
    @patch('src.backtester.engine.Backtester')
    def test_quick_backtest_creates_engine(self, mock_backtester_class):
        """Test that quick_backtest creates Backtester instance."""
        from src.backtester.engine import quick_backtest
        
        # Setup mock
        mock_instance = Mock()
        mock_instance.run_full_backtest.return_value = {'test': 'results'}
        mock_backtester_class.return_value = mock_instance
        
        # Call quick_backtest
        result = quick_backtest(config_path='config.yaml')
        
        # Verify Backtester was created
        self.assertTrue(mock_backtester_class.called)
        
        # Verify run_full_backtest was called
        mock_instance.run_full_backtest.assert_called_once()
        
        # Verify results returned
        self.assertEqual(result, {'test': 'results'})


class TestPortfolioMetadata(unittest.TestCase):
    """Test portfolio metadata generation."""
    
    def test_metadata_includes_date_range(self):
        """Test that portfolio metadata includes analysis date range."""
        config = {
            'data': {
                'start_date': '2020-01-01',
                'end_date': '2023-01-01'
            },
            'assets': {
                'base': 'ACWI',
                'hedges': [{'ticker': 'TLT', 'name': 'Treasury', 'max_weight': 0.5}]
            },
            'regime': {'method': 'drawdown'},
            'optimization': {'max_total_hedge_weight': 0.5, 'weight_step': 0.01},
            'metrics': {'cvar_confidence': 0.95},
            'rebalancing': {'frequency': 'quarterly'}
        }
        
        bt = Backtester(config)
        
        dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
        bt.returns = pd.DataFrame({
            'ACWI': np.random.normal(0.0003, 0.01, len(dates)),
            'TLT': np.random.normal(0.0001, 0.008, len(dates))
        }, index=dates)
        bt.prices = (1 + bt.returns).cumprod()
        bt.regime_labels = pd.Series(0, index=bt.returns.index)
        
        bt.downloader = Mock()
        bt.downloader.asset_inception_dates = {
            'ACWI': dates[0],
            'TLT': dates[0]
        }
        
        with patch('src.backtester.engine.greedy_sequential_allocation') as mock_opt:
            mock_opt.return_value = {'TLT': 0.15}
            
            with patch('src.backtester.engine.simulate_rebalanced_portfolio') as mock_sim:
                # Return DataFrame matching expected structure
                sim_df = pd.DataFrame({
                    'portfolio_return': bt.returns['ACWI'],
                    'portfolio_value': 1.0,
                    'rebalance_flag': 0,
                    'drift': 0.0
                }, index=bt.returns.index)
                mock_sim.return_value = sim_df
                
                result = bt.build_optimal_portfolio(target_cvar_reduction=0.1)
                
                # Check metadata structure
                self.assertIn('portfolio_metadata', result)
                metadata = result['portfolio_metadata']
                
                self.assertIn('portfolio_start_date', metadata)
                self.assertIn('portfolio_end_date', metadata)
                self.assertIn('asset_inception_dates', metadata)
                
                # Verify dates are datetime objects
                self.assertIsInstance(metadata['portfolio_start_date'], pd.Timestamp)
                self.assertIsInstance(metadata['portfolio_end_date'], pd.Timestamp)


class TestBaseMetrics(unittest.TestCase):
    """Test base_metrics computation in individual hedge analysis."""
    
    def test_base_metrics_present_in_results(self):
        """Test that base_metrics is included in individual hedge results."""
        from src.backtester.engine import _analyze_hedge_worker
        
        # Setup test data
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        np.random.seed(42)
        
        base_returns = pd.Series(np.random.normal(0.0003, 0.01, len(dates)), index=dates)
        hedge_returns = pd.Series(np.random.normal(0.0001, 0.008, len(dates)), index=dates)
        regime_labels = pd.Series(0, index=dates)
        rf_rate = pd.Series(0.03 / 252, index=dates)
        
        # Create args tuple
        args = (
            'TLT',  # ticker
            base_returns,
            hedge_returns,
            regime_labels,
            [0.10, 0.25],  # targets
            0.50,  # max_weight
            0.01,  # weight_step
            0.95,  # alpha
            1000,  # n_bootstrap
            0.05,  # hypothesis_alpha
            0.25,  # target_reduction
            rf_rate,
            'monthly',  # cvar_frequency
            0.001,  # tie_break_tolerance
            'daily',  # hedge_frequency
            'quarterly',  # rebalance_frequency
            10.0  # trading_fee_bps
        )
        
        # Call worker
        ticker, result = _analyze_hedge_worker(args)
        
        # Verify base_metrics exists
        self.assertIn('base_metrics', result)
        self.assertIsInstance(result['base_metrics'], dict)
        
        # Verify expected keys in base_metrics
        expected_keys = ['cagr', 'sharpe', 'cvar_95', 'max_drawdown']
        for key in expected_keys:
            self.assertIn(key, result['base_metrics'], 
                         f"Expected key '{key}' not found in base_metrics")
    
    def test_base_metrics_computed_on_same_period(self):
        """Test that optimization results include CAGR and Sharpe per row."""
        from src.backtester.engine import _analyze_hedge_worker
        
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        np.random.seed(42)
        
        base_returns = pd.Series(np.random.normal(0.0003, 0.01, len(dates)), index=dates)
        hedge_returns = pd.Series(np.random.normal(0.0001, 0.008, len(dates)), index=dates)
        regime_labels = pd.Series(0, index=dates)
        rf_rate = pd.Series(0.03 / 252, index=dates)
        
        args = (
            'TLT', base_returns, hedge_returns, regime_labels,
            [0.10, 0.25], 0.50, 0.01, 0.95, 1000, 0.05, 0.25,
            rf_rate, 'monthly', 0.001, 'daily', 'quarterly', 10.0
        )
        
        ticker, result = _analyze_hedge_worker(args)
        
        # Base metrics should exist
        self.assertIn('base_metrics', result)
        
        # Optimization results should include CAGR/Sharpe per row
        if result['optimization']:
            for opt in result['optimization']:
                self.assertIn('base_cagr', opt)
                self.assertIn('hedged_cagr', opt)
                self.assertIn('base_sharpe', opt)
                self.assertIn('hedged_sharpe', opt)
    
    def test_base_metrics_empty_when_insufficient_data(self):
        """Test that base_metrics is empty when data is insufficient."""
        from src.backtester.engine import _analyze_hedge_worker
        
        # Very short period (insufficient)
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        np.random.seed(42)
        
        base_returns = pd.Series(np.random.normal(0.0003, 0.01, len(dates)), index=dates)
        hedge_returns = pd.Series(np.random.normal(0.0001, 0.008, len(dates)), index=dates)
        regime_labels = pd.Series(0, index=dates)
        rf_rate = pd.Series(0.03 / 252, index=dates)
        
        args = (
            'TLT', base_returns, hedge_returns, regime_labels,
            [0.10, 0.25], 0.50, 0.01, 0.95, 1000, 0.05, 0.25,
            rf_rate, 'monthly', 0.001, 'daily', 'quarterly', 10.0
        )
        
        ticker, result = _analyze_hedge_worker(args)
        
        # Should have base_metrics key but empty dict
        self.assertIn('base_metrics', result)
        self.assertEqual(result['base_metrics'], {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
