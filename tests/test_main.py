"""
Unit Tests for Main Entry Point

Author: L.Bassetti
Tests for the main.py entry point and pipeline orchestration.
"""

import unittest
from unittest.mock import patch, MagicMock, call
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import main


class TestMainEntryPoint(unittest.TestCase):
    """Test cases for main.py entry point."""
    
    @patch('main.generate_html_report')
    @patch('main.quick_backtest')
    @patch('builtins.print')
    def test_main_pipeline_execution(self, mock_print, mock_backtest, mock_report):
        """Test that main() orchestrates the complete pipeline correctly."""
        # Setup mock backtest results
        mock_results = {
            'config': {'test': 'config'},
            'individual_hedges': {},
            'portfolios': {},
            'returns': MagicMock(),
            'regime_labels': MagicMock()
        }
        mock_backtest.return_value = mock_results
        
        # Execute main
        main()
        
        # Verify backtest was called with config
        mock_backtest.assert_called_once_with(config_path='config.yaml')
        
        # Verify report generation was called
        self.assertTrue(mock_report.called)
        call_args = mock_report.call_args
        
        # Check that returns and regime_labels were extracted
        self.assertIsNotNone(call_args[1]['returns'])
        self.assertIsNotNone(call_args[1]['regime_labels'])
        
        # Check output path format
        output_path = call_args[1]['output_path']
        self.assertTrue(output_path.startswith('output/tail_risk_analysis_'))
        self.assertTrue(output_path.endswith('.html'))
    
    @patch('main.generate_html_report')
    @patch('main.quick_backtest')
    def test_main_handles_missing_returns(self, mock_backtest, mock_report):
        """Test that main() handles missing returns data gracefully."""
        # Setup mock backtest results without returns
        mock_results = {
            'config': {},
            'individual_hedges': {},
            'portfolios': {}
        }
        mock_backtest.return_value = mock_results
        
        # Execute main - should not raise exception
        main()
        
        # Verify report was still called
        self.assertTrue(mock_report.called)
    
    @patch('main.generate_html_report')
    @patch('main.quick_backtest')
    def test_timestamp_format(self, mock_backtest, mock_report):
        """Test that output filename has correct timestamp format."""
        mock_results = {'returns': None, 'regime_labels': None}
        mock_backtest.return_value = mock_results
        
        # Freeze time for testing
        with patch('main.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 12, 7, 14, 30, 45)
            mock_datetime.strftime = datetime.strftime
            
            main()
            
            # Check filename format
            call_args = mock_report.call_args
            output_path = call_args[1]['output_path']
            self.assertIn('2025.12.07.14.30.45', output_path)
    
    @patch('main.generate_html_report')
    @patch('main.quick_backtest')
    def test_main_preserves_results_structure(self, mock_backtest, mock_report):
        """Test that main() preserves all results except returns/regime_labels."""
        # Setup comprehensive mock results
        mock_results = {
            'config': {'test': 'data'},
            'individual_hedges': {'TLT': {}},
            'portfolios': {'10pct': {}},
            'data_info': {},
            'regime_stats': {},
            'returns': MagicMock(),
            'regime_labels': MagicMock(),
            'baseline_values': MagicMock()
        }
        mock_backtest.return_value = mock_results.copy()
        
        main()
        
        # Get the results passed to report generator
        call_args = mock_report.call_args
        results_passed = call_args[0][0]
        
        # Verify non-returns data is preserved
        self.assertIn('config', results_passed)
        self.assertIn('individual_hedges', results_passed)
        self.assertIn('portfolios', results_passed)
        
        # Verify returns/regime_labels were removed from results dict
        self.assertNotIn('returns', results_passed)
        self.assertNotIn('regime_labels', results_passed)


class TestMainErrorHandling(unittest.TestCase):
    """Test error handling in main pipeline."""
    
    @patch('main.generate_html_report')
    @patch('main.quick_backtest')
    def test_backtest_exception_propagates(self, mock_backtest, mock_report):
        """Test that exceptions from backtest are propagated."""
        mock_backtest.side_effect = ValueError("Invalid configuration")
        
        with self.assertRaises(ValueError) as context:
            main()
        
        self.assertIn("Invalid configuration", str(context.exception))
        self.assertFalse(mock_report.called)
    
    @patch('main.generate_html_report')
    @patch('main.quick_backtest')
    def test_report_generation_exception_propagates(self, mock_backtest, mock_report):
        """Test that exceptions from report generation are propagated."""
        mock_backtest.return_value = {'returns': None, 'regime_labels': None}
        mock_report.side_effect = IOError("Cannot write to output directory")
        
        with self.assertRaises(IOError) as context:
            main()
        
        self.assertIn("Cannot write to output directory", str(context.exception))


class TestMainIntegration(unittest.TestCase):
    """Integration tests for main pipeline (requires config.yaml)."""
    
    @unittest.skipUnless(os.path.exists('config.yaml'), "Requires config.yaml")
    def test_main_config_file_exists(self):
        """Test that required config file exists."""
        self.assertTrue(os.path.exists('config.yaml'))
    
    @unittest.skipUnless(os.path.exists('config.yaml'), "Requires config.yaml")
    @patch('main.generate_html_report')
    def test_main_loads_actual_config(self, mock_report):
        """Test that main() can load and process actual config file."""
        # This is a smoke test - just ensure it doesn't crash
        try:
            with patch('main.quick_backtest') as mock_backtest:
                mock_backtest.return_value = {
                    'config': {},
                    'returns': None,
                    'regime_labels': None
                }
                main()
            success = True
        except Exception as e:
            success = False
            print(f"Integration test failed: {e}")
        
        self.assertTrue(success)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
