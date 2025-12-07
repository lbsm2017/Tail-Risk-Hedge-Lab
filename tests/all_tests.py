"""
Comprehensive Test Suite Runner

Author: L.Bassetti
Master test runner that executes all unit and integration tests.

Usage:
    python tests/all_tests.py
    
Or via Makefile:
    make tests
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_tests():
    """Discover and run all tests in the tests directory."""
    
    print("=" * 70)
    print("Tail-Risk Hedge Lab - Comprehensive Test Suite")
    print("=" * 70)
    print()
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Discover all tests in the tests directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # Count total tests
    def count_tests(test_suite):
        """Recursively count test cases in a suite."""
        count = 0
        for test in test_suite:
            if isinstance(test, unittest.TestSuite):
                count += count_tests(test)
            else:
                count += 1
        return count
    
    total_tests = count_tests(suite)
    print(f"Discovered {total_tests} tests across multiple modules")
    print()
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


def list_available_tests():
    """List all available test modules and test cases."""
    
    print("=" * 70)
    print("Available Test Modules")
    print("=" * 70)
    print()
    
    test_modules = [
        ('test_main.py', 'Tests for main.py entry point and pipeline orchestration'),
        ('test_engine.py', 'Tests for backtester engine and portfolio construction'),
        ('test_integration.py', 'End-to-end integration tests for complete pipeline'),
        ('test_phase3.py', 'Tests for Phase 3 features (hypothesis testing)'),
        ('test_downloader.py', 'Tests for data downloader module')
    ]
    
    for module, description in test_modules:
        module_path = os.path.join(os.path.dirname(__file__), module)
        exists = "✓" if os.path.exists(module_path) else "✗"
        print(f"{exists} {module:25s} - {description}")
    
    print()
    print("=" * 70)


if __name__ == '__main__':
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['--list', '-l']:
        list_available_tests()
        sys.exit(0)
    
    # Run all tests
    exit_code = run_all_tests()
    sys.exit(exit_code)
