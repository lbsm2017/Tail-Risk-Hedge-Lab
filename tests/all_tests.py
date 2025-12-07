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
from io import StringIO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ANSI color codes for Windows console
class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    @staticmethod
    def enable_windows_colors():
        """Enable ANSI color support on Windows."""
        if sys.platform == 'win32':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except:
                pass


class ColoredTextTestResult(unittest.TextTestResult):
    """Custom test result class with colored output."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.success_count = 0
        self.verbosity_level = verbosity
        
    def addSuccess(self, test):
        super().addSuccess(test)
        self.success_count += 1
        if self.verbosity_level >= 1:
            self.stream.write(f"{Colors.GREEN}✓ PASS{Colors.RESET}\n")
            self.stream.flush()
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity_level >= 1:
            self.stream.write(f"{Colors.RED}✗ ERROR{Colors.RESET}\n")
            self.stream.flush()
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity_level >= 1:
            self.stream.write(f"{Colors.RED}✗ FAIL{Colors.RESET}\n")
            self.stream.flush()
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity_level >= 1:
            self.stream.write(f"{Colors.YELLOW}⊘ SKIP{Colors.RESET} ({reason})\n")
            self.stream.flush()
    
    def startTest(self, test):
        super().startTest(test)
        if self.verbosity_level >= 2:
            self.stream.write(f"{Colors.WHITE}{self.getDescription(test)} ... {Colors.RESET}")
            self.stream.flush()


class ColoredTextTestRunner(unittest.TextTestRunner):
    """Custom test runner with colored output."""
    resultclass = ColoredTextTestResult


def run_all_tests():
    """Discover and run all tests in the tests directory."""
    
    # Enable colored output on Windows
    Colors.enable_windows_colors()
    
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.WHITE}Tail-Risk Hedge Lab - Comprehensive Test Suite{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")
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
    print(f"{Colors.WHITE}Discovered {total_tests} tests across multiple modules{Colors.RESET}")
    print()
    
    # Run tests with colored verbose output
    runner = ColoredTextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print colored summary
    print()
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.WHITE}Test Summary{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    
    successes = result.testsRun - len(result.failures) - len(result.errors)
    
    print(f"{Colors.WHITE}Tests run: {result.testsRun}{Colors.RESET}")
    
    if successes > 0:
        print(f"{Colors.GREEN}Successes: {successes}{Colors.RESET}")
    
    if len(result.failures) > 0:
        print(f"{Colors.RED}Failures: {len(result.failures)}{Colors.RESET}")
    
    if len(result.errors) > 0:
        print(f"{Colors.RED}Errors: {len(result.errors)}{Colors.RESET}")
    
    if len(result.skipped) > 0:
        print(f"{Colors.YELLOW}Skipped: {len(result.skipped)}{Colors.RESET}")
    
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    
    # Print final status with color
    if result.wasSuccessful():
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED{Colors.RESET}\n")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.RESET}\n")
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


def list_available_tests():
    """List all available test modules and test cases."""
    
    # Enable colored output on Windows
    Colors.enable_windows_colors()
    
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.WHITE}Available Test Modules{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")
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
        if os.path.exists(module_path):
            print(f"{Colors.GREEN}✓{Colors.RESET} {Colors.WHITE}{module:25s}{Colors.RESET} - {description}")
        else:
            print(f"{Colors.RED}✗{Colors.RESET} {Colors.WHITE}{module:25s}{Colors.RESET} - {description}")
    
    print()
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")


if __name__ == '__main__':
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['--list', '-l']:
        list_available_tests()
        sys.exit(0)
    
    # Run all tests
    exit_code = run_all_tests()
    sys.exit(exit_code)
