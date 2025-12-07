"""
Statistical hypothesis testing module.

Author: L.Bassetti
"""

from .tests import (
    bootstrap_cvar_test,
    bootstrap_mdd_test,
    safe_haven_regression,
    comprehensive_hypothesis_tests,
    correlation_stability_test
)

__all__ = [
    'HypothesisResult',
    'test_cvar_reduction',
    'test_mdd_reduction',
    'safe_haven_regression',
    'variance_ratio_test'
]
