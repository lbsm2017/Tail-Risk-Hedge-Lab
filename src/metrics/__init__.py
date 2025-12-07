"""
Tail-risk metrics and correlation analysis module.

Author: L.Bassetti
"""

from .tail_risk import (
    cvar,
    max_drawdown,
    downside_deviation,
    sortino_ratio,
    calmar_ratio,
    lower_partial_moment,
    compute_all_metrics
)

from .correlations import (
    conditional_correlation,
    downside_beta,
    correlation_breakdown,
    quantile_correlation
)
from .correlations import (
    conditional_correlation,
    downside_beta,
    rolling_correlation,
    quantile_correlation
)

__all__ = [
    'cvar',
    'max_drawdown',
    'downside_deviation',
    'sortino_ratio',
    'calmar_ratio',
    'lower_partial_moment',
    'conditional_correlation',
    'downside_beta',
    'rolling_correlation',
    'quantile_correlation'
]
