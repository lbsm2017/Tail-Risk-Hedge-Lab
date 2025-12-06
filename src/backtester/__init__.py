"""Main backtesting engine module."""

from .engine import Backtester, quick_backtest
from .rebalancing import (
    simulate_rebalanced_portfolio,
    simulate_single_hedge_rebalanced,
    get_rebalance_dates,
    get_rebalancing_summary
)

__all__ = [
    'Backtester',
    'quick_backtest',
    'simulate_rebalanced_portfolio',
    'simulate_single_hedge_rebalanced',
    'get_rebalance_dates',
    'get_rebalancing_summary'
]
