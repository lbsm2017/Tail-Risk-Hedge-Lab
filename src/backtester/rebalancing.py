"""
Portfolio Rebalancing Module

Author: L.Bassetti
Implements periodic rebalancing to maintain target weights.
Simulates realistic portfolio behavior with quarterly (or other frequency) resets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def get_rebalance_dates(
    dates: pd.DatetimeIndex,
    frequency: str = "quarterly"
) -> List[pd.Timestamp]:
    """
    Get dates when rebalancing should occur.
    
    Args:
        dates: DatetimeIndex of trading days
        frequency: 'daily', 'weekly', 'monthly', 'quarterly', 'annual'
        
    Returns:
        List of rebalancing dates (last trading day of each period)
    """
    if len(dates) == 0:
        return []
    
    dates_series = pd.Series(dates, index=dates)
    
    if frequency == "daily":
        return list(dates)
    elif frequency == "weekly":
        # Last trading day of each week
        return list(dates_series.groupby(dates_series.index.to_period('W')).last())
    elif frequency == "monthly":
        # Last trading day of each month
        return list(dates_series.groupby(dates_series.index.to_period('M')).last())
    elif frequency == "quarterly":
        # Last trading day of each quarter
        return list(dates_series.groupby(dates_series.index.to_period('Q')).last())
    elif frequency == "annual":
        # Last trading day of each year
        return list(dates_series.groupby(dates_series.index.to_period('Y')).last())
    else:
        raise ValueError(f"Unknown frequency: {frequency}. Use: daily, weekly, monthly, quarterly, annual")


def simulate_rebalanced_portfolio(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    rebalance_frequency: str = "quarterly",
    initial_value: float = 1.0,
    trading_fee_bps: float = 0.0
) -> pd.DataFrame:
    """
    Simulate portfolio with periodic rebalancing to target weights.
    
    Between rebalance dates, weights drift based on relative asset performance.
    On rebalance dates, weights are reset to target allocations.
    Trading fees are applied on rebalance days based on turnover.
    
    Args:
        returns: DataFrame with asset returns (columns = asset tickers)
        weights: Target weights dict {'ACWI': 0.75, 'TLT': 0.15, 'GLD': 0.10}
        rebalance_frequency: 'daily', 'weekly', 'monthly', 'quarterly', 'annual'
        initial_value: Starting portfolio value (default 1.0)
        trading_fee_bps: Trading cost in basis points per turnover (default 0.0)
                         Fee is applied to one-way turnover on each rebalance.
                         First day is fee-free (initial portfolio construction).
        
    Returns:
        DataFrame with columns:
        - portfolio_value: Cumulative portfolio value
        - portfolio_return: Daily portfolio returns
        - rebalance_flag: 1 on rebalance days, 0 otherwise
        - drift: Total weight drift from target before rebalance
    """
    # Validate weights
    assets = list(weights.keys())
    missing = set(assets) - set(returns.columns)
    if missing:
        raise ValueError(f"Assets not in returns DataFrame: {missing}")
    
    # Use only relevant assets
    returns = returns[assets].copy()
    
    # Align and drop NaN
    returns = returns.dropna()
    
    if len(returns) == 0:
        raise ValueError("No valid data after alignment")
    
    # Get rebalance dates
    rebalance_dates = set(get_rebalance_dates(returns.index, rebalance_frequency))
    
    # Initialize tracking arrays
    n_days = len(returns)
    portfolio_values = np.zeros(n_days)
    portfolio_returns = np.zeros(n_days)
    rebalance_flags = np.zeros(n_days, dtype=int)
    drifts = np.zeros(n_days)
    
    # Initialize current weights and portfolio value
    current_weights = np.array([weights[asset] for asset in assets])
    target_weights = current_weights.copy()
    portfolio_value = initial_value
    
    # Trading fee as decimal (10 bps = 0.0010)
    fee_rate = trading_fee_bps / 10000.0
    
    # Track if first rebalance (no fee on initial construction)
    is_first_rebalance = True
    
    # Simulate day by day
    for i, (date, row) in enumerate(returns.iterrows()):
        asset_returns = row.values
        
        # Check if rebalance day (before applying today's returns)
        is_rebalance = date in rebalance_dates
        rebalance_flags[i] = 1 if is_rebalance else 0
        
        # Calculate drift before potential rebalance
        drifts[i] = calculate_drift(
            dict(zip(assets, current_weights)),
            dict(zip(assets, target_weights))
        )
        
        # Rebalance if needed
        if is_rebalance:
            # Apply trading fee (skip first rebalance - initial portfolio construction)
            if not is_first_rebalance and fee_rate > 0:
                # One-way turnover = sum of absolute weight changes / 2
                turnover = np.sum(np.abs(current_weights - target_weights)) / 2.0
                fee_cost = turnover * fee_rate * portfolio_value
                portfolio_value -= fee_cost
            
            is_first_rebalance = False
            current_weights = target_weights.copy()
        
        # Calculate portfolio return for the day
        daily_return = np.sum(current_weights * asset_returns)
        portfolio_returns[i] = daily_return
        
        # Update portfolio value
        portfolio_value *= (1 + daily_return)
        portfolio_values[i] = portfolio_value
        
        # Update weights based on relative performance (drift)
        # New weight = old_weight * (1 + asset_return) / (1 + portfolio_return)
        if abs(daily_return + 1) > 1e-10:  # Avoid division by zero
            weight_factors = (1 + asset_returns) / (1 + daily_return)
            current_weights = current_weights * weight_factors
            # Normalize to ensure sum = 1 (handles numerical errors)
            current_weights = current_weights / current_weights.sum()
    
    # Build result DataFrame
    result = pd.DataFrame({
        'portfolio_value': portfolio_values,
        'portfolio_return': portfolio_returns,
        'rebalance_flag': rebalance_flags,
        'drift': drifts
    }, index=returns.index)
    
    return result


def calculate_drift(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float]
) -> float:
    """
    Calculate total absolute weight drift from target.
    
    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        
    Returns:
        Sum of absolute weight differences (0 = perfect alignment)
    """
    total_drift = 0.0
    for asset in target_weights:
        current = current_weights.get(asset, 0.0)
        target = target_weights[asset]
        total_drift += abs(current - target)
    return total_drift


def get_rebalancing_summary(simulation_result: pd.DataFrame) -> Dict:
    """
    Summarize rebalancing activity from simulation results.
    
    Args:
        simulation_result: DataFrame from simulate_rebalanced_portfolio()
        
    Returns:
        Dict with rebalancing statistics
    """
    rebalance_days = simulation_result[simulation_result['rebalance_flag'] == 1]
    
    return {
        'n_rebalances': len(rebalance_days),
        'rebalance_dates': list(rebalance_days.index),
        'avg_drift_at_rebalance': rebalance_days['drift'].mean() if len(rebalance_days) > 0 else 0.0,
        'max_drift_at_rebalance': rebalance_days['drift'].max() if len(rebalance_days) > 0 else 0.0,
        'total_return': simulation_result['portfolio_value'].iloc[-1] - 1.0,
        'annualized_return': (simulation_result['portfolio_value'].iloc[-1] ** (252 / len(simulation_result))) - 1
    }


def simulate_single_hedge_rebalanced(
    base_returns: pd.Series,
    hedge_returns: pd.Series,
    hedge_weight: float,
    rebalance_frequency: str = "quarterly",
    trading_fee_bps: float = 0.0
) -> pd.DataFrame:
    """
    Convenience function for single-hedge portfolio simulation.
    
    Args:
        base_returns: Base asset (ACWI) returns
        hedge_returns: Hedge asset returns
        hedge_weight: Weight allocated to hedge (base gets 1 - hedge_weight)
        rebalance_frequency: Rebalancing frequency
        trading_fee_bps: Trading cost in basis points per turnover (default 0.0)
        
    Returns:
        Simulation result DataFrame
    """
    # Combine into DataFrame
    returns = pd.DataFrame({
        'base': base_returns,
        'hedge': hedge_returns
    }).dropna()
    
    weights = {
        'base': 1 - hedge_weight,
        'hedge': hedge_weight
    }
    
    return simulate_rebalanced_portfolio(
        returns=returns,
        weights=weights,
        rebalance_frequency=rebalance_frequency,
        trading_fee_bps=trading_fee_bps
    )


def get_asset_inception_dates(returns: pd.DataFrame) -> Dict[str, pd.Timestamp]:
    """
    Get the first date each asset has valid data.
    
    Args:
        returns: DataFrame with asset returns
        
    Returns:
        Dict mapping asset ticker to first available date
    """
    inception = {}
    for col in returns.columns:
        first_valid = returns[col].first_valid_index()
        if first_valid is not None:
            inception[col] = first_valid
    return inception


def get_portfolio_data_window(
    returns: pd.DataFrame,
    weights: Dict[str, float]
) -> Tuple[pd.Timestamp, pd.Timestamp, Dict[str, pd.Timestamp]]:
    """
    Get the date range where ALL assets in portfolio are available.
    
    Args:
        returns: DataFrame with all asset returns
        weights: Portfolio weights dict (keys are asset tickers, values are weights)
        
    Returns:
        Tuple of (start_date, end_date, asset_inception_dates)
        where start_date is when last asset became available (intersection point)
    """
    portfolio_assets = [a for a, w in weights.items() if w > 0 and a in returns.columns]
    
    if not portfolio_assets:
        raise ValueError("No valid assets in portfolio")
    
    # Get inception date for each asset in portfolio
    inceptions = {}
    for asset in portfolio_assets:
        first_valid = returns[asset].first_valid_index()
        if first_valid is not None:
            inceptions[asset] = first_valid
    
    if not inceptions:
        raise ValueError("No valid data for any portfolio assets")
    
    # Start date is when the last asset became available (intersection)
    start_date = max(inceptions.values())
    # End date is the last date in the return series
    end_date = returns.index[-1]
    
    return start_date, end_date, inceptions
