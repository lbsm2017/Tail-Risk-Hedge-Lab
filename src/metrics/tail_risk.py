"""
Tail-Risk Metrics Module

Author: L.Bassetti
Implements various tail-risk measures including CVaR, Maximum Drawdown,
Downside Deviation, and other risk-adjusted performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
from datetime import datetime


def resample_returns(returns: Union[pd.Series, np.ndarray], frequency: str = 'monthly') -> pd.Series:
    """
    Resample daily log returns to specified frequency.
    
    For log returns, aggregated returns are the sum of daily log returns.
    This preserves the mathematical properties of log returns.
    
    Args:
        returns: Daily log returns (Series with DatetimeIndex)
        frequency: Target frequency - 'daily', 'weekly', or 'monthly'
        
    Returns:
        Resampled log returns as pandas Series
    """
    if isinstance(returns, np.ndarray):
        raise ValueError("Array input not supported for resampling. Use pandas Series with DatetimeIndex.")
    
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns must have a DatetimeIndex for resampling.")
    
    frequency = frequency.lower()
    
    if frequency == 'daily':
        # No resampling needed
        return returns
    elif frequency == 'weekly':
        # Sum log returns within each week (Friday close)
        return returns.resample('W-FRI').sum()
    elif frequency == 'monthly':
        # Sum log returns within each month
        return returns.resample('M').sum()
    else:
        raise ValueError(f"Invalid frequency '{frequency}'. Must be 'daily', 'weekly', or 'monthly'.")


def resample_to_monthly(returns: Union[pd.Series, np.ndarray]) -> pd.Series:
    """
    Backwards compatibility wrapper for resample_returns with monthly frequency.
    
    Args:
        returns: Daily log returns (Series with DatetimeIndex)
        
    Returns:
        Monthly log returns as pandas Series
    """
    return resample_returns(returns, frequency='monthly')


def cvar(returns: Union[pd.Series, np.ndarray], alpha: float = 0.95, 
         frequency: str = 'monthly') -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    CVaR is the expected loss given that the loss exceeds VaR.
    
    Args:
        returns: Array or Series of returns (log returns if resampling from daily)
        alpha: Confidence level (e.g., 0.95 for 95%)
        frequency: Return frequency for CVaR calculation - 'daily', 'weekly', or 'monthly'
                  If not 'daily', resamples daily returns before computing CVaR
        
    Returns:
        CVaR value (positive number representing expected loss)
    """
    # Handle resampling for pandas Series (if frequency is not daily)
    if frequency and frequency.lower() != 'daily':
        if isinstance(returns, pd.Series):
            if isinstance(returns.index, pd.DatetimeIndex):
                returns = resample_returns(returns, frequency=frequency)
                returns = returns.values  # Convert to array for calculation
            else:
                raise ValueError(f"CVaR with frequency '{frequency}' requires DatetimeIndex. Use frequency='daily' for array input.")
        else:
            raise ValueError(f"CVaR with frequency '{frequency}' requires pandas Series with DatetimeIndex. Use frequency='daily' for array input.")
    
    # Convert to numpy array if needed
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if len(returns) == 0:
        return np.nan
    
    # Calculate VaR threshold
    var_threshold = np.percentile(returns, (1 - alpha) * 100)
    
    # Calculate CVaR as mean of returns below VaR
    tail_returns = returns[returns <= var_threshold]
    
    if len(tail_returns) == 0:
        return np.nan
    
    return -tail_returns.mean()  # Return as positive value


def var(returns: np.ndarray, alpha: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Array of returns
        alpha: Confidence level
        
    Returns:
        VaR value (positive number)
    """
    if len(returns) == 0:
        return np.nan
    
    return -np.percentile(returns, (1 - alpha) * 100)


def max_drawdown(prices: pd.Series) -> Tuple[float, Optional[datetime], Optional[datetime]]:
    """
    Calculate maximum drawdown and identify peak/trough dates.
    
    Args:
        prices: Series of prices (with datetime index)
        
    Returns:
        Tuple of (max_drawdown, peak_date, trough_date)
        max_drawdown is returned as positive number (e.g., 0.50 for 50% drawdown)
    """
    if len(prices) == 0:
        return np.nan, None, None
    
    # Calculate running maximum
    running_max = prices.expanding().max()
    
    # Calculate drawdown
    drawdown = (prices - running_max) / running_max
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    
    if pd.isna(max_dd):
        return np.nan, None, None
    
    # Find trough date
    trough_date = drawdown.idxmin()
    
    # Find peak date (last maximum before trough)
    peak_date = running_max.loc[:trough_date].idxmax()
    
    return -max_dd, peak_date, trough_date  # Return as positive value


def drawdown_series(prices: pd.Series) -> pd.Series:
    """
    Calculate drawdown series over time.
    
    Args:
        prices: Series of prices
        
    Returns:
        Series of drawdowns (negative values)
    """
    running_max = prices.expanding().max()
    drawdown = (prices - running_max) / running_max
    return drawdown


def downside_deviation(returns: np.ndarray, mar: float = 0.0) -> float:
    """
    Calculate downside deviation (semi-standard deviation).
    
    Only considers returns below the Minimum Acceptable Return (MAR).
    
    Args:
        returns: Array of returns
        mar: Minimum Acceptable Return threshold
        
    Returns:
        Downside deviation
    """
    if len(returns) == 0:
        return np.nan
    
    # Select returns below MAR
    downside_returns = returns[returns < mar]
    
    if len(downside_returns) == 0:
        return 0.0
    
    # Calculate semi-variance
    downside_diff = downside_returns - mar
    downside_var = np.mean(downside_diff ** 2)
    
    return np.sqrt(downside_var)


def sortino_ratio(returns: np.ndarray, mar: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate Sortino Ratio.
    
    Risk-adjusted return using downside deviation instead of total volatility.
    
    Args:
        returns: Array of returns
        mar: Minimum Acceptable Return
        periods_per_year: Number of periods per year (252 for daily, 52 for weekly)
        
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return np.nan
    
    # Annualized excess return
    mean_return = np.mean(returns)
    excess_return = (mean_return - mar) * periods_per_year
    
    # Annualized downside deviation
    dd = downside_deviation(returns, mar)
    ann_dd = dd * np.sqrt(periods_per_year)
    
    if ann_dd == 0:
        return np.nan
    
    return excess_return / ann_dd


def calmar_ratio(returns: np.ndarray, prices: Optional[pd.Series] = None) -> float:
    """
    Calculate Calmar Ratio (CAGR / Maximum Drawdown).
    
    Args:
        returns: Array of returns
        prices: Optional price series for MDD calculation (if not provided, computed from returns)
        
    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return np.nan
    
    # Calculate CAGR
    total_return = np.exp(np.sum(returns)) - 1
    n_years = len(returns) / 252  # Assume daily returns
    cagr = (1 + total_return) ** (1 / n_years) - 1
    
    # Calculate MDD
    if prices is not None:
        mdd, _, _ = max_drawdown(prices)
    else:
        # Reconstruct prices from returns
        cum_returns = np.exp(np.cumsum(returns))
        prices_reconstructed = pd.Series(cum_returns * 100)
        mdd, _, _ = max_drawdown(prices_reconstructed)
    
    if mdd == 0:
        return np.nan
    
    return cagr / mdd


def lower_partial_moment(returns: np.ndarray, threshold: float, order: int) -> float:
    """
    Calculate Lower Partial Moment (LPM).
    
    Args:
        returns: Array of returns
        threshold: Threshold return (e.g., 0 for target return)
        order: Order of moment (0, 1, or 2)
            0: Probability of underperformance
            1: Expected shortfall below threshold
            2: Semi-variance
            
    Returns:
        LPM value
    """
    if len(returns) == 0:
        return np.nan
    
    # Calculate deviations below threshold
    deviations = np.maximum(threshold - returns, 0)
    
    # Calculate moment
    lpm = np.mean(deviations ** order)
    
    return lpm


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Calculate Omega Ratio.
    
    Ratio of probability-weighted gains to losses relative to threshold.
    
    Args:
        returns: Array of returns
        threshold: Threshold return
        
    Returns:
        Omega ratio
    """
    if len(returns) == 0:
        return np.nan
    
    # Returns above threshold
    gains = np.maximum(returns - threshold, 0)
    
    # Returns below threshold
    losses = np.maximum(threshold - returns, 0)
    
    total_gains = np.sum(gains)
    total_losses = np.sum(losses)
    
    if total_losses == 0:
        return np.inf if total_gains > 0 else 1.0
    
    return total_gains / total_losses


def sharpe_ratio(returns: np.ndarray, rf_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe Ratio.
    
    Args:
        returns: Array of returns
        rf_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return np.nan
    
    excess_returns = returns - (rf_rate / periods_per_year)
    
    if np.std(excess_returns) == 0:
        return np.nan
    
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


def cagr(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized return (CAGR)
    """
    if len(returns) == 0:
        return np.nan
    
    total_return = np.exp(np.sum(returns)) - 1
    n_years = len(returns) / periods_per_year
    
    if n_years == 0:
        return np.nan
    
    return (1 + total_return) ** (1 / n_years) - 1


def annualized_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility.
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized volatility
    """
    if len(returns) == 0:
        return np.nan
    
    return np.std(returns) * np.sqrt(periods_per_year)


def tail_ratio(returns: np.ndarray, percentile: float = 95) -> float:
    """
    Calculate tail ratio (right tail / left tail).
    
    Measures asymmetry in return distribution tails.
    
    Args:
        returns: Array of returns
        percentile: Percentile for tail definition
        
    Returns:
        Tail ratio (>1 indicates positive skew)
    """
    if len(returns) == 0:
        return np.nan
    
    right_tail = np.abs(np.percentile(returns, percentile))
    left_tail = np.abs(np.percentile(returns, 100 - percentile))
    
    if left_tail == 0:
        return np.nan
    
    return right_tail / left_tail


def ulcer_index(prices: pd.Series) -> float:
    """
    Calculate Ulcer Index.
    
    Measures the depth and duration of drawdowns.
    
    Args:
        prices: Series of prices
        
    Returns:
        Ulcer Index
    """
    if len(prices) == 0:
        return np.nan
    
    dd = drawdown_series(prices)
    squared_dd = dd ** 2
    
    return np.sqrt(np.mean(squared_dd))


def recovery_time(prices: pd.Series) -> pd.Series:
    """
    Calculate time to recover from each drawdown.
    
    Args:
        prices: Series of prices
        
    Returns:
        Series indicating days since last peak
    """
    running_max = prices.expanding().max()
    is_at_peak = (prices == running_max)
    
    # Count days since last peak
    days_since_peak = pd.Series(0, index=prices.index)
    days = 0
    
    for i, at_peak in enumerate(is_at_peak):
        if at_peak:
            days = 0
        else:
            days += 1
        days_since_peak.iloc[i] = days
    
    return days_since_peak


def compute_all_metrics(returns: pd.Series, prices: Optional[pd.Series] = None,
                        rf_rate: float = 0.0, cvar_frequency: str = 'monthly') -> dict:
    """
    Compute all tail-risk and performance metrics.
    
    Args:
        returns: Series of returns
        prices: Optional series of prices
        rf_rate: Risk-free rate
        
    Returns:
        Dictionary of all metrics
    """
    ret_array = returns.values
    
    metrics = {
        'cagr': cagr(ret_array),
        'volatility': annualized_volatility(ret_array),
        'sharpe': sharpe_ratio(ret_array, rf_rate),
        'sortino': sortino_ratio(ret_array),
        'var_95': var(ret_array, 0.95),
        'cvar_95': cvar(returns, 0.95, frequency=cvar_frequency),  # Pass Series for resampling
        'var_99': var(ret_array, 0.99),
        'cvar_99': cvar(returns, 0.99, frequency=cvar_frequency),  # Pass Series for resampling
        'downside_dev': downside_deviation(ret_array),
        'omega': omega_ratio(ret_array),
        'tail_ratio': tail_ratio(ret_array),
        'skewness': pd.Series(ret_array).skew(),
        'kurtosis': pd.Series(ret_array).kurtosis(),
    }
    
    # Metrics requiring prices
    if prices is not None:
        mdd, peak, trough = max_drawdown(prices)
        metrics['max_drawdown'] = mdd
        metrics['mdd_peak_date'] = peak
        metrics['mdd_trough_date'] = trough
        metrics['calmar'] = calmar_ratio(ret_array, prices)
        metrics['ulcer_index'] = ulcer_index(prices)
    else:
        metrics['max_drawdown'] = np.nan
        metrics['mdd_peak_date'] = None
        metrics['mdd_trough_date'] = None
        metrics['calmar'] = calmar_ratio(ret_array, None)
        metrics['ulcer_index'] = np.nan
    
    return metrics
