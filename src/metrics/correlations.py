"""
Correlation Analysis Module

Implements conditional correlations, downside beta, and tail dependence measures.
"""

import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats


def conditional_correlation(
    equity_returns: pd.Series,
    hedge_returns: pd.Series,
    regime_labels: pd.Series,
    regime_value: int = 1
) -> float:
    """
    Calculate Pearson correlation during a specific regime only.
    
    Args:
        equity_returns: Equity return series
        hedge_returns: Hedge asset return series
        regime_labels: Binary series (0=Normal, 1=Crisis)
        regime_value: Regime to condition on (default 1 = Crisis)
        
    Returns:
        Correlation during specified regime
    """
    # Align indices
    aligned = pd.DataFrame({
        'equity': equity_returns,
        'hedge': hedge_returns,
        'regime': regime_labels
    }).dropna()
    
    # Filter to regime
    regime_data = aligned[aligned['regime'] == regime_value]
    
    if len(regime_data) < 2:
        return np.nan
    
    return regime_data['equity'].corr(regime_data['hedge'])


def downside_beta(equity_ret: pd.Series, hedge_ret: pd.Series, threshold: float = 0.0) -> float:
    """
    Calculate beta when equity returns are below threshold.
    
    Beta = Cov(hedge, equity | equity < threshold) / Var(equity | equity < threshold)
    
    Args:
        equity_ret: Equity return series
        hedge_ret: Hedge asset return series
        threshold: Threshold for downside (default 0)
        
    Returns:
        Downside beta
    """
    # Align data
    aligned = pd.DataFrame({
        'equity': equity_ret,
        'hedge': hedge_ret
    }).dropna()
    
    # Filter to downside
    downside = aligned[aligned['equity'] < threshold]
    
    if len(downside) < 2:
        return np.nan
    
    # Calculate beta
    cov = downside['hedge'].cov(downside['equity'])
    var = downside['equity'].var()
    
    if var == 0:
        return np.nan
    
    return cov / var


def rolling_correlation(
    equity_ret: pd.Series,
    hedge_ret: pd.Series,
    window: int = 63
) -> pd.Series:
    """
    Calculate rolling Pearson correlation.
    
    Args:
        equity_ret: Equity return series
        hedge_ret: Hedge asset return series
        window: Rolling window size (e.g., 63 for ~3 months)
        
    Returns:
        Series of rolling correlations
    """
    # Align data
    aligned = pd.DataFrame({
        'equity': equity_ret,
        'hedge': hedge_ret
    }).dropna()
    
    return aligned['equity'].rolling(window).corr(aligned['hedge'])


def quantile_correlation(
    equity_ret: pd.Series,
    hedge_ret: pd.Series,
    quantile: float = 0.05
) -> float:
    """
    Calculate correlation at worst quantile of equity returns.
    
    Focuses on correlation during extreme equity downturns.
    
    Args:
        equity_ret: Equity return series
        hedge_ret: Hedge asset return series
        quantile: Quantile threshold (e.g., 0.05 for worst 5%)
        
    Returns:
        Correlation in tail
    """
    # Align data
    aligned = pd.DataFrame({
        'equity': equity_ret,
        'hedge': hedge_ret
    }).dropna()
    
    # Find quantile threshold
    threshold = aligned['equity'].quantile(quantile)
    
    # Filter to tail
    tail_data = aligned[aligned['equity'] <= threshold]
    
    if len(tail_data) < 2:
        return np.nan
    
    return tail_data['equity'].corr(tail_data['hedge'])


def rank_correlation(
    equity_ret: pd.Series,
    hedge_ret: pd.Series,
    method: str = 'spearman'
) -> float:
    """
    Calculate rank-based correlation (Spearman or Kendall).
    
    Args:
        equity_ret: Equity return series
        hedge_ret: Hedge asset return series
        method: 'spearman' or 'kendall'
        
    Returns:
        Rank correlation
    """
    # Align data
    aligned = pd.DataFrame({
        'equity': equity_ret,
        'hedge': hedge_ret
    }).dropna()
    
    if len(aligned) < 2:
        return np.nan
    
    if method == 'spearman':
        corr, _ = stats.spearmanr(aligned['equity'], aligned['hedge'])
    elif method == 'kendall':
        corr, _ = stats.kendalltau(aligned['equity'], aligned['hedge'])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return corr


def tail_dependence_clayton(equity_ret: np.ndarray, hedge_ret: np.ndarray) -> float:
    """
    Estimate lower tail dependence coefficient using Clayton copula.
    
    Uses method of moments (Kendall's tau inversion).
    
    Args:
        equity_ret: Equity returns
        hedge_ret: Hedge returns
        
    Returns:
        Lower tail dependence coefficient (0 to 1)
    """
    # Remove NaN
    mask = ~(np.isnan(equity_ret) | np.isnan(hedge_ret))
    equity_clean = equity_ret[mask]
    hedge_clean = hedge_ret[mask]
    
    if len(equity_clean) < 2:
        return np.nan
    
    # Calculate Kendall's tau
    tau, _ = stats.kendalltau(equity_clean, hedge_clean)
    
    if tau <= 0:
        # No positive dependence, cannot fit Clayton
        return 0.0
    
    # Estimate Clayton parameter: theta = 2*tau / (1 - tau)
    theta = 2 * tau / (1 - tau)
    
    # Lower tail dependence: lambda_L = 2^(-1/theta)
    if theta > 0:
        lambda_l = 2 ** (-1 / theta)
    else:
        lambda_l = 0.0
    
    return lambda_l


def correlation_matrix_by_regime(
    returns: pd.DataFrame,
    regime_labels: pd.Series
) -> dict:
    """
    Calculate correlation matrices for different regimes.
    
    Args:
        returns: DataFrame of returns for multiple assets
        regime_labels: Binary series (0=Normal, 1=Crisis)
        
    Returns:
        Dictionary with 'normal' and 'crisis' correlation matrices
    """
    # Align data
    aligned = returns.copy()
    aligned['regime'] = regime_labels
    aligned = aligned.dropna()
    
    # Split by regime
    normal_returns = aligned[aligned['regime'] == 0].drop('regime', axis=1)
    crisis_returns = aligned[aligned['regime'] == 1].drop('regime', axis=1)
    
    result = {}
    
    if len(normal_returns) >= 2:
        result['normal'] = normal_returns.corr()
    else:
        result['normal'] = None
    
    if len(crisis_returns) >= 2:
        result['crisis'] = crisis_returns.corr()
    else:
        result['crisis'] = None
    
    return result


def dynamic_correlation(
    equity_ret: pd.Series,
    hedge_ret: pd.Series,
    window: int = 63,
    min_periods: int = 20
) -> pd.DataFrame:
    """
    Calculate time-varying correlation with rolling windows.
    
    Args:
        equity_ret: Equity return series
        hedge_ret: Hedge return series
        window: Rolling window size
        min_periods: Minimum observations for calculation
        
    Returns:
        DataFrame with correlation and confidence bands
    """
    # Align data
    aligned = pd.DataFrame({
        'equity': equity_ret,
        'hedge': hedge_ret
    }).dropna()
    
    # Rolling correlation
    corr = aligned['equity'].rolling(window, min_periods=min_periods).corr(aligned['hedge'])
    
    # Fisher transformation for confidence intervals
    z = np.arctanh(corr)
    se_z = 1 / np.sqrt(window - 3)
    
    # 95% confidence bands
    z_lower = z - 1.96 * se_z
    z_upper = z + 1.96 * se_z
    
    corr_lower = np.tanh(z_lower)
    corr_upper = np.tanh(z_upper)
    
    result = pd.DataFrame({
        'correlation': corr,
        'lower_95': corr_lower,
        'upper_95': corr_upper
    }, index=aligned.index)
    
    return result


def beta(equity_ret: pd.Series, hedge_ret: pd.Series) -> float:
    """
    Calculate standard beta (full sample).
    
    Args:
        equity_ret: Equity return series
        hedge_ret: Hedge return series
        
    Returns:
        Beta coefficient
    """
    # Align data
    aligned = pd.DataFrame({
        'equity': equity_ret,
        'hedge': hedge_ret
    }).dropna()
    
    if len(aligned) < 2:
        return np.nan
    
    cov = aligned['hedge'].cov(aligned['equity'])
    var = aligned['equity'].var()
    
    if var == 0:
        return np.nan
    
    return cov / var


def correlation_breakdown(
    equity_ret: pd.Series,
    hedge_ret: pd.Series,
    regime_labels: Optional[pd.Series] = None
) -> dict:
    """
    Comprehensive correlation analysis.
    
    Args:
        equity_ret: Equity return series
        hedge_ret: Hedge return series
        regime_labels: Optional regime labels
        
    Returns:
        Dictionary with all correlation metrics
    """
    result = {
        'pearson_full': equity_ret.corr(hedge_ret),
        'spearman_full': rank_correlation(equity_ret, hedge_ret, 'spearman'),
        'kendall_full': rank_correlation(equity_ret, hedge_ret, 'kendall'),
        'beta_full': beta(equity_ret, hedge_ret),
        'downside_beta': downside_beta(equity_ret, hedge_ret),
        'quantile_corr_5pct': quantile_correlation(equity_ret, hedge_ret, 0.05),
        'quantile_corr_10pct': quantile_correlation(equity_ret, hedge_ret, 0.10),
        'tail_dependence_lower': tail_dependence_clayton(
            equity_ret.values, hedge_ret.values
        )
    }
    
    if regime_labels is not None:
        result['correlation_normal'] = conditional_correlation(
            equity_ret, hedge_ret, regime_labels, 0
        )
        result['correlation_crisis'] = conditional_correlation(
            equity_ret, hedge_ret, regime_labels, 1
        )
    
    return result
