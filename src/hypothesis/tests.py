"""
Statistical Hypothesis Testing Module

Author: L.Bassetti
Implements bootstrap tests and safe-haven regression analysis.
Optimized with vectorized numpy operations and Numba JIT for performance.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import stats
from ..metrics.tail_risk import (
    cvar, max_drawdown, cvar_batch, compute_month_labels
)

# Optional numba import for JIT acceleration
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# =============================================================================
# Vectorized Batch MDD for Bootstrap (Numba + NumPy fallback)
# =============================================================================

@jit(nopython=True, parallel=True, cache=True)
def _mdd_batch_numba(samples_2d: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated batch maximum drawdown computation.
    
    Computes MDD for each row (bootstrap sample) in parallel.
    Each sample is a sequence of returns that we convert to cumulative prices.
    
    Args:
        samples_2d: 2D array of shape (n_bootstrap, n_days) containing returns
        
    Returns:
        1D array of MDD values (positive numbers) for each sample
    """
    n_samples = samples_2d.shape[0]
    n_days = samples_2d.shape[1]
    mdd_values = np.empty(n_samples)
    
    for i in prange(n_samples):
        # Convert returns to cumulative prices
        cum_prices = np.empty(n_days)
        cum_prices[0] = 1.0 + samples_2d[i, 0]
        for j in range(1, n_days):
            cum_prices[j] = cum_prices[j-1] * (1.0 + samples_2d[i, j])
        
        # Calculate running maximum
        running_max = cum_prices[0]
        max_drawdown = 0.0
        
        for j in range(n_days):
            if cum_prices[j] > running_max:
                running_max = cum_prices[j]
            
            drawdown = (cum_prices[j] - running_max) / running_max
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        
        mdd_values[i] = -max_drawdown  # Return as positive value
    
    return mdd_values


def _mdd_batch_numpy(samples_2d: np.ndarray) -> np.ndarray:
    """
    NumPy fallback for batch MDD computation.
    
    Args:
        samples_2d: 2D array of shape (n_bootstrap, n_days) containing returns
        
    Returns:
        1D array of MDD values for each sample
    """
    n_samples = samples_2d.shape[0]
    mdd_values = np.empty(n_samples)
    
    for i in range(n_samples):
        # Convert returns to prices
        prices = (1 + samples_2d[i, :]).cumprod()
        
        # Calculate drawdown
        running_max = np.maximum.accumulate(prices)
        drawdown = (prices - running_max) / running_max
        
        mdd_values[i] = -drawdown.min()
    
    return mdd_values


def mdd_batch(samples_2d: np.ndarray) -> np.ndarray:
    """
    Vectorized batch MDD computation for bootstrap samples.
    
    Uses Numba JIT acceleration when available, falls back to NumPy.
    
    Args:
        samples_2d: 2D array of shape (n_bootstrap, n_days) containing returns
        
    Returns:
        1D array of MDD values (positive numbers) for each sample
    """
    if NUMBA_AVAILABLE:
        return _mdd_batch_numba(samples_2d)
    else:
        return _mdd_batch_numpy(samples_2d)


def bootstrap_cvar_test(
    base_returns: pd.Series,
    hedge_returns: pd.Series,
    hedge_weight: float,
    n_bootstrap: int = 500,  # Default from config.yaml - vectorized for 10-50x speedup
    alpha: float = 0.05,
    confidence_level: float = 0.95,
    cvar_frequency: str = 'monthly'
) -> Dict:
    """
    Bootstrap test for CVaR reduction significance - VECTORIZED.
    
    Uses batch CVaR computation with Numba JIT acceleration for 10-50x speedup.
    All bootstrap samples computed in parallel using matrix operations.
    
    H0: Hedge does not reduce CVaR
    H1: Hedge reduces CVaR
    
    Args:
        base_returns: Base portfolio returns (daily log returns with DatetimeIndex)
        hedge_returns: Hedge asset returns (daily log returns with DatetimeIndex)
        hedge_weight: Weight allocated to hedge
        n_bootstrap: Number of bootstrap samples (can be increased with vectorization)
        alpha: Significance level
        confidence_level: CVaR confidence level
        cvar_frequency: 'daily' or 'monthly' for CVaR computation
        
    Returns:
        Dictionary with test results
    """
    # Align data
    aligned = pd.DataFrame({
        'base': base_returns,
        'hedge': hedge_returns
    }).dropna()
    
    n = len(aligned)
    base_arr = aligned['base'].values
    hedge_arr = aligned['hedge'].values
    
    # Actual portfolio
    portfolio_returns = (1 - hedge_weight) * aligned['base'] + hedge_weight * aligned['hedge']
    portfolio_returns.index = aligned.index
    
    # Calculate actual CVaR reduction
    base_cvar = cvar(aligned['base'], alpha=confidence_level, frequency=cvar_frequency)
    portfolio_cvar = cvar(portfolio_returns, alpha=confidence_level, frequency=cvar_frequency)
    actual_reduction = base_cvar - portfolio_cvar
    
    # === VECTORIZED BOOTSTRAP ===
    # Generate all bootstrap indices at once (n_bootstrap x n matrix)
    rng = np.random.default_rng(42)
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    
    # Create bootstrap samples for base and hedge (n_bootstrap x n matrices)
    boot_base_samples = base_arr[boot_indices]
    boot_hedge_samples = hedge_arr[boot_indices]
    
    # Construct portfolio returns for all bootstraps (vectorized)
    boot_portfolio_samples = (1 - hedge_weight) * boot_base_samples + hedge_weight * boot_hedge_samples
    
    # Pre-compute month labels if using monthly frequency
    if cvar_frequency.lower() == 'monthly':
        month_labels, n_months = compute_month_labels(aligned.index)
        boot_portfolio_cvars = cvar_batch(
            boot_portfolio_samples, 
            alpha=confidence_level,
            frequency='monthly',
            month_labels=month_labels,
            n_months=n_months
        )
    else:
        # Daily frequency
        boot_portfolio_cvars = cvar_batch(
            boot_portfolio_samples,
            alpha=confidence_level,
            frequency='daily'
        )
    
    # Calculate bootstrap reductions
    bootstrap_reductions = base_cvar - boot_portfolio_cvars
    
    # Calculate p-value (one-sided test)
    p_value = np.mean(bootstrap_reductions >= actual_reduction)
    
    # Confidence interval
    ci_lower = np.percentile(bootstrap_reductions, alpha * 100 / 2)
    ci_upper = np.percentile(bootstrap_reductions, 100 - alpha * 100 / 2)
    
    return {
        'test': f'Bootstrap CVaR Reduction ({cvar_frequency.title()})',
        'actual_reduction': actual_reduction,
        'p_value': p_value,
        'significant': p_value < alpha,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_bootstrap': n_bootstrap,
        'base_cvar': base_cvar,
        'portfolio_cvar': portfolio_cvar,
        'method': 'Vectorized (Numba JIT)' if NUMBA_AVAILABLE else 'Vectorized (NumPy)'
    }


def bootstrap_mdd_test(
    base_returns: pd.Series,
    hedge_returns: pd.Series,
    hedge_weight: float,
    n_bootstrap: int = 500,  # Default from config.yaml - vectorized for 10-20x speedup
    alpha: float = 0.05
) -> Dict:
    """
    Bootstrap test for Maximum Drawdown reduction significance - VECTORIZED.
    
    Uses batch MDD computation with Numba JIT acceleration for 10-20x speedup.
    All bootstrap samples computed in parallel using matrix operations.
    
    H0: Hedge does not reduce MDD
    H1: Hedge reduces MDD
    
    Args:
        base_returns: Base portfolio returns
        hedge_returns: Hedge asset returns
        hedge_weight: Weight allocated to hedge
        n_bootstrap: Number of bootstrap samples (can be increased with vectorization)
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    # Align data
    aligned = pd.DataFrame({
        'base': base_returns,
        'hedge': hedge_returns
    }).dropna()
    
    n = len(aligned)
    base_arr = aligned['base'].values
    hedge_arr = aligned['hedge'].values
    
    # Actual portfolio
    portfolio_returns = (1 - hedge_weight) * base_arr + hedge_weight * hedge_arr
    
    # Calculate actual MDD reduction
    base_mdd, _, _ = max_drawdown((1 + aligned['base']).cumprod())
    portfolio_mdd, _, _ = max_drawdown((1 + pd.Series(portfolio_returns)).cumprod())
    actual_reduction = base_mdd - portfolio_mdd
    
    # === VECTORIZED BOOTSTRAP ===
    # Generate all bootstrap indices at once (n_bootstrap x n matrix)
    rng = np.random.default_rng(42)
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    
    # Create bootstrap samples for base and hedge (n_bootstrap x n matrices)
    boot_base_samples = base_arr[boot_indices]
    boot_hedge_samples = hedge_arr[boot_indices]
    
    # Construct portfolio returns for all bootstraps (vectorized)
    boot_portfolio_samples = (1 - hedge_weight) * boot_base_samples + hedge_weight * boot_hedge_samples
    
    # Batch MDD computation (vectorized with Numba/NumPy)
    boot_portfolio_mdds = mdd_batch(boot_portfolio_samples)
    
    # Calculate bootstrap reductions
    bootstrap_reductions = base_mdd - boot_portfolio_mdds
    
    # P-value (one-sided test)
    p_value = np.mean(bootstrap_reductions >= actual_reduction)
    
    # Confidence interval
    ci_lower = np.percentile(bootstrap_reductions, alpha * 100 / 2)
    ci_upper = np.percentile(bootstrap_reductions, 100 - alpha * 100 / 2)
    
    return {
        'test': 'Bootstrap MDD Reduction',
        'actual_reduction': actual_reduction,
        'p_value': p_value,
        'significant': p_value < alpha,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_bootstrap': n_bootstrap,
        'base_mdd': base_mdd,
        'portfolio_mdd': portfolio_mdd,
        'method': 'Vectorized (Numba JIT)' if NUMBA_AVAILABLE else 'Vectorized (NumPy)'
    }


def safe_haven_regression(
    equity_returns: pd.Series,
    hedge_returns: pd.Series,
    regime_labels: pd.Series,
    window: Optional[int] = None
) -> Dict:
    """
    Baur-Lucey safe-haven regression test.
    
    Tests if hedge has negative correlation with equity during crisis.
    
    Regression: hedge_ret = alpha + beta_normal * equity_ret * (1-crisis) 
                           + beta_crisis * equity_ret * crisis + epsilon
    
    Safe haven: beta_crisis < 0 and significantly different from beta_normal
    
    Args:
        equity_returns: Equity return series
        hedge_returns: Hedge asset return series
        regime_labels: Binary series (0=Normal, 1=Crisis)
        window: Optional rolling window (if None, use full sample)
        
    Returns:
        Dictionary with regression results
    """
    # Align data
    aligned = pd.DataFrame({
        'equity': equity_returns,
        'hedge': hedge_returns,
        'regime': regime_labels
    }).dropna()
    
    # Create interaction terms
    aligned['equity_normal'] = aligned['equity'] * (1 - aligned['regime'])
    aligned['equity_crisis'] = aligned['equity'] * aligned['regime']
    
    # Run regression
    from scipy.stats import linregress
    
    # Normal regime beta
    normal_data = aligned[aligned['regime'] == 0]
    if len(normal_data) >= 10:
        slope_normal, intercept_normal, r_normal, p_normal, se_normal = linregress(
            normal_data['equity'], normal_data['hedge']
        )
    else:
        slope_normal = np.nan
        p_normal = np.nan
    
    # Crisis regime beta
    crisis_data = aligned[aligned['regime'] == 1]
    if len(crisis_data) >= 10:
        slope_crisis, intercept_crisis, r_crisis, p_crisis, se_crisis = linregress(
            crisis_data['equity'], crisis_data['hedge']
        )
    else:
        slope_crisis = np.nan
        p_crisis = np.nan
    
    # Test for difference in betas using Chow test approximation
    if not np.isnan(slope_normal) and not np.isnan(slope_crisis):
        # Z-test for difference
        se_diff = np.sqrt(se_normal**2 + se_crisis**2)
        z_stat = (slope_crisis - slope_normal) / se_diff if se_diff > 0 else 0
        p_diff = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Safe haven criteria
        is_safe_haven = (slope_crisis < 0) and (p_diff < 0.05)
    else:
        z_stat = np.nan
        p_diff = np.nan
        is_safe_haven = False
    
    return {
        'test': 'Safe Haven Regression',
        'beta_normal': slope_normal,
        'beta_crisis': slope_crisis,
        'p_value_normal': p_normal,
        'p_value_crisis': p_crisis,
        'beta_difference': slope_crisis - slope_normal if not np.isnan(slope_crisis) else np.nan,
        'p_value_difference': p_diff,
        'is_safe_haven': is_safe_haven,
        'n_normal': len(normal_data),
        'n_crisis': len(crisis_data)
    }


def correlation_stability_test(
    equity_returns: pd.Series,
    hedge_returns: pd.Series,
    regime_labels: pd.Series,
    alpha: float = 0.05
) -> Dict:
    """
    Test if correlation differs significantly between regimes.
    
    Uses Fisher's Z-transformation to test:
    H0: corr_normal = corr_crisis
    H1: corr_normal ≠ corr_crisis
    
    Args:
        equity_returns: Equity return series
        hedge_returns: Hedge asset return series
        regime_labels: Binary series (0=Normal, 1=Crisis)
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    # Align data
    aligned = pd.DataFrame({
        'equity': equity_returns,
        'hedge': hedge_returns,
        'regime': regime_labels
    }).dropna()
    
    # Split by regime
    normal_data = aligned[aligned['regime'] == 0]
    crisis_data = aligned[aligned['regime'] == 1]
    
    if len(normal_data) < 3 or len(crisis_data) < 3:
        return {
            'test': 'Correlation Stability',
            'error': 'Insufficient data for test'
        }
    
    # Calculate correlations
    corr_normal = normal_data['equity'].corr(normal_data['hedge'])
    corr_crisis = crisis_data['equity'].corr(crisis_data['hedge'])
    
    # Fisher Z-transformation
    z_normal = np.arctanh(corr_normal)
    z_crisis = np.arctanh(corr_crisis)
    
    # Standard error of difference
    se_diff = np.sqrt(1/(len(normal_data)-3) + 1/(len(crisis_data)-3))
    
    # Z-statistic
    z_stat = (z_crisis - z_normal) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {
        'test': 'Correlation Stability',
        'corr_normal': corr_normal,
        'corr_crisis': corr_crisis,
        'corr_difference': corr_crisis - corr_normal,
        'z_statistic': z_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'n_normal': len(normal_data),
        'n_crisis': len(crisis_data)
    }


def tail_dependence_test(
    equity_returns: pd.Series,
    hedge_returns: pd.Series,
    quantile: float = 0.05,
    n_bootstrap: int = 500,  # Default from config.yaml
    alpha: float = 0.05
) -> Dict:
    """
    Test for tail dependence using quantile correlation.
    
    H0: No tail dependence (tail correlation = full correlation)
    H1: Significant tail dependence
    
    Args:
        equity_returns: Equity return series
        hedge_returns: Hedge asset return series
        quantile: Quantile threshold for tail
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    # Align data
    aligned = pd.DataFrame({
        'equity': equity_returns,
        'hedge': hedge_returns
    }).dropna()
    
    # Full sample correlation
    full_corr = aligned['equity'].corr(aligned['hedge'])
    
    # Tail correlation
    threshold = aligned['equity'].quantile(quantile)
    tail_data = aligned[aligned['equity'] <= threshold]
    
    if len(tail_data) < 3:
        return {
            'test': 'Tail Dependence',
            'error': 'Insufficient tail data'
        }
    
    tail_corr = tail_data['equity'].corr(tail_data['hedge'])
    
    # Bootstrap to test if tail correlation differs from full correlation
    bootstrap_diffs = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Resample
        boot_sample = aligned.sample(n=len(aligned), replace=True)
        
        # Full correlation
        boot_full_corr = boot_sample['equity'].corr(boot_sample['hedge'])
        
        # Tail correlation
        boot_threshold = boot_sample['equity'].quantile(quantile)
        boot_tail = boot_sample[boot_sample['equity'] <= boot_threshold]
        
        if len(boot_tail) >= 3:
            boot_tail_corr = boot_tail['equity'].corr(boot_tail['hedge'])
            bootstrap_diffs.append(boot_tail_corr - boot_full_corr)
    
    if len(bootstrap_diffs) == 0:
        return {
            'test': 'Tail Dependence',
            'error': 'Bootstrap failed'
        }
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    actual_diff = tail_corr - full_corr
    
    # P-value (two-sided)
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(actual_diff))
    
    return {
        'test': 'Tail Dependence',
        'full_correlation': full_corr,
        'tail_correlation': tail_corr,
        'difference': actual_diff,
        'p_value': p_value,
        'significant': p_value < alpha,
        'n_tail': len(tail_data),
        'n_bootstrap': len(bootstrap_diffs)
    }


def comprehensive_hypothesis_tests(
    base_returns: pd.Series,
    hedge_returns: pd.Series,
    hedge_weight: float,
    regime_labels: Optional[pd.Series] = None,
    n_bootstrap: int = 500,  # Default from config.yaml - passed from backtester
    alpha: float = 0.05,
    cvar_frequency: str = 'monthly'
) -> Dict[str, Dict]:
    """
    Run all hypothesis tests for hedge effectiveness.
    
    Args:
        base_returns: Base portfolio returns
        hedge_returns: Hedge asset returns
        hedge_weight: Weight allocated to hedge
        regime_labels: Optional regime labels
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        
    Returns:
        Dictionary with all test results
    """
    results = {}
    
    # Bootstrap tests
    results['cvar_test'] = bootstrap_cvar_test(
        base_returns, hedge_returns, hedge_weight, n_bootstrap, alpha,
        cvar_frequency=cvar_frequency
    )
    
    results['mdd_test'] = bootstrap_mdd_test(
        base_returns, hedge_returns, hedge_weight, n_bootstrap, alpha
    )
    
    # Regime-based tests (if regime labels provided)
    if regime_labels is not None:
        results['safe_haven'] = safe_haven_regression(
            base_returns, hedge_returns, regime_labels
        )
        
        results['correlation_stability'] = correlation_stability_test(
            base_returns, hedge_returns, regime_labels, alpha
        )
    
    # Tail dependence (use same n_bootstrap for consistency)
    results['tail_dependence'] = tail_dependence_test(
        base_returns, hedge_returns, n_bootstrap=n_bootstrap, alpha=alpha
    )
    
    return results


def portfolio_performance_test(
    base_returns: pd.Series,
    hedged_returns: pd.Series,
    metric: str = 'sharpe',
    n_bootstrap: int = 500,  # Default from config.yaml
    alpha: float = 0.05
) -> Dict:
    """
    Test if hedged portfolio has significantly better performance.
    
    Args:
        base_returns: Base portfolio returns
        hedged_returns: Hedged portfolio returns
        metric: 'sharpe', 'sortino', or 'calmar'
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    from ..metrics.tail_risk import sharpe_ratio, sortino_ratio, calmar_ratio
    
    # Align data
    aligned = pd.DataFrame({
        'base': base_returns,
        'hedged': hedged_returns
    }).dropna()
    
    n = len(aligned)
    
    # Calculate actual metric difference
    if metric == 'sharpe':
        base_metric = sharpe_ratio(aligned['base'])
        hedged_metric = sharpe_ratio(aligned['hedged'])
    elif metric == 'sortino':
        base_metric = sortino_ratio(aligned['base'])
        hedged_metric = sortino_ratio(aligned['hedged'])
    elif metric == 'calmar':
        base_metric = calmar_ratio(aligned['base'])
        hedged_metric = calmar_ratio(aligned['hedged'])
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    actual_diff = hedged_metric - base_metric
    
    # Bootstrap
    bootstrap_diffs = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Resample pairs
        indices = np.random.choice(n, size=n, replace=True)
        boot_base = aligned['base'].iloc[indices].values
        boot_hedged = aligned['hedged'].iloc[indices].values
        
        # Calculate metrics
        if metric == 'sharpe':
            boot_base_metric = sharpe_ratio(pd.Series(boot_base))
            boot_hedged_metric = sharpe_ratio(pd.Series(boot_hedged))
        elif metric == 'sortino':
            boot_base_metric = sortino_ratio(pd.Series(boot_base))
            boot_hedged_metric = sortino_ratio(pd.Series(boot_hedged))
        else:  # calmar
            boot_base_metric = calmar_ratio(pd.Series(boot_base))
            boot_hedged_metric = calmar_ratio(pd.Series(boot_hedged))
        
        bootstrap_diffs.append(boot_hedged_metric - boot_base_metric)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # P-value (one-sided: hedged > base)
    p_value = np.mean(bootstrap_diffs <= 0)
    
    # Confidence interval
    ci_lower = np.percentile(bootstrap_diffs, alpha * 100 / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 - alpha * 100 / 2)
    
    return {
        'test': f'Portfolio Performance ({metric})',
        'base_metric': base_metric,
        'hedged_metric': hedged_metric,
        'improvement': actual_diff,
        'p_value': p_value,
        'significant': p_value < alpha,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_bootstrap': n_bootstrap
    }
