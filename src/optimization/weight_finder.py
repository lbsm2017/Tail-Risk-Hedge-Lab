"""
Weight Optimization Module

Author: L.Bassetti
Finds optimal hedge weights to achieve target risk reduction levels.
Weights float freely within constraints to hit CVaR/MDD reduction targets.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor
from ..metrics.tail_risk import cvar, max_drawdown, cagr


def compute_portfolio_risk(
    base_returns: pd.Series,
    hedge_returns: pd.Series,
    weight: float,
    metric: str = 'cvar',
    alpha: float = 0.95,
    base_resampled: Optional[pd.Series] = None,
    hedge_resampled: Optional[pd.Series] = None,
    cvar_frequency: str = 'monthly'
) -> float:
    """Compute risk metric for a given hedge weight. Returns positive value for risk.
    
    Args:
        base_returns: Base returns (daily)
        hedge_returns: Hedge returns (daily)
        weight: Hedge weight
        metric: 'cvar' or 'mdd'
        alpha: CVaR confidence level
        base_resampled: Pre-resampled base returns (for performance)
        hedge_resampled: Pre-resampled hedge returns (for performance)
        cvar_frequency: CVaR frequency - 'daily', 'weekly', or 'monthly'
    """
    if metric == 'cvar':
        # Use pre-resampled data if provided (for performance in grid search)
        if base_resampled is not None and hedge_resampled is not None:
            portfolio_resampled = (1 - weight) * base_resampled + weight * hedge_resampled
            # Already resampled, so use daily (no resampling)
            return cvar(portfolio_resampled, alpha=alpha, frequency='daily')
        else:
            # Construct portfolio returns and resample
            portfolio = (1 - weight) * base_returns + weight * hedge_returns
            return cvar(portfolio, alpha=alpha, frequency=cvar_frequency)
    else:  # mdd
        # MDD uses cumulative product of daily returns
        portfolio = (1 - weight) * base_returns + weight * hedge_returns
        cumulative = (1 + portfolio).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        # Return absolute value (positive) 
        return abs(drawdowns.min())


def find_weight_for_target_reduction(
    base_returns: pd.Series,
    hedge_returns: pd.Series,
    target_reduction: float,
    metric: str = 'cvar',
    min_weight: float = 0.0,
    max_weight: float = 0.50,
    weight_step: float = 0.01,
    alpha: float = 0.95,
    tolerance: float = 0.02,
    tie_break_tolerance: float = 0.001,
    **kwargs
) -> Dict:
    """
    Find the MINIMAL hedge weight that achieves the target risk reduction.
    
    Strategy: Among all weights that achieve the target reduction, select the
    smallest weight. This minimizes hedge allocation while meeting risk goals.
    The optimal weight is then maintained through quarterly rebalancing.
    
    Args:
        base_returns: Base portfolio returns (100% ACWI)
        hedge_returns: Hedge asset returns
        target_reduction: Target reduction (0.10 = 10% reduction)
        metric: 'cvar' or 'mdd'
        min_weight: Minimum hedge weight
        max_weight: Maximum hedge weight (determines min base weight)
        weight_step: Weight precision
        alpha: CVaR confidence level
        tolerance: Acceptable deviation from target
        tie_break_tolerance: Risk difference threshold for CAGR tie-breaking
        
    Returns:
        Dict with optimal_weight, achieved_reduction, baseline, hedged, feasible
    """
    # Align data and keep as Series for monthly CVaR calculation
    aligned = pd.DataFrame({
        'base': base_returns,
        'hedge': hedge_returns
    }).dropna()
    
    base_aligned = aligned['base']
    hedge_aligned = aligned['hedge']
    
    # Get CVaR frequency from kwargs or default to monthly
    cvar_frequency = kwargs.get('cvar_frequency', 'monthly')
    
    # Pre-resample once for performance (if metric is CVaR and not daily)
    if metric == 'cvar' and cvar_frequency != 'daily':
        from ..metrics.tail_risk import resample_returns
        base_resampled = resample_returns(base_aligned, frequency=cvar_frequency)
        hedge_resampled = resample_returns(hedge_aligned, frequency=cvar_frequency)
    else:
        base_resampled = None
        hedge_resampled = None
    
    # Calculate baseline risk (100% base asset) - returns positive value
    baseline = compute_portfolio_risk(base_aligned, hedge_aligned, 0.0, metric, alpha, 
                                     base_resampled, hedge_resampled, cvar_frequency)
    
    # Target risk value after reduction
    target_risk = baseline * (1 - target_reduction)
    
    # Grid search to find weight achieving target
    weights = np.arange(min_weight, max_weight + weight_step, weight_step)
    
    best_weight = 0.0
    best_risk = baseline
    best_reduction = 0.0
    
    # First pass: find all weights that meet or exceed target reduction
    # Goal: Find MINIMUM weight that achieves the target
    viable_candidates = []
    
    for w in weights:
        risk = compute_portfolio_risk(base_aligned, hedge_aligned, w, metric, alpha,
                                     base_resampled, hedge_resampled, cvar_frequency)
        
        # Only consider if it reduces risk
        if risk < baseline:
            reduction = (baseline - risk) / baseline
            
            # Check if this meets or exceeds target reduction
            if reduction >= target_reduction - tolerance:
                viable_candidates.append({
                    'weight': w,
                    'risk': risk,
                    'reduction': reduction
                })
    
    # Second pass: among viable candidates, select MINIMUM weight
    # Goal: Find the smallest weight that achieves the target risk reduction
    if viable_candidates:
        # Find the minimum weight among candidates that achieve target
        min_weight_value = min(c['weight'] for c in viable_candidates)
        min_weight_candidates = [c for c in viable_candidates if c['weight'] == min_weight_value]
        
        # Third pass: if multiple solutions at minimum weight exist (edge case), use CAGR tie-breaking
        if len(min_weight_candidates) > 1:
            best_cagr = -np.inf
            best_candidate = None
            
            for candidate in min_weight_candidates:
                w = candidate['weight']
                # Construct portfolio returns
                portfolio_returns = (1 - w) * base_aligned + w * hedge_aligned
                portfolio_cagr = cagr(portfolio_returns.values, periods_per_year=252)
                
                if portfolio_cagr > best_cagr:
                    best_cagr = portfolio_cagr
                    best_candidate = candidate
            
            if best_candidate is not None:
                best_weight = best_candidate['weight']
                best_risk = best_candidate['risk']
                best_reduction = best_candidate['reduction']
        else:
            # Only one minimum weight candidate - use it
            best_weight = min_weight_candidates[0]['weight']
            best_risk = min_weight_candidates[0]['risk']
            best_reduction = min_weight_candidates[0]['reduction']
    else:
        # No viable candidates found - target is not achievable
        # Return the best effort (maximum achievable reduction)
        all_results = []
        for w in weights:
            risk = compute_portfolio_risk(base_aligned, hedge_aligned, w, metric, alpha,
                                         base_resampled, hedge_resampled, cvar_frequency)
            if risk < baseline:
                reduction = (baseline - risk) / baseline
                all_results.append({
                    'weight': w,
                    'risk': risk,
                    'reduction': reduction
                })
        
        if all_results:
            # Find maximum achievable reduction at max weight
            max_reduction_result = max(all_results, key=lambda x: x['reduction'])
            best_weight = max_reduction_result['weight']
            best_risk = max_reduction_result['risk']
            best_reduction = max_reduction_result['reduction']
    
    # Determine feasibility
    feasible = best_reduction >= target_reduction - tolerance
    at_max_weight = best_weight >= max_weight - weight_step
    
    return {
        'target_reduction': target_reduction,
        'optimal_weight': best_weight,
        'achieved_reduction': best_reduction,
        'baseline_risk': baseline,
        'hedged_risk': best_risk,
        'feasible': feasible,
        'at_constraint': at_max_weight and not feasible,
        'metric': metric
    }


def find_weights_for_all_targets(
    base_returns: pd.Series,
    hedge_returns: pd.Series,
    targets: List[float] = [0.10, 0.25, 0.50],
    metrics: List[str] = ['cvar', 'mdd'],
    min_weight: float = 0.0,
    max_weight: float = 0.50,
    weight_step: float = 0.01,
    alpha: float = 0.95,
    tolerance: float = 0.02,
    **kwargs
) -> pd.DataFrame:
    """
    Find optimal weights for multiple target reductions.
    Parallelized for performance. Includes efficiency metric calculation.
    
    Args:
        base_returns: Base portfolio returns
        hedge_returns: Hedge asset returns
        targets: List of target reductions [0.10, 0.25, 0.50]
        metrics: List of metrics ['cvar', 'mdd']
        min_weight: Minimum hedge weight
        max_weight: Maximum hedge weight
        weight_step: Weight precision
        alpha: CVaR confidence level
        tolerance: Acceptable deviation from target
        
    Returns:
        DataFrame with results for each (target, metric) combination.
        Includes 'efficiency' column: risk reduction per unit weight.
    """
    def optimize_single(args):
        metric, target = args
        return find_weight_for_target_reduction(
            base_returns=base_returns,
            hedge_returns=hedge_returns,
            target_reduction=target,
            metric=metric,
            min_weight=min_weight,
            max_weight=max_weight,
            weight_step=weight_step,
            alpha=alpha,
            tolerance=tolerance,
            **kwargs
        )
    
    # All combinations
    combinations = [(m, t) for m in metrics for t in targets]
    
    # Parallel execution
    results = []
    with ThreadPoolExecutor(max_workers=len(combinations)) as executor:
        futures = [executor.submit(optimize_single, c) for c in combinations]
        for future in futures:
            results.append(future.result())
    
    # Convert to DataFrame and add efficiency metric
    df = pd.DataFrame(results)
    
    # Efficiency = risk reduction per unit weight
    # Higher efficiency = more risk reduction per 1% allocation
    # Zero efficiency when no weight allocated OR no reduction achieved
    df['efficiency'] = df.apply(
        lambda row: row['achieved_reduction'] / row['optimal_weight'] 
        if row['optimal_weight'] > 0 and row['achieved_reduction'] > 0 else 0.0,
        axis=1
    )
    
    return df


# Backwards compatibility alias
def optimize_for_multiple_targets(
    base_returns: pd.Series,
    hedge_returns: pd.Series,
    targets: list = [0.10, 0.25, 0.50],
    metrics: list = ['cvar', 'mdd'],
    max_weight: float = 0.50,
    weight_step: float = 0.01,
    alpha: float = 0.95,
    **kwargs
) -> pd.DataFrame:
    """Backwards compatible wrapper."""
    return find_weights_for_all_targets(
        base_returns=base_returns,
        hedge_returns=hedge_returns,
        targets=targets,
        metrics=metrics,
        min_weight=0.0,
        max_weight=max_weight,
        weight_step=weight_step,
        alpha=alpha,
        **kwargs
    )


def efficient_frontier(
    base_returns: pd.Series,
    hedge_returns: pd.Series,
    max_weight: float = 0.50,
    n_points: int = 50
) -> pd.DataFrame:
    """
    Calculate efficient frontier for base + hedge portfolio.
    
    Args:
        base_returns: Base portfolio returns
        hedge_returns: Hedge asset returns
        max_weight: Maximum hedge weight
        n_points: Number of points on frontier
        
    Returns:
        DataFrame with return, volatility, and Sharpe ratio for each weight
    """
    weights = np.linspace(0, max_weight, n_points)
    results = []
    
    for w in weights:
        # Construct portfolio
        hedged_returns = (1 - w) * base_returns + w * hedge_returns
        
        # Calculate metrics
        mean_ret = hedged_returns.mean() * 252  # Annualized
        vol = hedged_returns.std() * np.sqrt(252)  # Annualized
        sharpe = mean_ret / vol if vol > 0 else 0
        
        results.append({
            'hedge_weight': w,
            'annual_return': mean_ret,
            'annual_volatility': vol,
            'sharpe_ratio': sharpe
        })
    
    return pd.DataFrame(results)


def risk_parity_weight(
    base_returns: pd.Series,
    hedge_returns: pd.Series,
    target_base_contribution: float = 0.50
) -> float:
    """
    Calculate weight for risk parity between base and hedge.
    
    Weight is chosen so base contributes target_base_contribution to total risk.
    
    Args:
        base_returns: Base portfolio returns
        hedge_returns: Hedge asset returns
        target_base_contribution: Target risk contribution from base (0 to 1)
        
    Returns:
        Optimal hedge weight
    """
    # Align returns
    aligned = pd.DataFrame({
        'base': base_returns,
        'hedge': hedge_returns
    }).dropna()
    
    # Calculate covariance matrix
    cov_matrix = aligned.cov()
    
    vol_base = aligned['base'].std()
    vol_hedge = aligned['hedge'].std()
    corr = aligned['base'].corr(aligned['hedge'])
    
    # For a two-asset portfolio: w_base + w_hedge = 1
    # Risk contribution from base: w_base * (w_base * vol_base^2 + w_hedge * cov_base_hedge) / portfolio_vol
    # 
    # Simplified approach: use inverse volatility weighting
    inv_vol_base = 1 / vol_base if vol_base > 0 else 0
    inv_vol_hedge = 1 / vol_hedge if vol_hedge > 0 else 0
    
    total_inv_vol = inv_vol_base + inv_vol_hedge
    
    if total_inv_vol == 0:
        return 0.0
    
    # Weight proportional to inverse volatility
    w_base = inv_vol_base / total_inv_vol
    w_hedge = inv_vol_hedge / total_inv_vol
    
    # Adjust to target contribution
    adjusted_w_base = w_base * (target_base_contribution / 0.5)  # Assuming 0.5 is default parity
    adjusted_w_hedge = 1 - adjusted_w_base
    
    # Ensure within bounds
    hedge_weight = max(0, min(1, adjusted_w_hedge))
    
    return hedge_weight
