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
from ..metrics.tail_risk import cvar, max_drawdown


def compute_portfolio_risk(
    base_returns: np.ndarray,
    hedge_returns: np.ndarray,
    weight: float,
    metric: str = 'cvar',
    alpha: float = 0.95
) -> float:
    """Compute risk metric for a given hedge weight. Returns positive value for risk."""
    portfolio = (1 - weight) * base_returns + weight * hedge_returns
    
    if metric == 'cvar':
        quantile = np.percentile(portfolio, (1 - alpha) * 100)
        # Return absolute value (positive) for easier comparison
        return abs(portfolio[portfolio <= quantile].mean())
    else:  # mdd
        cumulative = np.cumprod(1 + portfolio)
        running_max = np.maximum.accumulate(cumulative)
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
    tolerance: float = 0.02
) -> Dict:
    """
    Find hedge weight that achieves exactly the target risk reduction.
    
    The weight floats between min and max to find the value that 
    produces the desired reduction in CVaR or MDD.
    
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
        
    Returns:
        Dict with optimal_weight, achieved_reduction, baseline, hedged, feasible
    """
    base_arr = base_returns.values
    hedge_arr = hedge_returns.values
    
    # Calculate baseline risk (100% base asset) - returns positive value
    baseline = compute_portfolio_risk(base_arr, hedge_arr, 0.0, metric, alpha)
    
    # Target risk value after reduction
    target_risk = baseline * (1 - target_reduction)
    
    # Grid search to find weight achieving target
    weights = np.arange(min_weight, max_weight + weight_step, weight_step)
    
    best_weight = 0.0
    best_risk = baseline
    best_reduction = 0.0
    
    # First pass: find weight that gets closest to target reduction
    # while actually reducing risk
    for w in weights:
        risk = compute_portfolio_risk(base_arr, hedge_arr, w, metric, alpha)
        
        # Only consider if it reduces risk
        if risk < baseline:
            reduction = (baseline - risk) / baseline
            
            # Check if this achieves target (or gets closer)
            if reduction >= target_reduction - tolerance:
                # Found a valid solution
                if best_reduction < target_reduction or reduction < best_reduction:
                    # Either first valid solution or closer to target
                    best_weight = w
                    best_risk = risk
                    best_reduction = reduction
                    if abs(reduction - target_reduction) <= tolerance:
                        break  # Close enough to target
            elif reduction > best_reduction:
                # Better than current best, even if not at target
                best_weight = w
                best_risk = risk
                best_reduction = reduction
    
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
    tolerance: float = 0.02
) -> pd.DataFrame:
    """
    Find optimal weights for multiple target reductions.
    Parallelized for performance.
    
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
        DataFrame with results for each (target, metric) combination
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
            tolerance=tolerance
        )
    
    # All combinations
    combinations = [(m, t) for m in metrics for t in targets]
    
    # Parallel execution
    results = []
    with ThreadPoolExecutor(max_workers=len(combinations)) as executor:
        futures = [executor.submit(optimize_single, c) for c in combinations]
        for future in futures:
            results.append(future.result())
    
    return pd.DataFrame(results)


# Backwards compatibility alias
def optimize_for_multiple_targets(
    base_returns: pd.Series,
    hedge_returns: pd.Series,
    targets: list = [0.10, 0.25, 0.50],
    metrics: list = ['cvar', 'mdd'],
    max_weight: float = 0.50,
    weight_step: float = 0.01,
    alpha: float = 0.95
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
        alpha=alpha
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
