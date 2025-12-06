"""
Multi-Asset Portfolio Optimization Module

Implements portfolio construction with multiple hedge assets.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from scipy.optimize import minimize
from ..metrics.tail_risk import cvar, max_drawdown, sharpe_ratio


def optimize_multi_asset_cvar(
    base_returns: pd.Series,
    hedge_returns: pd.DataFrame,
    target_cvar_reduction: float = 0.25,
    max_total_weight: float = 0.50,
    max_weights: Optional[Dict[str, float]] = None,
    alpha: float = 0.95
) -> Dict[str, float]:
    """
    Find optimal weights for multiple hedge assets to minimize CVaR.
    
    Args:
        base_returns: Base portfolio returns
        hedge_returns: DataFrame with hedge asset returns (columns = assets)
        target_cvar_reduction: Target CVaR reduction (e.g., 0.25 for 25%)
        max_total_weight: Maximum total hedge weight
        max_weights: Dict of max weight per asset (optional)
        alpha: Confidence level for CVaR
        
    Returns:
        Dictionary with optimal weights for each hedge asset
    """
    # Align data
    all_data = pd.concat([base_returns.rename('base'), hedge_returns], axis=1).dropna()
    base_aligned = all_data['base']
    hedge_aligned = all_data.drop('base', axis=1)
    
    n_assets = len(hedge_aligned.columns)
    
    # Calculate baseline CVaR
    baseline_cvar = cvar(base_aligned, alpha=alpha)
    target_cvar = baseline_cvar * (1 - target_cvar_reduction)
    
    # Define objective: minimize CVaR
    def objective(weights):
        # Construct portfolio
        hedge_weight_total = np.sum(weights)
        base_weight = 1 - hedge_weight_total
        
        portfolio_returns = base_weight * base_aligned
        for i, asset in enumerate(hedge_aligned.columns):
            portfolio_returns += weights[i] * hedge_aligned[asset]
        
        return cvar(portfolio_returns, alpha=alpha)
    
    # Define constraint: target CVaR reduction
    def cvar_constraint(weights):
        portfolio_cvar = objective(weights)
        # Return positive if constraint satisfied (CVaR <= target)
        return target_cvar - portfolio_cvar
    
    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda w: max_total_weight - np.sum(w)},  # Total weight <= max
        {'type': 'ineq', 'fun': lambda w: np.sum(w)},  # Non-negative total
        {'type': 'eq', 'fun': cvar_constraint}  # Meet CVaR target
    ]
    
    # Bounds for each asset
    bounds = []
    for asset in hedge_aligned.columns:
        if max_weights and asset in max_weights:
            bounds.append((0, max_weights[asset]))
        else:
            bounds.append((0, max_total_weight))
    
    # Initial guess: equal weights
    x0 = np.ones(n_assets) * (max_total_weight / n_assets)
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    # Extract weights
    if result.success:
        weights_dict = {asset: weight for asset, weight in zip(hedge_aligned.columns, result.x)}
    else:
        # If optimization fails, return zero weights
        weights_dict = {asset: 0.0 for asset in hedge_aligned.columns}
    
    return weights_dict


def optimize_multi_asset_max_sharpe(
    base_returns: pd.Series,
    hedge_returns: pd.DataFrame,
    max_total_weight: float = 0.50,
    max_weights: Optional[Dict[str, float]] = None,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Find weights that maximize Sharpe ratio with hedge assets.
    
    Args:
        base_returns: Base portfolio returns
        hedge_returns: DataFrame with hedge asset returns
        max_total_weight: Maximum total hedge weight
        max_weights: Dict of max weight per asset
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Dictionary with optimal weights
    """
    # Align data
    all_data = pd.concat([base_returns.rename('base'), hedge_returns], axis=1).dropna()
    base_aligned = all_data['base']
    hedge_aligned = all_data.drop('base', axis=1)
    
    n_assets = len(hedge_aligned.columns)
    
    # Define objective: maximize Sharpe (minimize negative Sharpe)
    def objective(weights):
        hedge_weight_total = np.sum(weights)
        base_weight = 1 - hedge_weight_total
        
        portfolio_returns = base_weight * base_aligned
        for i, asset in enumerate(hedge_aligned.columns):
            portfolio_returns += weights[i] * hedge_aligned[asset]
        
        sharpe = sharpe_ratio(portfolio_returns, risk_free_rate=risk_free_rate / 252)
        return -sharpe  # Minimize negative Sharpe
    
    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda w: max_total_weight - np.sum(w)},
        {'type': 'ineq', 'fun': lambda w: np.sum(w)}
    ]
    
    # Bounds
    bounds = []
    for asset in hedge_aligned.columns:
        if max_weights and asset in max_weights:
            bounds.append((0, max_weights[asset]))
        else:
            bounds.append((0, max_total_weight))
    
    # Initial guess
    x0 = np.ones(n_assets) * (max_total_weight / n_assets)
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    if result.success:
        weights_dict = {asset: weight for asset, weight in zip(hedge_aligned.columns, result.x)}
    else:
        weights_dict = {asset: 0.0 for asset in hedge_aligned.columns}
    
    return weights_dict


def greedy_sequential_allocation(
    base_returns: pd.Series,
    hedge_returns: pd.DataFrame,
    target_reduction: float = 0.25,
    metric: str = 'cvar',
    max_total_weight: float = 0.50,
    max_weights: Optional[Dict[str, float]] = None,
    weight_step: float = 0.01,
    alpha: float = 0.95
) -> Dict[str, float]:
    """
    Greedy algorithm: sequentially add hedge assets with best marginal improvement.
    
    Args:
        base_returns: Base portfolio returns
        hedge_returns: DataFrame with hedge asset returns
        target_reduction: Target risk reduction
        metric: 'cvar' or 'mdd'
        max_total_weight: Maximum total hedge weight
        max_weights: Dict of max weight per asset
        weight_step: Weight increment for search
        alpha: Confidence level for CVaR
        
    Returns:
        Dictionary with optimal weights
    """
    # Align data
    all_data = pd.concat([base_returns.rename('base'), hedge_returns], axis=1).dropna()
    base_aligned = all_data['base']
    hedge_aligned = all_data.drop('base', axis=1)
    
    # Calculate baseline risk
    if metric == 'cvar':
        baseline_risk = cvar(base_aligned, alpha=alpha)
    else:  # mdd
        baseline_risk, _, _ = max_drawdown((1 + base_aligned).cumprod())
    
    target_risk = baseline_risk * (1 - target_reduction)
    
    # Initialize weights
    weights = {asset: 0.0 for asset in hedge_aligned.columns}
    current_portfolio = base_aligned.copy()
    current_risk = baseline_risk
    total_weight = 0.0
    
    # Greedy allocation
    while total_weight < max_total_weight and current_risk > target_risk:
        best_asset = None
        best_increment = 0.0
        best_new_risk = current_risk
        
        # Try adding weight_step to each asset
        for asset in hedge_aligned.columns:
            # Check if we can add more to this asset
            max_asset_weight = max_weights.get(asset, max_total_weight) if max_weights else max_total_weight
            
            if weights[asset] + weight_step <= max_asset_weight and total_weight + weight_step <= max_total_weight:
                # Try this allocation
                test_weights = weights.copy()
                test_weights[asset] += weight_step
                
                # Build test portfolio
                test_total_weight = sum(test_weights.values())
                test_portfolio = (1 - test_total_weight) * base_aligned
                
                for test_asset, w in test_weights.items():
                    test_portfolio += w * hedge_aligned[test_asset]
                
                # Calculate risk
                if metric == 'cvar':
                    test_risk = cvar(test_portfolio, alpha=alpha)
                else:
                    test_risk, _, _ = max_drawdown((1 + test_portfolio).cumprod())
                
                # Check if this is best improvement
                if test_risk < best_new_risk:
                    best_asset = asset
                    best_increment = weight_step
                    best_new_risk = test_risk
        
        # If no improvement found, stop
        if best_asset is None:
            break
        
        # Apply best increment
        weights[best_asset] += best_increment
        total_weight += best_increment
        current_risk = best_new_risk
    
    return weights


def portfolio_analytics(
    base_returns: pd.Series,
    hedge_returns: pd.DataFrame,
    weights: Dict[str, float],
    alpha: float = 0.95
) -> Dict:
    """
    Calculate comprehensive analytics for multi-asset portfolio.
    
    Args:
        base_returns: Base portfolio returns
        hedge_returns: DataFrame with hedge asset returns
        weights: Dictionary with hedge weights
        alpha: Confidence level for CVaR
        
    Returns:
        Dictionary with portfolio metrics
    """
    # Align data
    all_data = pd.concat([base_returns.rename('base'), hedge_returns], axis=1).dropna()
    base_aligned = all_data['base']
    hedge_aligned = all_data.drop('base', axis=1)
    
    # Construct portfolio
    total_hedge_weight = sum(weights.values())
    base_weight = 1 - total_hedge_weight
    
    portfolio_returns = base_weight * base_aligned
    for asset, w in weights.items():
        if w > 0 and asset in hedge_aligned.columns:
            portfolio_returns += w * hedge_aligned[asset]
    
    # Calculate metrics
    portfolio_cvar = cvar(portfolio_returns, alpha=alpha)
    portfolio_mdd, peak, trough = max_drawdown((1 + portfolio_returns).cumprod())
    portfolio_sharpe = sharpe_ratio(portfolio_returns)
    
    # Baseline metrics
    baseline_cvar = cvar(base_aligned, alpha=alpha)
    baseline_mdd, _, _ = max_drawdown((1 + base_aligned).cumprod())
    baseline_sharpe = sharpe_ratio(base_aligned)
    
    # Reductions
    cvar_reduction = (baseline_cvar - portfolio_cvar) / baseline_cvar * 100
    mdd_reduction = (baseline_mdd - portfolio_mdd) / baseline_mdd * 100
    
    # Returns
    portfolio_cagr = (portfolio_returns.mean() * 252) * 100
    baseline_cagr = (base_aligned.mean() * 252) * 100
    
    analytics = {
        'weights': weights,
        'total_hedge_weight': total_hedge_weight,
        'portfolio_cvar': portfolio_cvar,
        'portfolio_mdd': portfolio_mdd,
        'portfolio_sharpe': portfolio_sharpe,
        'portfolio_cagr': portfolio_cagr,
        'baseline_cvar': baseline_cvar,
        'baseline_mdd': baseline_mdd,
        'baseline_sharpe': baseline_sharpe,
        'baseline_cagr': baseline_cagr,
        'cvar_reduction_pct': cvar_reduction,
        'mdd_reduction_pct': mdd_reduction,
        'sharpe_improvement': portfolio_sharpe - baseline_sharpe
    }
    
    return analytics


def equal_risk_contribution(
    base_returns: pd.Series,
    hedge_returns: pd.DataFrame,
    max_total_weight: float = 0.50
) -> Dict[str, float]:
    """
    Equal risk contribution portfolio (risk parity).
    
    Each asset contributes equally to portfolio risk.
    
    Args:
        base_returns: Base portfolio returns
        hedge_returns: DataFrame with hedge asset returns
        max_total_weight: Maximum total hedge weight
        
    Returns:
        Dictionary with optimal weights
    """
    # Align data
    all_data = pd.concat([base_returns.rename('base'), hedge_returns], axis=1).dropna()
    
    # Calculate covariance matrix
    cov_matrix = all_data.cov().values
    
    n_assets = len(all_data.columns)
    
    # Define objective: minimize sum of squared differences in risk contribution
    def objective(weights):
        # Full weights including base
        full_weights = np.array([1 - np.sum(weights)] + list(weights))
        
        # Portfolio variance
        port_var = full_weights @ cov_matrix @ full_weights
        
        if port_var <= 0:
            return 1e10
        
        # Risk contributions
        marginal_contrib = cov_matrix @ full_weights
        risk_contrib = full_weights * marginal_contrib / np.sqrt(port_var)
        
        # Target: equal contributions
        target_contrib = 1.0 / n_assets
        
        return np.sum((risk_contrib - target_contrib) ** 2)
    
    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda w: max_total_weight - np.sum(w)},
        {'type': 'ineq', 'fun': lambda w: np.sum(w)}
    ]
    
    # Bounds
    bounds = [(0, max_total_weight) for _ in range(n_assets - 1)]  # Exclude base
    
    # Initial guess
    x0 = np.ones(n_assets - 1) * (max_total_weight / n_assets)
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    if result.success:
        weights_dict = {asset: weight for asset, weight in zip(hedge_returns.columns, result.x)}
    else:
        # Fallback: inverse volatility
        vols = hedge_returns.std()
        inv_vols = 1 / vols
        weights_dict = {asset: (inv_vol / inv_vols.sum()) * max_total_weight 
                       for asset, inv_vol in inv_vols.items()}
    
    return weights_dict
