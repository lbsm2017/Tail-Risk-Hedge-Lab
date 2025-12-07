"""
Multi-Asset Portfolio Optimization Module

Author: L.Bassetti
Implements portfolio construction with multiple hedge assets.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from scipy.optimize import minimize
from ..metrics.tail_risk import cvar, max_drawdown, sharpe_ratio, cagr


def align_mixed_frequency_returns(
    base_returns: pd.Series,
    hedge_returns: pd.DataFrame,
    target_frequency: str = 'auto'
) -> pd.DataFrame:
    """
    Align daily and monthly returns to a common frequency.
    
    When mixing daily assets with monthly assets (like MAN AHL), we need to:
    1. Detect which assets are monthly (sparse data with ~monthly gaps)
    2. Aggregate daily assets to monthly using geometric compounding
    3. Return aligned monthly data
    
    This ensures proper portfolio analytics when combining assets with different
    data frequencies.
    
    Args:
        base_returns: Base portfolio returns (typically daily)
        hedge_returns: DataFrame with hedge asset returns (mixed frequencies)
        target_frequency: 'auto' (detect from data), 'monthly', or 'daily'
        
    Returns:
        DataFrame with 'base' column and hedge columns, all at same frequency
    """
    # Detect frequency of each asset
    def get_asset_frequency(returns: pd.Series) -> str:
        """Determine if asset is daily or monthly based on data density."""
        returns = returns.dropna()
        if len(returns) < 2:
            return 'unknown'
        
        # Calculate average days between observations
        total_days = (returns.index[-1] - returns.index[0]).days
        avg_days = total_days / (len(returns) - 1)
        
        if avg_days > 20:  # Monthly data has ~30 days between observations
            return 'monthly'
        elif avg_days > 5:  # Weekly
            return 'weekly'
        else:  # Daily
            return 'daily'
    
    # Check base frequency
    base_freq = get_asset_frequency(base_returns)
    
    # Check hedge frequencies
    hedge_freqs = {}
    has_monthly = False
    for col in hedge_returns.columns:
        freq = get_asset_frequency(hedge_returns[col])
        hedge_freqs[col] = freq
        if freq == 'monthly':
            has_monthly = True
    
    # Determine target frequency
    if target_frequency == 'auto':
        # If any asset is monthly, convert all to monthly
        target_frequency = 'monthly' if has_monthly else 'daily'
    
    if target_frequency == 'monthly' and base_freq == 'daily':
        # Need to convert daily base returns to monthly
        # Use geometric compounding: (1+r1) * (1+r2) * ... * (1+rn) - 1
        base_monthly = (1 + base_returns).resample('ME').prod() - 1
        base_monthly = base_monthly.dropna()
    else:
        base_monthly = base_returns
    
    # Convert hedge returns as needed
    hedge_monthly = pd.DataFrame()
    for col in hedge_returns.columns:
        if target_frequency == 'monthly' and hedge_freqs[col] == 'daily':
            # Convert daily to monthly
            monthly = (1 + hedge_returns[col]).resample('ME').prod() - 1
            monthly = monthly.dropna()
            hedge_monthly[col] = monthly
        else:
            # Already monthly or keeping as-is
            hedge_monthly[col] = hedge_returns[col]
    
    # Combine and align
    all_data = pd.concat([base_monthly.rename('base'), hedge_monthly], axis=1).dropna()
    
    return all_data


def optimize_multi_asset_cvar(
    base_returns: pd.Series,
    hedge_returns: pd.DataFrame,
    target_cvar_reduction: float = 0.25,
    max_total_weight: float = 0.50,
    max_weights: Optional[Dict[str, float]] = None,
    alpha: float = 0.95,
    cvar_frequency: str = 'monthly'
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
    baseline_cvar = cvar(base_aligned, alpha=alpha, frequency=cvar_frequency)
    target_cvar = baseline_cvar * (1 - target_cvar_reduction)
    
    # Define objective: minimize CVaR
    def objective(weights):
        # Construct portfolio
        hedge_weight_total = np.sum(weights)
        base_weight = 1 - hedge_weight_total
        
        portfolio_returns = base_weight * base_aligned
        for i, asset in enumerate(hedge_aligned.columns):
            portfolio_returns += weights[i] * hedge_aligned[asset]
        
        return cvar(portfolio_returns, alpha=alpha, frequency=cvar_frequency)
    
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
    alpha: float = 0.95,
    cvar_frequency: str = 'monthly',
    tie_break_tolerance: float = 0.001,
    hedge_efficiency: Optional[Dict[str, float]] = None,
    tie_break_method: str = 'efficiency',
    tolerance: float = 0.005
) -> Dict[str, float]:
    """
    Greedy algorithm: find MINIMAL hedge allocation to achieve target risk reduction.
    
    Tracks BOTH CVaR and MaxDD, stops when EITHER metric achieves target (within tolerance).
    This maximizes baseline exposure while providing tail-risk protection.
    
    Tie-breaking priority:
    1. Lowest risk (primary selection based on 'metric' parameter)
    2. If multiple within tie_break_tolerance:
       - 'efficiency': Highest risk reduction per weight unit (from individual analysis)
       - 'cagr': Highest portfolio CAGR
       - 'crisis_correlation': Most negative crisis correlation
    
    Args:
        base_returns: Base portfolio returns
        hedge_returns: DataFrame with hedge asset returns
        target_reduction: Target risk reduction (e.g., 0.25 for 25%) - applies to BOTH CVaR and MDD
        metric: 'cvar' or 'mdd' - primary metric for candidate selection
        max_total_weight: Maximum total hedge weight
        max_weights: Dict of max weight per asset
        weight_step: Weight increment for search
        alpha: Confidence level for CVaR
        cvar_frequency: CVaR frequency ('daily', 'weekly', 'monthly')
        tie_break_tolerance: Risk difference threshold for tie-breaking
        hedge_efficiency: Dict mapping ticker to efficiency score (risk reduction per weight)
        tie_break_method: 'efficiency', 'cagr', or 'crisis_correlation'
        tolerance: Accept solutions within this fraction of target (e.g., 0.005 = 0.5%)
        
    Returns:
        Dictionary with minimal weights achieving target on CVaR OR MDD (or best effort if unachievable)
    """
    # Align data with proper frequency handling (converts daily to monthly when needed)
    all_data = align_mixed_frequency_returns(base_returns, hedge_returns)
    base_aligned = all_data['base']
    hedge_aligned = all_data.drop('base', axis=1)
    
    # Calculate baseline risks for BOTH metrics
    baseline_cvar = cvar(base_aligned, alpha=alpha, frequency=cvar_frequency)
    baseline_mdd, _, _ = max_drawdown((1 + base_aligned).cumprod())
    
    # Calculate target risks for both metrics
    target_cvar = baseline_cvar * (1 - target_reduction)
    target_mdd = baseline_mdd * (1 - target_reduction)
    
    # Initialize weights
    weights = {asset: 0.0 for asset in hedge_aligned.columns}
    current_portfolio = base_aligned.copy()
    current_cvar = baseline_cvar
    current_mdd = baseline_mdd
    total_weight = 0.0
    
    # Track which metric we're primarily optimizing for
    primary_metric = metric
    
    # Greedy allocation - stop when EITHER metric achieves target
    # Continue only if BOTH metrics are still above target (neither achieved yet)
    while (total_weight < max_total_weight and 
           not (current_cvar <= target_cvar or current_mdd <= target_mdd)):
        best_asset = None
        best_increment = 0.0
        # Initialize best risk based on primary metric
        best_new_risk = current_cvar if primary_metric == 'cvar' else current_mdd
        
        # Try adding weight_step to each asset
        candidates = []  # Store all valid candidates for tie-breaking
        
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
                
                # Calculate BOTH risks
                test_cvar = cvar(test_portfolio, alpha=alpha, frequency=cvar_frequency)
                test_mdd, _, _ = max_drawdown((1 + test_portfolio).cumprod())
                
                # Use primary metric for selection, but track both
                if primary_metric == 'cvar':
                    test_risk = test_cvar
                else:
                    test_risk = test_mdd
                
                # Store candidate with both metrics
                candidates.append({
                    'asset': asset,
                    'risk': test_risk,
                    'cvar': test_cvar,
                    'mdd': test_mdd,
                    'portfolio': test_portfolio
                })
                
                # Track overall best risk
                if test_risk < best_new_risk:
                    best_new_risk = test_risk
        
        # Select best asset using tie-breaking logic
        if candidates:
            # Filter candidates within tie_break_tolerance of best risk
            top_candidates = [c for c in candidates if abs(c['risk'] - best_new_risk) <= tie_break_tolerance]
            
            if len(top_candidates) > 1:
                # Multiple assets with similar risk - use configured tie-breaking method
                if tie_break_method == 'efficiency' and hedge_efficiency:
                    # Prefer assets with higher efficiency (risk reduction per weight unit)
                    best_efficiency = -np.inf
                    for candidate in top_candidates:
                        eff = hedge_efficiency.get(candidate['asset'], 0.0)
                        if eff > best_efficiency:
                            best_efficiency = eff
                            best_asset = candidate['asset']
                    
                    # If all have same efficiency (or efficiency not available), fall back to CAGR
                    if best_asset is None or best_efficiency == 0.0:
                        tie_break_method = 'cagr'  # Fallback
                
                if tie_break_method == 'cagr' or best_asset is None:
                    # Use CAGR tie-breaking (original logic)
                    best_cagr = -np.inf
                    for candidate in top_candidates:
                        portfolio_cagr = cagr(candidate['portfolio'].values, periods_per_year=252)
                        if portfolio_cagr > best_cagr:
                            best_cagr = portfolio_cagr
                            best_asset = candidate['asset']
            elif len(top_candidates) == 1:
                best_asset = top_candidates[0]['asset']
            
            best_increment = weight_step
        
        # If no improvement found, stop
        if best_asset is None:
            break
        
        # Apply best increment
        weights[best_asset] += best_increment
        total_weight += best_increment
        
        # Update current risks based on best candidate
        best_candidate = next((c for c in candidates if c['asset'] == best_asset), None)
        if best_candidate:
            current_cvar = best_candidate['cvar']
            current_mdd = best_candidate['mdd']
        
        # Calculate achieved reductions for BOTH metrics
        achieved_cvar_reduction = (baseline_cvar - current_cvar) / baseline_cvar
        achieved_mdd_reduction = (baseline_mdd - current_mdd) / baseline_mdd
        
        # Early stopping: if EITHER metric achieves target (within tolerance), return minimal weights
        if (achieved_cvar_reduction >= (target_reduction - tolerance) or 
            achieved_mdd_reduction >= (target_reduction - tolerance)):
            break
    
    return weights


def build_portfolio_analytics(
    base_returns: pd.Series,
    hedge_returns: pd.DataFrame,
    weights: Dict[str, float],
    alpha: float = 0.95,
    rf_rate: float = 0.0,
    cvar_frequency: str = 'monthly'
) -> Dict:
    """
    Calculate comprehensive analytics for multi-asset portfolio.
    
    Properly aligns daily and monthly returns before computing metrics.
    When mixing daily assets with monthly assets (like MAN AHL), daily returns
    are aggregated to monthly using geometric compounding.
    
    Args:
        base_returns: Base portfolio returns (daily data)
        hedge_returns: DataFrame with hedge asset returns (mixed frequencies)
        weights: Dictionary with hedge weights
        alpha: Confidence level for CVaR
        rf_rate: Annualized risk-free rate
        
    Returns:
        Dictionary with portfolio metrics
    """
    # Align data with proper frequency handling (converts daily to monthly when needed)
    all_data = align_mixed_frequency_returns(base_returns, hedge_returns)
    base_aligned = all_data['base']
    hedge_aligned = all_data.drop('base', axis=1)
    
    # Detect hedged portfolio frequency (after alignment)
    if len(base_aligned) > 1:
        avg_days_between = (base_aligned.index[-1] - base_aligned.index[0]).days / (len(base_aligned) - 1)
        if avg_days_between > 20:  # Monthly data (~30 days between observations)
            hedged_periods_per_year = 12
            hedged_frequency = 'monthly'
        elif avg_days_between > 5:  # Weekly data (~7 days between observations)
            hedged_periods_per_year = 52
            hedged_frequency = 'weekly'
        else:  # Daily data (~1-2 days between observations)
            hedged_periods_per_year = 252
            hedged_frequency = 'daily'
    else:
        hedged_periods_per_year = 252
        hedged_frequency = 'daily'
    
    # For fair comparison, baseline uses the SAME aligned data as hedged portfolio
    # This ensures risk reduction percentages match what the optimizer calculated
    # Both baseline and hedged are computed on the same dates/frequency
    base_unhedged = base_aligned
    baseline_periods_per_year = hedged_periods_per_year
    baseline_frequency = hedged_frequency
    
    # Construct hedged portfolio using aligned data
    total_hedge_weight = sum(weights.values())
    base_weight = 1 - total_hedge_weight
    
    portfolio_returns = base_weight * base_aligned
    for asset, w in weights.items():
        if w > 0 and asset in hedge_aligned.columns:
            portfolio_returns += w * hedge_aligned[asset]
    
    # Hedged portfolio metrics (uses aligned frequency)
    portfolio_cvar = cvar(portfolio_returns, alpha=alpha, frequency=cvar_frequency)
    portfolio_mdd, peak, trough = max_drawdown((1 + portfolio_returns).cumprod())
    portfolio_sharpe = sharpe_ratio(portfolio_returns, rf_rate=rf_rate, periods_per_year=hedged_periods_per_year)
    
    # Baseline metrics (uses original daily data for same date range)
    baseline_cvar = cvar(base_unhedged, alpha=alpha, frequency=cvar_frequency)
    baseline_mdd, _, _ = max_drawdown((1 + base_unhedged).cumprod())
    baseline_sharpe = sharpe_ratio(base_unhedged, rf_rate=rf_rate, periods_per_year=baseline_periods_per_year)
    
    # Reductions
    cvar_reduction = (baseline_cvar - portfolio_cvar) / baseline_cvar * 100
    mdd_reduction = (baseline_mdd - portfolio_mdd) / baseline_mdd * 100
    
    # Returns - Use proper CAGR with detected frequencies
    from ..metrics.tail_risk import cagr as compute_cagr
    portfolio_cagr = compute_cagr(portfolio_returns.values, periods_per_year=hedged_periods_per_year) * 100
    baseline_cagr = compute_cagr(base_unhedged.values, periods_per_year=baseline_periods_per_year) * 100
    
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
        'sharpe_improvement': portfolio_sharpe - baseline_sharpe,
        'hedged_frequency': hedged_frequency,
        'baseline_frequency': baseline_frequency,
        'hedged_periods_per_year': hedged_periods_per_year,
        'baseline_periods_per_year': baseline_periods_per_year,
        'hedged_n_periods': len(base_aligned),
        'baseline_n_periods': len(base_unhedged),
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


# Alias for backwards compatibility
portfolio_analytics = build_portfolio_analytics
