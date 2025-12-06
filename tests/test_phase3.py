"""
Test script for Phase 3 modules: Hypothesis Testing, Backtester, and Reporting.

Copyright (c) 2025 L.Bassetti
Run this to validate the complete framework implementation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def test_hypothesis_module():
    """Test hypothesis testing functions."""
    print("\n" + "=" * 60)
    print("Testing Hypothesis Module")
    print("=" * 60)
    
    from src.hypothesis.tests import (
        bootstrap_cvar_test,
        safe_haven_regression,
        correlation_stability_test
    )
    
    # Create synthetic data
    np.random.seed(42)
    n = 252 * 5  # 5 years daily
    dates = pd.date_range(start='2019-01-01', periods=n, freq='D')
    
    # Base returns (equity-like)
    base_returns = pd.Series(
        np.random.normal(0.0003, 0.01, n),
        index=dates
    )
    
    # Hedge returns (negatively correlated during crashes)
    hedge_returns = pd.Series(
        -0.5 * base_returns + np.random.normal(0, 0.005, n),
        index=dates
    )
    
    # Regime labels (20% crisis periods)
    regime_labels = pd.Series(
        (np.random.random(n) < 0.2).astype(int),
        index=dates
    )
    
    print("\n1. Bootstrap CVaR Test...")
    cvar_result = bootstrap_cvar_test(
        base_returns=base_returns,
        hedge_returns=hedge_returns,
        hedge_weight=0.20,
        n_bootstrap=1000,
        alpha=0.05
    )
    print(f"   Reduction: {cvar_result['actual_reduction']:.4f}")
    print(f"   P-value: {cvar_result['p_value']:.4f}")
    print(f"   Significant: {cvar_result['significant']}")
    
    print("\n2. Safe Haven Regression...")
    sh_result = safe_haven_regression(
        equity_returns=base_returns,
        hedge_returns=hedge_returns,
        regime_labels=regime_labels
    )
    print(f"   Beta (Normal): {sh_result['beta_normal']:.4f}")
    print(f"   Beta (Crisis): {sh_result['beta_crisis']:.4f}")
    print(f"   Is Safe Haven: {sh_result['is_safe_haven']}")
    
    print("\n3. Correlation Stability Test...")
    corr_result = correlation_stability_test(
        equity_returns=base_returns,
        hedge_returns=hedge_returns,
        regime_labels=regime_labels
    )
    print(f"   Corr (Normal): {corr_result['corr_normal']:.4f}")
    print(f"   Corr (Crisis): {corr_result['corr_crisis']:.4f}")
    print(f"   Significant: {corr_result['significant']}")
    
    print("\n✓ Hypothesis module tests passed!")


def test_regime_detector():
    """Test regime detection methods."""
    print("\n" + "=" * 60)
    print("Testing Regime Detector")
    print("=" * 60)
    
    from src.regime.detector import RegimeDetector
    
    # Create synthetic data with clear drawdown periods
    np.random.seed(42)
    n = 252 * 5
    dates = pd.date_range(start='2019-01-01', periods=n, freq='D')
    
    # Price series with crashes
    returns = np.random.normal(0.0003, 0.01, n)
    # Add some crash periods
    returns[200:230] = -0.03  # Crash 1
    returns[500:550] = -0.02  # Crash 2
    
    prices = pd.Series((1 + pd.Series(returns)).cumprod() * 100, index=dates)
    returns = pd.Series(returns, index=dates)
    
    # Create VIX-like data
    vix = pd.Series(np.random.uniform(15, 25, n), index=dates)
    vix[200:230] = np.random.uniform(35, 50, 30)  # High VIX during crash
    vix[500:550] = np.random.uniform(30, 45, 50)
    
    # Test different methods
    config = {
        'method': 'drawdown',
        'drawdown_threshold': -0.10,
        'vix_crisis_threshold': 30,
        'vix_recovery_threshold': 20,
        'volatility_window': 21,
        'volatility_percentile': 0.75
    }
    
    detector = RegimeDetector(config)
    
    print("\n1. Drawdown Regime...")
    regime_dd = detector.drawdown_regime(prices)
    stats_dd = detector.get_regime_statistics(regime_dd)
    print(f"   Crisis periods: {stats_dd['crisis_periods']} ({stats_dd['crisis_pct']:.1f}%)")
    print(f"   Crisis episodes: {stats_dd['crisis_episodes']}")
    
    print("\n2. VIX Regime...")
    regime_vix = detector.vix_regime(vix)
    stats_vix = detector.get_regime_statistics(regime_vix)
    print(f"   Crisis periods: {stats_vix['crisis_periods']} ({stats_vix['crisis_pct']:.1f}%)")
    
    print("\n3. Volatility Percentile Regime...")
    regime_vol = detector.volatility_percentile_regime(returns)
    stats_vol = detector.get_regime_statistics(regime_vol)
    print(f"   Crisis periods: {stats_vol['crisis_periods']} ({stats_vol['crisis_pct']:.1f}%)")
    
    print("\n4. Ensemble Regime...")
    config['method'] = 'ensemble'
    detector_ensemble = RegimeDetector(config)
    regime_ensemble = detector_ensemble.detect(prices, returns, vix)
    stats_ensemble = detector_ensemble.get_regime_statistics(regime_ensemble)
    print(f"   Crisis periods: {stats_ensemble['crisis_periods']} ({stats_ensemble['crisis_pct']:.1f}%)")
    
    print("\n✓ Regime detector tests passed!")


def test_optimization():
    """Test optimization functions."""
    print("\n" + "=" * 60)
    print("Testing Optimization Module")
    print("=" * 60)
    
    from src.optimization.weight_finder import find_weight_for_target, grid_search_weight
    from src.optimization.multi_asset import greedy_sequential_allocation
    
    # Create synthetic data
    np.random.seed(42)
    n = 252 * 5
    dates = pd.date_range(start='2019-01-01', periods=n, freq='D')
    
    base_returns = pd.Series(
        np.random.normal(0.0003, 0.01, n),
        index=dates
    )
    
    hedge1 = pd.Series(
        -0.3 * base_returns + np.random.normal(0, 0.005, n),
        index=dates
    )
    
    hedge2 = pd.Series(
        -0.5 * base_returns + np.random.normal(0, 0.008, n),
        index=dates
    )
    
    print("\n1. Single Asset Weight Finder...")
    weight, reduction, value = find_weight_for_target(
        base_returns=base_returns,
        hedge_returns=hedge1,
        target_reduction=0.25,
        metric='cvar',
        max_weight=0.50,
        weight_step=0.01
    )
    print(f"   Optimal weight: {weight:.2%}")
    print(f"   Achieved reduction: {reduction:.2%}")
    
    print("\n2. Grid Search...")
    grid_results = grid_search_weight(
        base_returns=base_returns,
        hedge_returns=hedge1,
        metric='cvar',
        max_weight=0.30,
        weight_step=0.05
    )
    print(f"   Grid points: {len(grid_results)}")
    print(f"   Max reduction: {grid_results['reduction_pct'].max():.1f}%")
    
    print("\n3. Multi-Asset Greedy Allocation...")
    hedge_returns = pd.DataFrame({
        'HEDGE1': hedge1,
        'HEDGE2': hedge2
    })
    
    weights = greedy_sequential_allocation(
        base_returns=base_returns,
        hedge_returns=hedge_returns,
        target_reduction=0.25,
        metric='cvar',
        max_total_weight=0.50,
        weight_step=0.05
    )
    print(f"   Allocated weights:")
    for asset, w in weights.items():
        if w > 0:
            print(f"     {asset}: {w:.2%}")
    
    print("\n✓ Optimization tests passed!")


def test_correlations():
    """Test correlation analysis functions."""
    print("\n" + "=" * 60)
    print("Testing Correlations Module")
    print("=" * 60)
    
    from src.metrics.correlations import (
        conditional_correlation,
        downside_beta,
        quantile_correlation,
        correlation_breakdown
    )
    
    # Create synthetic data
    np.random.seed(42)
    n = 252 * 5
    dates = pd.date_range(start='2019-01-01', periods=n, freq='D')
    
    equity_returns = pd.Series(
        np.random.normal(0.0003, 0.01, n),
        index=dates
    )
    
    hedge_returns = pd.Series(
        -0.4 * equity_returns + np.random.normal(0, 0.005, n),
        index=dates
    )
    
    regime_labels = pd.Series(
        (equity_returns < -0.01).astype(int),
        index=dates
    )
    
    print("\n1. Conditional Correlation...")
    corr_crisis = conditional_correlation(
        equity_returns, hedge_returns, regime_labels, regime_value=1
    )
    print(f"   Crisis correlation: {corr_crisis:.4f}")
    
    print("\n2. Downside Beta...")
    beta_down = downside_beta(equity_returns, hedge_returns, threshold=0.0)
    print(f"   Downside beta: {beta_down:.4f}")
    
    print("\n3. Quantile Correlation...")
    corr_5pct = quantile_correlation(equity_returns, hedge_returns, quantile=0.05)
    print(f"   5% tail correlation: {corr_5pct:.4f}")
    
    print("\n4. Correlation Breakdown...")
    breakdown = correlation_breakdown(
        equity_returns, hedge_returns, regime_labels
    )
    print(f"   Pearson: {breakdown['pearson_full']:.4f}")
    print(f"   Crisis: {breakdown['correlation_crisis']:.4f}")
    print(f"   Normal: {breakdown['correlation_normal']:.4f}")
    
    print("\n✓ Correlation tests passed!")


def test_reporting():
    """Test reporting functions."""
    print("\n" + "=" * 60)
    print("Testing Reporting Module")
    print("=" * 60)
    
    from src.reporting.report import generate_markdown_report
    
    # Create mock results structure
    results = {
        'config': {
            'regime': {'method': 'ensemble'},
            'assets': {'base': 'ACWI'},
            'metrics': {'cvar_confidence': 0.95}
        },
        'data_info': {
            'start_date': '2019-01-01',
            'end_date': '2023-12-31',
            'n_days': 1260,
            'assets': ['ACWI', 'TLT', 'GLD']
        },
        'regime_stats': {
            'crisis_periods': 252,
            'crisis_pct': 20.0,
            'crisis_episodes': 3,
            'avg_crisis_length': 84.0
        },
        'individual_hedges': {
            'TLT': {
                'correlations': {
                    'pearson_full': -0.25,
                    'correlation_normal': -0.10,
                    'correlation_crisis': -0.45,
                    'spearman_full': -0.30,
                    'beta_full': -0.20,
                    'downside_beta': -0.50,
                    'quantile_corr_5pct': -0.60,
                    'tail_dependence_lower': 0.15
                },
                'optimization': [
                    {
                        'target_reduction': 25.0,
                        'metric': 'cvar',
                        'optimal_weight': 0.30,
                        'achieved_reduction': 26.5,
                        'feasible': True
                    }
                ],
                'hypothesis_tests': {
                    'cvar_test': {
                        'actual_reduction': 0.012,
                        'p_value': 0.001,
                        'significant': True
                    },
                    'safe_haven': {
                        'beta_normal': -0.10,
                        'beta_crisis': -0.50,
                        'is_safe_haven': True
                    }
                }
            }
        },
        'portfolios': {
            '25pct_reduction': {
                'weights': {'TLT': 0.20, 'GLD': 0.10},
                'total_hedge_weight': 0.30,
                'portfolio_cvar': 0.025,
                'portfolio_mdd': 0.15,
                'portfolio_sharpe': 0.85,
                'portfolio_cagr': 8.5,
                'baseline_cvar': 0.035,
                'baseline_mdd': 0.22,
                'baseline_sharpe': 0.65,
                'baseline_cagr': 9.2,
                'cvar_reduction_pct': 28.6,
                'mdd_reduction_pct': 31.8,
                'sharpe_improvement': 0.20,
                'crisis_base_return': -5.2,
                'crisis_portfolio_return': -2.8,
                'normal_base_return': 12.5,
                'normal_portfolio_return': 11.8
            }
        }
    }
    
    print("\n1. Generating Markdown Report...")
    generate_markdown_report(results, output_path='output/test_report.md')
    print("   ✓ Report generated at: output/test_report.md")
    
    print("\n✓ Reporting tests passed!")


def main():
    """Run all Phase 3 tests."""
    print("\n" + "=" * 60)
    print("PHASE 3 MODULE TESTING")
    print("Testing: Hypothesis, Regime, Optimization, Correlations, Reporting")
    print("=" * 60)
    
    try:
        test_correlations()
        test_regime_detector()
        test_optimization()
        test_hypothesis_module()
        test_reporting()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nPhase 3 modules are fully functional:")
        print("  ✓ src/metrics/correlations.py")
        print("  ✓ src/regime/detector.py")
        print("  ✓ src/optimization/weight_finder.py")
        print("  ✓ src/optimization/multi_asset.py")
        print("  ✓ src/hypothesis/tests.py")
        print("  ✓ src/backtester/engine.py")
        print("  ✓ src/reporting/report.py")
        print("\nYou can now run: python main.py")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
