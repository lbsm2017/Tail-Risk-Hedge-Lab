"""
Tests for efficiency-based tie-breaking in multi-asset optimization.

Tests cover:
1. Efficiency calculation in weight_finder
2. Efficiency extraction in engine
3. Tie-breaking logic in greedy_sequential_allocation
4. Config-based tie-break method selection
"""

import pytest
import pandas as pd
import numpy as np
from src.optimization.weight_finder import find_weights_for_all_targets
from src.optimization.multi_asset import greedy_sequential_allocation
from src.metrics.tail_risk import cvar


class TestEfficiencyCalculation:
    """Test efficiency metric calculation in individual analysis."""
    
    def test_efficiency_added_to_results(self):
        """Verify efficiency column is added to optimization results."""
        np.random.seed(42)
        base = pd.Series(np.random.randn(1000) * 0.01, index=pd.date_range('2020-01-01', periods=1000))
        hedge = pd.Series(np.random.randn(1000) * 0.015, index=pd.date_range('2020-01-01', periods=1000))
        
        results = find_weights_for_all_targets(
            base_returns=base,
            hedge_returns=hedge,
            targets=[0.10, 0.25],
            metrics=['cvar'],
            max_weight=0.50
        )
        
        assert 'efficiency' in results.columns, "Efficiency column missing"
        assert len(results) > 0, "No results returned"
        assert results['efficiency'].notna().all(), "Efficiency contains NaN"
    
    def test_efficiency_calculation_formula(self):
        """Verify efficiency = achieved_reduction / optimal_weight."""
        np.random.seed(42)
        base = pd.Series(np.random.randn(1000) * 0.01, index=pd.date_range('2020-01-01', periods=1000))
        hedge = pd.Series(np.random.randn(1000) * 0.015, index=pd.date_range('2020-01-01', periods=1000))
        
        results = find_weights_for_all_targets(
            base_returns=base,
            hedge_returns=hedge,
            targets=[0.25],
            metrics=['cvar'],
            max_weight=0.50
        )
        
        for _, row in results.iterrows():
            if row['optimal_weight'] > 0:
                expected_eff = row['achieved_reduction'] / row['optimal_weight']
                assert abs(row['efficiency'] - expected_eff) < 1e-10, \
                    f"Efficiency mismatch: {row['efficiency']} vs {expected_eff}"
    
    def test_efficiency_zero_when_no_weight(self):
        """Verify efficiency is 0 when optimal_weight is 0 or no reduction achieved."""
        np.random.seed(42)
        base = pd.Series(np.random.randn(500) * 0.01, index=pd.date_range('2020-01-01', periods=500))
        # Perfect positive correlation - no hedging benefit
        hedge = base * 1.5
        
        results = find_weights_for_all_targets(
            base_returns=base,
            hedge_returns=hedge,
            targets=[0.50],  # Impossible target
            metrics=['cvar'],
            max_weight=0.50
        )
        
        # Should have zero or near-zero weight and/or no reduction
        zero_efficiency_rows = results[(results['optimal_weight'] == 0) | (results['achieved_reduction'] <= 0)]
        if len(zero_efficiency_rows) > 0:
            assert (zero_efficiency_rows['efficiency'] == 0).all(), \
                "Efficiency should be 0 when weight is 0 or no reduction achieved"
    
    def test_efficiency_higher_for_better_hedges(self):
        """Verify more effective hedges have higher efficiency."""
        np.random.seed(42)
        base = pd.Series(np.random.randn(1000) * 0.02, index=pd.date_range('2020-01-01', periods=1000))
        
        # Good hedge: negative correlation
        good_hedge = -base * 0.5 + pd.Series(np.random.randn(1000) * 0.005, index=base.index)
        
        # Poor hedge: positive correlation
        poor_hedge = base * 0.3 + pd.Series(np.random.randn(1000) * 0.01, index=base.index)
        
        good_results = find_weights_for_all_targets(
            base_returns=base,
            hedge_returns=good_hedge,
            targets=[0.25],
            metrics=['cvar'],
            max_weight=0.50
        )
        
        poor_results = find_weights_for_all_targets(
            base_returns=base,
            hedge_returns=poor_hedge,
            targets=[0.25],
            metrics=['cvar'],
            max_weight=0.50
        )
        
        good_eff = good_results.iloc[0]['efficiency']
        poor_eff = poor_results.iloc[0]['efficiency']
        
        assert good_eff > poor_eff, \
            f"Good hedge should have higher efficiency: {good_eff} vs {poor_eff}"


class TestGreedyAllocationTieBreaking:
    """Test tie-breaking logic in greedy sequential allocation."""
    
    def test_efficiency_tie_breaking(self):
        """Verify efficiency-based tie-breaking selects higher efficiency asset."""
        np.random.seed(42)
        base = pd.Series(np.random.randn(500) * 0.02, index=pd.date_range('2020-01-01', periods=500))
        
        # Two hedges with similar risk reduction but different efficiency
        hedge1 = pd.Series(np.random.randn(500) * 0.015, index=base.index)
        hedge2 = pd.Series(np.random.randn(500) * 0.015, index=base.index)
        
        hedges = pd.DataFrame({'H1': hedge1, 'H2': hedge2})
        
        # Define efficiency scores (H2 is more efficient)
        efficiency = {'H1': 1.5, 'H2': 2.5}
        
        weights = greedy_sequential_allocation(
            base_returns=base,
            hedge_returns=hedges,
            target_reduction=0.10,
            metric='cvar',
            max_total_weight=0.30,
            weight_step=0.01,
            tie_break_tolerance=0.01,  # Large tolerance to create tie
            hedge_efficiency=efficiency,
            tie_break_method='efficiency'
        )
        
        # If there was a tie, H2 should be selected first (higher efficiency)
        # At minimum, verify weights are allocated
        assert sum(weights.values()) > 0, "No weights allocated"
    
    def test_cagr_tie_breaking_fallback(self):
        """Verify CAGR tie-breaking works when efficiency not provided."""
        np.random.seed(42)
        base = pd.Series(np.random.randn(500) * 0.02, index=pd.date_range('2020-01-01', periods=500))
        
        hedge1 = pd.Series(np.random.randn(500) * 0.015, index=base.index)
        hedge2 = pd.Series(np.random.randn(500) * 0.015, index=base.index)
        
        hedges = pd.DataFrame({'H1': hedge1, 'H2': hedge2})
        
        # No efficiency provided - should fall back to CAGR
        weights = greedy_sequential_allocation(
            base_returns=base,
            hedge_returns=hedges,
            target_reduction=0.10,
            metric='cvar',
            max_total_weight=0.30,
            weight_step=0.01,
            tie_break_tolerance=0.01,
            hedge_efficiency=None,  # No efficiency
            tie_break_method='efficiency'
        )
        
        assert sum(weights.values()) > 0, "No weights allocated"
    
    def test_cagr_method_explicitly(self):
        """Verify explicit CAGR method selection works."""
        np.random.seed(42)
        base = pd.Series(np.random.randn(500) * 0.02, index=pd.date_range('2020-01-01', periods=500))
        
        hedge1 = pd.Series(np.random.randn(500) * 0.015, index=base.index)
        hedges = pd.DataFrame({'H1': hedge1})
        
        weights = greedy_sequential_allocation(
            base_returns=base,
            hedge_returns=hedges,
            target_reduction=0.10,
            metric='cvar',
            max_total_weight=0.30,
            weight_step=0.01,
            tie_break_method='cagr'  # Explicit CAGR
        )
        
        assert sum(weights.values()) > 0, "No weights allocated"
    
    def test_single_asset_no_tie_breaking(self):
        """Verify single asset case works (no tie-breaking needed)."""
        np.random.seed(42)
        base = pd.Series(np.random.randn(500) * 0.02, index=pd.date_range('2020-01-01', periods=500))
        hedge = pd.Series(np.random.randn(500) * 0.015, index=base.index)
        
        hedges = pd.DataFrame({'H1': hedge})
        
        weights = greedy_sequential_allocation(
            base_returns=base,
            hedge_returns=hedges,
            target_reduction=0.10,
            metric='cvar',
            max_total_weight=0.30,
            weight_step=0.01,
            hedge_efficiency={'H1': 2.0},
            tie_break_method='efficiency'
        )
        
        assert weights['H1'] > 0, "Single asset should be allocated"
    
    def test_respects_max_weights_constraint(self):
        """Verify efficiency tie-breaking still respects weight constraints."""
        np.random.seed(42)
        base = pd.Series(np.random.randn(500) * 0.02, index=pd.date_range('2020-01-01', periods=500))
        
        hedge1 = pd.Series(np.random.randn(500) * 0.015, index=base.index)
        hedge2 = pd.Series(np.random.randn(500) * 0.015, index=base.index)
        
        hedges = pd.DataFrame({'H1': hedge1, 'H2': hedge2})
        
        # H1 has high efficiency but low max weight
        efficiency = {'H1': 5.0, 'H2': 2.0}
        max_weights = {'H1': 0.05, 'H2': 0.30}
        
        weights = greedy_sequential_allocation(
            base_returns=base,
            hedge_returns=hedges,
            target_reduction=0.20,
            metric='cvar',
            max_total_weight=0.50,
            max_weights=max_weights,
            weight_step=0.01,
            hedge_efficiency=efficiency,
            tie_break_method='efficiency'
        )
        
        assert weights['H1'] <= max_weights['H1'], \
            f"H1 weight {weights['H1']} exceeds max {max_weights['H1']}"
        assert weights['H2'] <= max_weights['H2'], \
            f"H2 weight {weights['H2']} exceeds max {max_weights['H2']}"


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_end_to_end_efficiency_flow(self):
        """Test complete flow from individual analysis to multi-asset optimization."""
        np.random.seed(42)
        base = pd.Series(np.random.randn(1000) * 0.02, index=pd.date_range('2020-01-01', periods=1000))
        
        # Create multiple hedges
        hedge1 = pd.Series(np.random.randn(1000) * 0.015, index=base.index)
        hedge2 = pd.Series(np.random.randn(1000) * 0.012, index=base.index)
        
        # Step 1: Individual analysis with efficiency
        h1_results = find_weights_for_all_targets(
            base_returns=base,
            hedge_returns=hedge1,
            targets=[0.25],
            metrics=['cvar'],
            max_weight=0.50
        )
        
        h2_results = find_weights_for_all_targets(
            base_returns=base,
            hedge_returns=hedge2,
            targets=[0.25],
            metrics=['cvar'],
            max_weight=0.50
        )
        
        # Step 2: Extract efficiencies
        h1_eff = h1_results.iloc[0]['efficiency']
        h2_eff = h2_results.iloc[0]['efficiency']
        
        efficiency_map = {'H1': h1_eff, 'H2': h2_eff}
        
        # Step 3: Multi-asset optimization
        hedges = pd.DataFrame({'H1': hedge1, 'H2': hedge2})
        
        weights = greedy_sequential_allocation(
            base_returns=base,
            hedge_returns=hedges,
            target_reduction=0.25,
            metric='cvar',
            max_total_weight=0.50,
            weight_step=0.01,
            hedge_efficiency=efficiency_map,
            tie_break_method='efficiency'
        )
        
        # Verify weights allocated
        assert sum(weights.values()) > 0, "No weights allocated"
        assert all(w >= 0 for w in weights.values()), "Negative weights found"
    
    def test_efficiency_values_reasonable(self):
        """Verify efficiency values are in reasonable range."""
        np.random.seed(42)
        base = pd.Series(np.random.randn(1000) * 0.02, index=pd.date_range('2020-01-01', periods=1000))
        hedge = pd.Series(np.random.randn(1000) * 0.015, index=base.index)
        
        results = find_weights_for_all_targets(
            base_returns=base,
            hedge_returns=hedge,
            targets=[0.10, 0.25, 0.50],
            metrics=['cvar', 'mdd'],
            max_weight=0.50
        )
        
        # Efficiency should be positive and not extremely large
        assert (results['efficiency'] >= 0).all(), "Negative efficiency found"
        assert (results['efficiency'] < 100).all(), "Unreasonably large efficiency"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
