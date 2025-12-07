"""Quick test to verify minimal allocation logic."""
import numpy as np
import pandas as pd
from src.optimization.multi_asset import greedy_sequential_allocation

# Create simple test data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=500)
base = pd.Series(np.random.randn(500) * 0.02, index=dates)

# Create a very effective hedge
hedge1 = -base * 0.8 + np.random.randn(500) * 0.005  # Highly negatively correlated
hedges = pd.DataFrame({'H1': hedge1})

# Test with 25% target
print("Testing 25% CVaR reduction target...")
weights = greedy_sequential_allocation(
    base_returns=base,
    hedge_returns=hedges,
    target_reduction=0.25,
    metric='cvar',
    max_total_weight=0.50,
    weight_step=0.01,
    tolerance=0.005
)

total_weight = sum(weights.values())
print(f"Total hedge weight: {total_weight*100:.1f}%")
print(f"Weights: {weights}")

# Calculate actual reduction
from src.metrics.tail_risk import cvar

baseline_cvar = cvar(base, alpha=0.95, frequency='monthly')
portfolio_returns = (1 - total_weight) * base + total_weight * hedge1
portfolio_cvar = cvar(portfolio_returns, alpha=0.95, frequency='monthly')
achieved_reduction = (baseline_cvar - portfolio_cvar) / baseline_cvar

print(f"\nBaseline CVaR: {baseline_cvar*100:.2f}%")
print(f"Portfolio CVaR: {portfolio_cvar*100:.2f}%")
print(f"Achieved reduction: {achieved_reduction*100:.2f}%")
print(f"Target: 25.0%")
print(f"Within tolerance: {abs(achieved_reduction - 0.25) <= 0.005}")

if total_weight > 0.15:
    print(f"\n⚠ WARNING: Total weight {total_weight*100:.1f}% seems too high for 25% target!")
else:
    print(f"\n✓ Total weight {total_weight*100:.1f}% looks reasonable for 25% target")
