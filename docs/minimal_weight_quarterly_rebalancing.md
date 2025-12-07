# Individual Asset Testing - Minimal Weight with Quarterly Rebalancing

## Overview

Individual hedge asset analysis now implements a **minimal weight strategy** with **quarterly rebalancing**:

1. **Find minimal weight**: Determine the smallest hedge allocation that achieves the target risk reduction over the entire test period
2. **Quarterly rebalancing**: Maintain that weight through regular portfolio rebalancing

## Methodology

### Minimal Weight Optimization

The optimization algorithm searches for the **minimum hedge weight** that achieves the target CVaR or MDD reduction:

```
Goal: min(weight) such that:
  - risk_reduction >= target_reduction
  - min_weight <= weight <= max_weight
```

**Algorithm:**
1. Grid search across weight range (0% to max_weight)
2. For each weight, calculate portfolio risk with that allocation
3. Identify all weights that meet or exceed the target reduction
4. Select the **smallest** weight among viable candidates
5. If multiple solutions at minimum weight, use CAGR as tie-breaker

**Example:**
- Target: 25% CVaR reduction
- Weight candidates that achieve target: [15%, 20%, 25%, 30%]
- **Selected weight: 15%** (minimum that achieves goal)

### Quarterly Rebalancing

Once the optimal weight is determined, the portfolio is simulated with **quarterly rebalancing**:

**Rebalancing Process:**
1. Start with target weights (e.g., 85% ACWI, 15% hedge)
2. Let weights drift naturally between rebalance dates
3. At end of each quarter, reset weights to target allocation
4. Track drift and rebalancing frequency

**Drift Calculation:**
```python
drift = Σ |current_weight[i] - target_weight[i]|
```

## Implementation Details

### Weight Finder (`src/optimization/weight_finder.py`)

Updated `find_weight_for_target_reduction()`:

```python
def find_weight_for_target_reduction(...) -> Dict:
    """
    Find the MINIMAL hedge weight that achieves the target risk reduction.
    
    Strategy: Among all weights that achieve the target reduction, select the
    smallest weight. This minimizes hedge allocation while meeting risk goals.
    The optimal weight is then maintained through quarterly rebalancing.
    """
    # Search for all viable candidates
    viable_candidates = []
    for w in weights:
        if achieves_target(w):
            viable_candidates.append(w)
    
    # Select MINIMUM weight
    optimal_weight = min(viable_candidates)
```

### Backtester Engine (`src/backtester/engine.py`)

Updated `_analyze_hedge_worker()` to apply quarterly rebalancing:

```python
# Compute hedged metrics with quarterly rebalancing
if optimal_weight > 0:
    # Create DataFrame with base and hedge returns
    returns_df = pd.DataFrame({
        'base': aligned_base,
        'hedge': aligned_hedge
    })
    
    # Target weights for quarterly rebalancing
    target_weights = {
        'base': 1 - optimal_weight,
        'hedge': optimal_weight
    }
    
    # Simulate portfolio with quarterly rebalancing
    rebalanced_portfolio = simulate_rebalanced_portfolio(
        returns=returns_df,
        weights=target_weights,
        rebalance_frequency='quarterly',
        initial_value=1.0
    )
    
    # Extract rebalanced portfolio returns
    hedged_returns = rebalanced_portfolio['portfolio_return']
```

## Metrics Output

The hedged portfolio metrics now include rebalancing information:

```python
hedged_metrics = {
    'cvar': 0.0234,
    'mdd': 0.1856,
    'cagr': 0.0892,
    'sharpe': 1.23,
    'rebalancing': {
        'frequency': 'quarterly',
        'n_rebalances': 68,       # Number of rebalance events
        'avg_drift': 0.0143       # Average weight drift between rebalances
    }
}
```

## Key Differences from Previous Approach

### Before (Simple Weighted Returns)
```python
hedged_returns = (1 - weight) * base_returns + weight * hedge_returns
```
- Weights fixed at constant ratio
- No transaction costs consideration
- Unrealistic for long-term portfolios

### After (Quarterly Rebalancing)
```python
# Weights drift between quarters
# Reset to target allocation quarterly
# More realistic portfolio behavior
```

## Configuration

Rebalancing frequency is set in `config.yaml`:

```yaml
rebalancing:
  frequency: "quarterly"  # Options: daily, weekly, monthly, quarterly, annual
  method: "calendar"      # calendar = rebalance at end of period
```

## Benefits

1. **Cost Efficiency**: Minimal weight reduces transaction costs and tracking error
2. **Realistic Simulation**: Quarterly rebalancing mirrors actual portfolio management
3. **Risk Control**: Achieves target risk reduction with smallest hedge allocation
4. **Transparency**: Reports number of rebalances and average drift

## Example Output

```
MAN AHL Evolution - Individual Analysis

Optimal Weight: 12%
  - Base (ACWI): 88%
  - Hedge: 12%

Target: 25% CVaR Reduction
Achieved: 26.3%

Rebalancing:
  - Frequency: Quarterly
  - Number of rebalances: 52
  - Average drift: 1.2%
  - Max drift: 3.8%

Baseline (100% ACWI):
  - CVaR (95%): 4.12%
  - MDD: 34.5%

Hedged Portfolio (88/12):
  - CVaR (95%): 3.04% (-26.3%)
  - MDD: 28.7% (-16.8%)
  - CAGR: 7.8%
```

## Testing

All tests pass with the new implementation:
- ✓ 33/33 rebalancing tests pass
- ✓ 8/8 engine tests pass
- ✓ Weight drift properly calculated
- ✓ Rebalance flags set correctly
- ✓ Portfolio value evolves realistically

Run tests:
```bash
python -m pytest tests/test_rebalancing.py -v
python -m pytest tests/test_engine.py -v
```

## Technical Notes

- Rebalancing uses **calendar method**: last trading day of each quarter
- Drift is calculated as sum of absolute weight deviations
- Initial portfolio value set to 1.0 (normalized)
- Returns are compounded properly between rebalances
- Risk-free rate aligned to portfolio dates
