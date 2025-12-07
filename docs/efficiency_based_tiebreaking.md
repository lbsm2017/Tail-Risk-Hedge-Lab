# Efficiency-Based Tie-Breaking for Multi-Asset Optimization

## Overview

Multi-asset portfolio optimization uses **efficiency-based tie-breaking**
This favors assets that deliver more risk reduction per unit weight, resulting in more cost-effective portfolios.

## Motivation

- Selects assets with highest **risk reduction per weight unit**
- Formula: `Efficiency = Achieved Reduction / Optimal Weight`
- Favors assets that are more "capital efficient" as hedges
- Example: Asset achieving 25% reduction with 10% weight (eff=2.5) beats asset achieving 25% reduction with 20% weight (eff=1.25)

## Implementation Details

### 1. Efficiency Calculation (Individual Analysis)

**Location:** `src/optimization/weight_finder.py`

The `find_weights_for_all_targets()` function now adds an `efficiency` column to results:

```python
df['efficiency'] = df.apply(
    lambda row: row['achieved_reduction'] / row['optimal_weight'] 
    if row['optimal_weight'] > 0 else 0.0,
    axis=1
)
```

**Output Structure:**
```python
{
    'target_reduction': 0.25,
    'optimal_weight': 0.15,
    'achieved_reduction': 0.258,
    'efficiency': 1.72,  # NEW: 0.258 / 0.15 = 1.72
    'metric': 'cvar',
    'feasible': True
}
```

### 2. Efficiency Extraction (Engine)

**Location:** `src/backtester/engine.py` (lines 630-658)

Before calling multi-asset optimization, extract efficiency scores from individual analysis:

```python
# Extract efficiency scores from individual hedge analysis
hedge_efficiency = {}
if hasattr(self, 'individual_hedges') and self.individual_hedges:
    for ticker in hedge_tickers:
        hedge_data = self.individual_hedges[ticker]
        opt_results = hedge_data.get('optimization', [])
        
        # Focus on CVaR results for primary target (25% reduction)
        cvar_results = [r for r in opt_results if r.get('metric') == 'cvar']
        
        if cvar_results:
            # Use primary target (25%) if available
            primary_result = next(
                (r for r in cvar_results if abs(r.get('target_reduction', 0) - 0.25) < 0.01),
                None
            )
            
            if primary_result and 'efficiency' in primary_result:
                hedge_efficiency[ticker] = primary_result['efficiency']
```

### 3. Tie-Breaking Logic (Multi-Asset Optimization)

**Location:** `src/optimization/multi_asset.py` (lines 263-291)

Updated `greedy_sequential_allocation()` function signature:

```python
def greedy_sequential_allocation(
    ...,
    hedge_efficiency: Optional[Dict[str, float]] = None,
    tie_break_method: str = 'efficiency'
) -> Dict[str, float]:
```

**Tie-breaking priority:**
1. **Primary selection**: Lowest risk (within candidates)
2. **If multiple within `tie_break_tolerance`:**
   - **'efficiency'**: Select asset with highest efficiency score
   - **'cagr'**: Select asset with highest portfolio CAGR (original)
   - **'crisis_correlation'**: Reserved for future use

**Code:**
```python
if len(top_candidates) > 1:
    if tie_break_method == 'efficiency' and hedge_efficiency:
        # Prefer assets with higher efficiency
        best_efficiency = -np.inf
        for candidate in top_candidates:
            eff = hedge_efficiency.get(candidate['asset'], 0.0)
            if eff > best_efficiency:
                best_efficiency = eff
                best_asset = candidate['asset']
        
        # Fallback to CAGR if efficiency not available
        if best_asset is None or best_efficiency == 0.0:
            tie_break_method = 'cagr'
    
    if tie_break_method == 'cagr' or best_asset is None:
        # Original CAGR tie-breaking
        best_cagr = -np.inf
        for candidate in top_candidates:
            portfolio_cagr = cagr(candidate['portfolio'].values, periods_per_year=252)
            if portfolio_cagr > best_cagr:
                best_cagr = portfolio_cagr
                best_asset = candidate['asset']
```

### 4. Configuration

**Location:** `config.yaml`

```yaml
optimization:
  tie_break_method: "efficiency"  # Options: 'efficiency', 'cagr', 'crisis_correlation'
  tie_break_tolerance: 0.001      # Risk difference threshold (0.1%) for tie-breaking
```

## Example Scenario

### Setup
- Base: 100% ACWI
- Target: 25% CVaR reduction
- Three hedge candidates with similar risk profiles (within 0.1% tolerance)

### Individual Analysis Results

| Asset | Optimal Weight | CVaR Reduction | Efficiency |
|-------|----------------|----------------|------------|
| TLT   | 12%           | 25.2%          | **2.10**   |
| GLD   | 18%           | 25.4%          | 1.41       |
| CHF   | 22%           | 25.1%          | 1.14       |

### Multi-Asset Selection

**With efficiency tie-breaking (NEW):**
- All three achieve ~25% reduction (within tolerance)
- TLT selected first: highest efficiency (2.10)
- Result: More capital-efficient portfolio

**With CAGR tie-breaking (OLD):**
- Would select based on portfolio CAGR
- Might select less capital-efficient asset

## Benefits

1. **Capital Efficiency**: Minimizes hedge allocation needed to achieve targets
2. **Cost Reduction**: Lower transaction costs, rebalancing costs, tracking error
3. **Consistency**: Aligns with individual analysis goal (minimal weight)
4. **Transparency**: Clear metric based on individual hedge effectiveness
5. **Robustness**: Falls back to CAGR if efficiency unavailable

## Testing

Comprehensive test suite in `tests/test_efficiency_tiebreaking.py`:

### Test Coverage
- ✓ Efficiency calculation formula
- ✓ Efficiency added to results DataFrame
- ✓ Zero efficiency when no weight allocated
- ✓ Higher efficiency for better hedges
- ✓ Efficiency-based tie-breaking selection
- ✓ CAGR fallback when efficiency unavailable
- ✓ Explicit CAGR method selection
- ✓ Weight constraints respected
- ✓ End-to-end integration

**Run tests:**
```bash
python -m pytest tests/test_efficiency_tiebreaking.py -v
```

**Result:** 11/11 tests passing

## Backward Compatibility

- Default tie-break method: `'efficiency'` (configurable)
- Falls back to CAGR if efficiency data unavailable
- Existing code works unchanged (efficiency is optional parameter)
- Old reports remain valid (CAGR tie-breaking still available)

## Configuration Options

### Enable efficiency tie-breaking (recommended):
```yaml
optimization:
  tie_break_method: "efficiency"
```

### Revert to original CAGR tie-breaking:
```yaml
optimization:
  tie_break_method: "cagr"
```

### Adjust tie-breaking sensitivity:
```yaml
optimization:
  tie_break_tolerance: 0.001  # Smaller = fewer ties, larger = more ties
```

## Future Enhancements

**Crisis correlation tie-breaking:**
- Tertiary tie-breaker after efficiency
- Select assets with most negative crisis correlation
- Useful when efficiency scores are equal

**Implementation placeholder:**
```python
tie_break_method: 'crisis_correlation'
```

## Summary

The efficiency-based tie-breaking mechanism:
1. Calculates efficiency (risk reduction per weight unit) in individual analysis
2. Extracts efficiency scores before multi-asset optimization
3. Uses efficiency to break ties between similarly-performing assets
4. Falls back to CAGR if efficiency unavailable
5. Configurable via `config.yaml`

This approach creates more capital-efficient portfolios while maintaining the robustness of the original greedy allocation algorithm.
