# Trading Fees Implementation Plan

## Overview

This document outlines the implementation of transaction cost modeling via basis-point trading fees at rebalancing, plus comparative Sharpe/CAGR metrics for individual hedge analysis, enabling realistic portfolio performance evaluation.

---

## Implementation Steps

### 1. Add Trading Fee Configuration to `config.yaml`

Add `trading_fee_bps` parameter under the `rebalancing` section:

```yaml
rebalancing:
  frequency: "quarterly"
  method: "calendar"
  trading_fee_bps: 10  # Trading cost in basis points (10 bps = 0.10%)
```

### 2. Enhance `simulate_rebalanced_portfolio()` in `src/backtester/rebalancing.py`

- Add `trading_fee_bps: float = 0.0` parameter to function signature
- On rebalance days (after weight drift calculation):
  - Calculate one-way turnover: `turnover = sum(abs(current_weights - target_weights)) / 2`
  - Deduct fee from portfolio value: `portfolio_value -= turnover * fee_bps / 10000 * portfolio_value`
- **First day handling**: No fee on initial portfolio construction (assume cash-to-asset conversion is free)

### 3. Update `_analyze_hedge_worker()` in `src/backtester/engine.py`

- Call `compute_all_metrics()` on aligned base returns to capture unhedged Sharpe/CAGR/CVaR/MDD
- Pass `trading_fee_bps` from config to rebalancing simulation
- Return `base_metrics` alongside existing `hedged_metrics` in result dictionary

### 4. Update Multi-Asset Portfolio Path in `engine.py`

- Pass `trading_fee_bps` to `simulate_rebalanced_portfolio()` calls in multi-asset optimization section

### 5. Enhance HTML Report in `src/reporting/report.py`

- Add "Performance Comparison" table per individual hedge showing unhedged vs. hedged:
  - Sharpe Ratio
  - CAGR
  - CVaR (95%)
  - Maximum Drawdown
- Add columns to individual hedge summary table for Sharpe improvement and CAGR impact
- Display trading fee assumptions in report header/methodology section

### 6. Add Comprehensive Tests

In `tests/test_rebalancing.py`:
- `TestTradingFees`: verify fees reduce portfolio value
- Zero fee matches original behavior
- Fee proportional to turnover

In `tests/test_engine.py`:
- Test that `base_metrics` is present in individual hedge results
- Verify expected keys exist

---

## Design Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| **Fee application timing** | After weight drift calculation | Standard industry approach - fees deducted on rebalance date after measuring drift |
| **Cumulative fee tracking** | No, just net performance | Setting fees to 0 shows gross performance; keeps implementation simple |
| **First-day fees** | No fees on initial construction | Assume cash-to-asset conversion is free; only charge on rebalancing turnover |

---

## Fee Calculation Formula

On each rebalance date (excluding first day):

```
turnover = Σ|w_current - w_target| / 2    # One-way turnover
fee_cost = turnover × (fee_bps / 10000) × portfolio_value
portfolio_value_after = portfolio_value - fee_cost
```

**Example**: 
- Portfolio value: $100,000
- Weight drift: Base 52% → 50%, Hedge 48% → 50%
- Turnover: (|0.52 - 0.50| + |0.48 - 0.50|) / 2 = 0.02 (2%)
- Fee (10 bps): 0.02 × 0.0010 × $100,000 = $2.00

---

## Files Modified

| File | Changes |
|------|---------|
| `config.yaml` | Add `trading_fee_bps` parameter |
| `src/backtester/rebalancing.py` | Add fee calculation to `simulate_rebalanced_portfolio()` |
| `src/backtester/engine.py` | Compute base_metrics, pass fees to simulation |
| `src/reporting/report.py` | Add performance comparison tables |
| `tests/test_rebalancing.py` | Add `TestTradingFees` class |
| `tests/test_engine.py` | Add base_metrics verification tests |

---

## Testing Strategy

1. **Unit tests**: Verify fee calculation math is correct
2. **Integration tests**: Verify fees flow through full pipeline
3. **Regression tests**: Verify zero fees produce identical results to pre-implementation
4. **Edge cases**: Handle zero turnover, single asset, extreme fee values
