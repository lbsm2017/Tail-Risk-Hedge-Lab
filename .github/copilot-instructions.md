# Tail-Risk Hedge Lab - AI Agent Instructions

## Architecture Overview

This is a **quantitative finance backtesting framework** for evaluating tail-risk hedging strategies against a global equity baseline (ACWI). The pipeline flow:

```
main.py → Backtester.run_full_backtest() → HTML Report
           ├── DataDownloader (yfinance + custom Excel)
           ├── RegimeDetector (crisis identification)
           ├── Individual hedge analysis (parallel)
           │   ├── correlation_breakdown()
           │   ├── optimize_for_multiple_targets()
           │   └── comprehensive_hypothesis_tests()
           └── Multi-asset portfolio optimization
```

## Key Module Patterns

### Data Flow Convention
- All returns/prices use **pandas Series/DataFrame with DatetimeIndex**
- Functions accept aligned data and handle NaN internally via `.dropna()`
- Risk metrics (CVaR, MDD) return **positive values** representing loss magnitude

### Backtester Engine (`src/backtester/engine.py`)
- Central orchestrator using `ThreadPoolExecutor` for parallel hedge analysis
- Worker function `_analyze_hedge_worker()` is module-level (not method) for pickle compatibility
- Loads config via `yaml.safe_load()` from `config.yaml`
- **Individual hedge testing**: Finds MINIMAL weight to achieve target risk reduction, applies quarterly rebalancing

### Adding New Hedge Assets
1. Add ticker to `config.yaml` under `assets.hedges` with `min_weight`/`max_weight`
2. Or place Excel file in `data/import/` with columns: Date, Price (auto-detected)

### Regime Detection (`src/regime/detector.py`)
- Methods: `drawdown`, `vix`, `volatility`, `markov`, `ensemble`
- Returns binary Series: 0=Normal, 1=Crisis
- Ensemble uses majority voting across methods

### Metrics Conventions (`src/metrics/`)
- `cvar(returns, alpha=0.95)` → Expected Shortfall at 95% confidence
- `max_drawdown(prices)` → Returns tuple: (mdd_value, peak_date, trough_date)
- Risk functions return positive values; callers use these directly without negation

## Development Commands

```bash
# Run full analysis pipeline
python main.py

# Run tests (no pytest, use direct execution)
python tests/test_phase3.py
python tests/test_downloader.py

# Install dependencies
pip install -r requirements.txt
```

## Configuration (`config.yaml`)

```yaml
optimization:
  targets: [0.10, 0.25, 0.50]  # CVaR reduction targets (10%, 25%, 50%)
  tolerance: 0.02              # Accept within 2% of target
regime:
  method: "ensemble"           # Crisis detection method
```

## Code Style Patterns

- **Type hints**: All functions use `typing` annotations
- **Docstrings**: Google-style with Args/Returns sections
- **Progress bars**: Conditional `tqdm` import with `TQDM_AVAILABLE` fallback
- **Parallel processing**: ThreadPoolExecutor for I/O, module-level workers for ProcessPoolExecutor
- **Rebalancing**: Quarterly rebalancing applied via `simulate_rebalanced_portfolio()` for realistic portfolio behavior

## Report Generation (`src/reporting/report.py`)

Generates self-contained HTML with embedded base64 charts. Key functions:
- `generate_html_report()` - main entry point
- `create_rolling_correlation_chart()` - matplotlib → base64 PNG
- Uses `matplotlib.use('Agg')` for headless rendering

## File Locations

| Purpose | Location |
|---------|----------|
| Entry point | `main.py` |
| Configuration | `config.yaml` |
| Price cache | `data/prices.parquet` |
| Custom data | `data/import/*.xlsx` |
| Output reports | `output/tail_risk_analysis_*.html` |
| Documentation | `docs/*.md` |
