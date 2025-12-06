# Tail-Risk Hedging Backtester

A comprehensive Python framework for testing tail-risk protection strategies against a 100% global equity portfolio baseline.

## Project Structure

```
hedgeTool/
├── docs/                          # Documentation
├── data/                          # Downloaded data cache
├── src/                           # Source code
│   ├── data/                      # Data acquisition
│   ├── regime/                    # Crisis detection
│   ├── metrics/                   # Risk metrics
│   ├── optimization/              # Weight optimization
│   ├── hypothesis/                # Statistical tests
│   ├── backtester/                # Main engine
│   └── reporting/                 # Reports & visualization
├── notebooks/                     # Jupyter notebooks
├── tests/                         # Unit tests
├── output/                        # Generated reports
├── config.yaml                    # Configuration
├── requirements.txt               # Dependencies
└── main.py                        # Entry point
```

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the complete analysis:

```bash
python main.py
```

## Configuration

Edit `config.yaml` to customize:
- Date range
- Asset tickers
- Regime detection parameters
- Optimization targets
- Risk metrics

## Features

- **Multiple Regime Detection Methods**: Drawdown-based, VIX thresholds, volatility percentiles, Markov-switching, ensemble
- **Comprehensive Risk Metrics**: CVaR, Maximum Drawdown, Downside Deviation, Sortino Ratio, Tail Dependence
- **Hypothesis Testing**: Bootstrap tests, Baur-Lucey safe-haven regression, variance ratio tests
- **Optimization**: Find optimal weights for 10%, 25%, 50% risk reduction targets
- **Multi-Asset Portfolio Construction**: Build diversified tail-risk hedged portfolios
- **Detailed Reporting**: Tables, charts, and markdown reports

## Documentation

See `docs/` folder for comprehensive documentation:
- `documentation.md`: Complete methodology
- `plan.md`: Implementation plan
- `tail_risk_hedging.md`: Scientific backdrop
- Additional methodology documents

## License

For research purposes.
