# Tail-Risk Hedge Lab

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**A comprehensive Python framework for quantitative analysis of tail-risk hedging strategies.**

Tail-Risk Hedge Lab is an open-source backtesting engine designed to evaluate how various asset classes (bonds, gold, volatility products, managed futures, cryptocurrencies) perform as hedges against extreme drawdowns in global equity portfolios. The framework provides rigorous statistical testing, optimization algorithms, and professional reporting for academic research and quantitative analysis.

---

## 🎯 Key Features

### **Comprehensive Analysis Pipeline**
- **Automated Data Acquisition**: Download price data via Yahoo Finance (`yfinance`) or import custom return series from Excel files
- **Crisis Regime Detection**: 5 methodologies including drawdown-based, VIX thresholds, volatility percentiles, Markov-switching, and ensemble voting
- **Risk Metrics**: CVaR (Expected Shortfall), Maximum Drawdown, Sortino Ratio, Downside Beta, Tail Dependence, and more
- **Portfolio Optimization**: Find optimal hedge weights to achieve 10%, 25%, or 50% risk reduction targets
- **Statistical Hypothesis Testing**: Bootstrap tests, Baur-Lucey safe-haven regression, correlation stability analysis
- **Multi-Asset Portfolios**: Construct diversified tail-risk hedged portfolios with quarterly rebalancing
- **Professional Reporting**: Auto-generated HTML reports with embedded charts and statistical tables

### **Performance Optimized**
- Parallel processing for analyzing multiple hedge assets simultaneously
- Efficient data caching to minimize redundant API calls
- Vectorized NumPy/Pandas operations throughout

---

## 📦 Installation

### Requirements
- Python 3.8 or higher
- See `requirements.txt` for dependencies

### Setup

```bash
# Clone the repository
git clone https://github.com/lbsm2017/Tail-Risk-Hedge-Lab.git
cd Tail-Risk-Hedge-Lab

pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Basic Usage

Run the complete analysis with default configuration:

```bash
python main.py
```

This will:
1. Download price data for all configured assets (ACWI, TLT, GLD, BTC-USD, etc.)
2. Identify crisis periods using ensemble regime detection
3. Analyze each hedge asset individually (correlation, optimization, hypothesis tests)
4. Build optimal multi-asset portfolios for 10%, 25%, and 50% risk reduction targets
5. Generate a timestamped HTML report in the `output` folder

### Configuration

All settings are controlled via `config.yaml`:

```yaml
data:
  start_date: "2008-04-01"  # Analysis start date
  end_date: null            # Latest available if null
  
assets:
  base: "ACWI"              # Baseline portfolio (global equities)
  hedges:
    - ticker: "TLT"         # Long Treasury ETF
      name: "Long Treasury (20+ Year)"
      max_weight: 0.50      # Maximum allocation
    - ticker: "GLD"
      name: "Gold"
      max_weight: 0.40
    # Add more hedges...

optimization:
  targets: [0.10, 0.25, 0.50]  # Risk reduction targets
  weight_step: 0.01             # Optimization granularity
  
rebalancing:
  frequency: "quarterly"        # Rebalancing frequency
```

### Adding Assets

#### Method 1: Via `config.yaml` (Programmatic)

Add any Yahoo Finance ticker to the hedges list:

```yaml
assets:
  hedges:
    - ticker: "SPY"
      name: "S&P 500"
      max_weight: 0.30
```

#### Method 2: Import Custom Data from Excel

Place Excel files in `data/import` with the following structure:

| Date       | Price | 
|------------|-------|
| 2020-01-01 | 100.0 |
| 2020-01-02 | 101.5 |
| ...        | ...   |

**Supported formats:**
- `.xlsx` or `.xls` files
- First column should be dates (any column name containing "date")
- Second column should be prices or returns
- Filename (without extension) becomes the asset name

The framework will automatically:
- Detect and load the files
- Compute returns if prices are provided
- Align dates with other assets
- Include the asset in the analysis

**Example:**

```
data/import/MAN_AHL_Evolution.xlsx
data/import/Custom_Strategy.xlsx
```

---

## 📊 Output

The framework generates a professional HTML report (`output/tail_risk_analysis_YYYY.MM.DD.HH.MM.SS.html`) containing:

### **Executive Summary**
- Analysis period and sample statistics
- Regime detection methodology and crisis statistics

### **Individual Hedge Analysis**
- Correlation breakdown (normal vs. crisis regimes)
- Optimal weights for 10%, 25%, 50% risk reduction targets
- Feasibility assessment (whether target is achievable within constraints)
- Rolling correlation charts with crisis period highlighting
- Statistical hypothesis test results

### **Optimal Multi-Asset Portfolios**
- Three portfolio strategies (10%, 25%, 50% risk reduction)
- Asset allocations with weights
- Risk-return comparison tables (CVaR, MDD, Sharpe, CAGR)
- Regime-conditional performance analysis

### **Key Findings**
- Strongest crisis hedge identification
- Recommended allocation strategy

---

## 🏗️ Architecture

```
Tail-Risk-Hedge-Lab/
├── config.yaml                 # Configuration file
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
│
├── src/
│   ├── backtester/             # Main engine
│   │   ├── engine.py           # Orchestration & parallel processing
│   │   └── rebalancing.py      # Portfolio rebalancing simulation
│   ├── data/
│   │   └── downloader.py       # Data acquisition (yfinance + Excel import)
│   ├── regime/
│   │   └── detector.py         # Crisis detection algorithms
│   ├── metrics/
│   │   ├── tail_risk.py        # Risk metrics (CVaR, MDD, etc.)
│   │   └── correlations.py     # Correlation analysis
│   ├── optimization/
│   │   ├── weight_finder.py    # Single-asset optimization
│   │   └── multi_asset.py      # Multi-asset portfolio construction
│   ├── hypothesis/
│   │   └── tests.py            # Statistical hypothesis testing
│   └── reporting/
│       └── report.py           # HTML report generation
│
├── data/
│   ├── prices.parquet          # Cached price data (auto-generated)
│   └── import/                 # Custom Excel data files (user-provided)
│
├── output/                     # Generated HTML reports
│
├── tests/                      # Unit tests
│   ├── test_phase3.py
│   └── test_downloader.py
│
└── docs/                       # Methodology documentation
```

---

## 🔬 Methodology

### Regime Detection Methods

1. **Drawdown-based**: Crisis when equity drawdown exceeds threshold (e.g., -10%)
2. **VIX threshold**: Crisis when VIX > 30 with hysteresis
3. **Volatility percentile**: Crisis when rolling volatility exceeds 75th percentile
4. **Markov-switching**: Hamilton 2-state regime model
5. **Ensemble**: Majority vote across all methods (default)

### Optimization Algorithms

- **Binary Search**: Finds minimum weight to achieve target risk reduction
- **Grid Search**: Evaluates risk-return trade-off across weight range
- **Greedy Sequential**: Adds assets iteratively with best marginal improvement (multi-asset)
- **CVaR Minimization**: SLSQP optimization with weight constraints (multi-asset)

### Hypothesis Tests

- **Bootstrap CVaR Test**: Tests if risk reduction is statistically significant (10,000 samples)
- **Safe-Haven Regression**: Baur-Lucey methodology testing if beta < 0 during crises
- **Correlation Stability**: Fisher Z-test for correlation differences across regimes
- **Tail Dependence**: Clayton copula lower tail coefficient

---

## 🧪 Testing

Run the test suite to validate all modules:

```bash
python tests/test_phase3.py
```

Tests cover:
- Data downloading and caching
- Regime detection algorithms
- Correlation analysis
- Optimization functions
- Hypothesis testing
- Report generation

---

## 🤝 Contributing

We welcome contributions from the research community! This is an open-source project designed for collaborative improvement.

### How to Contribute

1. **Report Issues**: Found a bug or have a feature request? [Open an issue](https://github.com/lbsm2017/Tail-Risk-Hedge-Lab/issues)
2. **Submit Pull Requests**: 
   - Fork the repository
   - Create a feature branch (`git checkout -b feature/your-feature`)
   - Commit your changes with clear messages
   - Push to your fork and submit a pull request
3. **Improve Documentation**: Help clarify methodology, add examples, or fix typos
4. **Add Features**:
   - New regime detection algorithms
   - Additional risk metrics
   - Alternative optimization methods
   - Enhanced visualizations

### Contribution Guidelines

- Follow existing code style (Google-style docstrings, type hints)
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📄 License

**Dual License Model:**

- **Non-Commercial Use (Default)**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)  
  Free for academic research, personal projects, and non-commercial open research.

- **Commercial Use**: Requires a separate commercial license.  
  Contact: lorenzo.bassetti@gmail.com

See **[License.md](License.md)** for full details.

---

## 📚 Documentation

Comprehensive methodology documentation available in `docs`:

- `tail_risk_hedging.md`: Theoretical foundation
- `tail_risk_metrics_and_proxies.md`: Risk measure definitions
- `regime_identification_methods.md`: Crisis detection methodologies
- `statistical_hypothesis_testing_framework.md`: Testing procedures
- `framework_methodologies.md`: Implementation details

---

## 🙏 Citation

If you use this framework in academic research, please cite:

```bibtex
@software{bassetti2025tailrisk,
  author = {Bassetti, Lorenzo},
  title = {Tail-Risk Hedge Lab: A Quantitative Framework for Tail-Risk Hedging Analysis},
  year = {2025},
  url = {https://github.com/lbsm2017/Tail-Risk-Hedge-Lab}
}
```

---

## 📧 Contact

**Lorenzo Bassetti**  
Email: lorenzo.bassetti@gmail.com  
GitHub: [@lbsm2017](https://github.com/lbsm2017)

---

## ⚠️ Disclaimer

This software is provided for research and educational purposes only. It does not constitute investment advice. Past performance is not indicative of future results. All investments involve risk, including the possible loss of principal. Users are solely responsible for any decisions made using this framework.

---

## 🌟 Acknowledgments

Built with:
- `yfinance` for financial data
- `pandas` & `numpy` for data manipulation
- `scipy` & `statsmodels` for statistical analysis
- `matplotlib` for visualizations

Special thanks to the open-source quantitative finance community.

---

**⭐ Star this repository if you find it useful!**  
**🐛 Report issues and contribute to make it better!**
