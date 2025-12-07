# Tail-Risk Hedge Lab: Methodology & Scientific Framework

## Overview

The **Tail-Risk Hedge Lab** is a quantitative finance research framework for evaluating how different asset classes can protect a global equity portfolio during market crises. It provides a rigorous, evidence-based approach to answering: *"Which assets truly protect when equities crash, and how much should you hold?"*

---

## Table of Contents

1. [Purpose & Motivation](#purpose--motivation)
2. [The Tail-Risk Problem](#the-tail-risk-problem)
3. [Framework Architecture](#framework-architecture)
4. [Regime Detection Methodology](#regime-detection-methodology)
5. [Tail-Risk Metrics](#tail-risk-metrics)
6. [Correlation Analysis](#correlation-analysis)
7. [Statistical Hypothesis Testing](#statistical-hypothesis-testing)
8. [Portfolio Optimization](#portfolio-optimization)
9. [Asset Class Analysis](#asset-class-analysis)
10. [Quarterly Rebalancing Simulation](#quarterly-rebalancing-simulation)
11. [Key Academic References](#key-academic-references)
12. [Limitations & Future Work](#limitations--future-work)

---

## Purpose & Motivation

### Why This Exists

Investors holding global equities face **tail risk**—the danger of rare but severe drawdowns during market crises. Historical episodes demonstrate the magnitude of this risk:

| Crisis Event | Peak-to-Trough Decline | Duration |
|--------------|----------------------|----------|
| 2008-2009 Global Financial Crisis | ~55% | 17 months |
| 2020 COVID-19 Crash | ~34% | 1 month |
| 2022 Inflation Bear Market | ~27% | 10 months |

Traditional portfolio analysis using **average correlations fails during crises**. Assets that appear uncorrelated in normal times may all tumble together when markets panic. This framework addresses that limitation by:

1. **Regime-aware analysis**: Measuring asset behavior specifically during crisis periods
2. **Tail-focused metrics**: Using CVaR, Maximum Drawdown, and tail dependence rather than just volatility
3. **Statistical rigor**: Formal hypothesis testing to validate hedge effectiveness
4. **Actionable output**: Optimal hedge weights for specific risk reduction targets

### What It Answers

- Which assets provide genuine protection during equity crashes?
- How much of each hedge asset reduces CVaR or Maximum Drawdown by 10%, 25%, or 50%?
- Are these benefits statistically significant or just random chance?
- How do correlations change during crisis periods vs. normal markets?

---

## The Tail-Risk Problem

### Static Correlations Are Misleading

Standard portfolio theory uses historical correlation matrices assuming these relationships are stable. Research shows this assumption breaks down precisely when hedging matters most:

> *"Correlations are dynamic and regime-dependent rather than fixed. Assets that appear uncorrelated on average may all tumble together in a crisis."*

**Key insight**: What matters is **conditional correlation during stress**, not unconditional averages.

### Hedge vs. Diversifier vs. Safe Haven

The academic literature distinguishes three concepts (Baur & Lucey, 2010):

| Concept | Definition | Example |
|---------|------------|---------|
| **Hedge** | Uncorrelated or negatively correlated on average | Gold vs. equities (historically ~0 correlation) |
| **Diversifier** | Positively but imperfectly correlated | International equities |
| **Safe Haven** | Uncorrelated or negatively correlated during market turmoil | U.S. Treasuries during 2008 crash |

This framework specifically tests for **safe-haven behavior**—assets that protect when protection is needed most.

---

## Framework Architecture

### Pipeline Flow

```
main.py → Backtester.run_full_backtest()
           ├── DataDownloader
           │   ├── yfinance price data
           │   └── Custom Excel imports
           │
           ├── RegimeDetector (crisis identification)
           │   ├── Drawdown method
           │   ├── VIX threshold method
           │   ├── Volatility percentile method
           │   ├── Markov regime-switching
           │   └── Ensemble (majority voting)
           │
           ├── Individual Hedge Analysis (parallel)
           │   ├── Correlation breakdown (overall, crisis, normal)
           │   ├── Optimal weight finder for 10%/25%/50% targets
           │   └── Statistical hypothesis tests
           │
           ├── Multi-Asset Portfolio Optimization
           │   ├── Greedy sequential allocation
           │   └── Portfolio analytics
           │
           └── HTML Report Generation
               ├── Summary tables
               ├── Rolling correlation charts
               └── Performance metrics
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/data/downloader.py` | Data acquisition from yfinance + custom Excel files |
| `src/regime/detector.py` | Crisis period detection using 5 methods |
| `src/metrics/tail_risk.py` | CVaR, Maximum Drawdown, Sortino, Calmar |
| `src/metrics/correlations.py` | Conditional correlations, tail dependence |
| `src/hypothesis/tests.py` | Bootstrap CVaR tests, Baur-Lucey regression |
| `src/optimization/weight_finder.py` | Binary search for optimal hedge weights |
| `src/optimization/multi_asset.py` | Multi-asset portfolio construction |
| `src/backtester/rebalancing.py` | Quarterly rebalancing simulation |
| `src/reporting/report.py` | Professional HTML report generation |

---

## Regime Detection Methodology

The framework identifies **crisis vs. normal market regimes** using an ensemble of five methods:

### 1. Drawdown-Based Detection

**Formula:**
$$D_t = \frac{P_t - M_t}{M_t}, \quad M_t = \max_{s \leq t} P_s$$

**Classification:**
- Crisis: Drawdown ≤ -10% (configurable)
- Normal: Drawdown > -10%

### 2. VIX Threshold

**Classification:**
- Crisis: VIX ≥ 30
- Elevated: 20 ≤ VIX < 30
- Normal: VIX < 20

### 3. Volatility Percentile

Rolling realized volatility compared to its historical distribution:

$$\text{Crisis}_t = \mathbf{1}\left[\sigma_t > Q_{80\%}(\sigma)\right]$$

Where $\sigma_t$ is rolling 63-day volatility.

### 4. Hamilton Markov Regime-Switching

A 2-state model where returns are drawn from state-dependent distributions:

$$r_t \mid S_t = j \sim \mathcal{N}(\mu_j, \sigma_j^2), \quad j \in \{1, 2\}$$

The Hamilton filter provides filtered probabilities:

$$P(S_t = \text{Crisis} \mid \mathcal{F}_t) > 0.5 \Rightarrow \text{Crisis}$$

### 5. Ensemble (Default)

**Majority voting** across all available methods. A period is classified as crisis if ≥50% of methods agree.

---

## Tail-Risk Metrics

### Conditional Value-at-Risk (CVaR / Expected Shortfall)

The expected loss given that loss exceeds VaR at confidence level α:

$$\text{CVaR}_\alpha = -\mathbb{E}[R \mid R \leq \text{VaR}_\alpha]$$

**Historical estimation (α = 0.95):**
$$\text{CVaR}_{95} = -\frac{1}{|\{r_t : r_t \leq q_{5\%}\}|} \sum_{r_t \leq q_{5\%}} r_t$$

*Interpretation*: CVaR₉₅ = 5.2% means "When losses exceed the 95th percentile, the average loss is 5.2%"

### Maximum Drawdown (MDD)

Largest peak-to-trough decline before a new peak:

$$\text{MDD} = \max_{t} \left( \frac{M_t - P_t}{M_t} \right)$$

### Downside Deviation

Standard deviation of returns below a minimum acceptable return (MAR = 0):

$$\sigma_d = \sqrt{\frac{1}{n} \sum_{t=1}^{n} \min(r_t, 0)^2}$$

### Sortino Ratio

Risk-adjusted return using downside deviation:

$$\text{Sortino} = \frac{\bar{R}}{\sigma_d}$$

### Calmar Ratio

Return per unit of drawdown risk:

$$\text{Calmar} = \frac{\text{CAGR}}{\text{MDD}}$$

### Lower Tail Dependence (Clayton Copula)

Probability that both assets experience extreme losses simultaneously:

$$\lambda_L = \lim_{u \to 0^+} P(U_2 \leq u | U_1 \leq u) = 2^{-1/\theta}$$

Where θ is the Clayton copula parameter estimated from Kendall's tau:

$$\theta = \frac{2\tau}{1 - \tau}$$

*Ideal hedge*: λ_L ≈ 0 (hedge doesn't crash when equity crashes)

---

## Correlation Analysis

The framework computes three types of correlation:

### Overall Correlation
Standard Pearson correlation across all periods.

### Crisis Correlation
Correlation measured only during crisis regimes:

$$\rho_{\text{crisis}} = \text{Corr}(R_{\text{equity}}, R_{\text{hedge}} \mid \text{Regime} = \text{Crisis})$$

### Normal Correlation
Correlation during non-crisis periods.

### Downside Beta

Sensitivity when equity returns are negative:

$$\beta_{\text{down}} = \frac{\text{Cov}(R_H, R_E \mid R_E < 0)}{\text{Var}(R_E \mid R_E < 0)}$$

---

## Statistical Hypothesis Testing

### CVaR Reduction Test (Bootstrap)

**Hypotheses:**
- H₀: CVaR(hedged) ≥ CVaR(unhedged) — hedge does not help
- H₁: CVaR(hedged) < CVaR(unhedged) — hedge reduces tail risk

**Method:** Percentile bootstrap with 5,000+ resamples to construct confidence intervals for the difference.

### Safe-Haven Regression (Baur-Lucey)

Tests whether an asset provides protection specifically during extreme equity downturns:

$$R_{A,t} = \alpha + \beta_0 R_{\text{equity},t} + \beta_1 D_{10\%,t} R_{\text{equity},t} + \beta_2 D_{5\%,t} R_{\text{equity},t} + \beta_3 D_{1\%,t} R_{\text{equity},t} + \varepsilon_t$$

Where $D_{q\%,t}$ are dummy variables for equity returns in the worst q% quantile.

**Interpretation:**
- If $\beta_0 + \beta_k < 0$ for extreme quantiles, the asset is a **safe haven** during that stress level
- If $\beta_0 \approx 0$ on average but $\beta_0 + \beta_k < 0$, the asset hedges specifically during crashes

---

## Portfolio Optimization

### Single-Asset Optimization

For each hedge asset, find the weight that achieves target risk reduction:

1. Create portfolio: $(1-w) \times \text{ACWI} + w \times \text{Hedge}$
2. Compute CVaR and MDD for weights $w \in [0, \text{max\_weight}]$ at 1% increments
3. Find smallest $w$ where reduction ≥ target (10%, 25%, 50%)
4. Report optimal weight + associated metrics

### Multi-Asset Optimization

**Greedy Sequential Allocation:**
1. Start with 100% equity baseline
2. Iteratively add the asset that provides best marginal CVaR reduction
3. Continue until target reduction achieved or weight limits reached
4. Respect minimum ACWI weight (default 33%)

**CVaR Minimization:**
Direct optimization via scipy.optimize with constraints:
- Sum of weights = 1
- All weights ≥ 0
- ACWI weight ≥ minimum
- Each hedge ≤ individual maximum

---

## Asset Class Analysis

### Government Bonds (TLT, IEF)

**Expected behavior:** Strong safe haven in deflationary crises (2008, 2020); potential failure in inflationary shocks (2022).

**Key finding from literature:** Stock-bond correlation varies by regime:
- 2000-2022 average: **-0.31** (good hedge)
- 1970-1999 average: **+0.35** (poor hedge during inflation)

### Gold (GLD)

**Expected behavior:** Classic safe haven with near-zero tail dependence to equities.

**Key finding:** Baur & Lucey (2010) showed gold is uncorrelated on average and maintains low/negative correlation during crashes.

### Silver (SLV)

**Expected behavior:** Weaker hedge than gold due to industrial demand component.

**Key finding:** Safe-haven effect is shorter-lived and has weakened post-2000.

### Managed Futures (DBMF)

**Expected behavior:** "Crisis alpha" from trend-following that can short falling markets.

**Key finding:** CTAs generated positive returns in most crisis periods by diversifying across asset classes and quickly reducing exposure to crashing sectors.

### Bitcoin (BTC-USD) and Ethereum (ETH-USD)

**Expected behavior:** Diversifier but **not** a reliable safe haven.

**Key finding:** Bitcoin's correlation with equities turns positive during high-volatility regimes. During March 2020 COVID crash, Bitcoin fell 39% alongside equities.

---

## Quarterly Rebalancing Simulation

The framework simulates realistic portfolio management with periodic rebalancing:

### Rebalancing Process

1. **Calendar-based reset** at quarter-end (configurable frequency)
2. Weights drift with relative price movements between rebalances
3. Full recalculation of portfolio returns with drift

### Formula

Between rebalances, weights drift according to relative asset performance:

$$w_{i,t} = \frac{w_{i,\text{rebal}} \times (1 + R_{i,t})}{\sum_j w_{j,\text{rebal}} \times (1 + R_{j,t})}$$

At each rebalance date, weights reset to target allocation.

---

## Key Academic References

1. **Baur, D.G. & Lucey, B.M. (2010)** — "Is Gold a Hedge or a Safe Haven?" *The Financial Review*. Defines safe haven as an asset uncorrelated with stocks in a crash.

2. **Hamilton, J.D. (1989)** — "A New Approach to the Economic Analysis of Nonstationary Time Series." *Econometrica*. Foundational Markov regime-switching model.

3. **Li, L. & Chen, C.R. (2024)** — "When Safe-Haven Asset Is Less than a Safe-Haven Play." *J. Financial Econometrics*. Four-state regime-switching showing Bitcoin has positive crisis correlation.

4. **Sarkar, S. et al. (2022)** — "Testing the safe-haven properties of gold and bitcoin." *Finance Research Letters*. Wavelet quantile correlation approach; gold time-varying, Bitcoin weak safe haven.

5. **Asif, R., Frömmel, M., & Mende, A. (2022)** — "The crisis alpha of managed futures." *International Review of Financial Analysis*. CTAs earn positive returns in most crises.

6. **Molenaar et al. (2023)** / **Swedroe (2023)** — Stock-bond correlation varies by inflation regime; +0.35 in 1970-99 vs. -0.31 in 2000-22.

---

## Limitations & Future Work

### Current Limitations

1. **Survivorship bias**: Only analyzes assets that exist today with sufficient history
2. **Look-ahead in regime detection**: Markov smoother uses full sample (filtered probabilities available for real-time)
3. **No transaction costs**: Rebalancing simulation doesn't include bid-ask spreads or commissions
4. **Single baseline**: Only ACWI tested; could extend to other equity benchmarks
5. **Limited CTA history**: DBMF starts 2019, limiting long-term analysis

### Potential Extensions

1. **Walk-forward validation**: Rolling out-of-sample testing
2. **Transaction cost modeling**: Include rebalancing friction (2-10 bps)
3. **Currency hedging analysis**: Safe-haven currencies (CHF, JPY)
4. **Options overlay**: Compare explicit tail hedges (puts) vs. asset-based hedges
5. **Regime-conditional allocation**: Dynamic weights based on current regime probability
6. **Inflation regime overlay**: Adjust bond allocation based on inflation expectations

---

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run full analysis
python main.py

# Run tests
make tests

# Output
# → output/tail_risk_analysis_YYYY.MM.DD.HH.MM.SS.html
```

### Configuration

Edit `config.yaml` to:
- Add/remove hedge assets
- Adjust target reduction levels (10%, 25%, 50%)
- Change regime detection method
- Modify weight constraints
- Set rebalancing frequency

---

## License

MIT License. This is open research intended for educational and research purposes.

---

*Developed as part of quantitative finance research into evidence-based portfolio tail-risk hedging.*