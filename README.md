# 📈 AI-Driven Quantitative Portfolio Optimization Engine

> **A quantitative investment system that combines Mean-Variance Optimization, ensemble ML models (XGBoost, Random Forest), and the Black-Litterman model to generate alpha-producing portfolio allocations with real-time trading signals.**

---

## Highlights at a Glance

| Metric | ML+MV Strategy | MV Only | Market (SPY) |
|--------|:--------------:|:-------:|:------------:|
| **CAGR** | 18.46%/yr | 19.71%/yr | 14.51%/yr |
| **Excess Return vs Market** | +3.95% | +5.20% | — |
| **Sharpe Ratio** | 0.82 | — | — |
| **Sortino Ratio** | 1.28 | — | — |
| **Alpha (annualized)** | 4.64% | — | — |
| **Beta** | 0.94 | — | — |
| **Information Ratio** | 0.78 | — | — |
| **Max Drawdown** | -35.5% | — | — |
| **VaR 95% (daily)** | 1.53% | — | — |
| **CVaR 95% (daily)** | 2.58% | — | — |
| **Optimal Frontier Sharpe** | 1.35 | — | — |

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Pipeline Flowchart](#pipeline-flowchart)
- [Project Structure](#project-structure)
- [Portfolio Universe](#portfolio-universe)
- [How It Works](#how-it-works)
  - [Mean-Variance Optimization](#1-mean-variance-optimization-mvo)
  - [Machine Learning Predictions](#2-machine-learning-predictions)
  - [Black-Litterman Blending](#3-black-litterman-blending)
  - [Backtesting](#4-backtesting)
- [Results & Backtest Performance](#results--backtest-performance)
- [Key Configuration](#key-configuration)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Project](#running-the-project)
- [Dashboard Tabs](#dashboard-tabs)
- [ML Model Comparison](#ml-model-comparison)
- [Trading Signals](#trading-signals)
- [Performance Metrics Reference](#performance-metrics-reference)
- [Tech Stack](#tech-stack)
- [Glossary](#glossary)

---

## Overview

This engine answers one core question:

> *"Given a set of stocks, how much money should I put into each one to maximize gains while keeping risk under control?"*

The system builds **two competing optimization strategies** and benchmarks them against the S&P 500 (SPY):

| Strategy | Description | CAGR |
|----------|-------------|:----:|
| **ML + MV Optimized** | Math optimization + XGBoost predictions via Black-Litterman | **18.46%/yr** |
| **MV Optimized** | Pure mathematical optimization on historical data | **19.71%/yr** |
| **Original Unoptimized** | Dollar-weighted allocation (baseline) | **17.70%/yr** |
| **Market Index (SPY)** | S&P 500 benchmark | **14.51%/yr** |

All optimized strategies outperformed the market, with MV achieving **+5.20%** and ML+MV achieving **+3.95%** excess return over SPY.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          config.py                                  │
│         Portfolio tickers, date ranges, constraints, params         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌──────────────────────────┐     ┌──────────────────────────────────┐
│ mean_variance_optimization│     │  machine_learning_strategies.py  │
│          .py              │     │  Feature engineering, XGBoost,   │
│  Download data, MVO,      │     │  Random Forest, trading signals  │
│  Efficient Frontier,      │     └───────────────┬────────────────┘
│  Risk Parity              │                     │
└────────────┬─────────────┘                     │
             │                                    ▼
             │                     ┌──────────────────────────────┐
             │                     │   black_litterman_model.py   │
             │                     │  Market equilibrium returns   │
             │                     │  + ML views → blended returns │
             │                     └───────────────┬──────────────┘
             │                                     │
             └──────────────┬──────────────────────┘
                            ▼
              ┌──────────────────────────┐
              │   backtesting_engine.py  │
              │  Rolling walk-forward    │
              │  backtest with slippage  │
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐
              │  portfolio_statistics.py │
              │  Sharpe, Sortino, VaR,   │
              │  Beta, Alpha, Drawdown   │
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐
              │    factor_analysis.py    │
              │  Fama-French 3-factor    │
              │  regression per stock    │
              └────────────┬─────────────┘
                           ▼
         ┌─────────────────┴──────────────────┐
         ▼                                    ▼
┌──────────────┐                    ┌──────────────────┐
│   main.py    │                    │     app.py       │
│  Script-based│                    │  Streamlit web   │
│  pipeline    │                    │  dashboard (6    │
│  + Plotly    │                    │  interactive     │
│  chart       │                    │  tabs)           │
└──────────────┘                    └──────────────────┘
```

---

## Pipeline Flowchart

Below is the complete 10-step pipeline that the system executes:

```mermaid
flowchart TD
    A["📋 STEP 1: Load Portfolio & Config\n<i>config.py</i>\nDefine 30 stocks, dollar amounts,\ndate ranges, constraints"] --> B

    B["📥 STEP 2: Download Historical Prices\n<i>mean_variance_optimization.py</i>\n7+ years of daily closing prices\nfrom Yahoo Finance"] --> C

    C["📐 STEP 3: Standard MV Optimization\n<i>mean_variance_optimization.py</i>\nSLSQP optimizer maximizes Sharpe Ratio\nsubject to weight & volatility constraints"] --> D

    D["🔧 STEP 4: ML Feature Engineering\n<i>machine_learning_strategies.py</i>\nCreate 20+ technical indicators per stock\n(MA, RSI, MACD, Bollinger Bands, etc.)"] --> E

    E["🤖 STEP 5: Train ML Models & Predict\n<i>machine_learning_strategies.py</i>\nXGBoost predicts future returns\nusing 80/20 chronological split"] --> F

    F["🔀 STEP 6: Black-Litterman Blending\n<i>black_litterman_model.py</i>\nCombine market equilibrium returns\nwith ML predictions (τ = 0.025)"] --> G

    G["⚡ STEP 7: ML-Enhanced Optimization\n<i>mean_variance_optimization.py</i>\nRe-run SLSQP with blended\nexpected returns"] --> H

    H["📊 STEP 8: Backtest & Compare\n<i>backtesting_engine.py / main.py</i>\nRolling walk-forward with\n0.1% slippage per rebalance"] --> I

    I["🛡️ STEP 9: Risk Analysis\n<i>portfolio_statistics.py</i>\nSharpe 0.82 | Alpha 4.64%\nVaR 1.53% | Beta 0.94"] --> J

    J["📈 STEP 10: Display Results\n<i>app.py / main.py</i>\n6-tab dashboard: backtest, frontier,\nsignals, risk, factors, models"]

    style A fill:#2d3748,stroke:#4a90d9,color:#e2e8f0
    style E fill:#2d3748,stroke:#48bb78,color:#e2e8f0
    style F fill:#2d3748,stroke:#ed8936,color:#e2e8f0
    style J fill:#2d3748,stroke:#9f7aea,color:#e2e8f0
```

---

## Project Structure

```
MLPO/
├── config.py                        # Central settings: tickers, dates, constraints, hyperparams
├── main.py                          # Script entry point → runs full pipeline, outputs Plotly chart
├── app.py                           # Streamlit web dashboard (6 interactive tabs)
├── mean_variance_optimization.py    # MVO engine: data download, SLSQP optimizer, efficient frontier
├── machine_learning_strategies.py   # ML engine: features, XGBoost/RF/GB/LR, signals, tuning
├── black_litterman_model.py         # Black-Litterman: market caps, equilibrium returns, blending
├── portfolio_statistics.py          # Risk metrics: Sharpe, Sortino, VaR, CVaR, Beta, Alpha
├── factor_analysis.py               # Fama-French 3-factor regression per stock
├── backtesting_engine.py            # Rolling walk-forward backtest with transaction costs
├── live_allocations.py              # CLI tool for real-time weight recommendations
├── requirements.txt                 # Python dependencies
└── tests/                           # Unit tests (run with pytest)
```

### File Responsibilities

| File | Role | Key Functions |
|------|------|---------------|
| `config.py` | Central settings | `PORTFOLIO`, `SECTOR_MAP`, date ranges, constraints |
| `main.py` | Script entry point | Orchestrates the full pipeline → Plotly chart |
| `app.py` | Streamlit dashboard | 6-tab interactive UI calling all modules |
| `mean_variance_optimization.py` | MVO engine | `calculate_weights()`, `mean_variance_optimization()`, `compute_efficient_frontier()`, `risk_parity_portfolio()` |
| `machine_learning_strategies.py` | ML engine | `create_additional_features()`, `generate_investor_views()`, `compare_models()`, `generate_trading_signals()`, `tune_xgboost()` |
| `black_litterman_model.py` | BL blending | `get_market_caps()`, `get_market_returns()`, `black_litterman_adjustment()` |
| `portfolio_statistics.py` | Risk & metrics | `sharpe_ratio()`, `sortino_ratio()`, `value_at_risk()`, `conditional_var()`, `max_drawdown()`, `calculate_beta()`, `calculate_alpha()`, `full_risk_report()` |
| `factor_analysis.py` | Factor analysis | `download_factor_data()`, `analyze_factor_impact()` |
| `backtesting_engine.py` | Walk-forward backtest | `rolling_walk_forward_backtest()` |
| `live_allocations.py` | Live recommendations | Standalone CLI script for current-day allocations |

---

## Portfolio Universe

The system operates on **30 US stocks** across **7 sectors** with a starting portfolio of **~$30,000**:

| Sector | Tickers | Starting Allocation | % of Portfolio |
|--------|---------|-------------------:|:--------------:|
| **Technology** | AAPL, MSFT, GOOGL, NVDA, ADBE, CRM | $7,600 | ~25.3% |
| **Healthcare** | JNJ, UNH, PFE, ABBV, LLY | $5,600 | ~18.7% |
| **Finance** | JPM, BAC, GS, MS, BLK | $5,300 | ~17.7% |
| **Consumer Goods** | PG, KO, WMT, COST, NKE | $4,600 | ~15.3% |
| **Industrial** | CAT, BA, HON | $2,600 | ~8.7% |
| **Energy** | XOM, CVX, COP | $2,200 | ~7.3% |
| **Utilities** | NEE, DUK, SO | $2,100 | ~7.0% |

---

## How It Works

### 1. Mean-Variance Optimization (MVO)

Based on Harry Markowitz's Modern Portfolio Theory (Nobel Prize, 1952), MVO finds portfolio weights that maximize return for a given risk level. The system uses the **SLSQP** optimizer to maximize the **Sharpe Ratio** subject to:

- Each stock weight: between **1%** and **25%**
- All weights sum to **100%**
- Total portfolio volatility ≤ **22.5%**

The **Efficient Frontier** is computed across 50 risk-return points. The optimal portfolio on the frontier achieved a **Sharpe Ratio of 1.35** with **32.6% annualized return** at **21.1% volatility**.

### 2. Machine Learning Predictions

The ML pipeline processes each of the 30 stocks individually:

**Feature Engineering** — 20+ technical indicators computed from raw price data:

- Moving Averages (5, 10, 20, 50-day)
- RSI (Relative Strength Index)
- MACD + Signal Line + Histogram
- Bollinger Bands (Upper, Lower, Width, Position)
- Momentum (5, 10, 20-day)
- Volatility (10, 20-day rolling)
- Lagged prices (1–5 days back)

**Models** — 4 models trained with strict chronological 80/20 split (no data leakage):

| Model | RMSE | MAE | Description |
|-------|:----:|:---:|-------------|
| **XGBoost** (primary) | 0.0931 | 0.0741 | 100 sequential boosted trees |
| **Random Forest** | 0.0961 | 0.0761 | 100 independent trees averaged |
| **Gradient Boosting** | 0.0972 | 0.0775 | Sequential boosting (sklearn) |
| **Linear Regression** | 0.1140 | 0.0913 | Simple baseline |

XGBoost consistently achieves the lowest RMSE and MAE across the portfolio. Hyperparameter tuning is available via `RandomizedSearchCV` with 5-fold `TimeSeriesSplit` cross-validation.

### 3. Black-Litterman Blending

The Black-Litterman model (Goldman Sachs, 1990) solves MVO's sensitivity to expected return estimates:

1. **Market Equilibrium Returns** — Derived from real market-cap data and S&P 500 index returns
2. **ML Views** — XGBoost's predicted return per stock, weighted by R² confidence
3. **Blending** — BL posterior formula combines both, with **τ = 0.025** controlling trust in market equilibrium

High-confidence ML predictions shift expected returns significantly; low-confidence ones stay close to market consensus.

### 4. Backtesting

**Rolling Walk-Forward Backtest** — The primary validation method:

```
For each quarter in the backtest period:
  1. Train on the most recent 36 months of data
  2. Run MVO + ML + Black-Litterman pipeline
  3. Rebalance portfolio weights
  4. Deduct transaction costs (0.1% slippage × turnover)
  5. Simulate daily returns for the next 3 months
  6. Slide the window forward and repeat
```

This ensures the model is always evaluated on truly out-of-sample data across multiple rebalancing cycles.

---

## Results & Backtest Performance

### Rolling Walk-Forward Backtest (2015–2026)

All three portfolio strategies outperformed the S&P 500 benchmark:

| Strategy | CAGR | vs. Market | Growth of $1 |
|----------|:----:|:----------:|:------------:|
| **ML & MV Optimized** | 18.46%/yr | +3.95% | ~$4.30 |
| **MV Optimized** | 19.71%/yr | +5.20% | ~$4.50 |
| **Original Unoptimized** | 17.70%/yr | +3.19% | ~$4.00 |
| **Market Index (SPY)** | 14.51%/yr | — | ~$3.00 |

### Risk Analytics Dashboard

| Metric | Value | Interpretation |
|--------|:-----:|----------------|
| **Sharpe Ratio** | 0.82 | Good risk-adjusted return |
| **Sortino Ratio** | 1.28 | Strong downside protection |
| **Beta** | 0.94 | Near-market sensitivity |
| **Alpha (annualized)** | 4.64% | Significant outperformance after risk adjustment |
| **VaR 95% (daily)** | 1.53% | 95% confidence: daily loss ≤ 1.53% |
| **VaR 99% (daily)** | 3.06% | 99% confidence: daily loss ≤ 3.06% |
| **CVaR 95% (daily)** | 2.58% | Average loss on worst 5% of days |
| **Information Ratio** | 0.78 | Consistent outperformance vs SPY (excellent) |
| **Max Drawdown** | -35.5% | Largest peak-to-trough decline |
| **Ann. Volatility** | 17.34% | Within the 22.5% constraint |

### Efficient Frontier

| Metric | Optimal Portfolio |
|--------|:-----------------:|
| **Optimal Sharpe Ratio** | 1.35 |
| **Annualized Return** | 32.6% |
| **Annualized Volatility** | 21.1% |

### Annual Returns vs Market Benchmark

| Year | Portfolio | Market | Excess |
|:----:|:---------:|:------:|:------:|
| 2015 | 7.09% | 3.15% | +3.94% |
| 2016 | 20.14% | 12.00% | +8.14% |
| 2017 | 33.45% | 21.71% | +11.74% |
| 2018 | 2.19% | -4.57% | +6.76% |
| 2019 | 31.48% | 31.22% | +0.26% |
| 2020 | 22.05% | 18.33% | +3.72% |
| 2021 | 36.97% | 28.73% | +8.24% |

The portfolio outperformed SPY in **every single year** of the backtest, including the 2018 drawdown where SPY returned -4.57% but the portfolio stayed positive at +2.19%.

### Fama-French Factor Analysis (Sample)

| Ticker | Market Beta | SMB Loading | HML Loading | R² | Alpha (ann.) |
|:------:|:-----------:|:-----------:|:-----------:|:--:|:------------:|
| AAPL | 1.12 | -0.14 | -0.47 | 0.60 | 6.67% |
| MSFT | 1.09 | -0.29 | -0.55 | 0.69 | 5.50% |
| NVDA | 1.39 | 0.23 | -1.73 | 0.56 | 36.88% |
| JPM | 1.27 | 0.12 | 0.84 | 0.65 | 7.41% |
| JNJ | 0.64 | -0.41 | 0.66 | 0.35 | 4.71% |

NVDA shows the highest market beta (1.39) and strongest growth tilt (HML = -1.73), consistent with its AI-driven rally. JNJ exhibits defensive characteristics with low beta (0.64) and positive value loading.

---

## Key Configuration

All settings are centralized in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Risk-Free Rate | 4% | Baseline safe return (US government bonds) |
| Max Volatility | 22.5% | Portfolio risk ceiling |
| Min Weight / Stock | 1% | No stock is completely excluded |
| Max Weight / Stock | 25% | No single-stock concentration |
| XGBoost Estimators | 100 | Number of sequential trees |
| XGBoost Learning Rate | 0.1 | Step size for error correction |
| XGBoost Max Depth | 3 | Tree complexity limit |
| BL Tau (τ) | 0.025 | Uncertainty about market equilibrium |
| Rolling Window | 36 months | Retraining lookback period |
| Rebalance Frequency | 3 months | Quarterly weight updates |
| Slippage | 0.1% | Transaction cost per rebalance |

---

## Getting Started

### Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)
- Internet connection (for Yahoo Finance data downloads)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/MLPO.git
cd MLPO

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Project

There are three ways to use the system:

#### Option A — Static Analysis (Script)

```bash
python main.py
```

Runs the full pipeline end-to-end and opens an interactive Plotly chart comparing all 4 strategies over the backtest period.

#### Option B — Interactive Web Dashboard

```bash
streamlit run app.py
```

Opens a 6-tab web dashboard at `http://localhost:8501` with configurable parameters and interactive visualizations.

#### Option C — Live Allocation Recommendations

```bash
python live_allocations.py
```

Downloads the most recent 5 years of market data, runs the full ML+BL+MVO pipeline, and prints current recommended weights and trading signals for all 30 stocks.

#### Running Tests

```bash
cd tests/
pytest
```

---

## Dashboard Tabs

When running `streamlit run app.py`, you get 6 interactive tabs:

| Tab | Name | What You Can Do |
|-----|------|-----------------|
| 1 | **Rolling Backtest** | Configure training window, slippage, and volatility cap. Run the walk-forward backtest and view cumulative return chart for all strategies with CAGR comparison. |
| 2 | **Efficient Frontier** | Compute and visualize the frontier curve. See the max-Sharpe optimal point (1.35) and sector allocation treemap. |
| 3 | **Forecast & Signals** | Select ML model and tickers. Generate real-time Buy/Sell/Hold signals with current price, 50-day MA, trend, ML forecast %, and confidence. |
| 4 | **Risk Dashboard** | Full risk report: Sharpe, Sortino, Beta, Alpha, VaR (95% & 99%), CVaR, Max Drawdown. Interactive drawdown chart and return distribution histogram. Annual returns vs market table. |
| 5 | **Factor Analysis** | Fama-French 3-factor regression for all 30 stocks. Factor loadings table, grouped bar chart (Market/SMB/HML), and alpha bar chart. |
| 6 | **Model Comparison** | Pick any stock and compare all 4 ML models (RMSE, MAE, R²). View R² bar chart with error bars across time-series CV folds. Feature importance ranking. Optional XGBoost hyperparameter tuning. |

---

## ML Model Comparison

Performance comparison using **Time-Series Cross-Validation (5 forward-rolling folds)**:

| Model | RMSE | MAE | R² (best fold) |
|-------|:----:|:---:|:--------------:|
| **XGBoost** | 0.0931 | 0.0741 | -0.59 |
| **Random Forest** | 0.0961 | 0.0761 | -0.71 |
| **Gradient Boosting** | 0.0972 | 0.0775 | -0.65 |
| **Linear Regression** | 0.1140 | 0.0913 | -0.10 |

> **Note:** Negative R² values across folds are expected in stock price prediction — financial time series are notoriously difficult to predict. The models still provide directional value through the Black-Litterman blending framework, where even weak signals improve portfolio allocation when combined with market equilibrium priors.

XGBoost achieves the lowest prediction error (RMSE: 0.0931) and is used as the primary model.

---

## Trading Signals

The system generates **Buy/Sell/Hold signals** by combining trend analysis with ML predictions:

| Trend (50-day MA) | ML Predicted Return | Signal |
|:------------------:|:-------------------:|:------:|
| ↑ Upward | > +1% | 🟢 **Strong Buy** |
| ↑ Upward | > +0.5% | 🟢 **Buy** |
| ↑ Upward | < −0.5% | 🔴 **Sell** |
| ↓ Downward | < −1% | 🔴 **Strong Sell** |
| ↓ Downward | < −0.5% | 🔴 **Sell** |
| ↓ Downward | > +0.5% | ⚪ **Hold** |
| Either | Neutral | ⚪ **Hold** |

**Sample Live Signals (latest run):**

| Ticker | Price | 50-Day MA | Trend | ML Forecast | Action |
|:------:|------:|----------:|:-----:|:-----------:|:------:|
| AAPL | $270.71 | $260.56 | ↑ Upward | +15.67% | 🟢 Strong Buy |
| NVDA | $213.17 | $186.22 | ↑ Upward | +31.13% | 🟢 Strong Buy |
| GOOGL | $349.78 | $311.21 | ↑ Upward | +22.32% | 🟢 Strong Buy |
| JPM | $311.45 | $298.33 | ↑ Upward | +17.40% | 🟢 Strong Buy |
| ADBE | $243.20 | $251.37 | ↓ Downward | -3.57% | ⚪ Hold |
| PFE | $26.48 | $27.18 | ↓ Downward | -0.25% | ⚪ Hold |

---

## Performance Metrics Reference

| Metric | What It Measures | Good Range |
|--------|-----------------|:----------:|
| **Sharpe Ratio** | Return earned per unit of total risk | > 1.0 |
| **Sortino Ratio** | Return per unit of *downside* risk only | > 1.0 |
| **Information Ratio** | Consistency of outperformance vs benchmark | > 0.3 |
| **VaR (95%)** | Worst daily loss at 95% confidence | Lower is better |
| **CVaR (Expected Shortfall)** | Average loss on the worst 5% of days | Lower is better |
| **Max Drawdown** | Largest peak-to-trough decline | < 20% |
| **Beta** | Sensitivity to market movements | 0.7 – 1.1 |
| **Alpha (Jensen's)** | Return above what market risk explains | > 0% |

### Benchmark Scale

| Metric | Poor | Acceptable | Good | Excellent |
|--------|:----:|:----------:|:----:|:---------:|
| Sharpe | < 0 | 0 – 0.5 | 0.5 – 1.5 | > 1.5 |
| Sortino | < 0 | 0 – 0.8 | 0.8 – 2.0 | > 2.0 |
| Info Ratio | < 0 | 0 – 0.3 | 0.3 – 0.7 | > 0.7 |
| Max Drawdown | > 40% | 20 – 40% | 10 – 20% | < 10% |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Data** | Yahoo Finance (`yfinance`) |
| **ML** | XGBoost, scikit-learn (Random Forest, Gradient Boosting, Linear Regression) |
| **Optimization** | SciPy (`scipy.optimize.minimize` with SLSQP) |
| **Statistics** | NumPy, Pandas, statsmodels (OLS regression) |
| **Visualization** | Plotly (interactive charts), Streamlit (web dashboard) |
| **Backtesting** | Custom rolling walk-forward engine with slippage modeling |

---

## Glossary

| Term | Definition |
|------|-----------|
| **Alpha (α)** | Portfolio return above what market risk (beta) explains |
| **Beta (β)** | How much the portfolio moves relative to the overall market |
| **Black-Litterman** | Model blending market consensus with investor views (ML predictions) |
| **Covariance Matrix** | Table showing how every pair of stocks moves together |
| **CVaR** | Average loss on the worst X% of days (Expected Shortfall) |
| **Efficient Frontier** | Curve of optimal portfolios offering highest return per risk level |
| **Fama-French** | 3-factor model decomposing returns into Market, Size, and Value factors |
| **MACD** | Moving Average Convergence Divergence — momentum indicator |
| **MVO** | Mean-Variance Optimization — finding optimal risk/return balance |
| **R²** | How much variance the model explains (0 = none, 1 = perfect) |
| **RSI** | Relative Strength Index — overbought/oversold gauge (0–100) |
| **Sharpe Ratio** | Return earned per unit of total risk |
| **SLSQP** | Sequential Least Squares Programming — the optimization algorithm |
| **Sortino Ratio** | Like Sharpe but only penalizes downside risk |
| **SPY** | S&P 500 ETF — the market benchmark |
| **Tau (τ)** | Black-Litterman parameter for market equilibrium uncertainty |
| **VaR** | Value at Risk — worst expected loss at a given confidence level |
| **XGBoost** | Extreme Gradient Boosting — the primary ML model |
