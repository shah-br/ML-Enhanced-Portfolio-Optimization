# 📈 AI-Driven Quantitative Portfolio Optimization Engine

> **A quantitative investment system combining Mean-Variance Optimization, XGBoost, and the Black-Litterman model to generate alpha-producing portfolio allocations with live trading signals.**

---

## Key Results

### Strategy Performance Comparison

| Strategy | CAGR | Excess vs Market |
|----------|:----:|:----------------:|
| **ML & MV Optimized** | 18.46%/yr | +3.95% |
| **MV Optimized** | 19.71%/yr | +5.20% |
| **Original Unoptimized** | 17.70%/yr | +3.19% |
| **Market Index (SPY)** | 14.51%/yr | — |

All three portfolio strategies beat SPY — and the portfolio **outperformed the market every single year** from 2015–2021, including 2018 where SPY returned -4.57% but the portfolio stayed positive at +2.19%.

### ML+MV Strategy — Risk & Performance Profile

| Metric | Value |
|--------|:-----:|
| **Sharpe Ratio** | 0.82 |
| **Sortino Ratio** | 1.28 |
| **Alpha (annualized)** | 4.64% |
| **Beta** | 0.94 |
| **Information Ratio** | 0.78 |
| **Annualized Volatility** | 17.34% |
| **Max Drawdown** | -35.5% |
| **VaR 95% (daily)** | 1.53% |
| **VaR 99% (daily)** | 3.06% |
| **CVaR 95% (daily)** | 2.58% |
| **Optimal Frontier Sharpe** | 1.35 |

---

## How It Works

```
 ┌──────────────┐
 │  config.py   │  30 stocks, 7 sectors, $30K portfolio, constraints
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │ Download     │  Historical prices from Yahoo Finance (yfinance)
 │ Stock Data   │
 └──────┬───────┘
        ├─────────────────────────────────────┐
        ▼                                     ▼
 ┌──────────────┐                    ┌─────────────────┐
 │ MV Optimize  │                    │ ML Feature Eng.  │  20+ indicators:
 │ (SLSQP)      │                    │ + XGBoost Train  │  RSI, MACD, Bollinger,
 │              │                    │                  │  momentum, volatility,
 │ Maximize     │                    │ 80/20 chrono     │  lagged prices
 │ Sharpe Ratio │                    │ split (no leak)  │
 └──────┬───────┘                    └────────┬────────┘
        │                                     ▼
        │                            ┌─────────────────┐
        │                            │ Black-Litterman  │  Blend market equilibrium
        │                            │ Model (τ=0.025)  │  with ML predictions
        │                            └────────┬────────┘
        │                                     ▼
        │                            ┌─────────────────┐
        │                            │ ML-Enhanced MVO  │  Re-optimize with
        │                            │ (SLSQP)          │  blended returns
        │                            └────────┬────────┘
        │                                     │
        └─────────────┬───────────────────────┘
                      ▼
               ┌──────────────┐
               │  Backtest    │  Rolling walk-forward, quarterly rebalance,
               │  Engine      │  0.1% slippage per turnover
               └──────┬───────┘
                      ▼
               ┌──────────────┐
               │  Risk &      │  Sharpe, Sortino, VaR, CVaR, Alpha, Beta,
               │  Analytics   │  Fama-French 3-factor decomposition
               └──────┬───────┘
                      ▼
               ┌──────────────┐
               │  Dashboard   │  6-tab Streamlit app + Plotly charts
               │  (app.py)    │  + live trading signals
               └──────────────┘
```

**Two strategies compete:** MV Optimized (pure math) vs ML+MV Optimized (math + XGBoost via Black-Litterman), both benchmarked against SPY.

---

## Project Structure

```
MLPO/
├── config.py                        # Tickers, dates, constraints, hyperparams
├── main.py                          # Script entry point → Plotly chart
├── app.py                           # Streamlit dashboard (6 tabs)
├── mean_variance_optimization.py    # SLSQP optimizer, efficient frontier
├── machine_learning_strategies.py   # XGBoost/RF/GB/LR, features, signals
├── black_litterman_model.py         # Market caps, equilibrium, BL blending
├── portfolio_statistics.py          # Sharpe, Sortino, VaR, CVaR, Beta, Alpha
├── factor_analysis.py               # Fama-French 3-factor regression
├── backtesting_engine.py            # Rolling walk-forward backtest
├── live_allocations.py              # CLI live recommendations
└── tests/                           # Unit tests (pytest)
```

---

## ML Model Comparison

Time-Series CV with 5 forward-rolling folds:

| Model | RMSE | MAE |
|-------|:----:|:---:|
| **XGBoost** (primary) | 0.0931 | 0.0741 |
| Random Forest | 0.0961 | 0.0761 |
| Gradient Boosting | 0.0972 | 0.0775 |
| Linear Regression | 0.1140 | 0.0913 |

XGBoost achieves the lowest prediction error and is used as the primary model. Hyperparameter tuning available via `RandomizedSearchCV` with `TimeSeriesSplit`.

---

## Dashboard

Run with `streamlit run app.py` → opens at `http://localhost:8501`

| Tab | What It Does |
|-----|-------------|
| **Rolling Backtest** | Walk-forward backtest with cumulative return chart (all 4 strategies) |
| **Efficient Frontier** | Frontier curve + max-Sharpe point (1.35) + sector allocation treemap |
| **Forecast & Signals** | Live Buy/Sell/Hold signals per stock with ML confidence scores |
| **Risk Dashboard** | Full risk suite: Sharpe, VaR, CVaR, drawdown chart, return histogram |
| **Factor Analysis** | Fama-French 3-factor loadings (Market, SMB, HML) + alpha per stock |
| **Model Comparison** | Compare XGBoost/RF/GB/LR head-to-head with R² chart + feature importance |

---

## Trading Signals (Sample)

| Ticker | Price | Trend | ML Forecast | Action |
|:------:|------:|:-----:|:-----------:|:------:|
| AAPL | $270.71 | ↑ | +15.67% | 🟢 Strong Buy |
| NVDA | $213.17 | ↑ | +31.13% | 🟢 Strong Buy |
| GOOGL | $349.78 | ↑ | +22.32% | 🟢 Strong Buy |
| JPM | $311.45 | ↑ | +17.40% | 🟢 Strong Buy |
| ADBE | $243.20 | ↓ | -3.57% | ⚪ Hold |
| PFE | $26.48 | ↓ | -0.25% | ⚪ Hold |

Signals combine **50-day MA trend** with **XGBoost predicted annualized return**. Upward trend + high forecast = Strong Buy. Conflicting signals = Hold.

---

## Getting Started

```bash
# Clone & install
git clone https://github.com/your-username/MLPO.git
cd MLPO
pip install -r requirements.txt

# Option A: Script-based analysis
python main.py

# Option B: Interactive dashboard
streamlit run app.py

# Option C: Live allocation recommendations
python live_allocations.py

# Run tests
cd tests/ && pytest
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Data** | Yahoo Finance (`yfinance`) |
| **ML** | XGBoost, scikit-learn |
| **Optimization** | SciPy (SLSQP) |
| **Statistics** | NumPy, Pandas, statsmodels |
| **Visualization** | Plotly, Streamlit |
| **Backtesting** | Custom walk-forward engine with slippage |
