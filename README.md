# Volatility Forecasting Pipeline  
### Multi-Horizon Statistical & Economic Evaluation

---

## Overview

This project implements a complete volatility forecasting pipeline in Python.

The objective is to:

- Construct multi-horizon realized volatility targets  
- Compare Rolling, EWMA and GARCH(1,1) models  
- Evaluate forecasts statistically (MAE, MSE, QLIKE)  
- Assess economic value via a volatility targeting strategy  
- Visualize volatility dynamics using interactive 3D surfaces  

The notebook follows a research-style structure and focuses on clarity, reproducibility and economic interpretation.
Full HTML notebook available here:  
https://Morales-Mathieu.github.io/volatility-forecasting/docs/index.html

---

## Models Implemented

### 1. Rolling Volatility
- 20-day window
- 60-day window
- Baseline model

### 2. EWMA (RiskMetrics)
- Recursive variance update
- Configurable persistence parameter (lambda)

### 3. GARCH(1,1)
- Zero-mean specification
- Normal innovations
- Walk-forward refitting
- Rolling training window

All volatility measures are annualized for consistency.

---

## Multi-Horizon Framework

Forecast horizons:

- 1 day
- 5 days
- 10 days

This allows analysis of volatility term structure and persistence dynamics.

---

## Statistical Evaluation

Forecasts are evaluated using:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- QLIKE (standard volatility forecasting loss)

Models are ranked primarily by QLIKE.

---

## Economic Evaluation

A volatility targeting strategy is implemented:

- Exposure scales inversely with forecast volatility
- Portfolio returns use lagged forecasts
- Performance metrics:
  - Annualized return
  - Annualized volatility
  - Sharpe ratio
  - Maximum drawdown

This tests whether better volatility forecasts translate into improved portfolio performance.

---

## 3D Volatility Surfaces

Interactive Plotly 3D surfaces are generated for:

- Realized volatility
- EWMA forecasts
- GARCH forecasts
- Forecast error surfaces

These visualizations allow inspection of:

- Volatility clustering
- Term structure evolution
- Model bias across regimes

---

## Project Structure

```bash
volatility-forecasting-pipeline/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_main.ipynb
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── metrics.py
│   ├── strategy.py
│   ├── plots.py
│   └── utils.py
│
├── docs/
│   └── index.html
│
├── requirements.txt
└── README.md
```
---

## How to Run

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```


## Run the Main Notebook

Open and execute:
```bash
    notebooks/01_main.ipynb
```

Outputs are automatically saved in:

- data/raw
- data/processed
- reports/figures
- reports/tables

---

## Stylized Facts Verified

The project confirms classical empirical properties:

- Low autocorrelation of returns
- Strong autocorrelation of squared returns
- Volatility clustering
- Heavy-tailed return distribution

---

## Robustness Analysis

Sensitivity tests are performed on:

- EWMA persistence parameter
- GARCH refitting frequency

This ensures conclusions are not driven by a specific calibration.

---

## Possible Extensions

- Student-t GARCH
- HAR-RV model
- Implied volatility integration
- Regime-switching models
- Multi-asset extension
- Transaction costs in backtests


