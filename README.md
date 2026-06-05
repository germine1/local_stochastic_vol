# 📊 Local & Stochastic Volatility Engine

A comprehensive Python framework for advanced volatility modeling, option pricing, and risk management. Built for quantitative analysts, traders, and researchers in Singapore and globally.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🚀 Features

### ** Volatility Models**

| Model | Type | Best For | Key Features |
|-------|------|----------|--------------|
| **Dupire (Local Vol)** | Deterministic | Vanilla options, exotic path-dependent | Perfect smile fit, fast calibration |
| **Heston** | Stochastic Vol | Vol derivatives, long-dated options | Mean-reverting vol, semi-analytical formulas |
| **SABR** | Stochastic Vol | Interest rate options, FX | Industry standard, Hagan formula |
| **Merton Jump** | Jump Diffusion | Earnings events, crashes | Lognormal jumps, analytical pricing |
| **Kou Jump** | Jump Diffusion | Asymmetric tail risk | Double exponential jumps, heavy tails |

### **3 Numerical Methods**

- **Monte Carlo Simulation**
  - European, Asian, Barrier, American (LSM) options
  - Variance reduction (antithetic variates, control variates)
  - Parallel processing support

- **Carr-Madan FFT**
  - O(N log N) complexity → price 1000s of strikes instantly
  - Works with any model's characteristic function
  - Spectral accuracy

- **Finite Difference PDE**
  - Explicit, Implicit, Crank-Nicolson schemes
  - European and American options
  - Free boundary problems

### **Comprehensive Risk Analytics**

- **15+ Greeks**: Delta, Gamma, Vega, Theta, Rho, Vanna, Volga, Charm, Speed, Zomma, Color, etc.
- **Analytical formulas** (where available) + **finite difference** (universal)
- **Portfolio Greeks** aggregation
- **Greeks heatmaps** and sensitivity analysis

### **Interactive Visualizations**

- **3D Volatility Surfaces** (Plotly interactive)
- **Price Path Simulations** with confidence bands
- **Greeks Dashboards**
- **Calibration Diagnostics**
- **Model Comparison** plots
- **Export to HTML** for web deployment (Vercel, GitHub Pages)

### **Market Data Integration**

- ✅ **yfinance** (free real-time data)
- ⚠️ **Bloomberg** (stub ready, requires license)
- Automatic implied volatility surface extraction
- Historical data analysis

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/volatility-engine.git
cd volatility-engine

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
