# Stock-Portfolio-Optimizer
An interactive stock portfolio optimizer built with Streamlit using Markowitz MPT, Sharpe Ratio Optimization, and Monte Carlo Simulation.
# 📊 Stock Portfolio Optimizer

An interactive web application built with Python and Streamlit that helps 
investors analyze and optimize their stock portfolios using quantitative 
finance techniques.

## 🚀 Features

- 📈 **Stock Comparison** — Normalized price history, annualized 
  returns, volatility, and correlation heatmap
- ⚖️ **Risk vs Return Analysis** — Individual stock scatter plot 
  with Capital Market Line
- 🎯 **Efficient Frontier** — Markowitz Modern Portfolio Theory with 
  Max Sharpe and Min Volatility portfolios
- 🏆 **Sharpe Ratio Optimization** — Mathematically optimal portfolio 
  allocation with investment calculator
- 🎲 **Monte Carlo Simulation** — Thousands of random portfolio 
  simulations with Sharpe ratio heatmap

## 🛠️ Tech Stack

- **Streamlit** — Web app framework
- **yfinance** — Real-time stock data from Yahoo Finance
- **SciPy (SLSQP)** — Portfolio optimization engine
- **NumPy / Pandas** — Data processing and matrix computations
- **Plotly** — Interactive charts and visualizations

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio-optimizer.git
cd portfolio-optimizer

# Create conda environment
conda create -n portfolio-optimizer python=3.11
conda activate portfolio-optimizer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📌 Usage

1. Enter stock tickers in the sidebar (e.g. `AAPL,MSFT,GOOGL` or 
   Indian stocks like `TCS.NS,RELIANCE.NS`)
2. Set your date range and risk-free rate
3. Click **Run Optimizer**
4. Explore each tab for full analysis

## 📐 Models Used

| Model | Purpose |
|---|---|
| Markowitz MPT | Efficient Frontier construction |
| SLSQP Optimizer | Maximum Sharpe Ratio portfolio |
| Monte Carlo | Random portfolio simulation |
| Capital Market Line | Optimal risk-return tradeoff |
| Max Drawdown | Worst-case loss analysis |

