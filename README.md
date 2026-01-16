# 📈 Bitcoin Hedged Grid Strategy Backtester

A research-grade backtesting engine designed to simulate a **Volatility Harvesting (Grid)** strategy protected by **Tail Risk Hedging (Put Options)**.

This project simulates the performance of a portfolio that accumulates Bitcoin using a dynamic ATR-based grid while simultaneously purchasing rolling protective Put options (modeled via Black-Scholes) to cap drawdowns during market crashes.

---

## 🚀 Key Features

* **Dynamic Grid Execution:**
* Uses **ATR (Average True Range)** to dynamically adjust grid spacing based on market volatility.
* Accumulates assets during dips and takes profits on local bounces.

* **Synthetic Options Hedging:**
* Simulates the purchase of OTM (Out-of-the-Money) Put options to hedge against drops >10%.
* Includes a **Black-Scholes Pricing Model** accounting for Volatility Risk Premium (VRP) and time decay (Theta).

* **Advanced Benchmarking:**
* Compares strategy performance against **Buy & Hold**, **DCA (Dollar Cost Averaging)**, and **Volatility Targeting** strategies.

* **Institutional-Grade Analytics:**
* Calculates **XIRR** (Extended Internal Rate of Return) for accurate cash flow analysis.
* Computes **Rolling VaR (95%)**, Sharpe, Sortino, and Calmar ratios.
* Visualizes "Fat Tail" risks and drawdown distributions.

## 🛠 Project Structure

* `main.py`: The entry point. Orchestrates data fetching, backtesting, and rendering the dashboard.
* `config.py`: Central configuration file (Capital, Grid settings, Hedging parameters, Timeframes).
* `backtest_engine.py`: The core logic loop handling positions, grid orders, and cash flow management.
* `options.py`: A dedicated module for Options pricing (Black-Scholes) and Mark-to-Market valuation.
* `advanced_benchmarks.py`: Generates comparative baselines (DCA, Vol-Targeting) for performance assessment.
* `logger_and_analytics.py`: Handles statistical analysis (Normal Distribution fitting, VaR) and matplotlib visualizations.
* `data_fetcher.py`: Wrapper around `ccxt` to download historical OHLCV data from Binance.

## 📊 Analytics Dashboard

The engine generates a comprehensive dashboard including:

1. **Equity Curve:** Net Strategy vs. Gross (Unhedged) vs. Benchmark.
2. **Drawdown Analysis:** Underwater plots for all strategies.
3. **Rolling Metrics:** 6-month Rolling Sharpe Ratio.
4. **Returns Distribution:** Histogram of daily returns vs. Normal Distribution (to visualize Kurtosis).

## ⚡ Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/DanRedelien/Grid-Protective-Put-on-BTCUSDT

```

2. **Install dependencies:**
```bash
pip install pandas numpy matplotlib scipy ccxt

```

3. **Configure:**
Edit `config.py` to set your desired start date, initial capital, and grid parameters.
4. **Run:**
```bash
python main.py

```

## 🤖 AI Contribution Note

> **Note:** The core logic, architecture, and mathematical modeling for this project were developed in collaboration with **Gemini 3.0 Pro**.
> *Localization Note:* Current code comments are a mix of English and Russian. A full translation and standardization of comments to English is planned for future updates.

---

## ⚠️ Critical Analysis & Limitations (The "Cons")

While this backtester provides valuable insights, users must be aware of the following limitations regarding real-world application:

1. **Intra-candle Path Dependency (Look-ahead Bias):**
* The backtester runs on 30-minute candles. It assumes that if the `Low` of the candle hit a buy order and the `High` hit a sell order, both happened favorably. In reality, the price path inside the candle matters (e.g., did it go High first, then Low? If so, the Sell wouldn't trigger if the Buy hadn't happened yet). This can slightly overstate grid profitability.

2. **Synthetic Options Liquidity:**
* The `options.py` module calculates theoretical Black-Scholes prices. In the real crypto options market (e.g., Deribit), liquidity is often low, and spreads are wide. You might pay a significantly higher premium than the theoretical model suggests, especially during high volatility (IV expansion).

3. **Volatility Skew (The "Smile"):**
* The model assumes a constant Volatility or a simplified Volatility surface. Real markets have a "Volatility Smile," meaning OTM Puts are often much more expensive (higher Implied Volatility) than the model predicts. This underestimates the cost of hedging.

4. **Execution Slippage:**
* While fees are accounted for, the impact of slippage during "Fast crashes" is difficult to model perfectly without tick-level data.

5. **Data Quality:**
* The standard `ccxt` fetcher does not account for exchange downtimes or historical data gaps, which are common in crypto API history
