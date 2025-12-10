# DSAP_Project_2025
Final project for the Data Science and Advanced Programming 2025 Course at HEC Lausanne.
# Monte Carlo Portfolio Simulator

### "Skill vs. Luck" Analysis for Active Traders

## Abstract
This project is a sophisticated financial simulation engine designed to audit active trading performance. It takes a real-world **Interactive Brokers (IBKR)** activity statement and simulates thousands of alternative "random walk" trading scenarios.

Unlike standard random walks, this engine is **Tax & Fee-Aware**. It rigorously models transaction frictions—including exchange-specific stamp duties, commission schedules, withholding taxes on dividends, and dynamic margin interest rates—to create a mathematically valid "Shadow Benchmark." This allows investors to isolate true Alpha (skill) from the noise of market beta and luck.

---

## Key Features

* **High-Fidelity Friction Modelling**:
    * **Transaction Fees**: Exact fee schedules for 15+ exchanges (NASDAQ, LSE, SIX, etc.).
    * **Stamp Duties**: Simulates UK Stamp Duty (0.5%), French FTT (0.4%), etc.
    * **Withholding Taxes**: Applies treaty-specific tax rates on dividends (e.g., 15% US, 35% CH).
* **Dynamic Margin Engine**:
    * Scrapes historical margin rates from IBKR and backfills using FRED/OECD data.
    * Calculates daily interest expenses on negative currency balances.
* **Corporate Action Handler**:
    * Automatically handles stock splits (retroactive adjustment) and dividends.
    * Projects future dividends to ensure simulated paths don't miss cash flows.
* **Shadow Benchmarking**:
    * Re-runs the *actual* historical trades through the simulation engine to normalize costs, ensuring an "apples-to-apples" comparison between reality and random scenarios.

---

## Project Structure

```text
your-project/
├── data/                     # Input folder for IBKR CSV reports
├── results/                  # Output folder for simulation artifacts (.pkl)
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── config.py             # Centralized settings (Fees, Taxes, Currencies)
│   ├── data_loader.py        # Parses complex stacked IBKR CSVs
│   ├── data_processor.py     # Vectorized split-adjustment engine
│   ├── market_data_loader.py # Fetches and resamples Yahoo Finance data
│   ├── margin_rates.py       # Scrapes and backfills interest rate data
│   ├── simulation_engine.py  # Core Monte Carlo logic
│   └── portfolio_reconstructor.py # Matrix-based valuation engine
├── main.py                   # Entry point
├── requirements.txt          # Dependency list
└── README.md                 # Project documentation