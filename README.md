# Portfolio Benchmarking & Stochastic Simulation Engine

**Author:** [Your Name]
**Student ID:** [Your Student ID]
**Course:** Advanced Programming (Fall 2025)
**Date:** January 11, 2026

---

## 1. Project Overview

This project implements a **Portfolio Benchmarking and Stochastic Simulation framework**.

The application serves as a simulation tool to compare actual portfolio performance against stochastic counterfactuals. By modeling full transaction economics—including dividends, margin rates, and transaction fees—the engine quantifies decision-making quality within the specific asset universe available to the portfolio manager.

### Key Features
* **Hybrid Data Sourcing:** Combines IBKR trade logs, `yfinance` market data, and a custom scraper for historical margin rates (with FRED fallback).
* **Full Economics Modelling:** Accounts for stock splits, withholding taxes, and margin costs to create a realistic control environment.
* **Stochastic Benchmarking:** Generates $N$ counterfactual portfolio paths to construct a rigorous performance baseline.
* **State Machine Architecture:** Orchestrates data ingestion, market data acquisition, and simulation steps to ensure data integrity.
* **Reproducible Analysis:** Automatically persists all numerical results (CSVs) and visualizations (PNGs) to a local `results/` directory.

---

## 2. Research Question

**"How does the realized performance of the portfolio compare against a distribution of stochastic counterfactuals derived from the same asset universe?"**

Standard benchmarking methods (e.g., vs. S&P 500) often fail to account for the specific opportunity set and constraints of the investor. This project solves this by constructing a custom benchmark:
1.  **Control Portfolio:** A faithful reconstruction of the actual portfolio using the simulation engine's logic to normalize fees and execution timing.
2.  **Competence Universe:** A distribution of simulated portfolios where trade timing and capital allocation are fixed, but assets are randomly selected from the pool of instruments the investor actually traded.

---

## 3. Project Structure

The project follows a modular structure in the `src/` directory, orchestrated by `main.py`.

```text
├── data/
│   ├── checkpoints/           # Serialized simulation states (.pkl)
│   └── [data].csv             # Input IBKR Activity Report (CSV)
├── results/                   # Auto-generated analysis artifacts (CSVs & PNGs)
├── src/
│   ├── __init__.py            # Package initialization
│   ├── config.py              # Fee schedules, currency lists, and constants
│   ├── data_loader.py         # ETL pipeline for IBKR "stacked" CSVs
│   ├── data_processor.py      # Logic for corporate actions and stock splits
│   ├── market_data_loader.py  # yfinance API wrapper with retry logic
│   ├── margin_rates.py        # Hybrid scraper (IBKR Website + FRED) for interest rates
│   ├── simulation_engine.py   # Core Monte Carlo logic & Competence Universe selection
│   ├── portfolio_reconstructor.py # Cash & Securities Engines for daily NAV calculation
│   └── portfolio_analytics.py # Statistical metrics (Sharpe, Drawdown) & plotting
├── main.py                    # Application Entry Point
├── requirements.txt           # Python dependencies
├── PROPOSAL.md                # Project Proposal
├── AI_USAGE.md                # AI Tools Declaration
└── README.md                  # Documentation