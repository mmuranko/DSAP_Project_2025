# Monte Carlo Portfolio Reconstruction

---

## Project Overview

This Python application reconstructs historical investment portfolios from Interactive Brokers (IBKR) activity statements and performs Monte Carlo simulations to assess performance attribution. By generating counterfactual portfolio histories based on the investor's specific set of tradeable assets, the application isolates skill from luck. 

---

## Setup

1.  **Clone Repository**  
2.  **Install Dependencies**  
    ```bash
    pip install -r requirements.txt
    ```
3.  **Provide Data**  
    **Portfolio Report** On IBKR navigate to Performance & Reports > Statements > Activity Statement, select any time period, and press Download CSV. Place the CSV file it in the `data/` folder of this application.  
    **Saved States:** If existing simulation checkpoints exist (`.pkl` or `.pkl.gz`), they must be placed in the `data/saved_states/` folder to be recognised by the loader.  

---

## Usage

### Starting the Application

Run the main application entry point:

```bash
python main.py
```

### Menu Options

The application uses a state-machine architecture. Initially, the user can decide to load a saved state and run the `5. Analyse Results` step to analyse and visualise the results. To run a new simulation the user can either run each step sequentially or select `0. RUN FULL PIPELINE`. The menu has the following main options:

- `0. RUN FULL PIPELINE`: Clears all data and executes Steps 1 through 5 sequentially. 
- `1. Parse IBKR Report`: Loads the report CSV file and parses the initial state, the event log, and portfolio metadata. 
- `2. Fetch Market Data`: Downloads historical asset market prices (Yahoo Finance) and generates daily margin rates (IBKR/FRED) for all assets found in Step 1. 
- `3. Run Control Reconstruction`: Rebuilds the *Real Portfolio* and generates the *Control Portfolio* (using the real trades but applying the simplified simulation logic). 
- `4. Run Monte Carlo Simulation`: Generates $N$ counterfactual portfolio paths by randomly replacing the asset for each trade in the event log. 
- `5. Analyse Results`: Calculates and plots risk and return metrics for the real and control portfolio and the simulated portfolios and saves them in `results/`. 
- `6. Save State`: Saves the current state of the application to `data/saved_states/`. 
- `7. Load State`: Loads a previous state from `data/saved_states/`. 

The user can exit the application from the main menu by typing `Q` and pressing Enter.  

When selecting `1. Parse IBKR Report` or `7. Load State`, the user can choose the file to be parsed or loaded from a list of existing files.  

---

## Project Structure

```text
├── data/
│   ├── saved_states/               # Saved application states
│   │   └── [data].pkl.gz
│   └── [data].csv                  # Input IBKR Activity Reports
│
├── results/                        
│   ├── [run]/                      # Generated results data of each run is saved in a dedicated subfolder
│   │   ├── [data].png
│   │   └── [data].csv
│   └── [data].png                  # Highlighted simulation results
│
├── src/
│   ├── __init__.py                 # Package initialisation
│   ├── config.py                   # Information on the risk-free rate, currencies, exchanges, taxes, fees, and margin spreads
│   ├── data_loader.py              # Parses the IBKR CSV report
│   ├── data_processor.py           # Split adjustment of the parsed data
│   ├── margin_rates.py             # Hybrid interest rate scraper (IBKR & FRED)
│   ├── market_data_loader.py       # Loads market data for relevant assets using yfinance
│   ├── portfolio_analytics.py      # Analyses the simulation results and plots key results
│   ├── portfolio_reconstructor.py  # Calculates the daily portfolio NAV from the initial state and event log
│   └── simulation_engine.py        # Simulates counterfactual portfolio histories based on the initial state of the real portfolio
│
├── tests/
│   ├── __init__.py                 # Package initialisation
│   └── test.py                     # Tests the some of the internal logic of the application
│
├── main.py                         # Application entry point
├── requirements.txt                # Python dependencies
│
├── README.md                       # Documentation
├── PROPOSAL.md                     # Original project proposal
└── AI_USAGE.md                     # AI usage summary
```

---

## Testing

To verify the applications arithmetic logic and simulation mechanics, a short test module can be run:

```bash
python -m unittest tests/test.py
```

---

**Author:** Marvin Muranko  
**Student ID:** 21955182  
**Course:** Advanced Programming 2025  
**Date:** January 11, 2026  