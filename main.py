import os
import sys
import argparse
import time
import pickle
import pandas as pd
import numpy as np
import random

# Import custom modules from src
from src import data_loader as dl
from src import data_processor as dp
from src import market_data_loader as mdl
from src import margin_rates as mr
from src import simulation_engine as sim
from src import portfolio_reconstructor as pr

# --- GLOBAL CONFIGURATION (Defaults) ---
DEFAULT_PATHS = 150
DEFAULT_BATCH_SIZE = 10
SEED = 42

def setup_environment(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # If you use other libraries like torch/tensorflow, seed them here too.
    print(f"[*] Random seed set to: {seed}")

def main():
    # 1. Parse Command Line Arguments (Industry Standard)
    parser = argparse.ArgumentParser(description="Tax-Aware Monte Carlo Portfolio Simulator")
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS, help="Total number of simulation paths")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for processing")
    parser.add_argument("--input", type=str, default="U13271915_20250101_20251029.csv", help="Input CSV filename")
    args = parser.parse_args()

    # 2. Setup
    start_time = time.time()
    print("==========================================")
    print("   MONTE CARLO PORTFOLIO SIMULATOR")
    print("==========================================")
    setup_environment(SEED)

    # Define Paths relative to the script location
    # This ensures it works on Mac, Windows, and Linux automatically
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    
    input_filepath = os.path.join(DATA_DIR, args.input)
    checkpoint_file = os.path.join(RESULTS_DIR, 'simulation_artifact.pkl')

    # Validation
    if not os.path.exists(input_filepath):
        print(f"\n[!] CRITICAL ERROR: Input file not found: {input_filepath}")
        print("    Please ensure the file is in the 'data' folder.")
        sys.exit(1) # Exit with error code

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 3. EXECUTION PIPELINE ---

    # Step 1: Data Loading
    print("\n[1/5] Loading & Processing Data...")
    raw_data = dl.load_ibkr_report(input_filepath)
    if raw_data is None:
        sys.exit(1)

    clean_data = dp.apply_split_adjustments(raw_data)
    start_date = clean_data['report_start_date']
    end_date = clean_data['report_end_date']
    print(f"      Period: {start_date.date()} to {end_date.date()}")

    # Step 2: External Data
    print("\n[2/5] Fetching Market Data & Margin Rates...")
    daily_rates = mr.get_ibkr_rates_hybrid(start_date, end_date)
    market_data = mdl.load_market_data(clean_data, report_filename=args.input)
    print(f"      Market Data: Loaded {len(market_data)} assets.")

    # Step 3: Shadow Benchmark
    print("\n[3/5] Generating Shadow Benchmark (Actual Trades)...")
    # Initialize Engine
    engine = sim.MonteCarloEngine(clean_data, market_data, daily_rates)
    
    # Run the "Actual" history through the simulator's logic
    shadow_scenario = engine.run_actual_with_sim_logic()[0]
    actual_res = pr.reconstruct_portfolio(shadow_scenario, market_data)
    actual_final_nav = actual_res['total_nav'].iloc[-1]
    print(f"      Benchmark Final NAV: {actual_final_nav:,.2f}")

    # Step 4: Monte Carlo Simulation
    print(f"\n[4/5] Running Simulation ({args.paths} paths)...")
    
    all_daily_navs = []
    
    # Reset engine to ensure clean state for random paths
    engine = sim.MonteCarloEngine(clean_data, market_data, daily_rates)

    for batch_start in range(0, args.paths, args.batch):
        current_batch_size = min(args.batch, args.paths - batch_start)
        
        # Run Batch
        scenarios = engine.generate_scenario(num_paths=current_batch_size)
        
        # Process Batch Results
        for scenario in scenarios:
            res = pr.reconstruct_portfolio(scenario, market_data)
            all_daily_navs.append(res['total_nav'])
        
        # Simple Progress Bar
        print(f"      Batch {(batch_start // args.batch) + 1} completed.")

    # Consolidate Results
    df_paths = pd.concat(all_daily_navs, axis=1)
    df_paths.columns = range(len(all_daily_navs))

    # Step 5: Save Artifacts
    print(f"\n[5/5] Saving Results to {checkpoint_file}...")
    
    artifact = {
        'metadata': {
            'paths': args.paths, 
            'start_date': start_date, 
            'end_date': end_date,
            'runtime_seconds': time.time() - start_time
        },
        'results': {
            'actual_nav_series': actual_res['total_nav'],
            'simulated_paths': df_paths,
            'benchmark_final_nav': actual_final_nav
        }
    }

    with open(checkpoint_file, 'wb') as f:
        pickle.dump(artifact, f)

    elapsed = time.time() - start_time
    print(f"\nSUCCESS: Simulation completed in {elapsed:.2f} seconds.")
    print("Run the 'notebooks/visualization.ipynb' to see the charts.")

if __name__ == "__main__":
    main()