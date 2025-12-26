"""
Portfolio Simulation Application Entry Point.

Executes the Portfolio Reconstruction and Monte Carlo Simulation framework.
Manages sequential dependencies between data ingestion, market data acquisition,
control portfolio reconstruction, and stochastic simulation generation.

Implements a state-machine architecture to ensure data integrity across execution
steps, preventing the use of stale datasets or invalid configurations. Handles
user interaction via a menu-driven command-line interface (CLI) to support
reproducible analysis and persistent storage of simulation artifacts.
"""
import os
import sys
import pickle
import gzip
import pandas as pd
import time
from datetime import datetime
from typing import Callable

# --- Import Custom Modules ---
# data_loader (dl): Parsing and ingestion of raw broker reports.
# data_processor (dp): Data cleaning and split adjustment logic.
# market_data_loader (mdl): Fetching historical price data.
# margin_rates (mr): Retrieval of historical margin rates.
# simulation_engine (sim): Core stochastic simulation logic.
# portfolio_reconstructor (pr): Rebuilds portfolios from trade logs.
# portfolio_analytics: Statistical analysis and visualization.
from src import data_loader as dl
from src import data_processor as dp
from src import market_data_loader as mdl
from src import margin_rates as mr
from src import simulation_engine as sim
from src import portfolio_reconstructor as pr
from src.portfolio_analytics import PortfolioAnalyser

# --- Configuration Constants ---
# Default input path used if file selection is skipped.
# Directory containing raw input reports
DATA_DIR = r'data' 

# Default input path used if file selection is skipped.
DEFAULT_REPORT_PATH = r'data\U13271915_20241224_20251224.csv'

# Directory for persistent application state storage (pickled sessions).
CHECKPOINT_DIR = r'data/saved_states'

# Simulation configuration:
# 150 paths: Balances statistical significance with runtime.
# Batch size 50: Limits memory consumption during vectorized operations.
SIMULATION_SEED = 42
DEFAULT_PATHS = 150
BATCH_SIZE = 50 
ENABLE_TIMING = False

class PortfolioSimulationApp:
    """
    Controls the portfolio reconstruction and Monte Carlo simulation workflow.

    Manages application state, data flow between modules, and the interactive 
    command-line interface. Enforces dependency chains to ensure data integrity 
    during reconstruction and simulation.
    """
    def __init__(self) -> None:
        """
        Initialize application state and artifact storage.
        """
        # Data containers act as state flags. 
        # None indicates the specific processing step has not occurred.
        self.raw_data = None
        self.clean_data = None
        self.flow_series = None
        self.market_data = None
        self.daily_rates = None
        self.engine = None
        
        # Result containers.
        # 'real_res'/'control_res': Deterministic portfolio runs.
        # 'simulation_results': Stochastic Monte Carlo output.
        self.real_res = None
        self.control_res = None
        self.control_nav = None
        self.real_nav = None
        self.simulation_results = None 
        
        # Ensure checkpoint directory exists to prevent IOErrors on save.
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def menu(self) -> None:
        """
        Display main menu and route input to pipeline steps.

        Loop continues until explicit exit command.
        """
        while True:
            self._print_header()
            print(" 0. RUN FULL PIPELINE")
            print(" 1. Parse IBKR Report")
            print(" 2. Fetch Market Data")
            print(" 3. Run Real & Control Portfolio Reconstruction")
            print(" 4. Run Monte Carlo Simulation")
            print(" 5. Analyse and Plot Results")
            print()
            print(" 6. Save State")
            print(" 7. Load State")
            print()
            print(" Q. Quit")
            print("-" * 60)
            
            # Construct dynamic status line to track pipeline progress.
            # Visual indicator for completed steps.
            flags = []
            flags.append("DATA: OK" if self.clean_data else "DATA: --")
            flags.append("MARKET: OK" if self.engine else "MARKET: --")
            flags.append("CONTROL: OK" if self.control_nav is not None else "CONTROL: --")
            flags.append("SIM: OK" if self.simulation_results else "SIM: --")
            
            print(f" STATUS: {' | '.join(flags)}")
            print("-" * 60)

            choice = input(" >> Select Option: ").upper().strip()
            time.sleep(0.5)

            """
            start_time_0 = time.time() if ENABLE_TIMING else 0.0
            time_step_load_data = time.time() - start_time_0
                    print(f" [t] Step 1 - Parse IBKR Report: {time_step_load_data:.2f}s")
            """

            if choice == '0':
                # Force state reset to ensure clean pipeline execution.
                self._reset_state()
                self.step_load_data()
                self.step_fetch_market_data()
                self.step_control_reconstruction()
                self.step_run_simulation()
                self.step_analyse()

            elif choice == '1':
                # Reset state to prevent mixing new data with old artifacts.
                self._reset_state()
                self.step_load_data()

            elif choice == '2':
                self.step_fetch_market_data()

            elif choice == '3':
                self.step_control_reconstruction()

            elif choice == '4':
                self.step_run_simulation()

            elif choice == '5':
                self.step_analyse()

            elif choice == '6':
                self._save_checkpoint()

            elif choice == '7':
                # Reset state before loading external checkpoint.
                self._reset_state()
                self._load_checkpoint()

            elif choice == 'Q':
                sys.exit()

            else:
                print(" [!] Invalid selection.")
                time.sleep(0.5)

    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    def step_load_data(self) -> None:
        """
        Ingests the raw IBKR activity report.

        Parses CSV, applies split adjustments, and isolates capital flow events 
        (deposits/withdrawals). Invalidates downstream data to prevent state 
        inconsistencies.
        """
        self._print_section_header("STEP 1: DATA LOADING")

        # Invalidates downstream components to force recalculation.
        self.market_data = None
        self.daily_rates = None
        self.engine = None
        self.real_res = None
        self.control_res = None
        self.control_nav = None
        self.real_nav = None
        self.simulation_results = None

        # --- File Selection Logic ---
        files = []
        if os.path.exists(DATA_DIR):
            files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.csv')]
            # Sorts by modification time (newest first).
            files.sort(key=lambda x: os.path.getmtime(os.path.join(DATA_DIR, x)), reverse=True)

        path = None

        if files:
            # Loop until a valid selection is made
            while True:
                print(f"\n Available Reports in '{DATA_DIR}':")
                print("   [0] Manual Path Entry")
                for idx, f in enumerate(files):
                    print(f"   [{idx+1}] {f}")
                
                choice = input("\n >> Select file number [Default: 1]: ").strip()
                time.sleep(0.5)

                if not choice:
                    # Default to the newest file
                    path = os.path.join(DATA_DIR, files[0])
                    break # Exit loop
                elif choice == '0':
                    path = None # Trigger manual entry logic below
                    break # Exit loop
                else:
                    try:
                        file_idx = int(choice) - 1
                        if 0 <= file_idx < len(files):
                            path = os.path.join(DATA_DIR, files[file_idx])
                            break # Exit loop
                        else:
                            print(" [!] Number out of range. Please try again.")
                            time.sleep(1)
                    except ValueError:
                        print(" [!] Invalid input. Please enter a number.")
                        time.sleep(1)

        # Fallback: Manual Entry (if no files found or user selected 0)
        if path is None:
            path = input(f" >> Enter path to CSV [Default: {DEFAULT_REPORT_PATH}]: ").strip()
            time.sleep(0.5)
            # Use default path if input is empty.
            if not path:
                path = DEFAULT_REPORT_PATH

        print(f" [>] Loading: {path}")

        # Verifies file existence.
        if not os.path.exists(path):
            print(f"\n [!] Error: File not found at {path}")
            time.sleep(0.5)
            return
        
        start_time = time.time() if ENABLE_TIMING else 0.0

        # Stores filename for output directory naming.
        self.report_name = os.path.basename(path)

        # Parses raw IBKR CSV into structured DataFrame.
        self.raw_data = dl.load_ibkr_report(path)

        if self.raw_data is not None:
            # Applies stock split adjustments to historical holdings.
            self.clean_data = dp.apply_split_adjustments(self.raw_data)
        else:
            print("\n [!] Data load failed.")
            time.sleep(0.5)
            return

        # Extracts events DataFrame safely.
        events_df = self.clean_data.get('events') if self.clean_data is not None else None

        # Processes capital flows or initializes empty structures.
        if events_df is None or events_df.empty:
            self.flow_events_df = pd.DataFrame(columns=['timestamp', 'event_type', 'cash_change_native', 'currency'])
            self.flow_series = pd.Series(dtype=float)
        else:
            # Validates schema.
            required_cols = {'cash_change_native', 'event_type', 'timestamp'}
            if not required_cols.issubset(events_df.columns):
                raise KeyError(f"events DataFrame must contain columns: {required_cols}")

            # Filters DEPOSIT and WITHDRAWAL events; normalizes timestamps.
            flows = events_df[events_df['event_type'].isin(['DEPOSIT', 'WITHDRAWAL'])].copy()
            flows['timestamp'] = pd.to_datetime(flows['timestamp']).dt.normalize()

            # Retains filtered event-level DataFrame for traceability.
            self.flow_events_df = flows

            # Aggregates daily net flows (amounts assumed in Base Currency).
            self.flow_series = flows.groupby('timestamp')['cash_change_native'].sum().sort_index()

        time_step = time.time() - start_time

        if start_time > 0.0:
            print(f"\n [t] Step 1: {time_step:.2f}s")

    # =========================================================================
    # STEP 2: MARKET DATA
    # =========================================================================
    def step_fetch_market_data(self) -> None:
        """
        Execute Step 2: Acquire historical market data and margin rates.

        Fetches market data via yfinance API and margin rates via hybrid scraper. 
        Invalidates downstream simulation results to maintain state consistency.
        """
        # --- Dependency Check ---
        # Verify Step 1 completion.
        self._check_dependency(self.clean_data is not None, "Step 1 (Load Data)", self.step_load_data)
        if self.clean_data is None: 
            return

        self._print_section_header("STEP 2: MARKET DATA ACQUISITION")
        
        # Invalidate previous results to enforce consistency with new market data.
        self.real_res = None
        self.control_res = None
        self.control_nav = None
        self.real_nav = None
        self.simulation_results = None

        start_time = time.time() if ENABLE_TIMING else 0.0

        # Derive fetch date range from loaded report.
        start = self.clean_data['report_start_date']
        end = self.clean_data['report_end_date']
        
        # Retrieve margin rates.
        self.daily_rates = mr.get_ibkr_rates_hybrid(start, end)

        # Acquire historical price data for all portfolio instruments.
        self.market_data = mdl.load_market_data(self.clean_data)
        
        # Initialize Monte Carlo Engine with acquired data.
        self.engine = sim.MonteCarloEngine(
            self.clean_data, 
            self.market_data, 
            self.daily_rates
        )
        
        time_step = time.time() - start_time
            
        print("\n [+] Market data setup complete.")
        time.sleep(0.5)

        if start_time > 0.0:
            print(f"\n [t] Step 2: {time_step:.2f}s")

    # =========================================================================
    # STEP 3: CONTROL & REAL PORTFOLIO RECONSTRUCTION
    # =========================================================================
    def step_control_reconstruction(self) -> None:
        """
        Execute Step 3: Reconstruct Real and Control Portfolios.

        'Real Portfolio': Reconstructed from trade logs.
        'Control Portfolio': Generated by passing trade history through the 
        simulation engine to normalize fees and execution timing.
        """
        # --- Dependency Check ---
        # Verify simulation engine initialization.
        self._check_dependency(self.engine is not None, "Step 2 (Market Data)", self.step_fetch_market_data)
        if self.engine is None: 
            return

        self._print_section_header("STEP 3: CONTROL & REAL PORTFOLIO RECONSTRUCTION")

        # Invalidate downstream simulation results if control portfolio changes.
        self.simulation_results = None

        if self.clean_data is None or self.market_data is None:
            print("\n [!] Critical Error: Data not loaded correctly.")
            time.sleep(0.5)
            return
        
        start_time = time.time() if ENABLE_TIMING else 0.0

        # Reconstruct real portfolio mirroring actual execution.
        self.real_res = pr.reconstruct_portfolio(self.clean_data, self.market_data, verbose=True)
        final_nav = self.real_res['total_nav'].iloc[-1]
        print(f"\n     - Real Portfolio Final NAV: {final_nav:,.2f}\n")
        time.sleep(0.5)

        # Process actual trades through simulation logic to create Control baseline.
        # This provides a methodological comparison for random paths.
        control_scenario = self.engine.run_real_with_sim_logic()[0]
        self.control_res = pr.reconstruct_portfolio(control_scenario, self.market_data, verbose=False)
        
        control_nav = self.control_res['total_nav'].iloc[-1]
        print(f"\n     - Control Portfolio Final NAV: {control_nav:,.2f}")
        time.sleep(0.5)
        
        # Store key NAV series in application state.
        self.control_nav = self.control_res['total_nav']
        self.real_nav = self.real_res['total_nav']

        time_step = time.time() - start_time

        if start_time > 0.0:
            print(f"\n [t] Step 3: {time_step:.2f}s")

    # =========================================================================
    # STEP 4: MONTE CARLO SIMULATION
    # =========================================================================
    def step_run_simulation(self) -> None:
        """
        Execute Step 4: Monte Carlo Simulation.

        Generates N random counterfactual portfolio paths based on tradable assets. 
        Batches results to manage memory usage.
        """
        # --- Dependency Check ---
        self._check_dependency(self.control_nav is not None, "Step 3 (Control Portfolio)", self.step_control_reconstruction)
        if self.control_nav is None: 
            return

        self._print_section_header("STEP 4: MONTE CARLO SIMULATION")

        # Guard against uninitialized engine.
        if self.engine is None:
            print(" [!] Engine is not initialized. Please run Step 2 first.")
            return

        # Default path count.
        n_paths = DEFAULT_PATHS
        
        # Input loop for path definition.
        while True:
            user_input = input(f" >> Enter number of paths to generate [Default: {DEFAULT_PATHS}]: ").strip()
            time.sleep(0.5)
            
            if not user_input:
                n_paths = DEFAULT_PATHS
                break
            
            try:
                val = int(user_input)
                if 1 <= val <= 100000:
                    n_paths = val
                    break
                else:
                    print(" [!] Error: Number must be between 1 and 100,000.")
            except ValueError:
                print(" [!] Error: Invalid input.")

        print(f"\n [>] Starting Monte Carlo Simulation ({n_paths} paths)...")
        time.sleep(0.5)

        start_time = time.time() if ENABLE_TIMING else 0.0

        all_navs = []
        paths_generated = 0
        print(f"     - Progress: 0/{n_paths} paths complete...", end='\r', flush=True)
        
        # --- Simulation Loop ---
        # Process simulation in batches to control memory footprint.
        for batch_start in range(0, n_paths, BATCH_SIZE):
            # Calculate current batch size handling final remainder.
            current_batch_size = min(BATCH_SIZE, n_paths - batch_start)
            
            # Use unique seed per batch to ensure reproducibility.
            batch_seed = SIMULATION_SEED + batch_start

            # Generate batch scenarios.
            scenarios = self.engine.generate_scenario(
                num_paths=current_batch_size, 
                verbose=False, 
                seed=batch_seed
            )
            
            # Reconstruct NAV series for generated scenarios.
            for sc in scenarios:
                res = pr.reconstruct_portfolio(sc, self.market_data, verbose=False)
                all_navs.append(res['total_nav'])
            
            # Update dynamic progress display.
            paths_generated += current_batch_size
            print(f"     - Progress: {paths_generated}/{n_paths} paths complete...", end='\r', flush=True)
            
            # Explicitly free memory.
            del scenarios

        if not all_navs:
            print("\n [!] Simulation produced no data.")
            time.sleep(0.5)
            return

        # Concatenate individual paths into single DataFrame for vectorized analysis.
        self.sim_paths_df = pd.concat(all_navs, axis=1)
        self.sim_paths_df.columns = range(len(all_navs))
        
        # Package complete result set.
        self.simulation_results = {
            'simulated_paths': self.sim_paths_df,
            'control_nav_series': self.control_nav,
            'real_nav_series': self.real_nav
        }

        time_step = time.time() - start_time

        print(f"\n [+] Simulation complete. Generated {len(all_navs)} paths.")

        if start_time > 0.0:
            print(f"\n [t] Step 4: {time_step:.2f}s")

        time.sleep(0.5)

    # =========================================================================
    # STEP 5: ANALYSIS & Visualisation
    # =========================================================================
    def step_analyse(self) -> None:
        """
        Execute Step 5: Statistical Analysis and Visualization.

        Analyzes simulation results to produce performance summaries, exports CSVs 
        for external validation, and generates standardized plots in a 
        timestamped directory.
        """
        # --- Dependency Check ---
        # Verify simulation data exists.
        self._check_dependency(self.simulation_results is not None, "Step 4 (Simulation)", self.step_run_simulation)
        if self.simulation_results is None: 
            return

        self._print_section_header("STEP 5: ANALYSIS & VISUALISATION")

        start_time = time.time() if ENABLE_TIMING else 0.0

        # --- Dynamic Output Folder Setup ---
        # Create unique subfolder to prevent overwrite.
        # Naming Convention: YYYYMMDD_HHMM_{InputFilename}_N{PathCount}
        
        # Extract path count for context.
        n_paths = self.simulation_results['simulated_paths'].shape[1]
        
        # Retrieve input filename tag.
        input_tag = getattr(self, 'report_name', 'Unknown').replace('.csv', '')
        
        folder_name = f"{input_tag}_N{n_paths}"
        output_dir = os.path.join('results', folder_name)
        
        # Create output directory.
        os.makedirs(output_dir, exist_ok=True)
        print(f" [+] Created output directory: {output_dir}")
        time.sleep(0.5)

        # --- Retrieve Flow Data ---
        # Retrieve from active state or results dictionary.
        flow_series = getattr(self, 'flow_series', None)
        
        if flow_series is None:
            flow_series = self.simulation_results.get('flow_series', pd.Series(dtype=float))

        analyser = PortfolioAnalyser(
            real_nav=self.simulation_results['real_nav_series'],
            control_nav=self.simulation_results['control_nav_series'],
            sim_paths_df=self.simulation_results['simulated_paths'],
            flow_series=flow_series
        )

        # --- Data Preservation (CSV) ---
        # Calculate statistics once to ensure consistency between CSVs and plots.
        summary_df, raw_stats, _ = analyser.get_summary_table()
        
        # Persist high-level summary.
        summary_df.to_csv(os.path.join(output_dir, 'performance_summary.csv'))
        
        # Save full daily data for all simulated paths.
        self.simulation_results['simulated_paths'].to_csv(os.path.join(output_dir, 'monte_carlo_paths.csv'))
        
        # Save metric distributions (Sharpe/Vol).
        raw_stats.to_csv(os.path.join(output_dir, 'simulation_stats_dist.csv'))

        time_step = time.time() - start_time

        print("\n [+] Saved CSVs: summary, paths, and distribution stats.")

        if start_time > 0.0:
            print(f"\n [t] Step 5: {time_step:.2f}s")

        time.sleep(0.5)
        """
        # --- Visualization Generation ---
        # Display summary to provide feedback.
        print("\nPERFORMANCE SUMMARY:")
        print( summary_df)
        print("\n [>] Launching visualisation windows...\n")
        time.sleep(1.0)

        # Generate and save plots sequentially.
        analyser.plot_confidence_intervals(
            save_path=os.path.join(output_dir, '1_confidence_intervals.png')
        )

        analyser.plot_simulation_traces(
            num_paths=100, 
            save_path=os.path.join(output_dir, '2_simulation_traces.png')
        )

        analyser.plot_drawdown_profile(
            save_path=os.path.join(output_dir, '3_drawdown_profile.png')
        )

        analyser.plot_distributions_NAV(
            raw_stats, 
            save_path=os.path.join(output_dir, '4_distribution_NAV.png')
        )
        
        analyser.plot_distributions_maxdd(
            raw_stats, 
            save_path=os.path.join(output_dir, '5_distribution_maxdd.png')
        )

        analyser.plot_distributions_volatility(
            raw_stats, 
            save_path=os.path.join(output_dir, '6_distribution_volatility.png')
        )

        analyser.plot_distributions_CAGR(
            raw_stats, 
            save_path=os.path.join(output_dir, '7_distribution_CAGR.png')
        )

        analyser.plot_distributions_sharpe(
            raw_stats, 
            save_path=os.path.join(output_dir, '8_distribution_sharpe.png')
        )
        """
        print(f"\n [+] All analysis files saved to: {output_dir}")
        time.sleep(0.5)
        input(f"\n >> Press Enter to return to menu...")
        time.sleep(0.5)

    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================
    def _reset_state(self) -> None:
        """
        Clear loaded data and processing results.

        Resets all state variables to None to ensure subsequent pipeline runs 
        do not inherit stale data from previous sessions.
        """
        print("\n [!] Clearing previous application state...")
        time.sleep(0.5)

        # Wipe input datasets.
        self.raw_data = None
        self.clean_data = None
        self.flow_series = None

        self.market_data = None
        self.daily_rates = None
        self.engine = None
        
        # Invalidate downstream results.
        self.real_res = None
        self.control_res = None
        self.control_nav = None
        self.real_nav = None
        self.simulation_results = None
        self.report_name = None

    def _print_section_header(self, title: str) -> None:
        """
        Display formatted section header.

        Args:
            title (str): Header text.
        """
        print("\n" + "="*60)
        print(f" {title}")
        print("="*60 + "\n")

    def _check_dependency(self, condition: bool, fix_action_name: str, fix_action_func: Callable[[], None]) -> bool:
        """
        Enforce pipeline integrity by verifying prerequisites.

        If condition is not met, flags the missing dependency and automatically 
        triggers the corrective action.

        Args:
            condition (bool): Validity check (True if dependency is met).
            fix_action_name (str): Name of the missing step.
            fix_action_func (callable): Method to execute if condition is False.

        Returns:
            bool: True if dependency was already met, False if fix was triggered.
        """
        # --- Dependency Check ---
        # Alert user and trigger missing step if condition is False.
        if not condition:
            print(f"\n [!] Missing dependency: {fix_action_name}")
            time.sleep(0.5)
            print(f" [>] Auto-triggering {fix_action_name}...")
            time.sleep(0.5)
            fix_action_func()
            return False 
        return True

    def _print_header(self) -> None:
        """
        Render main application title banner.
        """
        print("\n" + "#"*60)
        print("       MONTE CARLO PORTFOLIO RECONSTRUCTION")
        print("#"*60)
    
    def _save_checkpoint(self) -> None:
        """
        Serialize current application state to a binary pickle file.

        Persists simulation results and loaded datasets.
        """
        self._print_section_header("SAVING STATE")
        
        # Verify data existence.
        if not self.simulation_results:
            print(" [!] Nothing to save (Run Simulation first).")
            time.sleep(0.5)
            input("\n >> Press Enter to return...")
            time.sleep(0.5)
            return

        # Package application state.
        # Includes metadata (timestamp) and datasets required to resume analysis.
        
        input_tag = getattr(self, 'report_name', 'unknown').replace('.csv', '')
        n_paths = self.simulation_results['simulated_paths'].shape[1]
        
        filename = f"{input_tag}_N{n_paths}.pkl.gz"
        save_path = os.path.join(CHECKPOINT_DIR, filename)
        
        # Ensure checkpoint directory exists.
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        artifact = {
            'metadata': {
                'timestamp': datetime.now(),
                'report_name': getattr(self, 'report_name', 'unknown')
            },
            'datasets': {
                'clean_data': self.clean_data,
                'market_data': self.market_data,
                'daily_rates': self.daily_rates,
                'flow_series': self.flow_series
            },
            'results': self.simulation_results
        }

        # Serialize artifact using gzip and pickle.
        with gzip.open(save_path, 'wb') as f:
            pickle.dump(artifact, f)
        print(f" [+] State saved to {save_path}")
        time.sleep(0.5)
        input("\n >> Press Enter to continue...")
        time.sleep(0.5)

    def _load_checkpoint(self) -> None:
        """
        Deserialize application state from a binary pickle file.

        Restores datasets and simulation results; re-initializes simulation engine.
        """
        self._print_section_header("LOADING STATE")

        # --- 1. Find available checkpoint files ---
        # List .pkl and .pkl.gz files.
        files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pkl') or f.endswith('.pkl.gz')]
        # Sort by modification time (newest first).
        files.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)), reverse=True)

        if not files:
            print(" [!] No checkpoint files found.")
            time.sleep(0.5)
            input("\n >> Press Enter to return...")
            time.sleep(0.5)
            return
        
        # --- 2. Simple File Selection Menu ---
        print("Available Checkpoints:")
        for idx, f in enumerate(files):
            print(f"   [{idx+1}] {f}")
            
        try:
            choice = input("\n >> Select file number (or Enter to cancel): ")
            if not choice.strip(): return
            selected_file = files[int(choice) - 1]
        except (ValueError, IndexError):
            print(" [!] Invalid selection.")
            time.sleep(1)
            return

        print(f" [>] Loading {selected_file}...")
        
        # --- 3. Load and Restore State ---
        # Detect compression based on extension.
        full_path = os.path.join(CHECKPOINT_DIR, selected_file)
        opener = gzip.open if selected_file.endswith('.gz') else open

        with opener(full_path, 'rb') as f:
            artifact = pickle.load(f)
        
        # Restore Datasets.
        self.clean_data = artifact['datasets']['clean_data']
        self.market_data = artifact['datasets']['market_data']
        self.daily_rates = artifact['datasets']['daily_rates']
        self.flow_series = artifact['datasets'].get('flow_series', pd.Series(dtype=float))
        self.simulation_results = artifact['results']

        # --- Restore Input Tag ---
        # Ensures visualisation step knows the original input name.
        self.report_name = artifact['metadata'].get('report_name', 'unknown')
        
        # Restore Nav shortcuts.
        self.control_nav = self.simulation_results['control_nav_series']
        self.real_nav = self.simulation_results['real_nav_series']
        
        # Re-initialize Engine.
        if self.clean_data is not None and self.market_data is not None:
             self.engine = sim.MonteCarloEngine(
                self.clean_data, 
                self.market_data, 
                self.daily_rates
            )
            
        print(" [+] Load complete.")
        time.sleep(0.5)
        input("\n >> Press Enter to continue...")
        time.sleep(0.5)

if __name__ == "__main__":
    app = PortfolioSimulationApp()
    try:
        app.menu()
    except KeyboardInterrupt:
        print("\n [!] Interrupted by user. Exiting.")
        time.sleep(0.5)
        sys.exit()