"""
Portfolio Simulation Application Entry Point.

This module serves as the primary execution entry point for the Portfolio Reconstruction
and Monte Carlo Simulation framework. It orchestrates the end-to-end data pipeline,
managing the sequential dependencies between data ingestion, market data acquisition,
control portfolio reconstruction, and stochastic simulation generation.

The module implements a state-machine architecture to ensure data integrity across
execution steps, preventing the utilisation of stale datasets or invalid configurations.
User interaction is handled via a menu-driven command-line interface (CLI), facilitating
reproducible analysis and the persistent storage of simulation artifacts.
"""
import os
import sys
import pickle
import pandas as pd
import time
from datetime import datetime
from typing import Callable

# --- Import Custom Modules ---
# Import the specialized modules that handle specific stages of the pipeline:
# - Data ingestion and cleaning (dl, dp)
# - Market data acquisition (mdl, mr)
# - Core simulation logic (sim)
# - Portfolio mathematics and analytics (pr, PortfolioAnalyser)
from src import data_loader as dl
from src import data_processor as dp
from src import market_data_loader as mdl
from src import margin_rates as mr
from src import simulation_engine as sim
from src import portfolio_reconstructor as pr
from src.portfolio_analytics import PortfolioAnalyser

# --- Configuration Constants ---
# Default path for the input CSV report, used if the user skips file selection.
DEFAULT_REPORT_PATH = r'data/U13271915_20250101_20251029.csv'

# Define the location for storing the application state (pickle files).
# This allows the user to save the session and resume analysis later without re-running simulations.
CHECKPOINT_DIR = r'data/checkpoints'
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'simulation_artifact.pkl')

# Simulation defaults: 150 paths provides a reasonable statistical sample,
# and a batch size of 50 manages memory usage during vectorised operations.
DEFAULT_PATHS = 150
BATCH_SIZE = 50 

class PortfolioSimulationApp:
    """
    Orchestrates the portfolio reconstruction and Monte Carlo simulation workflow.

    This class serves as the central controller, managing application state, data flow 
    between modules, and the interactive command-line interface. It enforces dependency 
    chains to ensure data integrity during the reconstruction and simulation processes.
    """
    def __init__(self) -> None:
        """
        Initialises the application state and prepares storage for simulation artifacts.
        """
        # Initialize input data containers to None.
        # These act as state flags: if None, the data loading step has not yet occurred.
        self.raw_data = None
        self.clean_data = None
        self.flow_series = None
        self.market_data = None
        self.daily_rates = None
        self.engine = None
        
        # Initialize result containers.
        # 'real_res' and 'control_res' store the deterministic portfolio runs.
        # 'simulation_results' stores the stochastic Monte Carlo output.
        self.real_res = None
        self.control_res = None
        self.control_nav = None
        self.real_nav = None
        self.simulation_results = None 
        
        # Create the checkpoint directory immediately to prevent IOErrors during save operations.
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def menu(self) -> None:
        """
        Displays the interactive main menu and routes user input to specific pipeline steps.

        This loop persists until the user explicitly selects the option to quit.
        """
        while True:
            self._print_header()
            print(" 0. ** RUN FULL PIPELINE ** (Start to Finish)")
            print(" 1. Load IBKR Report")
            print(" 2. Fetch Market Data")
            print(" 3. Run Control Portfolio Reconstruction")
            print(" 4. Run Monte Carlo Simulation")
            print(" 5. Analyse Results")
            print(" 6. Save State")
            print(" 7. Load State")
            print(" Q. Quit")
            print("-" * 60)
            
            # A dynamic status line is constructed to provide a quick visual overview 
            # of which pipeline stages have been successfully completed. 
            # This aids the user in knowing which step must be performed next.
            flags = []
            flags.append("DATA: OK" if self.clean_data else "DATA: --")
            flags.append("MARKET: OK" if self.engine else "MARKET: --")
            flags.append("CONTROL: OK" if self.control_nav is not None else "CONTROL: --")
            flags.append("SIM: OK" if self.simulation_results else "SIM: --")
            
            print(f" STATUS: {' | '.join(flags)}")
            print("-" * 60)

            choice = input(" >> Select Option: ").upper().strip()
            time.sleep(0.5)

            if choice == '0':
                # Enforce a full reset to ensure the pipeline executes from a clean state
                self._reset_state()
                self.step_load_data()
                self.step_fetch_market_data()
                self.step_control_reconstruction()
                self.step_run_simulation()
                self.step_analyse()
            elif choice == '1':
                # Enforce a full reset to ensure the pipeline executes from a clean state
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
                # Enforce a full reset to ensure the pipeline executes from a clean state
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
        Executes Step 1: Ingestion of the raw IBKR activity report.

        The CSV file is parsed, split adjustments are applied, and capital flow 
        events (deposits/withdrawals) are isolated. Downstream data is 
        invalidated to prevent state inconsistencies.
        """
        self._print_section_header("STEP 1: DATA LOADING")

        # Downstream components are invalidated to force a clean recalculation
        self.market_data = None
        self.daily_rates = None
        self.engine = None
        self.real_res = None
        self.control_res = None
        self.control_nav = None
        self.real_nav = None
        self.simulation_results = None

        # User is prompted to define IBKR report CSV file path
        path = input(f" >> Enter path to CSV [Default: {DEFAULT_REPORT_PATH}]: ").strip()
        time.sleep(0.5)
        
        # Default path is used if input is empty
        if not path:
            path = DEFAULT_REPORT_PATH

        # File existence is verified before proceeding
        if not os.path.exists(path):
            print(f" [!] Error: File not found at {path}")
            time.sleep(0.5)
            return

        # Filename is stored for later use in results directory naming
        self.report_name = os.path.basename(path)

        # Raw IBKR CSV report is parsed into a structured DataFrame
        self.raw_data = dl.load_ibkr_report(path)

        if self.raw_data is not None:
            # Stock split adjustments are applied to the historic holding quantities
            self.clean_data = dp.apply_split_adjustments(self.raw_data)
        else:
            print(" [!] Data load failed.")
            time.sleep(0.5)
            return

        # Events DataFrame is extracted safely
        events_df = self.clean_data.get('events') if self.clean_data is not None else None

        # Capital flows are processed, or empty structures are initialized if no events exist
        if events_df is None or events_df.empty:
            self.flow_events_df = pd.DataFrame(columns=['timestamp', 'event_type', 'cash_change_native', 'currency'])
            self.flow_series = pd.Series(dtype=float)
        else:
            # Schema is validated to ensure required columns are present
            required_cols = {'cash_change_native', 'event_type', 'timestamp'}
            if not required_cols.issubset(events_df.columns):
                raise KeyError(f"events DataFrame must contain columns: {required_cols}")

            # Only DEPOSIT and WITHDRAWAL events are filtered, and timestamps are normalized
            flows = events_df[events_df['event_type'].isin(['DEPOSIT', 'WITHDRAWAL'])].copy()
            flows['timestamp'] = pd.to_datetime(flows['timestamp']).dt.normalize()

            # Filtered event-level DataFrame is kept for traceability
            self.flow_events_df = flows

            # Daily net flows are aggregated (amounts are assumed to be in base currency)
            self.flow_series = flows.groupby('timestamp')['cash_change_native'].sum().sort_index()

    # =========================================================================
    # STEP 2: MARKET DATA
    # =========================================================================
    def step_fetch_market_data(self) -> None:
        """
        Executes Step 2: Acquisition of historical market data and margin rates.

        Market data is fetched via the yfinance API, and margin rates are retrieved
        via the hybrid scraper/FRED proxy. Downstream simulation results are
        invalidated to ensure state consistency.
        """
        # --- Dependency Check ---
        # Verification that Step 1 has completed is required before proceeding
        self._check_dependency(self.clean_data is not None, "Step 1 (Load Data)", self.step_load_data)
        if self.clean_data is None: 
            return  # Operation is aborted if Step 1 failed

        self._print_section_header("STEP 2: MARKET DATA ACQUISITION")
        
        # Previous simulation results are invalidated to ensure consistency with new market data
        self.real_res = None
        self.control_res = None
        self.control_nav = None
        self.real_nav = None
        self.simulation_results = None

        # Date range for data fetching is derived directly from the loaded report
        start = self.clean_data['report_start_date']
        end = self.clean_data['report_end_date']
        
        # Margin rates are retrieved via the hybrid scraper
        self.daily_rates = mr.get_ibkr_rates_hybrid(start, end)

        # Historical price data is acquired for all instruments in the portfolio
        self.market_data = mdl.load_market_data(self.clean_data)
        
        # The Monte Carlo Engine is initialized with the newly acquired data
        self.engine = sim.MonteCarloEngine(
            self.clean_data, 
            self.market_data, 
            self.daily_rates
        )
        
        print("\n [+] Market data setup complete.")
        time.sleep(0.5)

    # =========================================================================
    # STEP 3: CONTROL PORTFOLIO RECONSTRUCTION
    # =========================================================================
    def step_control_reconstruction(self) -> None:
        """
        Executes Step 3: Reconstruction of Real and Control Portfolios.

        The 'Real Portfolio' is reconstructed faithfully from trade logs. The
        'Control Portfolio' is generated by passing the trade history through the
        simulation engine to normalize fees and execution timing.
        """
        # --- Dependency Check ---
        # Verification that the simulation engine is initialized is required
        self._check_dependency(self.engine is not None, "Step 2 (Market Data)", self.step_fetch_market_data)
        if self.engine is None: 
            return

        self._print_section_header("STEP 3: CONTROL PORTFOLIO & REALITY CHECK")

        if self.clean_data is None or self.market_data is None:
            print(" [!] Critical Error: Data not loaded correctly.")
            time.sleep(0.5)
            return

        # Downstream simulation results are invalidated to force a fresh run in Step 4
        # if the control portfolio definition has changed
        self.simulation_results = None

        # Real portfolio is reconstructed in 'faithful' mode to mirror actual execution
        self.real_res = pr.reconstruct_portfolio(self.clean_data, self.market_data, verbose=True)
        final_nav = self.real_res['total_nav'].iloc[-1]
        print(f"\n     - Real Portfolio Final NAV: {final_nav:,.2f}\n")
        time.sleep(0.5)

        # Actual trades are processed through simulation logic to create a scientific
        # baseline (Control) that is methodologically comparable to random paths
        control_scenario = self.engine.run_real_with_sim_logic()[0]
        self.control_res = pr.reconstruct_portfolio(control_scenario, self.market_data, verbose=False)
        
        control_nav = self.control_res['total_nav'].iloc[-1]
        print(f"\n     - Control Portfolio Final NAV: {control_nav:,.2f}")
        time.sleep(0.5)
        
        # Key NAV series are stored in the application state for later analysis
        self.control_nav = self.control_res['total_nav']
        self.real_nav = self.real_res['total_nav']

    # =========================================================================
    # STEP 4: MONTE CARLO SIMULATION
    # =========================================================================
    def step_run_simulation(self) -> None:
        """
        Executes Step 4: Monte Carlo Simulation.

        Generates N random counterfactual portfolio paths based on the user's
        'Competence Universe'. Results are batched to manage memory usage efficiently.
        """
        # --- Dependency Check ---
        # The Control Portfolio is a prerequisite, as it defines the baseline for comparison
        self._check_dependency(self.control_nav is not None, "Step 3 (Control Portfolio)", self.step_control_reconstruction)
        if self.control_nav is None: 
            return

        self._print_section_header("STEP 4: MONTE CARLO SIMULATION")

        # Number of simulation paths defaults to the configuration constant
        n_paths = DEFAULT_PATHS
        
        # User input loop for defining path count
        while True:
            user_input = input(f" >> Enter number of paths to generate [Default: {DEFAULT_PATHS}]: ").strip()
            time.sleep(0.5)
            
            # Default value is used if no input is provided
            if not user_input:
                n_paths = DEFAULT_PATHS
                break
            
            # User input is validated for type and range
            try:
                val = int(user_input)
                if 1 <= val <= 100000:
                    n_paths = val
                    break
                else:
                    print(" [!] Error: Number must be between 1 and 100,000. Please try again.")
                    time.sleep(0.5)
            except ValueError:
                print(" [!] Error: Invalid input. Please enter a positive whole number.")
                time.sleep(0.5)

        print(f"\n [>] Starting Monte Carlo Simulation ({n_paths} paths)...")
        time.sleep(0.5)

        # Safety check to ensure engine availability (redundant if dependency chain holds, but safe)
        if self.engine is None:
            print(" [!] Engine is not initialized. Please run Step 2 first.")
            time.sleep(0.5)
            return

        all_navs = []
        paths_generated = 0
        print(f"     - Progress: 0/{n_paths} paths complete...", end='\r', flush=True)
        
        # --- Simulation Loop ---
        # Simulation is processed in batches to control memory footprint
        for batch_start in range(0, n_paths, BATCH_SIZE):
            # Current batch size is calculated to handle the final remainder correctly
            current_batch_size = min(BATCH_SIZE, n_paths - batch_start)
            
            # The engine is invoked silently to prevent console clutter
            scenarios = self.engine.generate_scenario(num_paths=current_batch_size, verbose=False)
            
            # Each generated scenario is immediately reconstructed into a NAV series
            for sc in scenarios:
                res = pr.reconstruct_portfolio(sc, self.market_data, verbose=False)
                all_navs.append(res['total_nav'])
            
            # Progress is displayed dynamically on the same line
            paths_generated += current_batch_size
            print(f"     - Progress: {paths_generated}/{n_paths} paths complete...", end='\r', flush=True)
            
            # Memory is explicitly freed to prevent accumulation during large simulations
            del scenarios

        print() # Newline printed after progress bar completion

        if not all_navs:
            print(" [!] Simulation produced no data.")
            time.sleep(0.5)
            return

        # Individual paths are concatenated into a single DataFrame for vectorized analysis
        self.sim_paths_df = pd.concat(all_navs, axis=1)
        self.sim_paths_df.columns = range(len(all_navs))
        
        # Complete result set is packaged for storage and analysis
        self.simulation_results = {
            'simulated_paths': self.sim_paths_df,
            'control_nav_series': self.control_nav,
            'real_nav_series': self.real_nav
        }
        
        print(f" [+] Simulation complete. Generated {len(all_navs)} paths.")
        time.sleep(0.5)

    # =========================================================================
    # STEP 5: ANALYSIS
    # =========================================================================
    def step_analyse(self) -> None:
        """
        Executes Step 5: Statistical Analysis and Visualisation.

        Simulation results are analyzed to produce performance summaries,
        CSVs are exported for external validation, and standardized plots
        are generated and saved to a timestamped directory.
        """
        # --- Dependency Check ---
        # Simulation data is required before attempting analysis
        self._check_dependency(self.simulation_results is not None, "Step 4 (Simulation)", self.step_run_simulation)
        if self.simulation_results is None: 
            return

        self._print_section_header("STEP 5: ANALYTICS & VISUALISATION")
        
        # --- Dynamic Output Folder Setup ---
        # A unique subfolder is created for every run to prevent overwriting previous results.
        # Naming Convention: YYYYMMDD_HHMM_{InputFilename}_N{PathCount}
        
        # Current time is captured for chronological sorting
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Number of paths is extracted to contextualize the results folder
        n_paths = self.simulation_results['simulated_paths'].shape[1]
        
        # Input filename tag is safely retrieved (defaults to 'Unknown' for legacy states)
        input_tag = getattr(self, 'report_name', 'Unknown')
        
        folder_name = f"{timestamp_str}_{input_tag}_N{n_paths}"
        output_dir = os.path.join('results', folder_name)
        
        # Output directory is created (existing directories are ignored)
        os.makedirs(output_dir, exist_ok=True)
        print(f" [+] Created output directory: {output_dir}")
        time.sleep(0.5)

        # --- Retrieve Flow Data ---
        # Flow data is retrieved from active state or results dictionary
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
        # Statistics are calculated once to ensure consistency between CSVs,
        # console output, and visualisation inputs
        summary_df, raw_stats, _ = analyser.get_summary_table()
        
        # High-level summary is persisted for quick performance review
        summary_df.to_csv(os.path.join(output_dir, 'performance_summary.csv'))
        
        # Full daily data for all simulated paths is saved for external validation
        self.simulation_results['simulated_paths'].to_csv(os.path.join(output_dir, 'monte_carlo_paths.csv'))
        
        # Distribution of metrics (Sharpe/Vol) is saved for statistical testing
        raw_stats.to_csv(os.path.join(output_dir, 'simulation_stats_dist.csv'))
        print(" [+] Saved CSVs: summary, paths, and distribution stats.")
        time.sleep(0.5)

        # --- Visualisation Generation ---
        # Summary is displayed immediately to provide feedback during plotting
        print("\nPERFORMANCE SUMMARY:")
        print( summary_df)
        print("\n [>] Launching visualisation windows...\n")
        time.sleep(1.0)

        # Plots are generated and saved sequentially. 
        # Specific filenames are enforced to ensure a complete set of grading artifacts.
        analyser.plot_confidence_intervals(
            save_path=os.path.join(output_dir, '1_confidence_intervals.png')
        )
        analyser.plot_simulation_traces(
            num_paths=100, 
            save_path=os.path.join(output_dir, '2_simulation_traces.png')
        ) 
        analyser.plot_distributions(
            raw_stats, 
            save_path=os.path.join(output_dir, '3_distributions.png')
        )
        analyser.plot_drawdown_profile(
            save_path=os.path.join(output_dir, '4_drawdown_profile.png')
        )
        
        print(f"\n [+] All analysis files saved to: {output_dir}")
        time.sleep(0.5)
        input(f"\n >> Press Enter to return to menu...")
        time.sleep(0.5)

    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================
    def _reset_state(self) -> None:
        """
        Clears all loaded data and processing results to facilitate a new execution cycle.

        This method ensures that subsequent pipeline runs do not inherit stale data
        from previous sessions by resetting all state variables to None.
        """
        print("\n [!] Clearing previous application state...")
        time.sleep(0.5)

        # Input datasets are wiped to ensure the next run starts with a clean slate
        self.raw_data = None
        self.clean_data = None
        self.flow_series = None

        self.market_data = None
        self.daily_rates = None
        self.engine = None
        
        # All downstream results are invalidated to prevent analysis of old results 
        # against new data
        self.real_res = None
        self.control_res = None
        self.control_nav = None
        self.real_nav = None
        self.simulation_results = None

    def _print_section_header(self, title: str) -> None:
        """
        Displays a formatted section header to the console for visual separation.

        Args:
            title (str): The text to be displayed within the header.
        """
        print("\n" + "="*60)
        print(f" {title}")
        print("="*60 + "\n")

    def _check_dependency(self, condition: bool, fix_action_name: str, fix_action_func: Callable[[], None]) -> bool:
        """
        Enforces pipeline integrity by verifying prerequisites before executing a step.

        If a required condition is not met (e.g., data not loaded), the missing
        dependency is flagged, and the appropriate corrective action is triggered automatically.

        Args:
            condition (bool): The validity check (True if dependency is met).
            fix_action_name (str): The human-readable name of the missing step.
            fix_action_func (callable): The method to execute if the condition is False.

        Returns:
            bool: True if the dependency was already met, False if a fix was triggered.
        """
        # --- Dependency Check ---
        # If the prerequisite condition is False, the user is alerted and the 
        # missing step is automatically triggered
        if not condition:
            print(f"\n [!] Missing dependency: {fix_action_name}")
            time.sleep(0.5)
            print(f" [>] Auto-triggering {fix_action_name}...")
            time.sleep(0.5)
            fix_action_func()
            return False # Indicates that a corrective action was required
        return True # Indicates that the workflow was already in a valid state

    def _print_header(self) -> None:
        """
        Renders the main application title banner to the standard output.
        """
        print("\n" + "#"*60)
        print("      PORTFOLIO RECONSTRUCTION & MONTE CARLO ENGINE")
        print("#"*60)
    
    def _save_checkpoint(self) -> None:
        """
        Serialises the current application state to a binary pickle file.

        This allows for the persistence of expensive simulation results and loaded datasets.
        """
        self._print_section_header("SAVING STATE")
        
        # A check is performed to ensure there is actually data to save
        if not self.simulation_results:
            print(" [!] Nothing to save (Run Simulation first).")
            time.sleep(0.5)
            input("\n >> Press Enter to return...")
            time.sleep(0.5)
            return

        # The entire application state is packaged into a dictionary. 
        # This includes metadata (timestamp) and all datasets required to resume analysis.
        artifact = {
            'metadata': {'timestamp': datetime.now()},
            'datasets': {
                'clean_data': self.clean_data,
                'market_data': self.market_data,
                'daily_rates': self.daily_rates,
                'flow_series': self.flow_series
            },
            'results': self.simulation_results
        }

        # The artifact is serialised to disk using pickle
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(artifact, f)
        print(f" [+] State saved to {CHECKPOINT_FILE}")
        time.sleep(0.5)
        input("\n >> Press Enter to continue...")
        time.sleep(0.5)

    def _load_checkpoint(self) -> None:
        """
        Deserialises application state from a binary pickle file.

        Restores datasets and simulation results, and re-initialises the simulation
        engine for immediate use.
        """
        self._print_section_header("LOADING STATE")
        
        if not os.path.exists(CHECKPOINT_FILE):
            print(f" [!] No checkpoint found at {CHECKPOINT_FILE}")
            time.sleep(0.5)
            input("\n >> Press Enter to return...")
            time.sleep(0.5)
            return
        
        # The artifact is deserialised from disk
        with open(CHECKPOINT_FILE, 'rb') as f:
            artifact = pickle.load(f)
        
        # State variables are populated from the loaded artifact
        self.clean_data = artifact['datasets']['clean_data']
        self.market_data = artifact['datasets']['market_data']
        self.daily_rates = artifact['datasets']['daily_rates']
        self.flow_series = artifact['datasets'].get('flow_series', pd.Series(dtype=float))
        self.simulation_results = artifact['results']
        
        # Navigation shortcuts are re-established for convenience
        self.control_nav = self.simulation_results['control_nav_series']
        self.real_nav = self.simulation_results['real_nav_series']
        
        # The simulation engine is re-initialised using the loaded data 
        # to allow for new simulations to be run if desired
        if self.clean_data and self.market_data is not None:
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