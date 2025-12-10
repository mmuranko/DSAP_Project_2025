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
from datetime import datetime
from typing import Callable

# --- Import Custom Modules ---
from src import data_loader as dl
from src import data_processor as dp
from src import market_data_loader as mdl
from src import margin_rates as mr
from src import simulation_engine as sim
from src import portfolio_reconstructor as pr
from src.portfolio_analytics import PortfolioAnalyser

# --- Configuration Constants ---
# Defines the default file path for the input Interactive Brokers (IBKR) activity report.
# Used as a fallback if the user does not specify a custom path during execution.
DEFAULT_REPORT_PATH = r'data/U13271915_20250101_20251029.csv'

# Defines the directory and filename for persisting the simulation state (pickled objects).
# Storing the state allows for analysis to be resumed subsequently without re-running 
# computationally expensive simulations.
CHECKPOINT_DIR = r'data/checkpoints'
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'simulation_artifact.pkl')

# Sets default parameters for the simulation to ensure the engine can run 
# even if the user bypasses detailed configuration.
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
        # State variables are explicitly set to None to indicate an empty or uninitialised state.
        # This allows the dependency checkers to easily verify if a step has been completed.
        self.raw_data = None
        self.clean_data = None
        self.market_data = None
        self.daily_rates = None
        self.engine = None
        
        # Results storage is segregated to distinguish between the deterministic inputs 
        # (Real/Control portfolios) and the stochastic outputs (Simulation results).
        self.real_res = None
        self.control_res = None
        self.control_nav = None
        self.real_nav = None
        self.simulation_results = None 
        
        # The checkpoint directory is created immediately to ensure save operations do not fail later.
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def reset_state(self) -> None:
        """
        Clears all loaded data and processing results to facilitate a new execution cycle.

        This method ensures that subsequent pipeline runs do not inherit stale data
        from previous sessions by resetting all state variables to None.
        """
        print("\n [!] Clearing previous application state...")

        # All input datasets are cleared to prevent the mixing of old and new data.
        self.raw_data = None
        self.clean_data = None
        self.market_data = None
        self.daily_rates = None
        self.engine = None
        
        # All downstream results are invalidated to ensure that subsequent analysis 
        # reflects only the most recent data load.
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

        If a required condition is not met (e.g. data not loaded), the missing
        dependency is flagged, and the appropriate corrective action is triggered automatically.

        Args:
            condition (bool): The validity check (True if dependency is met).
            fix_action_name (str): The human-readable name of the missing step.
            fix_action_func (callable): The method to execute if the condition is False.

        Returns:
            bool: True if the dependency was already met, False if a fix was triggered.
        """
        if not condition:
            print(f"\n [!] Missing dependency: {fix_action_name}")
            print(f" [>] Auto-triggering {fix_action_name}...")
            fix_action_func()
            return False # Indicates that a corrective action was required.
        return True # Indicates that the workflow was already in a valid state.

    def print_header(self) -> None:
        """
        Renders the main application title banner to the standard output.
        """
        print("\n" + "#"*60)
        print("      PORTFOLIO RECONSTRUCTION & MONTE CARLO ENGINE")
        print("#"*60)

    def menu(self) -> None:
        """
        Displays the interactive main menu and routes user input to specific pipeline steps.

        This loop persists until the user explicitly selects the option to quit.
        """
        while True:
            self.print_header()
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

            if choice == '0':
                # Enforce a full reset to ensure the pipeline executes from a clean state
                self.reset_state()
                self.step_analyse()
            elif choice == '1':
                # Enforce a full reset to ensure the pipeline executes from a clean state
                self.reset_state()
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
                self.save_checkpoint()
            elif choice == '7':
                # Enforce a full reset to ensure the pipeline executes from a clean state
                self.reset_state()
                self.load_checkpoint()
            elif choice == 'Q':
                sys.exit()
            else:
                print(" [!] Invalid selection.")

    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    def step_load_data(self) -> None:
        """
        Executes Step 1: Ingestion of the raw IBKR activity report.

        This step parses the CSV file, applies necessary split adjustments, and
        invalidates all downstream data to prevent state inconsistencies.
        """
        self._print_section_header("STEP 1: DATA LOADING")

        # Downstream components are invalidated to force a recalcualtion with the new data
        self.market_data = None
        self.daily_rates = None
        self.engine = None
        self.real_res = None
        self.control_res = None
        self.control_nav = None
        self.real_nav = None
        self.simulation_results = None

        path = input(f"Enter path to CSV [Default: {DEFAULT_REPORT_PATH}]: ").strip()
        if not path:
            path = DEFAULT_REPORT_PATH
        
        if not os.path.exists(path):
            print(f" [!] Error: File not found at {path}")
            return
        
        # Parses the raw IBKR CSV report into a structured DataFrame for processing.
        self.raw_data = dl.load_ibkr_report(path) #
        
        if self.raw_data is not None:
            # Applies stock split adjustments to the historic holding quantities
            self.clean_data = dp.apply_split_adjustments(self.raw_data) #
        else:
            print(" [!] Data load failed.")

    # =========================================================================
    # STEP 2: MARKET DATA
    # =========================================================================
    def step_fetch_market_data(self) -> None:
        """
        Executes Step 2: Acquisition of historical market data and margin rates.

        Market data is fetched via the yfinance API, and margin rates are retrieved
        via the hybrid scraper/FRED proxy. Downstream results are invalidated upon execution.
        """
        # Verification that Step 1 has completed successfully is required before proceeding.
        self._check_dependency(self.clean_data is not None, "Step 1 (Load Data)", self.step_load_data)
        if self.clean_data is None: return # Abort if Step 1 failed

        self._print_section_header("STEP 2: MARKET DATA ACQUISITION")
        
        # If market data changes, previous simulation results are rendered invalid 
        # and must be cleared to ensure consistency.
        self.real_res = None
        self.control_res = None
        self.control_nav = None
        self.real_nav = None
        self.simulation_results = None

        # The date range for data fetching is derived directly from the loaded report.
        start = self.clean_data['report_start_date']
        end = self.clean_data['report_end_date']
        
        # Margin rates and asset prices are fetched concurrently to prepare the simulation environment.
        self.daily_rates = mr.get_ibkr_rates_hybrid(start, end) #

        # Fetch historical price data for all instruments in the portfolio.
        self.market_data = mdl.load_market_data(self.clean_data) #

        print("Initializing Engine...")
        # The Monte Carlo Engine is initialised with the newly acquired data.
        self.engine = sim.MonteCarloEngine(
            self.clean_data, 
            self.market_data, 
            self.daily_rates
        ) #
        
        print("\n[+] Market data setup complete.")

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
        # Ensure the simulation engine is initialised (implies market data is ready).
        self._check_dependency(self.engine is not None, "Step 2 (Market Data)", self.step_fetch_market_data)
        if self.engine is None: return

        self._print_section_header("STEP 3: CONTROL PORTFOLIO & REALITY CHECK")

        # Downstream simulation results are invalidated. This ensures that Step 5 triggers 
        # a fresh simulation (Step 4) if the control portfolio definition has changed.
        self.simulation_results = None

        # The real portfolio is reconstructed using the 'faithful' mode (verbose=True) to mirror actual execution.
        self.real_res = pr.reconstruct_portfolio(self.clean_data, self.market_data, verbose=True) #
        final_nav = self.real_res['total_nav'].iloc[-1]
        print(f"\n   -> Real Portfolio Final NAV: {final_nav:,.2f}")

        # The actual trades are run through the simulation logic to create a scientific baseline 
        # (Control) that is comparable to the random paths.
        print("\n [>] Generating Control Portfolio...")
        control_scenario = self.engine.run_real_with_sim_logic()[0]
        self.control_res = pr.reconstruct_portfolio(control_scenario, self.market_data, verbose=False) #
        
        control_nav = self.control_res['total_nav'].iloc[-1]
        print(f"   -> Control Portfolio Final NAV: {control_nav:,.2f}")
        
        # Key navigation series are stored in the application state for later analysis.
        self.control_nav = self.control_res['total_nav']
        self.real_nav = self.real_res['total_nav']

    # =========================================================================
    # STEP 4: MONTE CARLO SIMULATION
    # =========================================================================
    def step_run_simulation(self) -> None:
        """
        Executes Step 4: Monte Carlo Simulation.

        Generates N random counterfactual portfolio paths based on the user's
        'Competence Universe'. Results are batched to manage memory usage.
        """
        # The Control Portfolio is a prerequisite, as it defines the baseline for comparison.
        self._check_dependency(
            self.control_nav is not None, 
            "Step 3 (Control Portfolio)", 
            self.step_control_reconstruction
        )
        if self.control_nav is None: return

        self._print_section_header("STEP 4: MONTE CARLO SIMULATION")

        # User input is requested for the number of paths, with a fallback to the default value.
        user_input = input(f"Enter number of paths to generate [Default: {DEFAULT_PATHS}]: ").strip()
        try:
            n_paths = int(user_input) if user_input else DEFAULT_PATHS
        except ValueError:
            n_paths = DEFAULT_PATHS

        # Enforce a valid positive integer for the number of paths.
        if n_paths <= 0:
            print(f" [!] Invalid number of paths ({n_paths}). Defaulting to {DEFAULT_PATHS}.")
            n_paths = DEFAULT_PATHS

        print(f"\n [>] Starting Monte Carlo Simulation ({n_paths} paths)...")
        
        all_navs = []
        paths_generated = 0
        
        # --- 3. SIMULATION LOOP ---
        for batch_start in range(0, n_paths, BATCH_SIZE):
            # The current batch size is calculated to handle the final remainder correctly.
            current_batch_size = min(BATCH_SIZE, n_paths - batch_start)
            
            # The engine is called silently (verbose=False) to prevent excessive console output.
            scenarios = self.engine.generate_scenario(num_paths=current_batch_size, verbose=False)
            
            # Each generated scenario is immediately reconstructed into a NAV series.
            for sc in scenarios:
                res = pr.reconstruct_portfolio(sc, self.market_data, verbose=False)
                all_navs.append(res['total_nav'])
            
            # Progress is displayed dynamically on the same line.
            paths_generated += current_batch_size
            print(f"     Progress: {paths_generated}/{n_paths} paths complete...", end='\r', flush=True)
            
            # Memory is explicitly freed to prevent accumulation during large simulations.
            del scenarios

        print() # Newline after progress bar finishes

        if not all_navs:
            print(" [!] Simulation produced no data.")
            return

        # All individual paths are concatenated into a single DataFrame for vectorised analysis.
        self.sim_paths_df = pd.concat(all_navs, axis=1)
        self.sim_paths_df.columns = range(len(all_navs))
        
        # The complete result set is packaged into a dictionary for easy saving and loading.
        self.simulation_results = {
            'simulated_paths': self.sim_paths_df,
            'control_nav_series': self.control_nav,
            'real_nav_series': self.real_nav
        }
        
        print(f"[+] Simulation complete. Generated {len(all_navs)} paths.")

    # =========================================================================
    # STEP 5: ANALYSIS
    # =========================================================================
    def step_analyse(self) -> None:
        """
        Executes Step 5: Statistical Analysis and Visualisation.

        Instantiates the PortfolioAnalyser to calculate performance metrics (CAGR,
        Sharpe, Drawdown) and generates comparative plots (Confidence Intervals,
        Distributions).
        """
        # Verification that simulation results exist is required before analysis can begin.
        self._check_dependency(self.simulation_results is not None, "Step 4 (Simulation)", self.step_run_simulation)
        if self.simulation_results is None: return

        self._print_section_header("STEP 5: ANALYTICS & VISUALISATION")
        
        # The Analysis module is instantiated with the prepared datasets.
        analyser = PortfolioAnalyser(
            real_nav=self.simulation_results['real_nav_series'],
            control_nav=self.simulation_results['control_nav_series'],
            sim_paths_df=self.simulation_results['simulated_paths']
        )

        # Performance summaries are generated and displayed in the console.
        summary_df, raw_stats, _ = analyser.get_summary_table()
        
        print("PERFORMANCE SUMMARY:")
        print(summary_df)
        print("\nLaunching visualisation windows...")

        # Visualisations are triggered sequentially.
        # Note: The application will pause here until the user closes each plot window.
        analyser.plot_confidence_intervals()
        analyser.plot_simulation_traces(num_paths=100) 
        analyser.plot_distributions(raw_stats)
        analyser.plot_drawdown_profile()
        
        input("Press Enter to return to menu...")

    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================
    def save_checkpoint(self) -> None:
        """
        Serialises the current application state to a binary pickle file.

        This allows for the persistence of expensive simulation results and loaded datasets.
        """
        self._print_section_header("SAVING STATE")
        # A check is performed to ensure there is actually data to save.
        if not self.simulation_results:
            print(" [!] Nothing to save (Run Simulation first).")
            input("Press Enter to return...")
            return

        # The entire application state is packaged into a dictionary. 
        # This includes metadata (timestamp) and all datasets required to resume analysis.
        artifact = {
            'metadata': {'timestamp': datetime.now()},
            'datasets': {
                'clean_data': self.clean_data,
                'market_data': self.market_data,
                'daily_rates': self.daily_rates
            },
            'results': self.simulation_results
        }

        # The artifact is serialised to disk using pickle.
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(artifact, f)
        print(f" [+] State saved to {CHECKPOINT_FILE}")
        input("Press Enter to continue...")

    def load_checkpoint(self) -> None:
        """
        Deserialises application state from a binary pickle file.

        Restores datasets and simulation results, and re-initialises the simulation
        engine for immediate use.
        """
        self._print_section_header("LOADING STATE")
        if not os.path.exists(CHECKPOINT_FILE):
            print(f" [!] No checkpoint found at {CHECKPOINT_FILE}")
            input("Press Enter to return...")
            return
        
        # The artifact is deserialised from disk.
        with open(CHECKPOINT_FILE, 'rb') as f:
            artifact = pickle.load(f)
        
        # State variables are populated from the loaded artifact.
        self.clean_data = artifact['datasets']['clean_data']
        self.market_data = artifact['datasets']['market_data']
        self.daily_rates = artifact['datasets']['daily_rates']
        self.simulation_results = artifact['results']
        
        # Navigation shortcuts are re-established for convenience.
        self.control_nav = self.simulation_results['control_nav_series']
        self.real_nav = self.simulation_results['real_nav_series']
        
        # The simulation engine is re-initialised using the loaded data 
        # to allow for new simulations to be run if desired.
        if self.clean_data and self.market_data is not None:
             self.engine = sim.MonteCarloEngine(
                self.clean_data, 
                self.market_data, 
                self.daily_rates
            ) #
            
        print(" [+] Load complete.")
        input("Press Enter to continue...")

if __name__ == "__main__":
    app = PortfolioSimulationApp()
    try:
        app.menu()
    except KeyboardInterrupt:
        print("\n [!] Interrupted by user. Exiting.")
        sys.exit()