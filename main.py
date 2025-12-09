import os
import sys
import pickle
import pandas as pd
from datetime import datetime

# --- Import Custom Modules ---
from src import data_loader as dl
from src import data_processor as dp
from src import market_data_loader as mdl
from src import margin_rates as mr
from src import simulation_engine as sim
from src import portfolio_reconstructor as pr
from src.portfolio_analytics import PortfolioAnalyzer

# --- Configuration Constants ---
DEFAULT_REPORT_PATH = r'data/U13271915_20250101_20251029.csv'
CHECKPOINT_DIR = r'data/checkpoints'
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'simulation_artifact.pkl')
DEFAULT_PATHS = 150
BATCH_SIZE = 10

class PortfolioSimulationApp:
    def __init__(self):
        # State Variables
        self.raw_data = None
        self.clean_data = None
        self.market_data = None
        self.daily_rates = None
        self.engine = None
        
        # Results Storage
        self.real_res = None
        self.shadow_res = None
        self.benchmark_nav = None
        self.real_nav = None
        self.simulation_results = None 
        
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def _print_section_header(self, title):
        print("\n" + "="*60)
        print(f" {title}")
        print("="*60 + "\n")

    def _check_dependency(self, condition, fix_action_name, fix_action_func):
        """
        Intelligent Chaining Logic:
        If 'condition' is False, announce the missing dependency 
        and automatically trigger 'fix_action_func'.
        """
        if not condition:
            print(f"\n [!] Missing dependency: {fix_action_name}")
            print(f" [>] Auto-triggering {fix_action_name}...")
            fix_action_func()
            return False # Signal that we had to fix something
        return True # Signal that dependency was already met

    def print_header(self):
        print("\n" + "#"*60)
        print("      PORTFOLIO RECONSTRUCTION & MONTE CARLO ENGINE")
        print("#"*60)

    def menu(self):
        while True:
            self.print_header()
            print(" 0. ** RUN FULL PIPELINE ** (Start to Finish)")
            print(" 1. Load IBKR Report")
            print(" 2. Fetch Market Data")
            print(" 3. Run Benchmark Reconstruction")
            print(" 4. Run Monte Carlo Simulation")
            print(" 5. Analyze Results")
            print(" 6. Save State")
            print(" 7. Load State")
            print(" Q. Quit")
            print("-" * 60)
            
            # Smart Status Line
            flags = []
            flags.append("DATA: OK" if self.clean_data else "DATA: --")
            flags.append("MARKET: OK" if self.engine else "MARKET: --")
            flags.append("BENCH: OK" if self.benchmark_nav is not None else "BENCH: --")
            flags.append("SIM: OK" if self.simulation_results else "SIM: --")
            
            print(f" STATUS: {' | '.join(flags)}")
            print("-" * 60)

            choice = input(" >> Select Option: ").upper().strip()

            if choice == '0':
                # Just ask for the final result, the chain handles the rest
                self.step_analyze()
            elif choice == '1':
                self.step_load_data()
            elif choice == '2':
                self.step_fetch_market_data()
            elif choice == '3':
                self.step_benchmark_reconstruction()
            elif choice == '4':
                self.step_run_simulation()
            elif choice == '5':
                self.step_analyze()
            elif choice == '6':
                self.save_checkpoint()
            elif choice == '7':
                self.load_checkpoint()
            elif choice == 'Q':
                sys.exit()
            else:
                print(" [!] Invalid selection.")

    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    def step_load_data(self):
        self._print_section_header("STEP 1: DATA INGESTION")
        
        path = input(f"Enter path to CSV [Default: {DEFAULT_REPORT_PATH}]: ").strip()
        if not path:
            path = DEFAULT_REPORT_PATH
        
        if not os.path.exists(path):
            print(f" [!] Error: File not found at {path}")
            return

        self.raw_data = dl.load_ibkr_report(path) #
        
        if self.raw_data is not None:
            self.clean_data = dp.apply_split_adjustments(self.raw_data) #
        else:
            print(" [!] Data load failed.")

    # =========================================================================
    # STEP 2: MARKET DATA
    # =========================================================================
    def step_fetch_market_data(self):
        # DEPENDENCY CHECK: Need Clean Data
        self._check_dependency(self.clean_data is not None, "Step 1 (Load Data)", self.step_load_data)
        if self.clean_data is None: return # Abort if Step 1 failed

        self._print_section_header("STEP 2: MARKET DATA ACQUISITION")
        
        start = self.clean_data['report_start_date']
        end = self.clean_data['report_end_date']
        
        self.daily_rates = mr.get_ibkr_rates_hybrid(start, end) #
        self.market_data = mdl.load_market_data(self.clean_data) #

        print("Initializing Engine...")
        self.engine = sim.MonteCarloEngine(
            self.clean_data, 
            self.market_data, 
            self.daily_rates
        ) #
        
        print("\n[+] Market data setup complete.")

    # =========================================================================
    # STEP 3: BENCHMARK RECONSTRUCTION
    # =========================================================================
    def step_benchmark_reconstruction(self):
        # DEPENDENCY CHECK: Need Engine
        self._check_dependency(self.engine is not None, "Step 2 (Market Data)", self.step_fetch_market_data)
        if self.engine is None: return

        self._print_section_header("STEP 3: BENCHMARK & REALITY CHECK")

        # 1. Real Portfolio
        self.real_res = pr.reconstruct_portfolio(self.clean_data, self.market_data, verbose=True) #
        final_nav = self.real_res['total_nav'].iloc[-1]
        print(f"\n   -> Real Portfolio Final NAV: {final_nav:,.2f}")

        # 2. Shadow Benchmark
        print("\n [>] Generating Shadow Benchmark...")
        shadow_scenario = self.engine.run_actual_with_sim_logic()[0] #
        self.shadow_res = pr.reconstruct_portfolio(shadow_scenario, self.market_data, verbose=False) #
        
        shadow_nav = self.shadow_res['total_nav'].iloc[-1]
        print(f"   -> Shadow Benchmark Final NAV: {shadow_nav:,.2f}")
        
        # Save results to state
        self.benchmark_nav = self.shadow_res['total_nav']
        self.real_nav = self.real_res['total_nav']

    # =========================================================================
    # STEP 4: MONTE CARLO SIMULATION
    # =========================================================================
    def step_run_simulation(self):
        # --- 1. DEPENDENCY CHECKS (CRITICAL) ---
        # We must ensure the Benchmark is ready. This implicitly checks Steps 1 & 2.
        self._check_dependency(
            self.benchmark_nav is not None, 
            "Step 3 (Benchmark)", 
            self.step_benchmark_reconstruction
        )
        if self.benchmark_nav is None: return

        self._print_section_header("STEP 4: MONTE CARLO SIMULATION")

        # --- 2. INPUT & CONFIGURATION ---
        user_input = input(f"Enter number of paths to generate [Default: {DEFAULT_PATHS}]: ").strip()
        try:
            n_paths = int(user_input) if user_input else DEFAULT_PATHS
        except ValueError:
            n_paths = DEFAULT_PATHS

        if n_paths <= 0:
            print(f" [!] Invalid number of paths ({n_paths}). Defaulting to {DEFAULT_PATHS}.")
            n_paths = DEFAULT_PATHS
        
        # Batching Logic: Target ~50 paths per batch to balance RAM and Updates
        BATCH_SIZE = 50 

        print(f"\n [>] Starting Monte Carlo Simulation ({n_paths} paths)...")
        
        all_navs = []
        paths_generated = 0
        
        # --- 3. SIMULATION LOOP ---
        for batch_start in range(0, n_paths, BATCH_SIZE):
            current_batch_size = min(BATCH_SIZE, n_paths - batch_start)
            
            # Call Engine SILENTLY (verbose=False) to avoid console spam
            scenarios = self.engine.generate_scenario(num_paths=current_batch_size, verbose=False)
            
            for sc in scenarios:
                res = pr.reconstruct_portfolio(sc, self.market_data, verbose=False)
                all_navs.append(res['total_nav'])
            
            # Update Progress Bar
            paths_generated += current_batch_size
            print(f"     Progress: {paths_generated}/{n_paths} paths complete...", end='\r', flush=True)
            
            del scenarios # Free memory

        print() # Newline after progress bar finishes

        if not all_navs:
            print(" [!] Simulation produced no data.")
            return

        # --- 4. STORE RESULTS ---
        self.sim_paths_df = pd.concat(all_navs, axis=1)
        self.sim_paths_df.columns = range(len(all_navs))
        
        self.simulation_results = {
            'simulated_paths': self.sim_paths_df,
            'actual_nav_series': self.benchmark_nav,
            'real_nav_series': self.real_nav
        }
        
        print(f"[+] Simulation complete. Generated {len(all_navs)} paths.")

    # =========================================================================
    # STEP 5: ANALYSIS
    # =========================================================================
    def step_analyze(self):
        # DEPENDENCY CHECK: Need Simulation Results
        self._check_dependency(self.simulation_results is not None, "Step 4 (Simulation)", self.step_run_simulation)
        if self.simulation_results is None: return

        self._print_section_header("STEP 5: ANALYTICS & VISUALIZATION")
        
        analyzer = PortfolioAnalyzer(
            real_nav=self.simulation_results['real_nav_series'],
            benchmark_nav=self.simulation_results['actual_nav_series'],
            sim_paths_df=self.simulation_results['simulated_paths']
        ) #

        summary_df, raw_stats, _ = analyzer.get_summary_table()
        
        print("PERFORMANCE SUMMARY:")
        print(summary_df)
        print("\nLaunching visualization windows...")

        # Updated method names to match portfolio_analytics.py
        analyzer.plot_confidence_intervals()  # Was plot_cone_of_uncertainty
        analyzer.plot_simulation_traces(num_paths=100) # Was plot_spaghetti
        analyzer.plot_distributions(raw_stats)
        analyzer.plot_drawdown_profile()      # Was plot_drawdown_curve
        
        input("Press Enter to return to menu...")

    # =========================================================================
    # UTILS
    # =========================================================================
    def save_checkpoint(self):
        self._print_section_header("SAVING STATE")
        if not self.simulation_results:
            print(" [!] Nothing to save (Run Simulation first).")
            input("Press Enter to return...")
            return
            
        artifact = {
            'metadata': {'timestamp': datetime.now()},
            'datasets': {
                'clean_data': self.clean_data,
                'market_data': self.market_data,
                'daily_rates': self.daily_rates
            },
            'results': self.simulation_results
        }
        
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(artifact, f)
        print(f" [+] State saved to {CHECKPOINT_FILE}")
        input("Press Enter to continue...")

    def load_checkpoint(self):
        self._print_section_header("LOADING STATE")
        if not os.path.exists(CHECKPOINT_FILE):
            print(f" [!] No checkpoint found at {CHECKPOINT_FILE}")
            input("Press Enter to return...")
            return
            
        with open(CHECKPOINT_FILE, 'rb') as f:
            artifact = pickle.load(f)
            
        self.clean_data = artifact['datasets']['clean_data']
        self.market_data = artifact['datasets']['market_data']
        self.daily_rates = artifact['datasets']['daily_rates']
        self.simulation_results = artifact['results']
        
        # Populate navigation properties
        self.benchmark_nav = self.simulation_results['actual_nav_series']
        self.real_nav = self.simulation_results['real_nav_series']
        
        # Re-init engine for potential future runs
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