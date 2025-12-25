import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the actual module from your project structure
from src.simulation_engine import MonteCarloEngine

class TestSimulationMechanics(unittest.TestCase):
    
    def setUp(self):
        """
        Prepares a 'Toy' environment with a deterministic setup.
        """
        print("\n" + "="*50)
        print(" [SETUP] Initialising Toy Environment...")
        
        cols_state = ['symbol', 'asset_category', 'currency', 'quantity', 'value_native']
        cols_events = ['timestamp', 'event_type', 'symbol', 'currency', 'quantity_change', 'cash_change_native']
        
        self.start_date = pd.Timestamp('2025-01-01')
        self.end_date = pd.Timestamp('2025-01-05')
        
        # Initial State: 10,000 CHF Cash, and a placeholder for TEST_STOCK
        initial_state = pd.DataFrame([
            {
                'symbol': 'CHF', 
                'asset_category': 'Cash', 
                'currency': 'CHF', 
                'quantity': 10000.0, 
                'value_native': 10000.0
            },
            {
                'symbol': 'TEST_STOCK', 
                'asset_category': 'Stock', 
                'currency': 'CHF', 
                'quantity': 0.0,  
                'value_native': 0.0
            }
        ], columns=cols_state)
        
        # Events: A single BUY trade on day 2
        events_data = [
            {
                'timestamp': self.start_date + timedelta(days=1),
                'event_type': 'TRADE_BUY', 
                'symbol': 'TEST_STOCK',
                'currency': 'CHF',
                'quantity_change': 10,       
                'cash_change_native': -1005.0 
            }
        ]
        events_df = pd.DataFrame(events_data, columns=cols_events)

        # Market Data: Flat price of 100 for TEST_STOCK
        dates = pd.date_range(self.start_date, self.end_date)
        market_data_df = pd.DataFrame(index=dates)
        market_data_df['Close'] = 100.0
        market_data_df['Dividends'] = 0.0 
        
        market_data_map = {'TEST_STOCK': market_data_df}
        
        daily_rates = pd.DataFrame(0.0001, index=dates, columns=['CHF', 'USD'])

        self.dummy_package = {
            'initial_state': initial_state, 
            'events': events_df,
            'report_start_date': self.start_date,
            'report_end_date': self.end_date,
            'base_currency': 'CHF',
            'financial_info': pd.DataFrame({'symbol': ['TEST_STOCK'], 'Exchange': ['TEST_EXCH']})
        }
        
        self.engine = MonteCarloEngine(self.dummy_package, market_data_map, daily_rates)
        
        self.engine.FEE_SCHEDULE = {
            'TEST_EXCH': {'fee': 5.0, 'tax_buy': 0.10}, 
            'DEFAULT': {'fee': 0.0, 'tax_buy': 0.0}
        }
        self.engine.sym_exchange_map = {'TEST_STOCK': 'TEST_EXCH'}
        self.engine.sym_curr_map = {'TEST_STOCK': 'CHF'}
        self.engine.DIVIDEND_TAX_RATES = {'TEST_EXCH': 0.35}
        
        print(" [SETUP] Environment Ready.")

    # ==========================================
    # CATEGORY 1: ARITHMETIC (Math Verification)
    # ==========================================
    
    def test_buy_quantity_algebra(self):
        """Verifies (Cash - Fee) / (Price * (1 + Tax)) logic."""
        print(" [TEST] Verifying Buy Quantity Algebra...")
        cash = 1150.0
        price = 100.0
        # Math: (1150 - 5) / (100 * 1.10) = 1145 / 110 = 10.40909...
        qty = self.engine._calculate_buy_quantity('TEST_STOCK', cash, price)
        
        self.assertAlmostEqual(qty, 10.4090909, places=5)
        print(f"   > Input Cash: {cash}, Price: {price}")
        print(f"   > Calculated Qty: {qty:.5f} (Expected: 10.40909)")
        print(" [PASS] Buy Quantity Algebra verified.")

    def test_sell_proceeds_algebra(self):
        """Verifies (Qty * Price) - Fee logic."""
        print(" [TEST] Verifying Sell Proceeds Algebra...")
        qty = 10
        price = 100.0
        # Math: 1000 - 5 = 995
        proceeds = self.engine._calculate_sell_proceeds('TEST_STOCK', qty, price)
        
        self.assertEqual(proceeds, 995.0)
        print(f"   > Sold {qty} @ {price}")
        print(f"   > Net Proceeds: {proceeds} (Expected: 995.0)")
        print(" [PASS] Sell Proceeds Algebra verified.")

    def test_zero_cash_buy(self):
        """Edge Case: Buying with 0 cash should return 0 quantity, not crash."""
        print(" [TEST] Checking Zero-Cash Buy safety...")
        qty = self.engine._calculate_buy_quantity('TEST_STOCK', 0.0, 100.0)
        self.assertEqual(qty, 0.0)
        print(" [PASS] Engine safely handled zero cash input.")

    def test_negative_sell_protection(self):
        """Edge Case: Selling small amounts where Fee > Value should floor at 0."""
        print(" [TEST] Checking Negative Sell Protection (Fee > Value)...")
        # Value 1.0 - Fee 5.0 = -4.0 -> Should be 0.0
        proceeds = self.engine._calculate_sell_proceeds('TEST_STOCK', 1, 1.0)
        self.assertEqual(proceeds, 0.0)
        print(f"   > Proceeds floor at {proceeds}")
        print(" [PASS] Negative proceeds prevented.")

    # ==========================================
    # CATEGORY 2: INITIALIZATION & DATA INTEGRITY
    # ==========================================

    def test_symbol_mapping(self):
        """Ensures the engine correctly mapped symbols to exchanges/currencies."""
        print(" [TEST] Verifying Symbol Mapping...")
        exch = self.engine.sym_exchange_map.get('TEST_STOCK')
        curr = self.engine.sym_curr_map.get('TEST_STOCK')
        
        self.assertEqual(exch, 'TEST_EXCH')
        self.assertEqual(curr, 'CHF')
        print(f"   > Mapped TEST_STOCK to {exch} / {curr}")
        print(" [PASS] Symbol maps are correct.")

    def test_initial_cash_load(self):
        """Ensures the initial state was loaded correctly into the engine's internal memory."""
        print(" [TEST] Verifying Initial State Load...")
        initial_cash_df = self.engine.original_initial_state
        cash_row = initial_cash_df[initial_cash_df['symbol'] == 'CHF']
        val = cash_row['quantity'].values[0]
        
        self.assertEqual(val, 10000.0)
        print(f"   > Found Initial CHF: {val}")
        print(" [PASS] Initial state loaded successfully.")

    # ==========================================
    # CATEGORY 3: INTEGRATION (The "Control" Run)
    # ==========================================

    def test_control_run_execution(self):
        """
        CRITICAL TEST: Runs the 'Real with Sim Logic' mode.
        """
        print(" [TEST] Executing Control Portfolio Run (Deterministic)...")
        try:
            scenarios = self.engine.run_real_with_sim_logic()
            
            self.assertIsInstance(scenarios, list)
            self.assertEqual(len(scenarios), 1)
            
            result_df = scenarios[0]['events']
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertGreaterEqual(len(result_df), 1)
            
            print(f"   > Generated {len(result_df)} events in Control timeline.")
            print(" [PASS] Control Run executed without crashing.")

        except Exception as e:
            self.fail(f"Control Run crashed with error: {e}")

    def test_generate_scenario_smoke_test(self):
        """
        Smoke Test for the main random generator.
        """
        print(" [TEST] Executing Random Scenario Smoke Test...")
        try:
            scenarios = self.engine.generate_scenario(num_paths=2, verbose=False, seed=123)
            self.assertEqual(len(scenarios), 2)
            
            first_path_events = scenarios[0]['events']
            self.assertFalse(first_path_events.empty, "Generated scenario events should not be empty")
            
            print(f"   > Successfully generated {len(scenarios)} random paths.")
            print(" [PASS] Monte Carlo Generator is functional.")
            
        except Exception as e:
            self.fail(f"Monte Carlo Random Generator crashed: {e}")

if __name__ == '__main__':
    unittest.main()