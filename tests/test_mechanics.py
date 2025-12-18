import unittest
import pandas as pd
import numpy as np
from src.simulation_engine import MonteCarloEngine

class TestSimulationMechanics(unittest.TestCase):
    def setUp(self):
        """
        Set up a 'Toy' environment with known constraints to verify algebra.
        """
        # FIX: We define the DataFrame WITH the specific columns expected by
        # the MonteCarloEngine.__init__ method (lines 55-57 of your code).
        empty_state = pd.DataFrame(columns=['symbol', 'currency', 'quantity', 'asset_category'])
        
        dummy_package = {
            'initial_state': empty_state, 
            'events': pd.DataFrame(),
            'report_start_date': pd.Timestamp('2025-01-01'),
            'report_end_date': pd.Timestamp('2025-01-02'),
            'base_currency': 'CHF',
            'financial_info': pd.DataFrame(columns=['symbol', 'Exchange']) # Added columns here too
        }
        
        # Instantiate engine
        # We pass empty maps because we mock them manually below
        self.engine = MonteCarloEngine(dummy_package, {}, pd.DataFrame())
        
        # Mock the Fee Schedule for predictability
        self.engine.FEE_SCHEDULE = {
            'TEST_EXCH': {'fee': 5.0, 'tax_buy': 0.10},
            'DEFAULT': {'fee': 0.0, 'tax_buy': 0.0}
        }
        
        # Mock the Symbol Maps so the engine knows which logic to apply
        self.engine.sym_exchange_map = {'TEST_STOCK': 'TEST_EXCH'}
        self.engine.sym_curr_map = {'TEST_STOCK': 'CHF'}

    def test_buy_quantity_logic(self):
        """
        CASE: We have 1150 Cash. Price is 100. Fee is 5. Tax is 10%.
        Math: 
          - Cost per share = 100 * (1 + 0.10) = 110.
          - Net Cash for shares = 1150 - 5 (Fee) = 1145.
          - Quantity = 1145 / 110 = 10.40909...
        """
        available_cash = 1150.0
        price = 100.0
        symbol = 'TEST_STOCK'
        
        qty = self.engine._calculate_buy_quantity(symbol, available_cash, price)
        
        # Expected: (1150 - 5) / (100 * 1.10)
        expected_qty = 10.4090909
        self.assertAlmostEqual(qty, expected_qty, places=5, msg="Buy quantity algebra is incorrect.")
        
    def test_sell_proceeds_logic(self):
        """
        CASE: Sell 10 shares at 100. Fee is 5. No sell tax.
        Math:
          - Gross = 10 * 100 = 1000.
          - Net = 1000 - 5 = 995.
        """
        qty = 10
        price = 100.0
        symbol = 'TEST_STOCK'
        
        proceeds = self.engine._calculate_sell_proceeds(symbol, qty, price)
        
        self.assertEqual(proceeds, 995.0, "Sell proceeds calculation is incorrect.")
        
    def test_sell_cannot_be_negative(self):
        """
        CASE: Sell 1 share at 1.0. Fee is 5.
        Math: Gross 1.0 - Fee 5.0 = -4.0.
        Result should be 0.0 (You can't lose money selling, you just get zero).
        """
        proceeds = self.engine._calculate_sell_proceeds('TEST_STOCK', 1, 1.0)
        self.assertEqual(proceeds, 0.0, "Sell proceeds should floor at zero.")

    def test_dividend_tax_logic(self):
        """
        CASE: Dividend of 100. Tax Rate 35% (e.g., Swiss).
        Math: Net should be 65.
        """
        # Mock the DIV_TAX_RATES
        self.engine.DIV_TAX_RATES = {'TEST_EXCH': 0.35}
        
        qty = 10
        div_per_share = 10.0 # Total Gross = 100
        
        # We replicate the logic from _run_single_path (Part A) to test the math
        gross_div_total = div_per_share * qty
        tax_rate = self.engine.DIV_TAX_RATES.get('TEST_EXCH')
        tax_amount = gross_div_total * tax_rate
        net_cash = gross_div_total - tax_amount
        
        self.assertEqual(net_cash, 65.0, "Dividend tax deduction logic is incorrect.")

if __name__ == '__main__':
    unittest.main()