import pandas as pd

# We only care about columns 0, 1, 2, 3, 4
USE_COLS = [0, 1, 2, 3, 4]
COL_NAMES = ['Section', 'RecordType', 'AssetCategory', 'Symbol', 'PriorQty']

def load_initial_state(filepath):
    """
    Loads the initial holdings and cash balances from an IBKR activity report.
    
    Args:
        filepath (str): The path to the IBKR CSV report.
        
    Returns:
        dict: A dictionary containing 'holdings' and 'cash' dictionaries.
    """
    try:
        # --- 1. Load the raw data (from the provided filepath) ---
        df_raw = pd.read_csv(
            filepath,
            header=None,
            usecols=USE_COLS,
            names=COL_NAMES,
            on_bad_lines='skip'
        )

        # --- 2. Filter for the MTM Summary Data ---
        df_mtm = df_raw[
            (df_raw['Section'] == 'Mark-to-Market Performance Summary') &
            (df_raw['RecordType'] == 'Data')
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        # --- 3. Vectorized Data Conversion (MUCH faster) ---
        # Convert the 'PriorQty' column to a numeric type all at once.
        df_mtm['Quantity'] = pd.to_numeric(df_mtm['PriorQty'], errors='coerce')

        # --- 4. Extract Initial Holdings (Stocks) ---
        # Filter for stocks with a quantity > 0
        df_stocks = df_mtm[
            (df_mtm['AssetCategory'] == 'Stocks') &
            (df_mtm['Quantity'] > 0)
        ]
        
        # Convert to dictionary directly (no loop needed)
        # This sets the 'Symbol' as the key and 'Quantity' as the value
        initial_holdings = df_stocks.set_index('Symbol')['Quantity'].to_dict()

        # --- 5. Extract Initial Cash (Forex) ---
        # Filter for forex (cash) balances
        df_cash = df_mtm[
            (df_mtm['AssetCategory'] == 'Forex') &
            (df_mtm['Quantity'].notna())
        ]
        
        # Convert to dictionary directly
        initial_cash = df_cash.set_index('Symbol')['Quantity'].to_dict()

        # --- 6. Combine and Return ---
        initial_state = {
            'holdings': initial_holdings,
            'cash': initial_cash
        }
        
        return initial_state

    except FileNotFoundError:
        print(f"Error: The file was not found at '{filepath}'")
        print("Please make sure the file is in the correct directory.")
        return None # Return None to signal an error
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- Testing Block ---
# This only runs when you execute this file directly
# (e.g., by running 'python src/data_loader.py')
# It will NOT run when you import it into your notebook.
if __name__ == "__main__":
    
    # Use the relative path from our project structure
    test_file_path = 'data/ibkr_sample.csv'
    
    print(f"--- Testing {__file__} ---")
    state = load_initial_state(test_file_path)
    
    if state:
        print("--- Initial Portfolio State (t=0) ---")
        import json # Import here since it's only for pretty-printing
        print(json.dumps(state, indent=2))