import pandas as pd
import json

# --- File and Column definitions ---
file_path = r'C:\Users\Marvin\OneDrive\[4] HEC Lausanne\1. Semester\Data Science\Project\U13271915_20250101_20251029.csv'

# We only care about columns 0, 1, 2, 3, 4
use_cols = [0, 1, 2, 3, 4]
col_names = ['Section', 'RecordType', 'AssetCategory', 'Symbol', 'PriorQty']

try:
    # --- 1. Load the raw data (from a file) ---
    df_raw = pd.read_csv(
        file_path,
        header=None,
        usecols=use_cols,
        names=col_names,
        on_bad_lines='skip' 
    )

    # --- 2. Filter for the MTM Summary Data ---
    df_mtm = df_raw[
        (df_raw['Section'] == 'Mark-to-Market Performance Summary') &
        (df_raw['RecordType'] == 'Data')
    ]

    # --- 3. Extract Initial Holdings (Stocks) ---
    df_stocks = df_mtm[df_mtm['AssetCategory'] == 'Stocks']
    
    initial_holdings = {}
    for _, row in df_stocks.iterrows():
        symbol = row['Symbol']
        quantity = pd.to_numeric(row['PriorQty'], errors='coerce')
        
        if pd.notna(quantity) and quantity > 0:
            initial_holdings[symbol] = quantity

    # --- 4. Extract Initial Cash (Forex) ---
    df_cash = df_mtm[df_mtm['AssetCategory'] == 'Forex']

    initial_cash = {}
    for _, row in df_cash.iterrows():
        currency = row['Symbol']
        balance = pd.to_numeric(row['PriorQty'], errors='coerce')
        
        if pd.notna(balance):
            initial_cash[currency] = balance

    # --- 5. Combine into the final initial_state dictionary ---
    initial_state = {
        'holdings': initial_holdings,
        'cash': initial_cash
    }

    print("--- Initial Portfolio State (t=0) ---")
    print(initial_state)

except FileNotFoundError:
    print(f"Error: The file was not found at '{file_path}'")
    print("Please make sure the file is in the correct directory.")
except Exception as e:
    print(f"An error occurred: {e}")
