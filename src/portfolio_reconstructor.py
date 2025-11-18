import pandas as pd

def _get_daily_changes(df_events, master_date_index):
    """
    Private helper to aggregate all daily *quantity* changes from the event log.
    """
    # Filter for events that actually change quantity
    df_qty_events = df_events[df_events['quantity_change'] != 0].copy()
    
    if df_qty_events.empty:
        return pd.DataFrame() # Return an empty DF if no qty changes
        
    df_changes = df_qty_events.groupby([
        df_qty_events['timestamp'].dt.date, 'symbol'
    ])['quantity_change'].sum()
    
    df_qty_changes = df_changes.unstack(level='symbol').fillna(0)
    # Reindex to master index and fill missing days with 0
    df_qty_changes = df_qty_changes.reindex(master_date_index.date, fill_value=0)
    df_qty_changes.index = master_date_index # Set back to full Timestamp index
    
    return df_qty_changes

def _get_daily_cash_flows(df_events, master_date_index):
    """
    Private helper to aggregate all daily *cash* flows, grouped by *currency*.
    This is the correct way to track cash balances.
    """
    # Filter for events that actually change cash
    df_cash_events = df_events[df_events['cash_change_native'] != 0].copy()
    
    if df_cash_events.empty:
        return pd.DataFrame() # Return an empty DF if no cash flows
        
    # Group by date and CURRENCY (this is the critical fix)
    df_flows = df_cash_events.groupby([
        df_cash_events['timestamp'].dt.date, 'currency'
    ])['cash_change_native'].sum()
    
    # Unstack to get currencies as columns (e.g., 'CHF', 'USD', 'EUR')
    df_flows_unstacked = df_flows.unstack(level='currency').fillna(0)
    
    # Reindex to the master daily index
    df_flows_reindexed = df_flows_unstacked.reindex(master_date_index.date, fill_value=0)
    df_flows_reindexed.index = master_date_index # Set back to full Timestamp index
    return df_flows_reindexed
    

def _get_aligned_prices(market_data_bundle, master_date_index):
    """
    Private helper to get all native prices and FX rates, aligned to the master index.
    """
    prices_native = {}
    
    for symbol, df in market_data_bundle['data'].items():
        if not df.empty:
            # Get only the 'Close' price, reindex to master, and forward-fill missing (weekends)
            # We align to the .date to match the master_date_index
            df.index = pd.to_datetime(df.index)
            prices_native[symbol] = df['Close'].reindex(master_date_index, method='ffill')
        else:
            # Create an empty Series if there's no price data
            prices_native[symbol] = pd.Series(index=master_date_index, dtype=float)
            
    return prices_native

def reconstruct_portfolio(adjusted_report_bundle, market_data_bundle, original_report):
    """
    Reconstructs the daily portfolio time series for each asset.

    Args:
        adjusted_report_bundle (dict): From portfolio_processor.adjust_for_splits()
                                 (This is the one with '_adj' keys)
        market_data_bundle (dict): From market_data_loader.fetch_market_data()
        original_report (dict): From data_loader.load_ibkr_report()

    Returns:
        dict: {symbol: pd.DataFrame}
              Each DataFrame contains the daily time series for:
              - quantity
              - price_native
              - value_native
              - fx_rate
              - value_base_chf
    """
    
    # --- 1. Get Key Inputs ---
    # --- FIX: Use the '_adj' keys from the processor ---
    df_initial_state = adjusted_report_bundle['initial_state_adj']
    df_events = adjusted_report_bundle['events_adj']
    # --- END FIX ---
    
    base_currency = original_report['base_currency']
    start_date = original_report['report_start_date']
    end_date = original_report['report_end_date']

    # --- 2. Create Master Date Index (for the report period only) ---
    master_date_index = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # --- 3. Pre-process all Price & Event Data ---
    
    aligned_prices = _get_aligned_prices(market_data_bundle, master_date_index)
    
    # Get daily *quantity* changes (grouped by symbol)
    df_qty_changes = _get_daily_changes(df_events, master_date_index)
    
    # Get daily *cash* flows (grouped by CURRENCY)
    df_cash_flows = _get_daily_cash_flows(df_events, master_date_index)
    
    portfolio_timeseries = {}
    
    # --- 4. Iterate Through Each Asset in the Initial State ---
    for _, row in df_initial_state.iterrows():
        symbol = row['symbol']
        asset_category = row['asset_category']
        currency = row['currency']
        initial_quantity = row['quantity']

        df_symbol_ts = pd.DataFrame(index=master_date_index)
        
        price_native_ts = aligned_prices.get(symbol, pd.Series(index=master_date_index, dtype=float))
        fx_rate_ts = aligned_prices.get(currency, pd.Series(index=master_date_index, dtype=float))

        # --- 5. Reconstruct Based on Asset Category ---
        
        if asset_category == 'Stock':
            if symbol in df_qty_changes.columns:
                qty_changes_ts = df_qty_changes[symbol]
            else:
                qty_changes_ts = pd.Series(0.0, index=master_date_index)
            
            df_symbol_ts['quantity'] = initial_quantity + qty_changes_ts.cumsum()
            df_symbol_ts['price_native'] = price_native_ts
            
        elif asset_category == 'Cash':
            if symbol in df_cash_flows.columns:
                cash_changes_ts = df_cash_flows[symbol]
            else:
                cash_changes_ts = pd.Series(0.0, index=master_date_index)

            df_symbol_ts['quantity'] = initial_quantity + cash_changes_ts.cumsum()
            df_symbol_ts['price_native'] = 1.0

        # --- 6. Calculate Values ---
        df_symbol_ts['value_native'] = df_symbol_ts['quantity'] * df_symbol_ts['price_native']
        df_symbol_ts['fx_rate'] = fx_rate_ts
        df_symbol_ts['value_base_chf'] = df_symbol_ts['value_native'] * df_symbol_ts['fx_rate']
        
        # Fill any missing values at the beginning (before first price)
        df_symbol_ts = df_symbol_ts.ffill()
        
        portfolio_timeseries[symbol] = df_symbol_ts

    print(f"Successfully reconstructed portfolio time series for {len(portfolio_timeseries)} assets.")
    return portfolio_timeseries


# --- Example of how to use this module ---
if __name__ == "__main__":
    # This block allows you to test this module directly
    # You'll need to run this from your main project directory
    
    import sys
    # Add 'src' to the path to find the other modules
    if 'src' not in sys.path:
        sys.path.append('src')
        
    try:
        import data_loader as dl
        import market_data_loader as mdl
        import data_processor as pp
    except ImportError:
        print("Error: Could not import other modules (data_loader, market_data_loader, portfolio_processor).")
        print("Please run this script from your main project folder (the one containing 'src').")
        sys.exit()

    FILE_PATH = r'data/U13271915_20250101_20251029.csv'
    EXCEL_OUTPUT_PATH = r'data/portfolio_reconstruction_output.xlsx'
    
    print(f"--- 1. Loading Report: {FILE_PATH} ---")
    original_report = dl.load_ibkr_report(FILE_PATH)
    
    if original_report:
        print("--- 2. Adjusting for Splits ---")
        # --- FIX: Use the 'adjusted_report_bundle' variable ---
        adjusted_report_bundle = pp.adjust_for_splits(original_report)
        
        print("--- 3. Fetching Market Data (Stocks & FX) ---")
        market_data_bundle = mdl.fetch_market_data(original_report)
        
        if market_data_bundle:
            print("--- 4. Reconstructing Portfolio ---")
            # --- FIX: Pass the correct bundle ---
            portfolio_ts = reconstruct_portfolio(adjusted_report_bundle, market_data_bundle, original_report)
            
            # --- 5. Save Results to Excel ---
            print(f"\nSaving reconstructed portfolio to: {EXCEL_OUTPUT_PATH}")
            try:
                with pd.ExcelWriter(EXCEL_OUTPUT_PATH) as writer:
                    for symbol, df in portfolio_ts.items():
                        df.to_excel(writer, sheet_name=symbol)
                print(f"Successfully saved all {len(portfolio_ts)} assets to Excel.")
            
            except ImportError:
                print("\n--- ERROR: 'openpyxl' is required to save to Excel ---")
                print("Please run: pip install openpyxl")
            except Exception as e:
                print(f"\nAn error occurred while saving to Excel: {e}")

            # --- 6. Show Sample Output ---
            print("\n--- Sample Time Series for 'ASML' (first 5 days) ---")
            if 'ASML' in portfolio_ts:
                print(portfolio_ts['ASML'].head())
            
            print("\n--- Sample Time Series for 'USD' (first 5 days) ---")
            if 'USD' in portfolio_ts:
                print(portfolio_ts['USD'].head())

            print("\n--- Sample Time Series for 'CHF' (first 5 days) ---")
            if 'CHF' in portfolio_ts:
                print(portfolio_ts['CHF'].head())