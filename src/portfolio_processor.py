import pandas as pd

def adjust_for_splits(report_data):
    """
    Adjusts initial holdings and event quantities for stock splits.
    
    This function takes the raw output from the data_loader and transforms
    all quantities to be compatible with split-adjusted price data (like yfinance).
    
    1. Creates a "split factor" time-series (e.g., IBKR is 4.0 before split, 1.0 after).
    2. Adjusts the 'initial_state' quantities by the factor on the start date.
    3. Adjusts all 'event_log' quantities by the correct factor for that event's date.
    4. Removes the 'SPLIT' events from the log, as they are now "baked in".
    
    Args:
        report_data (dict): The dictionary from load_ibkr_report()
                            containing 'initial_state' and 'events'.
                            
    Returns:
        dict: A new dictionary with the adjusted 'initial_state' and 'events' DataFrames.
    """
    
    df_initial_state = report_data['initial_state'].copy()
    df_events = report_data['events'].copy()

    # --- 1. Isolate stocks and split events ---
    
    # Get all unique stock symbols from the initial state
    stock_symbols = df_initial_state[
        df_initial_state['asset_category'] == 'Stock'
    ]['symbol'].unique()
    
    # Get all split events
    df_splits = df_events[df_events['event_type'] == 'SPLIT'].sort_values(by='timestamp')

    # If there are no stocks or no splits, no adjustments are needed.
    if len(stock_symbols) == 0 or df_splits.empty:
        print("No splits found. Returning data as-is.")
        # Return a copy to avoid changing the original data
        return {
            'initial_state': df_initial_state,
            'events': df_events
        }

    # --- 2. Create the Time-Series Adjustment Factor DataFrame ---
    
    # Get the full date range of the report
    start_date = df_events['timestamp'].min().date()
    end_date = df_events['timestamp'].max().date()
    date_index = pd.date_range(start_date, end_date, freq='D')
    
    # Create the "Split-event" DataFrame you described. Default factor is 1.0.
    df_adj_factors = pd.DataFrame(1.0, index=date_index, columns=stock_symbols)
    
    # This is the core logic:
    # We loop through each split and apply its ratio to all dates *before* it.
    for _, split in df_splits.iterrows():
        symbol = split['symbol']
        ratio = split['split_ratio']
        split_date = split['timestamp'].date()
        
        if symbol in df_adj_factors.columns:
            # Multiply all factors *before* the split_date by the ratio
            df_adj_factors.loc[:split_date - pd.Timedelta(days=1), symbol] *= ratio

    # --- 3. Adjust df_initial_state ---
    
    # Get the adjustment factors for the very first day
    initial_factors = df_adj_factors.iloc[0]
    
    # Create a copy to modify
    df_initial_state_adj = df_initial_state.set_index('symbol')
    
    # Vectorized multiplication:
    # For all symbols in our initial_factors, multiply their 'quantity'
    df_initial_state_adj.loc[initial_factors.index, 'quantity'] *= initial_factors
    
    # Reset the index to return it to its original format
    df_initial_state_adj = df_initial_state_adj.reset_index()


    # --- 4. Adjust df_event_log ---
    
    # Create a copy to modify
    df_events_adj = df_events.copy()
    
    # We need to map each event's date to its adjustment factor.
    # First, transform df_adj_factors from wide (dates x symbols) to long
    df_adj_factors_long = df_adj_factors.stack().reset_index(name='adj_factor')
    df_adj_factors_long.columns = ['event_date', 'symbol', 'adj_factor']
    
    # Ensure dates are in the same format for merging
    df_adj_factors_long['event_date'] = pd.to_datetime(df_adj_factors_long['event_date'])
    df_events_adj['event_date'] = pd.to_datetime(df_events_adj['timestamp'].dt.date)
    
    # Merge the adjustment factor into every event
    df_events_adj = pd.merge(
        df_events_adj,
        df_adj_factors_long,
        on=['event_date', 'symbol'],
        how='left'
    )
    
    # Fill non-stock events (like dividends, cash) with a factor of 1.0
    df_events_adj['adj_factor'] = df_events_adj['adj_factor'].fillna(1.0)
    
    # Vectorized adjustment of all quantities at once
    df_events_adj['quantity_change'] = df_events_adj['quantity_change'] * df_events_adj['adj_factor']

# --- 5. Clean Up and Return ---
    
    # Remove the 'SPLIT' events, as they are now "baked in"
    df_events_adj_final = df_events_adj[df_events_adj['event_type'] != 'SPLIT'].copy()
    
    # Drop the temporary helper columns
    df_events_adj_final = df_events_adj_final.drop(columns=['event_date', 'adj_factor'])
    
    # Return a bigger dictionary with all the intermediate pieces
    return {
        'initial_state_adj': df_initial_state_adj,
        'events_adj': df_events_adj_final,
        'adj_factors_wide': df_adj_factors,     # <-- The DataFrame you want
        'adj_factors_long': df_adj_factors_long,   # <-- The other DataFrame you want
        'events_with_factors': df_events_adj,      # <-- A bonus: events *before* cleanup
        'splits_found': df_splits                  # <-- Also very useful for debugging
    }


# --- Example of how to use this module ---
if __name__ == "__main__":
    
    # This test assumes:
    # 1. You have a 'src' folder with 'data_loader.py' in it.
    # 2. This file ('portfolio_processor.py') is also in 'src'.
    # 3. Your data is in a 'data' folder, one level up.
    
    try:
        from src import data_loader as dl
    except ImportError:
        print("Could not import data_loader. Make sure this file is in the 'src' folder.")
        exit()

    # Define paths (relative to the project root, not 'src')
    filepath = r'data/U13271915_20250101_20251029.csv'
    output_excel_path = r'data/report_output_ADJUSTED.xlsx'
    
    print(f"Loading report from: {filepath}...")
    original_report = dl.load_ibkr_report(filepath)
    
    if original_report:
        print("Report loaded successfully. Adjusting for splits...")
        
        # --- THIS IS THE NEW STEP ---
        adjusted_report = adjust_for_splits(original_report)
        
        print("Data adjusted. Saving to Excel...")
        
        # --- Save the ADJUSTED data to a new file ---
        try:
            with pd.ExcelWriter(output_excel_path) as writer:
                adjusted_report['initial_state'].to_excel(writer, sheet_name='Initial_State_Adj', index=False)
                adjusted_report['events'].to_excel(writer, sheet_name='Event_Log_Adj', index=False)
            
            print(f"Successfully saved ADJUSTED data to: {output_excel_path}")
            
        except ImportError:
            print("Error: 'openpyxl' is required to save to Excel. `pip install openpyxl`")
        except Exception as e:
            print(f"An error occurred while saving to Excel: {e}")
        
        # --- Print a comparison to prove it worked ---
        print("\n" + "="*80)
        print("Split Adjustment Demonstration (IBKR)")
        
        # 1. Initial State
        orig_qty = original_report['initial_state'].set_index('symbol').loc['IBKR', 'quantity']
        adj_qty = adjusted_report['initial_state'].set_index('symbol').loc['IBKR', 'quantity']
        print(f"\nInitial State 'IBKR' Quantity:")
        print(f"  Original:  {orig_qty}")
        print(f"  Adjusted:  {adj_qty}  <-- (Original * 4.0)")
        
        # 2. Event Log
        orig_trade = original_report['events'][
            (original_report['events']['symbol'] == 'IBKR') & 
            (original_report['events']['event_type'] == 'TRADE_SELL')
        ].iloc[0]
        
        adj_trade = adjusted_report['events'][
            (adjusted_report['events']['symbol'] == 'IBKR') & 
            (adjusted_report['events']['event_type'] == 'TRADE_SELL')
        ].iloc[0]

        print(f"\nEvent Log 'IBKR' Trade (2025-07-23):")
        print(f"  Original Qty:  {orig_trade['quantity_change']}")
        print(f"  Adjusted Qty:  {adj_trade['quantity_change']}  <-- (This trade was *after* the split, so factor is 1.0)")
        
        print(f"\nEvent Log 'SPLIT' Events:")
        print(f"  Original Log:  {len(original_report['events'][original_report['events']['event_type'] == 'SPLIT'])} split event(s) found.")
        print(f"  Adjusted Log:  {len(adjusted_report['events'][adjusted_report['events']['event_type'] == 'SPLIT'])} split event(s) found. <-- (Removed)")