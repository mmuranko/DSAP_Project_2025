import pandas as pd
import numpy as np

def apply_split_adjustments(data_package):
    df_initial = data_package['initial_state'].copy()
    df_events = data_package['events'].copy()
    
    # --- 1. Build the Multiplier Timeline ---
    # (This part remains similar, but we add an 'end_timestamp')
    
    mask_splits = df_events['event_type'] == 'SPLIT'
    df_splits = df_events.loc[mask_splits, ['timestamp', 'symbol', 'split_ratio']].copy()

    if df_splits.empty:
        return {
            'df_initial_state_adj': df_initial,
            'df_event_log_adj': df_events,
            'financial_info': data_package['financial_info'],
            'report_start_date': data_package['report_start_date'],
            'report_end_date': data_package['report_end_date'],
            'base_currency': data_package['base_currency']
        }

    df_splits = df_splits.sort_values(by=['symbol', 'timestamp'])

    # Calculate Multipliers (Same math as before)
    total_factors = df_splits.groupby('symbol')['split_ratio'].transform('prod')
    running_factors = df_splits.groupby('symbol')['split_ratio'].cumprod()
    df_splits['multiplier'] = total_factors / running_factors
    
    # Create Timeline: Start at 'min', include actual splits
    split_symbols = df_splits['symbol'].unique()
    start_rows = []
    for sym in split_symbols:
        tf = df_splits[df_splits['symbol'] == sym]['split_ratio'].prod()
        start_rows.append({'timestamp': pd.Timestamp.min, 'symbol': sym, 'multiplier': tf})
        
    df_timeline = pd.concat([pd.DataFrame(start_rows), df_splits[['timestamp', 'symbol', 'multiplier']]])
    df_timeline = df_timeline.sort_values(by=['symbol', 'timestamp'])
    
    # CRITICAL NEW STEP: Create Time Intervals
    # We shift the timestamp column up to create an "end_date" for each multiplier period
    df_timeline['next_timestamp'] = df_timeline.groupby('symbol')['timestamp'].shift(-1).fillna(pd.Timestamp.max)
    
    # Now df_timeline has: symbol | multiplier | timestamp (start) | next_timestamp (end)

    # --- 2. Adjust Initial State (Dictionary Mapping Strategy) ---
    # This strategy preserves your index order perfectly.
    
    report_start = pd.to_datetime(data_package['report_start_date'])
    
    # Filter timeline to find the ONE valid row per symbol for the report_start_date
    # Logic: Start time must be before report start, End time must be after report start
    mask_valid = (df_timeline['timestamp'] <= report_start) & (df_timeline['next_timestamp'] > report_start)
    valid_multipliers = df_timeline.loc[mask_valid].set_index('symbol')['multiplier']
    
    # Map the multipliers. 
    # .map() aligns by the Index (Symbol) and returns a Series matching the original df_initial order.
    # fillna(1.0) handles symbols that had no splits.
    multipliers_aligned = df_initial['symbol'].map(valid_multipliers).fillna(1.0)
    
    df_initial_state_adj = df_initial.copy()
    df_initial_state_adj['quantity'] = df_initial['quantity'] * multipliers_aligned

    # --- 3. Adjust Event Log (Standard Merge Strategy) ---
    # Since we are adjusting events, sorting by time is usually acceptable here, 
    # but this method allows you to sort however you want later.
    
    # Left join on SYMBOL only. 
    # This creates pairs of (Event, Multiplier_Range)
    df_merged = pd.merge(df_events, df_timeline, on='symbol', how='left')
    
    # Filter: Keep rows where the Event Time is inside the Multiplier Range
    # If multiplier is NaN (no splits for this stock), we keep the row (handled by filling NaNs later)
    mask_range = (
        (df_merged['timestamp_x'] >= df_merged['timestamp_y']) & 
        (df_merged['timestamp_x'] < df_merged['next_timestamp'])
    )
    
    # Rows with no splits will have NaN in 'timestamp_y', so they fail the mask. 
    # We must keep them OR rows that satisfy the mask.
    mask_keep = mask_range | df_merged['multiplier'].isna()
    
    df_event_log_adj = df_merged[mask_keep].copy()
    
    # Fill NaNs (for stocks with no splits) and Calculate
    df_event_log_adj['multiplier'] = df_event_log_adj['multiplier'].fillna(1.0)
    df_event_log_adj['quantity_change'] = df_event_log_adj['quantity_change'] * df_event_log_adj['multiplier']
    
    # Cleanup: Restore original column names (merge created _x and _y suffixes)
    df_event_log_adj = df_event_log_adj.rename(columns={'timestamp_x': 'timestamp'})
    
    # Select only original columns
    original_cols = df_events.columns
    df_event_log_adj = df_event_log_adj[original_cols]

    return {
        'df_initial_state_adj': df_initial_state_adj,
        'df_event_log_adj': df_event_log_adj,
        'financial_info': data_package['financial_info'],
        'report_start_date': data_package['report_start_date'],
        'report_end_date': data_package['report_end_date'],
        'base_currency': data_package['base_currency']
    }