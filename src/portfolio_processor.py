import pandas as pd
import numpy as np

def apply_split_adjustments(data_package):
    """
    Adjusts historical quantities in initial_state and event_log based on 
    subsequent stock splits to normalize everything to 'end-of-period' share equivalents.
    """
    # Unpack
    df_initial = data_package['initial_state'].copy()
    df_events = data_package['events'].copy()
    
    # --- 1. Build the Multiplier Timeline ---
    
    # Filter only split events
    # We need 'symbol', 'timestamp', and 'split_ratio'
    mask_splits = df_events['event_type'] == 'SPLIT'
    df_splits = df_events.loc[mask_splits, ['timestamp', 'symbol', 'split_ratio']].copy()

    if df_splits.empty:
        # Optimization: If no splits exist in the entire log, return originals immediately
        return {
            'df_initial_state_adj': df_initial,
            'df_event_log_adj': df_events,
            'financial_info': data_package['financial_info'],
            'report_start_date': data_package['report_start_date'],
            'report_end_date': data_package['report_end_date'],
            'base_currency': data_package['base_currency']
        }

    # Sort splits to ensure cumulative math is correct
    df_splits = df_splits.sort_values(by=['symbol', 'timestamp'])

    # Calculate the "Multiplier After Split"
    # Example: Splits 10:1 then 2:1. Total = 20.
    # Split 1 (10:1): Cumulative=10. Total=20. Factor AFTER this split = 20/10 = 2.
    # Split 2 (2:1):  Cumulative=20. Total=20. Factor AFTER this split = 20/20 = 1.
    
    # 1. Total accumulated split factor per symbol (The "End State")
    total_factors = df_splits.groupby('symbol')['split_ratio'].transform('prod')
    
    # 2. Cumulative split factor up to specific point in time
    running_factors = df_splits.groupby('symbol')['split_ratio'].cumprod()
    
    # 3. The multiplier valid AFTER this split occurred
    df_splits['multiplier'] = total_factors / running_factors
    
    # --- 2. Create a Lookup Table (Timeline) ---
    
    # We need to handle the time BEFORE the first split.
    # The multiplier valid from the "beginning of time" until the first split 
    # is actually the Total Factor.
    
    # Extract unique symbols that have splits
    split_symbols = df_splits['symbol'].unique()
    
    # Create a "Start of Time" row for each split-affected symbol
    # We use a timestamp strictly before any possible report date
    start_rows = []
    for sym in split_symbols:
        # Get total factor for this symbol
        tf = df_splits[df_splits['symbol'] == sym]['split_ratio'].prod()
        start_rows.append({
            'timestamp': pd.Timestamp.min,
            'symbol': sym,
            'multiplier': tf
        })
        
    df_start = pd.DataFrame(start_rows)
    
    # Combine: Start rows + Split rows
    # This creates a lookup: "From this timestamp onwards, use this multiplier"
    df_timeline = pd.concat([df_start, df_splits[['timestamp', 'symbol', 'multiplier']]])
    df_timeline = df_timeline.sort_values(by='timestamp')

    # --- 3. Adjust the Event Log ---
    
    # Sort events for merge_asof
    df_events = df_events.sort_values(by='timestamp')
    
    # Perform an As-Of Merge
    # This finds the latest multiplier in df_timeline that is <= event timestamp
    df_merged_events = pd.merge_asof(
        df_events,
        df_timeline,
        on='timestamp',
        by='symbol',
        direction='backward' # Look for the closest split/start point in the past
    )
    
    # Fill NaN multipliers (symbols with no splits) with 1.0
    df_merged_events['multiplier'] = df_merged_events['multiplier'].fillna(1.0)
    
    # Apply adjustment
    df_merged_events['quantity_change'] = df_merged_events['quantity_change'] * df_merged_events['multiplier']
    
    # Clean up temporary columns
    df_event_log_adj = df_merged_events.drop(columns=['multiplier'])
    
    # --- 4. Adjust the Initial State ---
    
    # Initial state is effectively at 'report_start_date'
    # We must determine if any splits happened BEFORE the report start (unlikely if MtM is fresh)
    # or if we just need to prep it for splits happening LATER in the report.
    
    df_initial_adj = df_initial.copy()
    
    if not df_initial_adj.empty:
        # Assign a temporary timestamp to allow merging
        # We use report_start_date. logic: The "Prior Quantity" is valid AT that moment.
        df_initial_adj['temp_timestamp'] = pd.to_datetime(data_package['report_start_date'])
        df_initial_adj = df_initial_adj.sort_values('temp_timestamp')
        
        # Merge timeline
        df_initial_merged = pd.merge_asof(
            df_initial_adj,
            df_timeline,
            left_on='temp_timestamp',
            right_on='timestamp',
            by='symbol',
            direction='backward'
        )
        
        # Fill NaNs and Multiply
        df_initial_merged['multiplier'] = df_initial_merged['multiplier'].fillna(1.0)
        df_initial_merged['quantity'] = df_initial_merged['quantity'] * df_initial_merged['multiplier']
        
        # Cleanup
        # Keep original columns + updated quantity
        cols_to_keep = df_initial.columns
        df_initial_state_adj = df_initial_merged[cols_to_keep]
    else:
        df_initial_state_adj = df_initial

    # --- 5. Return ---
    return {
        'df_initial_state_adj': df_initial_state_adj,
        'df_event_log_adj': df_event_log_adj,
        'financial_info': data_package['financial_info'],
        'report_start_date': data_package['report_start_date'],
        'report_end_date': data_package['report_end_date'],
        'base_currency': data_package['base_currency']
    }