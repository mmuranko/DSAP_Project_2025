import pandas as pd
import numpy as np

def apply_split_adjustments(data_package):
    """
    Adjusts 'initial_state' and 'events' for stock splits using vectorized 
    cumulative products. Returns dictionary with EXACT structure as input.
    """
    # Unpack and copy to avoid side-effects
    df_initial = data_package['initial_state'].copy()
    df_events = data_package['events'].copy()
    report_start = data_package['report_start_date']

    # --- 1. Prepare Event Data ---
    # Sort by symbol and time to ensure cumulative math is correct
    df_events = df_events.sort_values(by=['symbol', 'timestamp'])

    # Calculate Split Factors
    # Logic: The multiplier needed for any record is the Total Split Ratio 
    # divided by the Cumulative Ratio at that specific moment.
    # Example (4:1 split): 
    # Pre-split: Total(4) / Running(1) = 4x Multiplier.
    # Post-split: Total(4) / Running(4) = 1x Multiplier.
    
    grouped_ratios = df_events.groupby('symbol')['split_ratio']
    
    # Calculate intermediate columns (we will drop them later)
    df_events['running_factor'] = grouped_ratios.cumprod()
    df_events['total_factor'] = grouped_ratios.transform('prod')
    df_events['multiplier'] = df_events['total_factor'] / df_events['running_factor']

    # --- 2. Adjust Initial State ---
    if not df_initial.empty:
        # Find the 'running_factor' valid at the exact moment of report_start.
        # This is the last known cumprod from events occurring <= report_start.
        mask_past = df_events['timestamp'] <= report_start
        last_known_factor = df_events.loc[mask_past].groupby('symbol')['running_factor'].last()
        
        # Map factors to initial state
        # 1. Get Total Factor for the symbol
        initial_total = df_initial['symbol'].map(
            df_events.groupby('symbol')['total_factor'].first()
        ).fillna(1.0)
        
        # 2. Get Running Factor at start date (fill with 1.0 if no past events exist)
        initial_running = df_initial['symbol'].map(last_known_factor).fillna(1.0)
        
        # 3. Apply Adjustment
        df_initial['quantity'] = df_initial['quantity'] * (initial_total / initial_running)

    # --- 3. Adjust Event Log ---
    # Simply multiply the pre-calculated multiplier
    df_events['quantity_change'] = df_events['quantity_change'] * df_events['multiplier']

    # --- 4. Cleanup and Restore Order ---
    # Drop helper columns
    df_events = df_events.drop(columns=['running_factor', 'total_factor', 'multiplier'])
    
    # Re-sort by timestamp to return to chronological order
    df_events = df_events.sort_values(by='timestamp')

    # Assign to specific variable names for clarity in return
    df_initial_state_adj = df_initial
    df_event_log_adj = df_events

    return {
        'initial_state': df_initial_state_adj,
        'events': df_event_log_adj,
        'financial_info': data_package['financial_info'],
        'report_start_date': data_package['report_start_date'],
        'report_end_date': data_package['report_end_date'],
        'base_currency': data_package['base_currency']
    }