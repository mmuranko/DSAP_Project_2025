import time
from typing import Dict, Any

def apply_split_adjustments(data_package: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adjusts the initial state and event log for stock splits and retroactive events.

    This function normalizes the portfolio history by applying a 'multiplier' to
    all share quantities. This ensures that a share held in the past is expressed
    in terms of today's share count (e.g., after a 4:1 split, 1 old share becomes
    4 current shares).

    Logic Summary:
    Instead of iterating row-by-row, it uses vectorized cumulative products to
    calculate a running split factor for every event.
    The Multiplier for any given event = (Total Cumulative Split Ratio) / (Running Cumulative Ratio).

    Key Operations:
    1. Retroactive Event Handling: Moves events like tax adjustments that occurred
       before the report start date to the start date itself.
    2. Vectorized Adjustment: Calculates multipliers for the entire dataset in
       one pass and updates trade quantities.
    3. Initial State Adjustment: Updates the starting portfolio positions to match
       the post-split reality.

    Args:
        data_package (Dict[str, Any]): Dictionary containing 'initial_state',
            'events', and report metadata.

    Returns:
        Dict[str, Any]: A new data package with adjusted 'initial_state' and 'events'.
    """
    print("   [>] Applying split adjustments and event processing...")
    time.sleep(0.5)

    # Unpack and copy to avoid side-effects (modifying the original dict passed by reference)
    df_initial = data_package['initial_state'].copy()
    df_events = data_package['events'].copy()
    report_start = data_package['report_start_date']

    # --- 0. Handle Retroactive Events ---

    # Goal: Identify retroactive events that take effect in the report period 
    # (e.g. tax adjustments) and move them to the report start date. To avoid 
    # the symbols being included in subsequent portfolio manipulations, their 
    # symbol is set to None.
    mask_prior = df_events['timestamp'] < report_start
    
    if mask_prior.any():
        # Step 1: Fix the Timeline
        print(f"       - Moved {mask_prior.sum()} historical events to start date {report_start.date()}.")
        df_events.loc[mask_prior, 'timestamp'] = report_start

        # Step 2: Define the Universe of "Real" Stocks
        # A stock is valid if it is in the Initial State.
        # The Cash Engine will still see the 'cash_change', but the 
        # Securities Engine will ignore the row because the symbol is missing.
        valid_universe = set(df_initial['symbol'].unique())
        
        ghost_mask = mask_prior & ~df_events['symbol'].isin(valid_universe)
        ghost_count = ghost_mask.sum()
        
        if ghost_count > 0:
            print(f"       - Detached symbol from {ghost_count} historical tax events (Ghost Assets).")
            df_events.loc[ghost_mask, 'symbol'] = None

    # --- 1. Prepare Event Data ---
    
    # EXPLANATION: Sorting is Critical
    # Cumulative products rely entirely on row order. We must ensure the timeline is strictly chronological
    # so the 'running factor' accumulates correctly from past to future.
    df_events = df_events.sort_values(by=['symbol', 'timestamp'])

    # EXPLANATION: GroupBy Object
    # We create a groupby object once to avoid repeatedly specifying ('symbol') in subsequent operations.
    # This prepares the engine to treat each stock's timeline independently.
    grouped_ratios = df_events.groupby('symbol')['split_ratio']
    
    # EXPLANATION: Vectorised Cumulative Product (.cumprod)
    # Calculates the running product of split ratios *down* the column for each symbol.
    # Row 1 (Ratio 1.0) -> Running 1.0
    # Row 2 (Ratio 1.0) -> Running 1.0
    # Row 3 (Split 4.0) -> Running 4.0
    # Row 4 (Ratio 1.0) -> Running 4.0
    df_events['running_factor'] = grouped_ratios.cumprod()
    
    # EXPLANATION: Vectorised Broadcast (.transform)
    # .transform('prod') calculates the TOTAL product for the group (e.g., 4.0) 
    # and broadcasts that single value to EVERY row in that group.
    # This allows us to divide the "Final State" by the "Current State" in one vector operation.
    df_events['total_factor'] = grouped_ratios.transform('prod')
    
    # Calculate the retroactive multiplier
    df_events['multiplier'] = df_events['total_factor'] / df_events['running_factor']

    # --- 2. Adjust Initial State ---

    if not df_initial.empty:
        # EXPLANATION: Time-Travel Masking
        # We need to know what the split factor was specifically at the moment the report started.
        # We filter for events that happened in the past (<= report_start).
        mask_past = df_events['timestamp'] <= report_start
        
        # We take the .last() running factor from the past events to define the "state at start date".
        last_known_factor = df_events.loc[mask_past].groupby('symbol')['running_factor'].last()
        
        # EXPLANATION: Map & Fill
        # We map these calculated factors back to the initial_state DataFrame based on 'symbol'.
        # .fillna(1.0) handles stocks that had no events/splits (default multiplier is 1).
        
        # 1. Get the Total Factor (The final target universe ratio)
        initial_total = df_initial['symbol'].map(
            df_events.groupby('symbol')['total_factor'].first()
        ).fillna(1.0)
        
        # 2. Get the Running Factor (The ratio at the start of the report)
        initial_running = df_initial['symbol'].map(last_known_factor).fillna(1.0)
        
        # 3. Apply Adjustment to Initial Quantity
        df_initial['quantity'] = df_initial['quantity'] * (initial_total / initial_running)

    # --- 3. Adjust Event Log ---

    # EXPLANATION: Vectorised Multiplication
    # We simply multiply the entire 'quantity_change' column by our calculated 'multiplier' column.
    # This adjusts pre-split trades (e.g., Buy 10 becomes Buy 40) while leaving post-split trades alone.
    df_events['quantity_change'] = df_events['quantity_change'] * df_events['multiplier']

    # --- 4. Cleanup and Restore Order ---

    # Remove the temporary math columns to keep the DataFrame clean
    df_events = df_events.drop(columns=['running_factor', 'total_factor', 'multiplier'])
    
    # Re-sort by timestamp strictly to ensure the returned log is chronological (mixed symbols allowed)
    df_events = df_events.sort_values(by='timestamp')

    # Assign to specific variable names for clarity in return
    df_initial_state_adj = df_initial
    df_event_log_adj = df_events

    # --- 5. Console Summary ---
    
    # Report actual splits found (High value information)
    # We check the 'split_ratio' column to see if any non-1.0 entries exist
    split_events = df_events[df_events['split_ratio'] != 1.0]
    if not split_events.empty:
        split_symbols = split_events['symbol'].unique()
        print(f"       - Detected corporate actions/splits for: {', '.join(split_symbols)}")
    
    # Final Success Message
    print(f"   [+] Processing complete. Final dataset: {len(df_events)} events.")
    print()
    time.sleep(0.5)

    # Return dict ensuring the structure matches the input 'data_package' exactly
    return {
        'initial_state': df_initial_state_adj,
        'events': df_event_log_adj,
        'financial_info': data_package['financial_info'],
        'report_start_date': data_package['report_start_date'],
        'report_end_date': data_package['report_end_date'],
        'base_currency': data_package['base_currency']
    }