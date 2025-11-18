import pandas as pd
# import numpy as np
import yfinance as yf

def fetch_market_data(data_package):
    """
    Generates tickers for Stocks and Cash based on specific mapping rules,
    downloads historical data via yfinance, and maps it back to the original symbol.
    """
    # Unpack necessary data
    df_initial = data_package['initial_state']
    df_info = data_package['financial_info']
    report_start = data_package['report_start_date']
    report_end = data_package['report_end_date']
    base_currency = data_package['base_currency']

    # 1. Define Date Range
    # Report start minus 5 years, respecting the instruction not to touch the dates otherwise
    start_dt = pd.to_datetime(report_start)
    download_start = start_dt - pd.DateOffset(years=5)
    download_end = report_end # Passed directly as requested

    # 2. Define Exchange Suffix Map
    exchange_map = {
        # USA (No Suffix)
        'NASDAQ': '', 'NYSE': '', 'ARCA': '',
        # Switzerland
        'EBS': '.SW',
        # Germany
        'IBIS': '.F',
        # France
        'SBF': '.PA',
        # UK
        'LSE': '.L',
        # Italy
        'BVME': '.MI',
        # Spain
        'BM': '.MC',
    }

    # Dictionary to store Symbol -> Market Data DataFrame
    market_data_results = {}
    
    # Dictionary to map Symbol -> YFinance Ticker (for internal use)
    ticker_map = {}

    # --- 3. Build Tickers for Stocks ---
    # Filter for Stocks in initial state
    stock_symbols = df_initial[df_initial['asset_category'] == 'Stock']['symbol'].unique()

    for symbol in stock_symbols:
        # Find the exchange for this symbol in financial_info
        # Using a safe lookup that handles potential missing info gracefully
        exchange_row = df_info.loc[df_info['symbol'] == symbol, 'Exchange']
        
        if not exchange_row.empty:
            exchange_code = exchange_row.iloc[0]
            # Get suffix from map, default to empty string if exchange not in map (or handle as error)
            suffix = exchange_map.get(exchange_code, '')
            ticker_map[symbol] = f"{symbol}{suffix}"
        else:
            # Fallback if symbol not found in financial_info (assumes no suffix)
            ticker_map[symbol] = symbol

    # --- 4. Build Tickers for Cash ---
    # Filter for Cash in initial state
    cash_symbols = df_initial[df_initial['asset_category'] == 'Cash']['symbol'].unique()

    for symbol in cash_symbols:
        # If the cash symbol is the same as base currency, we don't need a rate (it is 1.0)
        # However, the prompt implies getting data for "entries... which are Cash".
        # Usually, we skip downloading USD=X if base is USD, but if required, we construct it.
        # We will skip only if symbol == base_currency to avoid download errors or redundant data.
        if symbol == base_currency:
            continue
            
        # Construct ticker: Base + Symbol + '=X' (e.g., 'USDCHF=X')
        ticker_map[symbol] = f"{base_currency}{symbol}=X"

    # --- 5. Download Data ---
    # We iterate through the map to download and assign back to the original symbol key
    for symbol, ticker in ticker_map.items():
        try:
            # Download with specified constraints
            data = yf.download(
                ticker, 
                start=download_start, 
                end=download_end, 
                timeout=30, 
                auto_adjust=False,
                progress=False # Disables the progress bar to keep logs clean
            )
            
            # Only store if we actually got data
            if not data.empty:
                market_data_results[symbol] = data
                
        except Exception as e:
            print(f"Error downloading data for {symbol} (Ticker: {ticker}): {e}")

    return market_data_results

def apply_split_adjustments(data_package):
    """
    Adjusts 'initial_state' and 'events' for stock splits using vectorized 
    cumulative products. Returns dictionary with EXACT structure as input,
    plus the added market data.
    """
    # Unpack and copy to avoid side-effects
    df_initial = data_package['initial_state'].copy()
    df_events = data_package['events'].copy()
    report_start = data_package['report_start_date']

    # --- 1. Prepare Event Data ---
    # Sort by symbol and time to ensure cumulative math is correct
    df_events = df_events.sort_values(by=['symbol', 'timestamp'])

    # Calculate Split Factors
    grouped_ratios = df_events.groupby('symbol')['split_ratio']
    
    # Calculate intermediate columns (we will drop them later)
    df_events['running_factor'] = grouped_ratios.cumprod()
    df_events['total_factor'] = grouped_ratios.transform('prod')
    df_events['multiplier'] = df_events['total_factor'] / df_events['running_factor']

    # --- 2. Adjust Initial State ---
    if not df_initial.empty:
        # Find the 'running_factor' valid at the exact moment of report_start.
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

    # Update the package with adjusted data so the fetch function uses the correct state if needed
    # (Though fetch currently only needs symbols, passing the updated state is cleaner)
    updated_package = {
        'initial_state': df_initial_state_adj,
        'events': df_event_log_adj,
        'financial_info': data_package['financial_info'],
        'report_start_date': data_package['report_start_date'],
        'report_end_date': data_package['report_end_date'],
        'base_currency': data_package['base_currency']
    }

    # --- 5. Fetch Market Data ---
    market_data = fetch_market_data(updated_package)

    return {
        'initial_state': df_initial_state_adj,
        'events': df_event_log_adj,
        'financial_info': data_package['financial_info'],
        'report_start_date': data_package['report_start_date'],
        'report_end_date': data_package['report_end_date'],
        'base_currency': data_package['base_currency'],
        'market_data': market_data
    }