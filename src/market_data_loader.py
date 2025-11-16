import yfinance as yf
import pandas as pd

# --- CONFIGURATION ---
# This map is based on YOUR logic.
# It converts IBKR exchange codes to yfinance ticker suffixes.
EXCHANGE_SUFFIX_MAP = {
    # USA (No Suffix)
    'NASDAQ': '',
    'NYSE': '',
    'ARCA': '',
    
    # Switzerland (.SW)
    'EBS': '.SW',
    
    # Germany (.F for Frankfurt)
    'IBIS': '.F', # Using your rule: IBIS -> .F
    
    # France (.PA for Paris)
    'SBF': '.PA',
    
    # UK (.L for London)
    'LSE': '.L',

    # Italy (.MI for Milan)
    'BVME': '.MI',
    
    # Spain (.MC for Madrid)
    'BM': '.MC',
}
# ---------------------


def _build_ticker_map(initial_state, financial_info):
    """
    Automatically builds the IBKR-to-yfinance ticker map
    using the 'Financial Instrument Information' section.
    """
    asset_tickers_map = {} # e.g., {'ANFO': 'ANFO.SW'}
    
    if financial_info is None or financial_info.empty:
        print("Error: 'Financial Instrument Information' was not found. Cannot map tickers.")
        return asset_tickers_map

    # Set 'Symbol' as the index for fast lookups
    try:
        info = financial_info.set_index('Symbol')
    except KeyError:
        print("Error: 'Financial Instrument Information' is missing 'Symbol' column.")
        return asset_tickers_map

    # Get all unique stock symbols from the initial state
    stock_symbols = initial_state[
        initial_state['asset_category'] == 'Stock'
    ]['symbol'].unique()
    
    for symbol in stock_symbols:
        try:
            # Look up the symbol (e.g., 'ANFO') in the info table
            # Handle cases where a symbol might be listed multiple times (e.g., VNA/VNA.DRTS)
            row = info.loc[symbol]
            exchange = row['Listing Exch'].iloc[0] if isinstance(row, pd.DataFrame) else row['Listing Exch']
            
            # Look up the exchange (e.g., 'EBS') in our suffix map
            if exchange in EXCHANGE_SUFFIX_MAP:
                suffix = EXCHANGE_SUFFIX_MAP[exchange]
                # Create the yfinance ticker (e.g., 'ANFO' + '.SW')
                asset_tickers_map[symbol] = f"{symbol}{suffix}"
            else:
                print(f"Warning: No suffix mapping for exchange '{exchange}' (Symbol: {symbol}). Will try simple ticker.")
                asset_tickers_map[symbol] = symbol # Default to no suffix

        except KeyError:
            print(f"Warning: Symbol '{symbol}' not found in 'Financial Instrument Information'. Skipping.")
            
    return asset_tickers_map


def fetch_market_data(report_data, base_currency='CHF'):
    """
    Fetches all required price and FX data from yfinance.
    
    Args:
        report_data (dict): The complete dictionary from load_ibkr_report(),
                            containing 'initial_state', 'events', and 'financial_info'.
        base_currency (str): Your home currency (e.g., 'CHF').
        
    Returns:
        dict: A dictionary containing 'prices' (DataFrame) and 'fx_rates' (DataFrame),
              or None on failure.
    """
    
    # Unpack the data
    initial_state = report_data['initial_state']
    events = report_data['events']
    financial_info = report_data['financial_info']
    
    # --- 1. Get Full Date Range ---
    start_date = events['timestamp'].min().date()
    end_date = events['timestamp'].max().date() + pd.Timedelta(days=2) # 2-day buffer
    
    # --- 2. Build Asset Ticker Map Automatically ---
    asset_tickers_map = _build_ticker_map(initial_state, financial_info)
    
    if not asset_tickers_map:
        print("Error: Could not build any asset tickers. Aborting market data fetch.")
        return None
        
    yf_stock_tickers = list(asset_tickers_map.values())
    
    # --- 3. Get FX Tickers to Fetch ---
    all_currencies = initial_state['currency'].unique()
    foreign_currencies = [c for c in all_currencies if c != base_currency and pd.notna(c)]
    
    # Format for yfinance, e.g., "USDCHF=X", "EURCHF=X"
    fx_tickers = [f"{c}{base_currency}=X" for c in foreign_currencies]
    
    print(f"Fetching {len(yf_stock_tickers)} asset prices and {len(fx_tickers)} FX rates...")
    
    try:
        # --- 4. Fetch Asset Prices (Adjusted Close) ---
        # This is what you wanted: split-adjusted prices in their original currency
        price_data = yf.download(yf_stock_tickers, start=start_date, end=end_date)
        if price_data.empty:
            raise ValueError(f"yfinance returned no price data for tickers: {yf_stock_tickers}")
            
        # Select 'Adj Close' and handle single/multi-ticker results
        if len(yf_stock_tickers) == 1:
            df_prices = price_data[['Adj Close']].rename(columns={'Adj Close': yf_stock_tickers[0]})
        else:
            df_prices = price_data['Adj Close']
            
        # Rename columns back to our internal symbols (e.g., 'ANFO.SW' -> 'ANFO')
        reverse_map = {v: k for k, v in asset_tickers_map.items()}
        df_prices = df_prices.rename(columns=reverse_map)
        
        # --- 5. Fetch FX Rates (Close) ---
        df_fx_rates = pd.DataFrame() # Create empty DF
        if fx_tickers: # Only fetch if there are foreign currencies
            fx_data = yf.download(fx_tickers, start=start_date, end=end_date)
            if fx_data.empty:
                print(f"Warning: yfinance returned no FX data for tickers: {fx_tickers}. Will proceed without FX.")
            else:
                # Select 'Close' and handle single/multi-ticker results
                if len(fx_tickers) == 1:
                    df_fx_rates = fx_data[['Close']].rename(columns={'Close': fx_tickers[0]})
                else:
                    df_fx_rates = fx_data['Close']
        
        # --- 6. Clean and Fill Missing Data (Holidays) ---
        # Forward-fill (ffill) carries the last known price over holidays/weekends
        df_prices = df_prices.ffill().bfill() # bfill for any initial NaNs
        df_fx_rates = df_fx_rates.ffill().bfill()
        
        # Add the base currency (CHF) to FX rates with a rate of 1.0
        df_fx_rates[base_currency] = 1.0
        
        print("Market data fetched successfully.")
        
        return {
            'prices': df_prices,
            'fx_rates': df_fx_rates
        }
        
    except Exception as e:
        print(f"Error during yfinance download: {e}")
        return None

# --- Example of how to use this module ---
if __name__ == "__main__":
    
    try:
        # This test assumes the file is in 'src' and you run it from the root folder
        from src import data_loader as dl
        from src import portfolio_processor as pp
    except ImportError:
        print("Could not import modules. Make sure this file is in the 'src' folder.")
        print("And that you are running this test from the parent directory (your project root).")
        exit()

    # Define paths (relative to the project root, not 'src')
    filepath = r'data/U13271915_20250101_20251029.csv'
    
    print(f"--- Step 1: Loading Report ---")
    # We now get 'financial_info' for free
    original_report = dl.load_ibkr_report(filepath)
    
    if original_report:
        print(f"\n--- Step 2: Adjusting for Splits ---")
        adjusted_report = pp.adjust_for_splits(original_report)
        
        if adjusted_report:
            print(f"\n--- Step 3: Fetching Market Data (Automapped) ---")
            
            # Pass the *original* report, which has the financial_info
            market_data = fetch_market_data(
                original_report, 
                base_currency='CHF'
            )
            
            if market_data:
                print("\n" + "="*80)
                print("--- Asset Prices (Head) ---")
                print(market_data['prices'].head())
                
                print("\n" + "="*80)
                print("--- FX Rates (Head) ---")
                print(market_data['fx_rates'].head())