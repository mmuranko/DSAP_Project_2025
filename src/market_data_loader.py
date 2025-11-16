import pandas as pd
import yfinance as yf
from datetime import timedelta

# --- DEFAULT CONFIGURATION ---

# This map is used as the default for the fetch_market_data function.
# It covers common exchanges reported by IBKR and maps them to yfinance suffixes.
DEFAULT_EXCHANGE_SUFFIX_MAP = {
    # USA (No Suffix)
    'NASDAQ': '',
    'NYSE': '',
    'ARCA': '',
    
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

# --- PRIVATE HELPER ---

def _build_ticker_map(all_stock_symbols, df_financial_info, exchange_suffix_map):
    """
    Private helper to build a map from IBKR symbol to the correct yfinance ticker.
    
    Returns:
        tuple: (ticker_map, ticker_info_map)
            - ticker_map: {'IBKR_SYMBOL': 'YFINANCE_TICKER'}
            - ticker_info_map: {'IBKR_SYMBOL': {details...}} (for debugging)
    """
    ticker_map = {}
    ticker_info_map = {}
    
    if df_financial_info is None:
        print("Warning: 'Financial Instrument Information' section is missing.")
        print("All tickers will be fetched using their base symbol (e.g., 'VNA', 'IBKR').")
        print("This may fail for non-US stocks.")
        # Use symbols directly as a fallback
        for symbol in all_stock_symbols:
            ticker_map[symbol] = symbol
            ticker_info_map[symbol] = {
                'exchange': 'Unknown',
                'suffix': '',
                'yfinance_ticker': symbol,
                'status': 'No financial info, using symbol as-is.'
            }
        return ticker_map, ticker_info_map

    # Build symbol -> exchange map from the financial info DataFrame
    df_fin_info_unique = df_financial_info[['Symbol', 'Exchange']].drop_duplicates()
    symbol_exchange_map = df_fin_info_unique.set_index('Symbol')['Exchange'].to_dict()

    for symbol in all_stock_symbols:
        exchange = symbol_exchange_map.get(symbol)
        suffix = ''
        status = 'OK'

        if not exchange:
            status = f"No exchange info found for {symbol}. Using symbol as-is."
            suffix = '' # Default to no suffix
        else:
            suffix = exchange_suffix_map.get(exchange)
            if suffix is None:
                status = f"No suffix mapping found for exchange '{exchange}' (symbol: {symbol}). Using symbol as-is."
                suffix = '' # Default to no suffix
        
        yfinance_ticker = f"{symbol}{suffix}"
        ticker_map[symbol] = yfinance_ticker
        ticker_info_map[symbol] = {
            'exchange': exchange,
            'suffix': suffix,
            'yfinance_ticker': yfinance_ticker,
            'status': status
        }
    
    return ticker_map, ticker_info_map

# --- PUBLIC FUNCTION ---

def fetch_market_data(report_data, exchange_suffix_map=DEFAULT_EXCHANGE_SUFFIX_MAP):
    """
    Fetches historical market data from yfinance for all stocks in the report.

    It uses the 'financial_info' DataFrame to map IBKR symbols to
    yfinance tickers (e.g., VNA on EBS -> VNA.SW).

    Args:
        report_data (dict): The dictionary from data_loader.load_ibkr_report().
        exchange_suffix_map (dict, optional): 
            A map of {exchange_name: yfinance_suffix}.
            If not provided, defaults to DEFAULT_EXCHANGE_SUFFIX_MAP.

    Returns:
        dict: A bundle containing market data and debug info.
              {
                  'data': {
                      'IBKR_SYMBOL_1': pd.DataFrame,
                      'IBKR_SYMBOL_2': pd.DataFrame
                  },
                  'ticker_info': {
                      'IBKR_SYMBOL_1': {'exchange': 'EBS', 'suffix': '.SW', ...}
                  }
              }
    """
    
    # --- 1. Extract data and get date range ---
    df_initial_state = report_data['initial_state']
    df_events = report_data['events']
    df_financial_info = report_data.get('financial_info') # Use .get() for safety

    if df_events.empty:
        print("Error: Event log is empty. Cannot determine date range.")
        return {'data': {}, 'ticker_info': {}}

    start_date = df_events['timestamp'].min().date()
    end_date = df_events['timestamp'].max().date()
    
    # Add a small buffer to ensure we get data for the start/end dates
    start_str = (start_date - timedelta(days=1)).strftime('%Y-%m-%d')
    end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')

    # --- 2. Identify all unique stock symbols ---
    # Get symbols from both initial state and events to be comprehensive
    stock_symbols_initial = set(df_initial_state[
        df_initial_state['asset_category'] == 'Stock'
    ]['symbol'].unique())
    
    # Find all symbols ever involved in a stock-related transaction
    stock_symbols_events = set(df_events[
        (df_events['symbol'].notna()) & 
        (df_events['event_type'].str.contains('TRADE|GIFT_VEST|CORP_ACTION|SPLIT'))
    ]['symbol'].unique())
    
    all_stock_symbols = stock_symbols_initial | stock_symbols_events
    
    if not all_stock_symbols:
        print("No stock symbols found in the report.")
        return {'data': {}, 'ticker_info': {}}

    # --- 3. Build the ticker map ---
    ticker_map, ticker_info_map = _build_ticker_map(
        all_stock_symbols, 
        df_financial_info, 
        exchange_suffix_map # Use the provided (or default) map
    )

    # --- 4. Fetch data from yfinance ---
    unique_yfinance_tickers = sorted(list(set(ticker_map.values())))
    
    if not unique_yfinance_tickers:
        print("No yfinance tickers to fetch.")
        return {'data': {}, 'ticker_info': ticker_info_map}

    print(f"Fetching {len(unique_yfinance_tickers)} tickers from {start_str} to {end_str}...")
    print(f"Tickers: {', '.join(unique_yfinance_tickers)}")
    
    # auto_adjust=True applies splits/dividends to the 'Close' price.
    # This is what we want, as portfolio_processor.py adjusts the *quantities*
    # to be compatible with *adjusted prices*.
    yf_data = yf.download(
        unique_yfinance_tickers, 
        start=start_str, 
        end=end_str, 
        auto_adjust=True, # Use adjusted close prices
        group_by='ticker',
        progress=False # Cleaner log for a module
    )
    
    # --- 5. Process yfinance results ---
    
    # Handle case for single ticker download (returns DataFrame, not dict-like)
    if len(unique_yfinance_tickers) == 1:
        ticker = unique_yfinance_tickers[0]
        if yf_data.empty:
            yf_data_map = {ticker: pd.DataFrame()} # Empty
        else:
            yf_data_map = {ticker: yf_data}
    else:
        # For multiple tickers, yf.download with group_by='ticker'
        # returns a dict-like object (pandas Panel)
        yf_data_map = {ticker: yf_data[ticker] for ticker in unique_yfinance_tickers}

    # --- 6. Map data back to original IBKR symbols ---
    market_data_by_symbol = {}
    for original_symbol, yfinance_ticker in ticker_map.items():
        data = yf_data_map.get(yfinance_ticker)
        
        if data is None or data.empty:
            print(f"Warning: No data returned for {original_symbol} (ticker: {yfinance_ticker})")
            ticker_info_map[original_symbol]['status'] = 'Error: yfinance returned no data.'
            # Provide an empty, but structured, DataFrame
            market_data_by_symbol[original_symbol] = pd.DataFrame(
                columns=['Open', 'High', 'Low', 'Close', 'Volume']
            ).rename_axis('Date')
        else:
            # Keep only the standard OHLCV columns
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            market_data_by_symbol[original_symbol] = data
    
    print("Market data fetch complete.")
    
    return {
        'data': market_data_by_symbol,
        'ticker_info': ticker_info_map
    }

# --- This main block is for testing ---
if __name__ == "__main__":
    # It assumes:
    # 1. You are running this file from the project's root directory (the folder *above* 'src').
    # 2. 'data_loader.py' is in 'src/' (e.g., 'src/data_loader.py').
    # 3. Your report CSV is in 'data/' (e.g., 'data/U13271915_20250101_20251029.csv').
    
    try:
        # Try to import the data_loader from the 'src' folder
        from src import data_loader as dl
    except ImportError:
        print("Error: Could not import 'src.data_loader'.")
        print("Please make sure you are running this script from the root project directory (the folder *above* 'src').")
        dl = None # Set to None to prevent running
    
    if dl:
        # --- 1. Define your report path ---
        file_path = r'data/U13271915_20250101_20251029.csv'
        
        # --- 2. Load Report ---
        print(f"Loading report data from: {file_path}")
        report_data = dl.load_ibkr_report(file_path)
        
        if report_data:
            print("Report loaded. Fetching market data using the default suffix map...")
            
            # --- 3. Fetch Data ---
            # No need to pass the map, it will use the default.
            market_data_bundle = fetch_market_data(report_data)
            
            print("\n" + "="*80)
            print("--- Ticker Info Map (Debugging) ---")
            # Use json for pretty-printing the debug map
            import json
            print(json.dumps(market_data_bundle['ticker_info'], indent=2))
            print("="*80)

            print("\n--- Fetched Data Summary ---")
            for symbol, df in market_data_bundle['data'].items():
                if not df.empty:
                    print(f"  Symbol: {symbol} (Ticker: {market_data_bundle['ticker_info'][symbol]['yfinance_ticker']})")
                    print(f"  Date Range: {df.index.min().date()} to {df.index.max().date()}")
                    print(f"  Rows: {len(df)}")
                    print(f"  First 2 rows:\n{df.head(2)}\n")
                else:
                    print(f"  Symbol: {symbol} (Ticker: {market_data_bundle['ticker_info'][symbol]['yfinance_ticker']})")
                    print(f"  Status: No data found.\n")