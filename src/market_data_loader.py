import pandas as pd
import yfinance as yf
from datetime import timedelta
from dateutil.relativedelta import relativedelta

# This map is still required to translate exchange codes to yfinance suffixes
DEFAULT_EXCHANGE_SUFFIX_MAP = {
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

# --- PRIVATE HELPER 1 (Stocks) ---

def _build_stock_ticker_map(all_stock_symbols, df_financial_info, exchange_suffix_map):
    """
    Builds the ticker map for STOCKS only.
    """
    ticker_map = {}
    ticker_info_map = {}

    if df_financial_info is None:
        print("Warning: 'Financial Instrument Information' section is missing for stocks.")
        for symbol in all_stock_symbols:
            ticker_map[symbol] = symbol
            ticker_info_map[symbol] = {'yfinance_ticker': symbol, 'status': 'Missing financial info'}
        return ticker_map, ticker_info_map

    df_financial_info.columns = [str(col).strip() for col in df_financial_info.columns]
    symbol_col, ticker_col, exchange_col = "Symbol", "Underlying", "Listing Exch"
    required_cols = [symbol_col, ticker_col, exchange_col]

    if not all(col in df_financial_info.columns for col in required_cols):
        print(f"CRITICAL ERROR: 'Financial Instrument Information' is missing required columns for stocks.")
        for symbol in all_stock_symbols:
            ticker_map[symbol] = symbol
            ticker_info_map[symbol] = {'yfinance_ticker': symbol, 'status': 'Missing required columns'}
        return ticker_map, ticker_info_map

    df_valid_instruments = df_financial_info[
        ~df_financial_info[ticker_col].str.contains(r'\.', na=False)
    ].copy()

    df_lookup = df_valid_instruments.drop_duplicates(subset=[symbol_col])
    symbol_to_ticker_map = df_lookup.set_index(symbol_col)[ticker_col].to_dict()
    symbol_to_exchange_map = df_lookup.set_index(symbol_col)[exchange_col].to_dict()

    for symbol in all_stock_symbols:
        cleaned_ticker = symbol_to_ticker_map.get(symbol)
        exchange = symbol_to_exchange_map.get(symbol)
        status = "OK"

        if not cleaned_ticker:
            status = f"Skipped (Stock): Symbol '{symbol}' not found in 'Underlying' list."
            ticker_info_map[symbol] = {'yfinance_ticker': None, 'status': status}
            continue 

        if not exchange:
            status = f"Warning (Stock): No exchange for '{symbol}'. Using ticker '{cleaned_ticker}' as-is."
            suffix = ''
        else:
            suffix = exchange_suffix_map.get(exchange)
            if suffix is None:
                status = f"Warning (Stock): No suffix for exchange '{exchange}'. Using '{cleaned_ticker}' as-is."
                suffix = ''

        yfinance_ticker = f"{cleaned_ticker}{suffix}"
        ticker_map[symbol] = yfinance_ticker
        ticker_info_map[symbol] = {
            'asset_type': 'Stock',
            'original_symbol': symbol,
            'yfinance_ticker': yfinance_ticker,
            'status': status
        }
    
    return ticker_map, ticker_info_map

# --- PRIVATE HELPER 2 (FX) ---

def _build_fx_ticker_map(report_data):
    """
    Builds the ticker map for FX rates only.
    """
    base_currency = report_data.get('base_currency')
    df_initial_state = report_data['initial_state']
    
    all_cash_symbols = set(df_initial_state[
        df_initial_state['asset_category'] == 'Cash'
    ]['symbol'].unique())
    
    fx_ticker_map = {} # Maps yfinance ticker to original symbol (e.g., 'USDCHF=X': 'USD')
    fx_info_map = {}   # Maps original symbol to info (e.g., 'USD': {...})

    if not base_currency:
        print("CRITICAL ERROR: No base_currency found in report_data. Cannot fetch FX rates.")
        return fx_ticker_map, fx_info_map, None

    for symbol in all_cash_symbols:
        if symbol == base_currency:
            # This is the base currency, no ticker needed.
            fx_info_map[symbol] = {
                'asset_type': 'FX',
                'original_symbol': symbol,
                'yfinance_ticker': 'N/A (Base Currency)',
                'status': 'OK'
            }
        else:
            # This is a foreign currency, build the ticker
            yfinance_ticker = f"{symbol}{base_currency}=X"
            fx_ticker_map[yfinance_ticker] = symbol
            fx_info_map[symbol] = {
                'asset_type': 'FX',
                'original_symbol': symbol,
                'yfinance_ticker': yfinance_ticker,
                'status': 'OK'
            }
            
    return fx_ticker_map, fx_info_map, base_currency

# --- MAIN PUBLIC FUNCTION ---

def fetch_market_data(report_data, exchange_suffix_map=DEFAULT_EXCHANGE_SUFFIX_MAP):
    """
    Fetches historical market data (Stocks and FX) from yfinance.
    Fetches data from 5 years *before* the report start date up to the report end date.
    """
    
    # 1. Extract data and get date range
    df_initial_state = report_data['initial_state']
    df_events = report_data['events']
    df_financial_info = report_data.get('financial_info')

    report_start_date = report_data.get('report_start_date')
    report_end_date = report_data.get('report_end_date')
    
    if not report_start_date or not report_end_date:
         print("Error: Report start/end date not found. Falling back to event log.")
         if df_events.empty:
             print("Error: Event log is empty. Cannot determine date range.")
             return {'data': {}, 'ticker_info': {}}
         report_start_date = df_events['timestamp'].min().date()
         report_end_date = df_events['timestamp'].max().date()

    fetch_start_date = report_start_date - relativedelta(years=5)
    start_str = fetch_start_date.strftime('%Y-%m-%d')
    end_str = (report_end_date + timedelta(days=1)).strftime('%Y-%m-%d')

    # 2. Identify all unique STOCK symbols (excluding cash)
    stock_symbols_initial = set(df_initial_state[
        df_initial_state['asset_category'] == 'Stock'
    ]['symbol'].unique())
    stock_symbols_events = set(df_events[df_events['symbol'].notna()]['symbol'].unique())
    all_cash_symbols = set(df_initial_state[
        df_initial_state['asset_category'] == 'Cash'
    ]['symbol'].unique())
    final_stock_symbols = (stock_symbols_initial | stock_symbols_events) - all_cash_symbols
    
    # 3. Build Ticker Maps
    stock_ticker_map, stock_info_map = _build_stock_ticker_map(
        final_stock_symbols, df_financial_info, exchange_suffix_map
    )
    fx_ticker_map, fx_info_map, base_currency = _build_fx_ticker_map(report_data)

    # Combine all info maps for the final report
    ticker_info_map = {**stock_info_map, **fx_info_map}

    # 4. Fetch data from yfinance
    stock_tickers_to_fetch = [t for t in stock_ticker_map.values() if t]
    fx_tickers_to_fetch = list(fx_ticker_map.keys())
    
    unique_yfinance_tickers = sorted(list(set(stock_tickers_to_fetch + fx_tickers_to_fetch)))
    
    if not unique_yfinance_tickers:
        print("No valid yfinance tickers to fetch.")
        return {'data': {}, 'ticker_info': ticker_info_map}

    print(f"Fetching {len(unique_yfinance_tickers)} tickers (Stocks & FX) from {start_str} to {end_str}...")
    print(f"Tickers: {', '.join(unique_yfinance_tickers)}")
    
    yf_data = yf.download(
        unique_yfinance_tickers, 
        start=start_str, 
        end=end_str, 
        auto_adjust=False,
        progress=False,
        timeout=30 
    )
    
    # 5. Process yfinance results
    market_data_by_symbol = {}
    yf_data_map = {} 
    
    if yf_data.empty:
        print("Warning: yfinance.download returned no data for all tickers.")
    elif len(unique_yfinance_tickers) == 1:
        ticker = unique_yfinance_tickers[0]
        if not yf_data.empty: yf_data_map = {ticker: yf_data}
    elif isinstance(yf_data.columns, pd.MultiIndex):
        # Use .xs to safely extract data for each ticker
        yf_data_map = {
            ticker: yf_data.xs(ticker, level=1, axis=1).dropna(how='all')
            for ticker in unique_yfinance_tickers 
            if ticker in yf_data.columns.get_level_values(1)
        }
    else:
        print("Warning: yfinance returned unexpected data format. Failures likely.")

    # 6. Map STOCK data back to original IBKR symbols
    for original_symbol, yfinance_ticker in stock_ticker_map.items():
        data = yf_data_map.get(yfinance_ticker) 
        if data is None or data.empty:
            if original_symbol in ticker_info_map and ticker_info_map[original_symbol]['status'] == 'OK':
                print(f"Warning: No data returned for {original_symbol} (ticker: {yfinance_ticker})")
                ticker_info_map[original_symbol]['status'] = 'Error: yfinance returned no data.'
            market_data_by_symbol[original_symbol] = pd.DataFrame().rename_axis('Date')
        else:
            data_processed = data[['Open', 'High', 'Low', 'Adj Close', 'Volume']].copy()
            data_processed = data_processed.rename(columns={'Adj Close': 'Close'})
            market_data_by_symbol[original_symbol] = data_processed
    
    # 7. Map FX data back to original symbols
    any_valid_df = next((df for df in market_data_by_symbol.values() if not df.empty), None)
    
    for yfinance_ticker, original_symbol in fx_ticker_map.items():
        data = yf_data_map.get(yfinance_ticker)
        if data is None or data.empty:
            if original_symbol in ticker_info_map and ticker_info_map[original_symbol]['status'] == 'OK':
                print(f"Warning: No data returned for {original_symbol} (ticker: {yfinance_ticker})")
                ticker_info_map[original_symbol]['status'] = 'Error: yfinance returned no data.'
            market_data_by_symbol[original_symbol] = pd.DataFrame().rename_axis('Date')
        else:
            # For FX, 'Adj Close' is the rate.
            data_processed = data[['Open', 'High', 'Low', 'Adj Close', 'Volume']].copy()
            data_processed = data_processed.rename(columns={'Adj Close': 'Close'})
            market_data_by_symbol[original_symbol] = data_processed
            if any_valid_df is None:
                any_valid_df = data_processed
                
    # 8. Create dummy 1.0 DataFrame for BASE CURRENCY
    if base_currency and any_valid_df is not None:
        print(f"Creating 1.0 price history for base currency: {base_currency}")
        # Use the index of the first valid df to get the right date range
        date_index = any_valid_df.index
        # Reindex to ensure it covers the full range (for assets with no data)
        full_date_index = pd.date_range(start=start_str, end=end_str, freq='D')
        full_date_index = full_date_index[full_date_index <= date_index.max()] # Truncate to match yf data
        
        df_base = pd.DataFrame(1.0, index=full_date_index, columns=['Open', 'High', 'Low', 'Close'])
        df_base['Volume'] = 0.0
        # Align with the main index from the other assets
        market_data_by_symbol[base_currency] = df_base.reindex(date_index, method='ffill')
        
    elif base_currency:
        print(f"Warning: Could not create 1.0 price history for {base_currency} (no valid date index found).")
        market_data_by_symbol[base_currency] = pd.DataFrame().rename_axis('Date')
        
    print("Market data fetch complete.")
    
    return {
        'data': market_data_by_symbol,
        'ticker_info': ticker_info_map
    }

# --- This main block is for testing ---
if __name__ == "__main__":
    
    try:
        # Assumes you are running from the root folder (above src)
        from src import data_loader as dl
    except ImportError:
        print("Error: Could not import 'src.data_loader'.")
        print("Please make sure you are running this script from the root project directory (the folder *above* 'src').")
        dl = None
    
    if dl:
        file_path = r'data/U13271915_20250101_20251029.csv'
        print(f"Loading report data from: {file_path}")
        report_data = dl.load_ibkr_report(file_path)
        
        if report_data:
            print("Report loaded.")
            print(f"  Stated Report Start: {report_data['report_start_date']}")
            print(f"  Stated Report End:   {report_data['report_end_date']}")
            print(f"  Base Currency: {report_data['base_currency']}")
            
            print("\nFetching market data (Stocks & FX)...")
            market_data_bundle = fetch_market_data(report_data)
            
            print("\n" + "="*80)
            print("--- Ticker Info Map (Debugging) ---")
            import json
            print(json.dumps(market_data_bundle['ticker_info'], indent=2))
            print("="*80)

            print("\n--- Fetched Data Summary ---")
            for symbol, df in market_data_bundle['data'].items():
                info = market_data_bundle['ticker_info'].get(symbol, {})
                ticker = info.get('yfinance_ticker', 'N/A')
                status = info.get('status', 'N/A')
                asset_type = info.get('asset_type', 'N/A')
                
                print(f"  Symbol: {symbol} (Type: {asset_type}, Ticker: {ticker})")
                if not df.empty:
                    print(f"  Date Range: {df.index.min().date()} to {df.index.max().date()}")
                    print(f"  Rows: {len(df)}")
                else:
                    print(f"  Status: {status}")
                print("-" * 20)