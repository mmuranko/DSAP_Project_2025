import pandas as pd
import yfinance as yf

def load_market_data(data_package):
    """
    Accepts a data package and
    constructs tickers for Stocks and Cash, and downloads daily market data.
    """
    # --- 1. Unpack Data ---
    df_initial_state_adj = data_package['initial_state'].copy()
    df_financial_info = data_package['financial_info']
    
    base_currency = data_package['base_currency']
    report_start_date = data_package['report_start_date']
    report_end_date = data_package['report_end_date']

    # --- 2. Define Ticker Logic ---
    
    # Map of Exchange codes to Yahoo Finance suffixes
    exchange_suffix_map = {
        'NASDAQ': '', 'NYSE': '', 'ARCA': '',  # USA
        'EBS': '.SW',   # Switzerland
        'IBIS': '.F',   # Germany
        'SBF': '.PA',   # France
        'LSE': '.L',    # UK
        'BVME': '.MI',  # Italy
        'BM': '.MC',    # Spain
    }

    # Dictionary to hold Symbol to Ticker mapping
    symbol_to_ticker = {}

    # A. Stocks
    # Filter for stocks in the initial state
    stocks_mask = df_initial_state_adj['asset_category'] == 'Stock'
    stock_symbols = df_initial_state_adj.loc[stocks_mask, 'symbol'].unique()

    for sym in stock_symbols:
        # Look up the Exchange in financial_info for each stock symbol
        exchange_entry = df_financial_info.loc[df_financial_info['symbol'] == sym, 'Exchange']
        
        if not exchange_entry.empty:
            exchange_code = exchange_entry.iloc[0]
            suffix = exchange_suffix_map.get(exchange_code, '') # Default to '' if not in map
            symbol_to_ticker[sym] = f"{sym}{suffix}"
        else:
            # Fallback if symbol not in financial_info
            symbol_to_ticker[sym] = sym

    # B. Cash
    # Filter for cash in the initial state
    cash_mask = df_initial_state_adj['asset_category'] == 'Cash'
    cash_symbols = df_initial_state_adj.loc[cash_mask, 'symbol'].unique()

    for sym in cash_symbols:
        # Skip if the cash symbol is the base currency itself (no exchange rate needed/valid)
        if sym == base_currency:
            continue
            
        # Construct ticker: Base + Symbol + '=X' (e.g., 'USDCHF=X')
        ticker = f"{base_currency}{sym}=X"
        symbol_to_ticker[sym] = ticker

    # --- 3. Download Market Data ---
    
    start_dt = report_start_date - pd.DateOffset(years=5)
    end_dt = report_end_date
    
    market_data_map = {}
    
    # 1. Collect unique tickers for bulk download (Reduces time from ~30s to ~2s)
    unique_tickers = list(set(symbol_to_ticker.values()))
    
    if unique_tickers:
        # threads=True enables parallel downloading
        # group_by='ticker' creates a hierarchy: Ticker -> Price Columns
        batch_data = yf.download(
            unique_tickers,
            start=start_dt,
            end=end_dt,
            timeout=30,
            auto_adjust=False,
            group_by='ticker', 
            threads=True,
            progress=False
        )

        # 2. Distribute data back to symbols
        for sym, ticker in symbol_to_ticker.items():
            try:
                df = pd.DataFrame()
                
                # Handle Single Ticker vs Multiple Ticker return structure
                if len(unique_tickers) == 1:
                    df = batch_data
                    # If single ticker returns MultiIndex (Price, Ticker), flatten it
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                else:
                    # For multiple tickers, selecting [ticker] automatically 
                    # drops the 'Ticker' column level, resulting in clean headers.
                    if ticker in batch_data.columns:
                        df = batch_data[ticker]

                # Drop rows where download failed (all NaNs)
                df = df.dropna(how='all')
                
                if not df.empty:
                    market_data_map[sym] = df
            except Exception:
                pass
            
    # --- 4. Return Package ---
    return market_data_map