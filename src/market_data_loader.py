import pandas as pd
import yfinance as yf
import time
from .config import EXCHANGE_SUFFIX_MAP

def load_market_data(data_package, report_filename=None):
    """
    Accepts a data package, constructs ticker symbols, and downloads daily market data.
    Uses yfinance's internal session management (curl_cffi).
    """
    
    # ======================================================
    # 1. PREPARE TICKERS
    # ======================================================
    df_initial_state_adj = data_package['initial_state'].copy()
    df_financial_info = data_package['financial_info']
    base_currency = data_package['base_currency']
    report_start_date = data_package['report_start_date']
    report_end_date = data_package['report_end_date']

    symbol_to_ticker = {}

    # A. Stocks
    stocks_mask = df_initial_state_adj['asset_category'] == 'Stock'
    stock_symbols = df_initial_state_adj.loc[stocks_mask, 'symbol'].unique()

    for sym in stock_symbols:
        exchange_entry = df_financial_info.loc[df_financial_info['symbol'] == sym, 'Exchange']
        if not exchange_entry.empty:
            exchange_code = exchange_entry.iloc[0]
            suffix = EXCHANGE_SUFFIX_MAP.get(exchange_code, '')
            symbol_to_ticker[sym] = f"{sym}{suffix}"
        else:
            symbol_to_ticker[sym] = sym

    # B. Cash
    cash_mask = df_initial_state_adj['asset_category'] == 'Cash'
    cash_symbols = df_initial_state_adj.loc[cash_mask, 'symbol'].unique()
    for sym in cash_symbols:
        if sym == base_currency: continue
        symbol_to_ticker[sym] = f"{sym}{base_currency}=X"

    # ======================================================
    # 2. SEQUENTIAL DOWNLOAD (Robust Mode)
    # ======================================================
    start_dt = report_start_date - pd.DateOffset(years=5)
    end_dt = report_end_date
    
    market_data_map = {}
    unique_tickers = list(set(symbol_to_ticker.values()))
    
    if unique_tickers:
        print(f"   [>] Downloading {len(unique_tickers)} tickers sequentially (Fresh Download)...")
        
        for i, ticker in enumerate(unique_tickers):
            
            # --- RETRY LOGIC ---
            max_retries = 3 
            success = False
            
            for attempt in range(max_retries):
                try:
                    # yfinance internal session handling
                    dat = yf.Ticker(ticker)
                    
                    df = dat.history(start=start_dt, end=end_dt, auto_adjust=False, actions=True)
                    
                    df = df.dropna(how='all')
                    if df.empty:
                        # Allow retry on empty data
                        if attempt < max_retries - 1:
                            raise ValueError("Received empty data")
                        else:
                            print(f"       [{i+1}/{len(unique_tickers)}] {ticker} NO DATA returned.")
                            break

                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(1)
                    
                    df = df.copy()

                    # Remove timezone (make it 'naive') & Normalize to midnight
                    df.index = df.index.tz_localize(None).normalize()
                    
                    # GBp to GBP Fix
                    if ticker.endswith('.L'):
                        divisor = 1.0
                        
                        # 1. Check Metadata (Source of Truth)
                        try:
                            # 'GBp' = Pence, 'GBP' = Pounds
                            # fast_info is efficient and reliable
                            currency = dat.fast_info['currency']
                            if currency == 'GBp':
                                divisor = 100.0
                        except:
                            # 2. Fallback Heuristic (Only if metadata fails)
                            # Assume > 500 is likely pence (e.g. 500p = £5)
                            if df['Close'].mean() > 500:
                                divisor = 100.0
                        
                        # Apply Correction
                        if divisor > 1.0:
                            cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Dividends'] if c in df.columns]
                            df[cols] = df[cols] / divisor

                    # Gap Fill (Weekends/Holidays)
                    df = df.resample('D').ffill().bfill()

                    # Save to Map
                    symbols = [k for k, v in symbol_to_ticker.items() if v == ticker]
                    for sym in symbols:
                        if '=X' in ticker:
                            market_data_map[ticker] = df
                        else:
                            market_data_map[sym] = df
                    
                    print(f"       [{i+1}/{len(unique_tickers)}] {ticker} success.")
                    success = True
                    break # Success, exit retry loop

                except Exception as e:
                    wait_time = 5 * (2 ** attempt) # 5s, 10s...
                    if attempt < max_retries - 1:
                        print(f"       [!] Issue with {ticker} ({e}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"       [!] Failed {ticker}: {e}")

            # Polite wait between tickers
            if success:
                time.sleep(0.5)

    return market_data_map