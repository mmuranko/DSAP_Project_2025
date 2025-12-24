"""
Market Data Loader Module.

Handles ticker construction, sequential data retrieval via yfinance, and
time-series normalisation for portfolio assets.
"""
import pandas as pd
import yfinance as yf
import time
from typing import Any, Optional
from .config import EXCHANGE_SUFFIX_MAP

def load_market_data(
    data_package: dict[str, Any], 
    report_filename: Optional[str] = None
) -> dict[str, pd.DataFrame]:
    """
    Constructs tickers and downloads daily market data for all portfolio assets.

    Maps internal symbols to provider tickers (e.g., adding '.L' for LSE), handles
    currency cross-rates for cash positions, and normalizes GBp/GBP pricing.
    Data is gap-filled to ensure a continuous daily index.

    Args:
        data_package: Dict containing 'initial_state', 'financial_info',
                      'base_currency', and report dates.
        report_filename: Optional report identifier (unused).

    Returns:
        Dict mapping internal symbols (or cash tickers) to cleaned DataFrames
        containing OHLCV and Dividend history.
    """
    # ======================================================
    # 1. PREPARE TICKERS
    # ======================================================
    # Extract core configuration and portfolio state from the provided package.
    df_initial_state_adj = data_package['initial_state'].copy()
    df_financial_info = data_package['financial_info']
    base_currency = data_package['base_currency']
    report_start_date = data_package['report_start_date']
    report_end_date = data_package['report_end_date']

    symbol_to_ticker = {}

    # A. Stocks
    # Identify equity assets and map internal symbols to vendor-specific tickers.
    stocks_mask = df_initial_state_adj['asset_category'] == 'Stock'
    stock_symbols = df_initial_state_adj.loc[stocks_mask, 'symbol'].unique()

    for sym in stock_symbols:
        # Retrieve the exchange code to determine the correct suffix (e.g., '.L', '.TO').
        exchange_entry = df_financial_info.loc[df_financial_info['symbol'] == sym, 'Exchange']
        if not exchange_entry.empty:
            exchange_code = exchange_entry.iloc[0]
            suffix = EXCHANGE_SUFFIX_MAP.get(exchange_code, '')
            symbol_to_ticker[sym] = f"{sym}{suffix}"
        else:
            # Default to the raw symbol if no exchange mapping exists.
            symbol_to_ticker[sym] = sym

    # B. Cash
    # Construct Forex tickers for non-base currency cash positions to facilitate 
    # cross-rate calculations.
    cash_mask = df_initial_state_adj['asset_category'] == 'Cash'
    cash_symbols = df_initial_state_adj.loc[cash_mask, 'symbol'].unique()
    for sym in cash_symbols:
        if sym == base_currency: continue
        symbol_to_ticker[sym] = f"{sym}{base_currency}=X"

    # ======================================================
    # 2. SEQUENTIAL DOWNLOAD
    # ======================================================
    # Define a lookback window of 1 year to ensure sufficient history for further analyses
    start_dt = report_start_date - pd.DateOffset(years=1)
    end_dt = report_end_date
    
    market_data_map = {}
    unique_tickers = list(set(symbol_to_ticker.values()))
    
    if unique_tickers:
        print(f"\n [>] Downloading {len(unique_tickers)} tickers sequentially...")
        time.sleep(0.5)
        
        for i, ticker in enumerate(unique_tickers):
            
            # Implement a retry mechanism with exponential backoff to handle transient network errors.
            max_retries = 3 
            success = False
            
            for attempt in range(max_retries):
                try:
                    # Instantiate the Ticker object (uses internal session handling).
                    dat = yf.Ticker(ticker)
                    
                    # Fetch historical price and dividend data.
                    df = dat.history(start=start_dt, end=end_dt, auto_adjust=False, actions=True)
                    
                    df = df.dropna(how='all')
                    if df.empty:
                        # Differentiate between a temporary failure and a permanent lack of data.
                        if attempt < max_retries - 1:
                            raise ValueError("Received empty data")
                        else:
                            print(f"     [{i+1}/{len(unique_tickers)}] {ticker} NO DATA returned.")
                            break

                    # Flatten MultiIndex columns if the provider returns grouped data.  
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(1)
                    
                    df = df.copy()

                    # Remove timezone information and normalize to midnight
                    # Convert index to DatetimeIndex explicitly before localizing
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    
                    # GBp to GBP Fix
                    # London Stock Exchange stocks often quote in Pence (GBp) while portfolios report in Pounds (GBP).
                    if ticker.endswith('.L'):
                        divisor = 1.0
                        
                        # 1. Check Metadata (Primary Method)
                        try:
                            # 'GBp' = Pence, 'GBP' = Pounds
                            # Query the fast_info attribute for the currency string.
                            currency = dat.fast_info['currency']
                            if currency == 'GBp':
                                divisor = 100.0
                        except:
                            # 2. Fallback Heuristic (Secondary Method)
                            # If metadata is unavailable, assume prices > 1000 indicate Pence.
                            if df['Close'].mean() > 1000:
                                divisor = 100.0
                        
                        # Apply the divisor to all monetary columns if scaling is required.
                        if divisor > 1.0:
                            cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Dividends'] if c in df.columns]
                            df[cols] = df[cols] / divisor

                    # Fill temporal gaps (weekends, holidays) using forward-fill to maintain a continuous daily time series.
                    df = df.resample('D').ffill()

                    # Problem: If data starts on Jan 3rd (post-holiday), Jan 1-2 remain NaN.
                    # Solution: If the data start is within 5 days of the requested 'start_dt', 
                    #           we assume it's a holiday/weekend gap and backfill it.
                    #           If the gap is larger (e.g., months), we assume the asset didn't exist yet.
                    
                    if not df.empty:
                        first_valid_dt = df.index.min()
                        # Calculate gap between requested start and actual data start
                        gap_days = (first_valid_dt - start_dt).days
                        
                        # Threshold: 5 days covers a long weekend + New Year's holiday
                        if 0 < gap_days <= 5:
                            # 1. Reindex to force the DataFrame to start at 'start_dt'
                            #    This creates the rows for Jan 1, Jan 2, etc. with NaN values.
                            full_idx = pd.date_range(start=start_dt, end=df.index.max(), freq='D')
                            df = df.reindex(full_idx)
                            
                            # 2. Backfill ONLY these newly created limited gaps
                            #    limit=6 ensures we don't accidentally fill a massive gap if logic fails
                            df.bfill(limit=6, inplace=True)

                    # Save the processed DataFrame to the map.
                    # Note: Multiple internal symbols (e.g., specific lots) may map to the same ticker.
                    symbols = [k for k, v in symbol_to_ticker.items() if v == ticker]
                    for sym in symbols:
                        if '=X' in ticker:
                            market_data_map[ticker] = df
                        else:
                            market_data_map[sym] = df
                    
                    print(f"       [{i+1}/{len(unique_tickers)}] {ticker} success.")
                    success = True
                    break # Exit the retry loop upon success.

                except Exception as e:
                    # Calculate exponential backoff wait time (5s, 10s, 20s).
                    wait_time = 5 * (2 ** attempt) # 5s, 10s...
                    if attempt < max_retries - 1:
                        print(f" [!] Issue with {ticker} ({e}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f" [!] Failed {ticker}: {e}")
                        time.sleep(0.5)

            # Introduce a delay between requests to respect API limits.
            if success:
                time.sleep(0.5)

    if unique_tickers:
        print(f" [+] Market data download complete.")
        time.sleep(0.5)

    return market_data_map