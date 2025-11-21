import pandas as pd
import yfinance as yf

def load_market_data(data_package):
    """
    Accepts a data package, constructs ticker symbols for Stocks and Cash, 
    and downloads daily market data using yfinance.
    
    NOTE: This module does NOT perform split adjustments. It downloads raw/adjusted 
    pricing data. Downstream processors should select the appropriate column 
    ('Adj Close' vs 'Close') to match the quantity basis (adjusted vs raw).
    """
    # --- 1. Unpack Data ---
    # EXPLANATION: Defensive Copy
    # Creating a copy prevents accidental modification of the master 'initial_state'
    # DataFrame while we use it for filtering assets here.
    df_initial_state_adj = data_package['initial_state'].copy()
    df_financial_info = data_package['financial_info']
    
    base_currency = data_package['base_currency']
    report_start_date = data_package['report_start_date']
    report_end_date = data_package['report_end_date']

    # --- 2. Define Ticker Logic ---
    
    # Map of Exchange codes to Yahoo Finance suffixes.
    # NOTE: This requires manual maintenance. New exchanges will default to '' (US),
    # potentially causing download errors if the suffix is actually required.
    exchange_suffix_map = {
        # --- Americas ---
        'NASDAQ': '',     # US
        'NYSE': '',       # US
        'ARCA': '',       # US ETF
        'AMEX': '',       # US
        'PINK': '',       # US OTC
        'TSE': '.TO',     # Toronto (Canada)
        'VENTURE': '.V',  # TSX Venture (Canada)
        'MEXI': '.MX',    # Mexico

        # --- Europe ---
        'LSE': '.L',      # London (UK)
        'IBIS': '.DE',    # Xetra (Germany) - Note: IBKR uses IBIS for Xetra
        'FWB': '.F',      # Frankfurt (Germany)
        'SBF': '.PA',     # Euronext Paris (France)
        'AEB': '.AS',     # Euronext Amsterdam (Netherlands)
        'EBS': '.SW',     # SIX Swiss Exchange (Switzerland)
        'VIRTX': '.SW',   # SIX Swiss (Blue chips)
        'BM': '.MC',      # Bolsa de Madrid (Spain)
        'BVME': '.MI',    # Borsa Italiana (Italy)
        'SB': '.ST',      # Stockholm (Sweden)
        'SFB': '.ST',     # Stockholm (Sweden)
        'OSE': '.OL',     # Oslo (Norway)
        'CPH': '.CO',     # Copenhagen (Denmark)
        'VIE': '.VI',     # Vienna (Austria)
        'PL': '.LS',      # Lisbon (Portugal)
        'EBR': '.BR',     # Brussels (Belgium)

        # --- Asia / Pacific ---
        'SEHK': '.HK',    # Hong Kong
        'ASX': '.AX',     # Australia
        'TSEJ': '.T',     # Tokyo (Japan)
        'SGX': '.SI',     # Singapore
        'NSE': '.NS',     # India (National)
        'BSE': '.BO',     # India (Bombay)
    }

    symbol_to_ticker = {}

    # A. Stocks
    # EXPLANATION: Boolean Masking
    # Efficiently filters the DataFrame for rows where asset_category is 'Stock'.
    stocks_mask = df_initial_state_adj['asset_category'] == 'Stock'
    stock_symbols = df_initial_state_adj.loc[stocks_mask, 'symbol'].unique()

    for sym in stock_symbols:
        # Look up the Exchange code in the financial_info table
        exchange_entry = df_financial_info.loc[df_financial_info['symbol'] == sym, 'Exchange']
        
        if not exchange_entry.empty:
            exchange_code = exchange_entry.iloc[0]
            
            # Explicit check with warning instead of silent default
            if exchange_code in exchange_suffix_map:
                suffix = exchange_suffix_map[exchange_code]
            else:
                # This alerts you to update your map rather than fetching wrong data
                print(f"Warning: Exchange '{exchange_code}' for {sym} not in map. Defaulting to US (no suffix).")
                suffix = ''
                
            symbol_to_ticker[sym] = f"{sym}{suffix}"
        else:
            # Fallback: assume ticker matches the symbol exactly
            symbol_to_ticker[sym] = sym

    # B. Cash (Forex)
    cash_mask = df_initial_state_adj['asset_category'] == 'Cash'
    cash_symbols = df_initial_state_adj.loc[cash_mask, 'symbol'].unique()

    for sym in cash_symbols:
        # Skip if the cash symbol is the base currency itself (exchange rate is always 1.0)
        if sym == base_currency:
            continue
            
        # "{sym}{base_currency}=X" (e.g., USDCHF=X). This is the price of 1 USD in CHF.
        ticker = f"{sym}{base_currency}=X"
        symbol_to_ticker[sym] = ticker

    # --- 3. Download Market Data ---
    
    # Fetch 5 years of history to allow for potential moving averages or lookback analysis
    start_dt = report_start_date - pd.DateOffset(years=5)
    end_dt = report_end_date
    
    market_data_map = {}
    
    # 1. Collect unique tickers for bulk download
    unique_tickers = list(set(symbol_to_ticker.values()))
    
    if unique_tickers:
        # EXPLANATION: Batch Download Optimization
        # Requesting all tickers in a single yfinance call is significantly faster 
        # (~2s vs ~30s) than looping. 'threads=True' enables parallel HTTP requests.
        # auto_adjust=False ensures we get both 'Close' (Raw) and 'Adj Close' (Split/Div Adjusted).
        batch_data = yf.download(
            unique_tickers,
            start=start_dt,
            end=end_dt,
            timeout=30,
            auto_adjust=False,
            group_by='ticker', 
            threads=True,
            progress=False,
            actions=True
        )

        # 2. Distribute data back to individual symbols
        for sym, ticker in symbol_to_ticker.items():
            try:
                df = pd.DataFrame()
                
                # Handle yfinance API structure variations (Single vs Multi Ticker returns)
                if len(unique_tickers) == 1:
                    df = batch_data
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                else:
                    if ticker in batch_data.columns:
                        df = batch_data[ticker]
                    else:
                        print(f"Warning: Ticker '{ticker}' for symbol '{sym}' not returned by API.")
                        continue

                # Clean: Remove rows where the exchange was closed (all NaNs)
                df = df.dropna(how='all')

                if not df.empty:
                    # create a distinct copy to avoid SettingWithCopyWarning
                    df = df.copy()

                    if ticker.endswith('.L'):
                        try:
                            # Create the object just for this specific ticker
                            t = yf.Ticker(ticker)

                            try:
                                curr = t.fast_info['currency']
                            except:
                                # Fallback to slower .info if fast_info fails
                                curr = t.info.get('currency', 'GBP')

                            if curr == 'GBp':
                                # Define the columns that are monetary values
                                price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']

                                # Find which of these are actually in your DataFrame columns
                                cols_to_fix = [c for c in price_cols if c in df.columns]

                                # Apply the math ONLY to those columns
                                df[cols_to_fix] = df[cols_to_fix] / 100

                        except Exception as e:
                            print(f"Could not verify currency for {ticker}: {e}")
                
                if not df.empty:
                    # --- KEY FIX START ---
                    # For Stocks: Store by Symbol (e.g. 'NESN') so Reconstructor finds it via df_holdings columns.
                    # For FX: Store by Ticker (e.g. 'USDCHF=X') so Reconstructor finds it via constructed FX string.
                    if '=X' in ticker:
                        market_data_map[ticker] = df
                    else:
                        market_data_map[sym] = df
                    # --- KEY FIX END ---
                else:
                    print(f"Warning: Downloaded data for '{sym}' ({ticker}) was empty.")
            
            except Exception as e:
                print(f"Error processing market data for '{sym}': {e}")
            
    # --- 4. Return Package ---
    return market_data_map