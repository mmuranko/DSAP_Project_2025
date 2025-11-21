import pandas as pd
import numpy as np

def reconstruct_portfolio(data_package, market_data_map=None, verbose=False):
    """
    Reconstructs Daily Holdings and performs Dual-Currency Valuation.
    
    Methodology:
    1. Quantity Engines: Separate logic for Cash (by Currency) and Securities (by Symbol).
    2. Linear Algebra Valuation: Matrix multiplication (Holdings * Prices * Rates).
    3. Robust Gap Filling: Uses ffill+bfill for missing market data.
    """

    # ==========================================
    # STEP 1: UNPACK & SETUP
    # ==========================================

    df_initial = data_package['initial_state'].copy()
    df_events = data_package['events'].copy()

    start_date = data_package['report_start_date'].normalize()
    end_date = data_package['report_end_date'].normalize()

    # We create a NEW column called 'date'. 
    # This does NOT delete the 'timestamp' info. It just gives us a "Day ID" for grouping.
    # Example: "2025-02-14 10:45:03" becomes "2025-02-14 00:00:00"
    df_events['date'] = df_events['timestamp'].dt.normalize()

    # Create the master timeline (Rows for our final matrix)
    # freq='D' ensures we have a row for EVERY day (including weekends)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # ==========================================
    # STEP 2: THE CASH ENGINE
    # ==========================================

    # 1. Aggregate Cash Flows
    # We sum 'cash_change_native' because it captures ALL money movement (Trades, Dividends, Fees)
    # We group by 'currency' (columns) and 'date' (index).
    cash_activity = df_events.groupby(['date', 'currency'])['cash_change_native'].sum()

    # 2. Pivot to Wide Format
    # Rows = Dates, Columns = Currencies (USD, CHF, EUR...)
    df_cash_changes = cash_activity.unstack(fill_value=0)

    # 3. Reindex to Master Timeline
    # This ensures we have a row for every single day (filling gaps with 0 change)
    df_cash_changes = df_cash_changes.reindex(all_dates, fill_value=0)

    # 4. Add Initial Cash State
    # We look for rows in initial_state where asset_category is 'Cash'
    initial_cash = df_initial[df_initial['asset_category'] == 'Cash']

    for _, row in initial_cash.iterrows():
        curr = row['currency']
        qty = row['quantity'] # In initial_state, 'quantity' holds the cash balance
        
        # Safety check: If we hold a currency (e.g. JPY) but never traded it, 
        # it won't be in df_cash_changes columns yet. We create it.
        if curr not in df_cash_changes.columns:
            df_cash_changes[curr] = 0.0
        
        # Add the starting balance to the VERY FIRST day
        # This acts as a "Deposit" at the beginning of time
        df_cash_changes.loc[start_date, curr] += qty

    # 5. Calculate Running Balance (Cumulative Sum)
    df_daily_cash_balance = df_cash_changes.cumsum()

    # ==========================================
    # STEP 3: THE SECURITIES ENGINE
    # ==========================================

    # 1. Aggregate Security Flows
    # We sum 'quantity_change' grouped by 'symbol' and 'date'.
    security_activity = df_events.groupby(['date', 'symbol'])['quantity_change'].sum()

    # 2. Pivot to Wide Format
    # Rows = Dates, Columns = Symbols (AAPL, NESN, ...)
    df_security_changes = security_activity.unstack(fill_value=0)

    # 3. Reindex to Master Timeline
    df_security_changes = df_security_changes.reindex(all_dates, fill_value=0)

    # 4. Add Initial Security State
    initial_stocks = df_initial[df_initial['asset_category'] == 'Stock']
    
    if verbose:
        print(f"Found {len(initial_stocks)} initial stock positions.")

    for _, row in initial_stocks.iterrows():
        sym = row['symbol']
        qty = row['quantity']
        
        # Safety check: If we hold a stock but never traded it in this period,
        # it won't be in the columns yet. We create it.
        if sym not in df_security_changes.columns:
            df_security_changes[sym] = 0.0
            
        # Add the starting shares to the VERY FIRST day
        df_security_changes.loc[start_date, sym] += qty

    # 5. Calculate Running Balance (Cumulative Sum)
    df_daily_security_balance = df_security_changes.cumsum()

    # Sanity Check: Are there any negative share counts?
    # (This usually implies a missing initial position or bad split logic)
    negatives = df_daily_security_balance[df_daily_security_balance < -1e-9].count().sum()
    if negatives > 0 and verbose:
        print(f"\nWARNING: Found {negatives} instances of negative share counts!")
        # detailed check
        for col in df_daily_security_balance.columns:
            if (df_daily_security_balance[col] < -1e-9).any():
                print(f"   - {col} goes negative.")

    # ==========================================
    # STEP 4: MERGE HOLDINGS
    # ==========================================

    # 1. Identify Overlap
    # The Security Engine might have created columns for 'USD' or 'CHF' because 
    # some events (like FX trades) have those symbols, even if quantity_change was 0.
    cols_overlap = df_daily_security_balance.columns.intersection(df_daily_cash_balance.columns)

    if not cols_overlap.empty:
        if verbose:
            print(f"Cleaning up redundant currency columns in Security Engine: {list(cols_overlap)}")
        # Drop them from the Security DataFrame so they don't duplicate the Cash DataFrame columns
        df_security_clean = df_daily_security_balance.drop(columns=cols_overlap)
    else:
        df_security_clean = df_daily_security_balance


    # 2. Merge
    # Now we are safe to concat. Cash columns come purely from the Cash Engine.
    df_holdings = pd.concat([df_security_clean, df_daily_cash_balance], axis=1).fillna(0)


    # ==========================================
    # STEP 5: CONSTRUCT PRICE MATRIX
    # ==========================================

    # 1. Setup Matrices (Aligned to Holdings Index/Columns)
    # Initialize with 1.0. This is crucial because:
    # - Cash Quantity * 1.0 = Cash Value (Native)
    # - Base Currency * 1.0 = Base Value (FX Rate)
    df_prices = pd.DataFrame(1.0, index=df_holdings.index, columns=df_holdings.columns)
    df_rates = pd.DataFrame(1.0, index=df_holdings.index, columns=df_holdings.columns)


    # 2. Fill Price Matrix (P)
    # We only update columns that are SECURITIES (Stocks). 
    for col in df_holdings.columns:
        # If this column is a Cash Account (e.g. 'USD'), its "Native Price" is 1.0.
        # We skip looking it up in market_data_map to avoid double-counting FX rates.
        if col in df_daily_cash_balance.columns:
            continue 

        if col in market_data_map:
            # It's a Security
            prices = market_data_map[col]
            price_series = prices['Close']
            
            # --- GAP FILLING STRATEGY ---
            # 1. Reindex to Master Timeline (introduces NaNs for weekends/holidays/missing dates)
            aligned_prices = price_series.reindex(df_holdings.index)
            
            # 2. Forward Fill (ffill): "Today's price is yesterday's closing price"
            # Handles weekends and holidays perfectly.
            aligned_prices = aligned_prices.ffill()
            
            # 3. Backward Fill (bfill): "If we have no past price, assume the first future price applies retroactively"
            # Handles assets that start trading mid-report or have missing initial data (like SPY5).
            aligned_prices = aligned_prices.bfill()
            
            # Assign to Matrix
            df_prices[col] = aligned_prices


    # ==========================================
    # STEP 6: CONSTRUCT EXCHANGE RATE MATRIX
    # ==========================================

    # Logic: For every asset in our holdings, find the multiplier to convert 
    # its Native Value -> Base Value.
    base_currency = data_package['base_currency']

    # Since it contains both Stocks and Cash rows, it maps every asset to its currency.
    asset_currency_map = df_initial.set_index('symbol')['currency'].to_dict()

    # 2. Iterate through every asset in our Holdings Matrix
    for col in df_holdings.columns:
        
        # A. Find the Currency of this asset
        asset_curr = asset_currency_map.get(col) # Try to get it without default first
    
        if asset_curr is None:
            if verbose:
                print(f"CRITICAL WARNING: Asset '{col}' appeared in holdings but was not in Initial State map! Defaulting to {base_currency}.")
            asset_curr = base_currency
        
        # B. Skip if no conversion needed
        if asset_curr == base_currency:
            continue 
            
        # C. Determine the Ticker Symbols for the FX Pair
        fx_ticker = f"{asset_curr}{base_currency}=X"   # Direct: "USDCHF=X"
        inv_ticker = f"{base_currency}{asset_curr}=X"   # Inverse: "CHFUSD=X"
        
        # D. Fetch the Rate Series (Safely)
        rate_series = None
        
        if fx_ticker in market_data_map:
            # We have the direct rate
            rate_series = market_data_map[fx_ticker]['Close']
        elif inv_ticker in market_data_map:
            # We have the inverse rate -> Calculate Reciprocal
            rate_series = 1.0 / market_data_map[inv_ticker]['Close']
            
        # E. Align and Assign
        if rate_series is not None:
            # 1. Reindex to master timeline
            aligned_rate = rate_series.reindex(df_holdings.index)

            # 2. Fill gaps (Forward first, then Backward for report start gaps)
            aligned_rate = aligned_rate.ffill()
            aligned_rate = aligned_rate.bfill()

            # 3. Insert into the Matrix
            df_rates[col] = aligned_rate
        else:
            if verbose:
                print(f"Warning: No FX rate found for {asset_curr} (Asset: {col})")

    # ==========================================
    # STEP 7: THE CALCULATION (Matrix Multiplication)
    # ==========================================

    # 1. Calculate Native Valuation
    # Formula: Quantity * Native Price
    # - For Stocks: Shares * Market Price
    # - For Cash:   Balance * 1.0
    df_val_native = df_holdings * df_prices

    # 2. Calculate Base Valuation
    # Formula: Quantity * Native Price * Exchange Rate
    # - For Stocks: Shares * Market Price * FX Rate
    # - For Cash:   Balance * 1.0 * FX Rate
    df_val_base = df_holdings * df_prices * df_rates

    daily_total_nav = df_val_base.sum(axis=1)

    # Return the essential datasets
    return {
        'holdings': df_holdings,
        'prices': df_prices,
        'valuation_native': df_val_native,
        'valuation_base': df_val_base,
        'total_nav': daily_total_nav
    }