import pandas as pd

def reconstruct_portfolio_history(adjusted_report, market_data, base_currency='CHF'):
    """
    Reconstructs the daily portfolio history from the adjusted data.
    
    Args:
        adjusted_report (dict): The split-adjusted data from portfolio_processor.
        market_data (dict): The price/fx data from market_data_loader.
        base_currency (str): The home currency for final valuation.
        
    Returns:
        pd.DataFrame: A DataFrame indexed by date, showing daily holdings,
                      cash, and total portfolio value in the base currency.
    """
    
    # --- 1. Unpack Inputs ---
    df_initial_state = adjusted_report['initial_state_adj']
    df_events = adjusted_report['events_adj']
    df_prices = market_data['prices']
    df_fx_rates = market_data['fx_rates']
    
    # --- 2. Get Full Date Range and All Assets ---
    start_date = df_prices.index.min()
    end_date = df_prices.index.max()
    date_index = pd.date_range(start_date, end_date, freq='D')

    # Get all stock/cash symbols
    stock_symbols = list(df_initial_state[df_initial_state['asset_category'] == 'Stock']['symbol'].unique())
    cash_symbols = list(df_initial_state[df_initial_state['asset_category'] == 'Cash']['symbol'].unique())
    
    # --- 3. Initialize State DataFrames ---
    # holdings_history stores (adjusted) *quantity* of shares
    holdings_history = pd.DataFrame(0.0, index=date_index, columns=stock_symbols)
    # cash_history stores *balance* of each currency
    cash_history = pd.DataFrame(0.0, index=date_index, columns=cash_symbols)

    # Set the starting (t=0) balances from the initial state
    for _, row in df_initial_state.iterrows():
        symbol = row['symbol']
        quantity = row['quantity']
        
        if row['asset_category'] == 'Stock' and symbol in holdings_history.columns:
            holdings_history.loc[start_date, symbol] = quantity
        elif row['asset_category'] == 'Cash' and symbol in cash_history.columns:
            cash_history.loc[start_date, symbol] = quantity
            
    # --- 4. Process the Event Log (Iterative State Machine) ---
    # Group events by date to process them in daily batches
    events_by_date = df_events.groupby(df_events['timestamp'].dt.date)
    
    for date, events in events_by_date:
        if date not in date_index:
            print(f"Warning: Event date {date} is outside market data range. Skipping.")
            continue
        
        # Get the day's changes
        changes = events.groupby(['symbol', 'event_type'])[['quantity_change', 'cash_change_native']].sum()
        
        for (symbol, event_type), change in changes.iterrows():
            if symbol in holdings_history.columns:
                holdings_history.loc[date, symbol] += change['quantity_change']
            
            if symbol in cash_history.columns:
                # This handles FX_TRADE, DIVIDEND, TAX, FEE, etc.
                cash_history.loc[date, symbol] += change['cash_change_native']
            
            # This handles the cash *part* of a stock trade
            if event_type in ['TRADE_BUY', 'TRADE_SELL']:
                currency = events.loc[events['symbol'] == symbol, 'currency'].iloc[0]
                if currency in cash_history.columns:
                    cash_history.loc[date, currency] += change['cash_change_native']
                else:
                    print(f"Warning: Currency '{currency}' from trade not in cash history. Skipping cash part of trade.")

    # --- 5. Forward-fill State and Calculate Daily Values ---
    
    # ffill() propagates the last known holding/cash balance to the next day
    holdings_history = holdings_history.ffill()
    cash_history = cash_history.ffill()

    # Align prices to our history (handles any missing symbols)
    df_prices_aligned = df_prices.reindex(columns=holdings_history.columns).ffill()
    
    # --- 6. Calculate Market Value in Native Currency ---
    # This is a vectorized operation (element-wise multiplication)
    holdings_value_native = holdings_history * df_prices_aligned
    
    # --- 7. Convert All Values to Base Currency (CHF) ---
    
    # Align FX rates to our holdings_value columns
    # We need to find the correct FX pair for each stock
    stock_to_currency_map = df_initial_state.set_index('symbol')['currency'].to_dict()
    
    holdings_value_chf = pd.DataFrame(0.0, index=date_index, columns=stock_symbols)
    
    for symbol in stock_symbols:
        currency = stock_to_currency_map.get(symbol)
        if currency == base_currency:
            holdings_value_chf[symbol] = holdings_value_native[symbol]
        elif currency in df_fx_rates.columns:
            # e.g., "USDCHF=X"
            fx_pair_name = f"{currency}{base_currency}=X" 
            if fx_pair_name in df_fx_rates.columns:
                holdings_value_chf[symbol] = holdings_value_native[symbol] * df_fx_rates[fx_pair_name]
            else:
                 print(f"Warning: FX pair '{fx_pair_name}' not found for {symbol}. Using 1.0.")
                 holdings_value_chf[symbol] = holdings_value_native[symbol]
        else:
            print(f"Warning: Currency '{currency}' for symbol {symbol} not in FX rates. Using 1.0.")
            holdings_value_chf[symbol] = holdings_value_native[symbol]

    # Convert cash balances to base currency
    cash_value_chf = pd.DataFrame(0.0, index=date_index, columns=cash_symbols)
    for symbol in cash_symbols:
        if symbol == base_currency:
            cash_value_chf[symbol] = cash_history[symbol]
        elif symbol in df_fx_rates.columns:
            # e.g., "USDCHF=X"
            fx_pair_name = f"{symbol}{base_currency}=X"
            if fx_pair_name in df_fx_rates.columns:
                 cash_value_chf[symbol] = cash_history[symbol] * df_fx_rates[fx_pair_name]
            else:
                 print(f"Warning: FX pair '{fx_pair_name}' not found for cash {symbol}. Using 1.0.")
                 cash_value_chf[symbol] = cash_history[symbol]
        else:
            print(f"Warning: Cash currency '{symbol}' not in FX rates. Using 1.0.")
            cash_value_chf[symbol] = cash_history[symbol]
            
    # --- 8. Combine Everything into One Final DataFrame ---
    
    # Total value of all stocks
    df_final = pd.DataFrame(index=date_index)
    df_final['Stock_Value_CHF'] = holdings_value_chf.sum(axis=1)
    
    # Total value of all cash (can be negative due to margin)
    df_final['Cash_Value_CHF'] = cash_value_chf.sum(axis=1)
    
    # Total Portfolio Value
    df_final['Total_Portfolio_CHF'] = df_final['Stock_Value_CHF'] + df_final['Cash_Value_CHF']
    
    # Add daily returns for analysis
    df_final['Daily_Return'] = df_final['Total_Portfolio_CHF'].pct_change()
    
    return {
        'portfolio_history': df_final,
        'holdings_history': holdings_history,
        'cash_history': cash_history,
        'holdings_value_chf': holdings_value_chf,
        'cash_value_chf': cash_value_chf
    }

# --- Example of how to use this module ---
if __name__ == "__main__":
    
    try:
        from src import data_loader as dl
        from src import portfolio_processor as pp
        from src import market_data_loader as mdl
    except ImportError:
        print("Could not import modules. Make sure this file is in the 'src' folder.")
        exit()

    # Define paths (relative to the project root, not 'src')
    filepath = r'data/U13271915_20250101_20251029.csv'
    output_excel_path = r'data/full_portfolio_reconstruction.xlsx'
    
    print(f"--- Step 1: Loading Report ---")
    original_report = dl.load_ibkr_report(filepath)
    
    if original_report:
        print(f"\n--- Step 2: Adjusting for Splits ---")
        adjusted_report = pp.adjust_for_splits(original_report)
        
        if adjusted_report:
            print(f"\n--- Step 3: Fetching Market Data ---")
            market_data = mdl.fetch_market_data(adjusted_report, base_currency='CHF')
            
            if market_data:
                print(f"\n--- Step 4: Reconstructing Portfolio History ---")
                
                final_portfolio = reconstruct_portfolio_history(
                    adjusted_report,
                    market_data,
                    base_currency='CHF'
                )
                
                print("\nPortfolio history reconstructed successfully!")
                
                # --- Save final results to Excel ---
                try:
                    with pd.ExcelWriter(output_excel_path) as writer:
                        final_portfolio['portfolio_history'].to_excel(writer, sheet_name='Portfolio_History')
                        final_portfolio['holdings_history'].to_excel(writer, sheet_name='Holdings_Qty')
                        final_portfolio['cash_history'].to_excel(writer, sheet_name='Cash_Balances')
                        final_portfolio['holdings_value_chf'].to_excel(writer, sheet_name='Holdings_Value_CHF')
                        final_portfolio['cash_value_chf'].to_excel(writer, sheet_name='Cash_Value_CHF')
                    
                    print(f"Successfully saved final portfolio to: {output_excel_path}")
                except ImportError:
                    print("Error: 'openpyxl' is required. `pip install openpyxl`")
                except Exception as e:
                    print(f"An error occurred while saving to Excel: {e}")

                print("\n" + "="*80)
                print("--- Final Portfolio History (Head) ---")
                print(final_portfolio['portfolio_history'].head())
                
                print("\n--- Final Portfolio History (Tail) ---")
                print(final_portfolio['portfolio_history'].tail())