import pandas as pd
import re
from datetime import datetime

# --- HELPER UTILITY ---

def get_section(df_raw, section_name):
    """
    Finds and extracts a specific section from the raw IBKR report.
    """
    try:
        # Find the header row for the section
        header_row_index = df_raw[
            (df_raw[0] == section_name) & (df_raw[1] == 'Header')
        ].index[0]
        
        # Get column names directly without stripping whitespace
        column_names = df_raw.iloc[header_row_index].values

        data_rows = df_raw[
            (df_raw[0] == section_name) & (df_raw[1] == 'Data')
        ]
        
        df_section = pd.DataFrame(data_rows.values, columns=column_names)
        df_section = df_section.dropna(axis=1, how='all').reset_index(drop=True)
        return df_section.iloc[:, 2:]
        
    except IndexError:
        return None
    except Exception as e:
        print(f"Error parsing section '{section_name}': {e}")
        return None

def clean_ibkr_symbol(symbol):
    """
    Cleans symbol according to specific rules:
    1. If symbol contains '.', return None (exclude entirely).
    2. If symbol ends with a lowercase letter, remove it.
    3. Otherwise return symbol as is.
    """
    if pd.isna(symbol) or symbol == '':
        return None
        
    s = str(symbol)
    
    # Rule 1: Exclude derivatives/rights with dots
    if '.' in s:
        return None
        
    # Rule 2: Strip trailing lowercase (e.g., 'VOW3d' -> 'VOW3')
    if s and s[-1].islower():
        return s[:-1]
        
    return s

# --- PARSER FUNCTIONS ---

def parse_initial_state(df_mtm, symbol_currency_map, master_symbol_set):
    """
    Parses the Mark-to-Market to build the initial state DataFrame.
    Uses vectorized operations and ensures symbols match the master set format.
    """
    if df_mtm is None:
        print("Warning: 'Mark-to-Market Performance Summary' section not found.")
        return pd.DataFrame(columns=['symbol', 'asset_category', 'currency', 'quantity', 'value_native'])

    # -----------------------
    # 1. Process Stocks
    # -----------------------
    df_stocks = df_mtm[df_mtm['Asset Category'] == 'Stocks'].copy()
    
    # CRITICAL: Apply the same cleaning logic to MtM symbols so they match the keys in your map
    df_stocks['symbol'] = df_stocks['Symbol'].apply(clean_ibkr_symbol)
    
    # Filter: Remove 'None' (dotted symbols) AND keep only those in your master set
    mask_valid = (df_stocks['symbol'].notna()) & (df_stocks['symbol'].isin(master_symbol_set))
    df_stocks = df_stocks[mask_valid].copy()

    # Numeric Conversions
    df_stocks['quantity'] = pd.to_numeric(df_stocks['Prior Quantity'], errors='coerce').fillna(0)
    prior_price = pd.to_numeric(df_stocks['Prior Price'], errors='coerce').fillna(0)
    
    # Calculate Values & Map Metadata
    df_stocks['value_native'] = df_stocks['quantity'] * prior_price
    df_stocks['currency'] = df_stocks['symbol'].map(symbol_currency_map).fillna('Unknown')
    df_stocks['asset_category'] = 'Stock'

    # Select final columns for stocks
    cols = ['symbol', 'asset_category', 'currency', 'quantity', 'value_native']
    df_stocks_final = df_stocks[cols]

    # -----------------------
    # 2. Process Forex (Cash)
    # -----------------------
    # Note: Cash symbols (e.g., 'USD') are usually not in the stock-only master_symbol_set, 
    # so we process them separately without filtering against that set.
    df_cash = df_mtm[df_mtm['Asset Category'] == 'Forex'].copy()
    
    df_cash['quantity'] = pd.to_numeric(df_cash['Prior Quantity'], errors='coerce').fillna(0)
    df_cash['symbol'] = df_cash['Symbol']  # Usually 'EUR', 'USD', etc.
    df_cash['currency'] = df_cash['Symbol']
    df_cash['value_native'] = df_cash['quantity'] # Cash value is just the quantity
    df_cash['asset_category'] = 'Cash'
    
    df_cash_final = df_cash[cols]

    # -----------------------
    # 3. Combine
    # -----------------------
    return pd.concat([df_stocks_final, df_cash_final], ignore_index=True)


def parse_event_log(df_raw, df_trades):
    """
    Parses all transactional sections into a single master event log.
    """
    all_events = []
    
    # --- Parse Stocks ---
    if df_trades is not None:
        df_trades_orders = df_trades[df_trades['DataDiscriminator'] == 'Order'].copy()
        
        # --- Stock Trades ---
        df_stock_trades = df_trades_orders[df_trades_orders['Asset Category'] == 'Stocks'].copy()
        
        # 1. Apply Symbol Cleaning
        df_stock_trades['Symbol'] = df_stock_trades['Symbol'].apply(clean_ibkr_symbol)
        
        # 2. Drop rows where Symbol became None (excluded) or Date is missing
        df_stock_trades = df_stock_trades.dropna(subset=['Symbol', 'Date/Time'])
        
        # 3. Pre-convert numerics (Handles commas strings efficiently)
        df_stock_trades['Quantity'] = pd.to_numeric(
            df_stock_trades['Quantity'].astype(str).str.replace(',', ''), 
            errors='coerce'
        ).fillna(0)
        
        df_stock_trades['Proceeds'] = pd.to_numeric(df_stock_trades['Proceeds'], errors='coerce').fillna(0)
        df_stock_trades['Comm/Fee'] = pd.to_numeric(df_stock_trades['Comm/Fee'], errors='coerce').fillna(0)

        # This handles "2025-02-03, 09:03:28"
        df_stock_trades['Date/Time'] = pd.to_datetime(
            df_stock_trades['Date/Time'], 
            format='%Y-%m-%d, %H:%M:%S'
        )

        # 4. Build Events
        for _, row in df_stock_trades.iterrows():
            timestamp = row['Date/Time']
            
            quantity_change = row['Quantity']
            
            all_events.append({
                'timestamp': timestamp,
                'event_type': 'TRADE_BUY' if quantity_change > 0 else 'TRADE_SELL',
                'symbol': row['Symbol'],
                'quantity_change': quantity_change,
                'cash_change_native': row['Proceeds'] + row['Comm/Fee'],
                'currency': row['Currency']
            })



        # --- Parse Forex ---
        df_fx_trades = df_trades_orders[df_trades_orders['Asset Category'] == 'Forex'].copy()
        
        # 1. Vectorized Symbol Parsing
        # IBKR Forex symbols are "BASE.QUOTE" (e.g. EUR.USD). 
        # We split them into two columns immediately.
        symbol_split = df_fx_trades['Symbol'].astype(str).str.split('.', n=1, expand=True)
        
        # Safety Check: Ensure the split actually resulted in 2 columns. 
        # If bad data exists (e.g. "EUR"), strict checking prevents crashing.
        if symbol_split.shape[1] == 2:
            df_fx_trades['curr_base'] = symbol_split[0]
            df_fx_trades['curr_quote'] = symbol_split[1]
            
            # Filter out rows where split failed (NaNs) or Date is missing
            df_fx_trades = df_fx_trades.dropna(subset=['curr_base', 'curr_quote', 'Date/Time'])
            
            # 2. Pre-convert Numerics
            df_fx_trades['Quantity'] = pd.to_numeric(
                df_fx_trades['Quantity'].astype(str).str.replace(',', ''), 
                errors='coerce'
            ).fillna(0)
            
            df_fx_trades['Proceeds'] = pd.to_numeric(
                df_fx_trades['Proceeds'].astype(str).str.replace(',', ''), 
                errors='coerce'
            ).fillna(0)

            # Vectorized conversion before the loop
            # Matches "2025-03-04, 22:13:15"
            df_fx_trades['Date/Time'] = pd.to_datetime(
                df_fx_trades['Date/Time'], 
                format='%Y-%m-%d, %H:%M:%S'
            )

            # 3. Build Events 
            for _, row in df_fx_trades.iterrows():
                timestamp = row['Date/Time']
                
                # Side A: The Base Currency (Quantity column applies here)
                all_events.append({
                    'timestamp': timestamp,
                    'event_type': 'FX_TRADE',
                    'symbol': row['curr_base'],
                    'quantity_change': 0,
                    'cash_change_native': row['Quantity'], 
                    'currency': row['curr_base']
                })
                
                # Side B: The Quote Currency (Proceeds column applies here)
                all_events.append({
                    'timestamp': timestamp,
                    'event_type': 'FX_TRADE',
                    'symbol': row['curr_quote'],
                    'quantity_change': 0,
                    'cash_change_native': row['Proceeds'],
                    'currency': row['curr_quote']
                })



    # --- Parse Dividends & Withholding Tax ---
    # Regex: Capture start word, ignore optional .Suffix, expect opening parenthesis
    extract_symbol_regex = r'^([A-Za-z0-9]+)(?:\.[A-Za-z0-9]+)?\('

    for section_name, event_type in [('Dividends', 'DIVIDEND'), ('Withholding Tax', 'TAX')]:
            
        df_section = get_section(df_raw, section_name)
            
        if df_section is not None:
            # 1. Clean Numerics
            df_section['Amount'] = pd.to_numeric(df_section['Amount'], errors='coerce').fillna(0)
            
            # Matches "24/04/2025" -> 2025-04-24 00:00:00
            df_section['Date'] = pd.to_datetime(
                df_section['Date'], 
                format='%d/%m/%Y'
            )

            # Drop rows where Date parsing failed (or was originally missing)
            df_section = df_section.dropna(subset=['Date'])
                
            # 2. Extract Symbols from Description
            df_section['Description'] = df_section['Description'].astype(str).str.strip()
            df_section['parsed_symbol'] = df_section['Description'].str.extract(extract_symbol_regex)[0]
                
            # 3. Apply Cleaner
            df_section['parsed_symbol'] = df_section['parsed_symbol'].apply(clean_ibkr_symbol)
                
            # 4. Build Events
            for _, row in df_section.iterrows():
                symbol = row['parsed_symbol']
                    
                if pd.isna(symbol):
                        continue

                all_events.append({
                    'timestamp': row['Date'],
                    'event_type': event_type,
                    'symbol': symbol,
                    'quantity_change': 0,
                    'cash_change_native': row['Amount'],
                    'currency': row['Currency']
                })



    # --- Parse Interest ---
    df_interest = get_section(df_raw, 'Interest')
        
    if df_interest is not None:
        # 1. Clean Numerics
        df_interest['Amount'] = pd.to_numeric(df_interest['Amount'], errors='coerce').fillna(0)
        
        # Matches "04/06/2025" -> 2025-06-04 00:00:00
        df_interest['Date'] = pd.to_datetime(
            df_interest['Date'], 
            format='%d/%m/%Y'
        )

        # Drop rows where Date is missing (or became NaT if parsing failed)
        df_interest = df_interest.dropna(subset=['Date'])
            
        # 2. Build Events
        for _, row in df_interest.iterrows():
            all_events.append({
                'timestamp': row['Date'],
                'event_type': 'INTEREST',
                'symbol': row['Currency'], 
                'quantity_change': 0,
                'cash_change_native': row['Amount'],
                'currency': row['Currency']
            })



    # --- 3. Parse Fees ---

    # --- 3a. General Fees (Account Level) ---
    # These are monthly fees, market data subscriptions, etc.
    df_fees = get_section(df_raw, 'Fees')
    if df_fees is not None:
        # 1. Clean Numerics
        df_fees['Amount'] = pd.to_numeric(df_fees['Amount'], errors='coerce').fillna(0)

        # 2. Parse Dates with Safety
        # errors='coerce' turns the empty dates in "Total" rows into NaT
        df_fees['Date'] = pd.to_datetime(
            df_fees['Date'], 
            format='%d/%m/%Y',
            errors='coerce' 
        )

        # 3. Filter out Total rows
        # This removes "Total", "Total in CHF", etc. because they have no Date.
        df_fees = df_fees.dropna(subset=['Date'])

        # 4. Build Events
        for _, row in df_fees.iterrows():
            amount = row['Amount']
            # Logic: Positive amount = Rebate, Negative = Fee
            event_type = 'FEE_REBATE' if amount > 0 else 'FEE'
            
            all_events.append({
                'timestamp': row['Date'],
                'event_type': event_type,
                'symbol': row['Currency'],
                'quantity_change': 0,
                'cash_change_native': amount,
                'currency': row['Currency']
            })

    # --- 3b. Transaction Fees (Trade Specific) ---
    df_trans_fees = get_section(df_raw, 'Transaction Fees')
    
    if df_trans_fees is not None:
        # 1. Clean Numerics
        df_trans_fees['Amount'] = pd.to_numeric(df_trans_fees['Amount'], errors='coerce').fillna(0)
        
        # 2. Apply Symbol Cleaning
        df_trans_fees['Symbol'] = df_trans_fees['Symbol'].apply(clean_ibkr_symbol)
        
        # Vectorized conversion matching "2025-03-18, 20:20:00"
        df_trans_fees['Date/Time'] = pd.to_datetime(
            df_trans_fees['Date/Time'], 
            format='%Y-%m-%d, %H:%M:%S'
        )

        # 3. Drop rows where Symbol or Date is missing
        df_trans_fees = df_trans_fees.dropna(subset=['Symbol', 'Date/Time'])

        # 4. Build Events
        for _, row in df_trans_fees.iterrows():
            amount = row['Amount']
            event_type = 'FEE_REBATE' if amount > 0 else 'FEE'
            
            all_events.append({
                'timestamp': row['Date/Time'],
                'event_type': event_type,
                'symbol': row['Symbol'],
                'quantity_change': 0,
                'cash_change_native': amount,
                'currency': row['Currency']
            })

    # --- 4. Parse Deposits & Withdrawals ---
    df_deposits = get_section(df_raw, 'Deposits & Withdrawals')
    
    if df_deposits is not None:
        # 1. Clean Numerics
        df_deposits['Amount'] = pd.to_numeric(
            df_deposits['Amount'].astype(str).str.replace(',', ''), 
            errors='coerce'
        ).fillna(0)
        
        # Vectorized conversion matching "25/02/2025"
        df_deposits['Settle Date'] = pd.to_datetime(
            df_deposits['Settle Date'], 
            format='%d/%m/%Y'
        )

        # Drop rows where Settle Date is missing
        df_deposits = df_deposits.dropna(subset=['Settle Date'])

        # 2. Build Events
        for _, row in df_deposits.iterrows():
            amount = row['Amount']
            event_type = 'DEPOSIT' if amount > 0 else 'WITHDRAWAL'
            
            all_events.append({
                'timestamp': row['Settle Date'],
                'event_type': event_type,
                'symbol': row['Currency'], 
                'quantity_change': 0, 
                'cash_change_native': amount,
                'currency': row['Currency']
            })

    # --- 5. Parse Grant Activity (Stock Awards) ---
    df_grants = get_section(df_raw, 'Grant Activity')
    
    if df_grants is not None:
        # 1. Clean Numerics
        df_grants['Quantity'] = pd.to_numeric(
            df_grants['Quantity'].astype(str).str.replace(',', ''), 
            errors='coerce'
        ).fillna(0)
        
        # 2. Apply Symbol Cleaning
        df_grants['Symbol'] = df_grants['Symbol'].apply(clean_ibkr_symbol)
        
        # Vectorized conversion matching "28/03/2025"
        df_grants['Vesting Date'] = pd.to_datetime(
            df_grants['Vesting Date'], 
            format='%d/%m/%Y'
        )

        # 3. Filter
        df_grants = df_grants.dropna(subset=['Symbol', 'Vesting Date'])

        # 4. Build Events
        for _, row in df_grants.iterrows():
            all_events.append({
                'timestamp': row['Vesting Date'],
                'event_type': 'GIFT_VEST', 
                'symbol': row['Symbol'],
                'quantity_change': row['Quantity'],
                'cash_change_native': 0, 
                'currency': 'USD' 
            })
            
# --- 6. Parse Corporate Actions ---
    df_corp_actions = get_section(df_raw, 'Corporate Actions')
    
    if df_corp_actions is not None:
        # 1. Clean Numerics & Dates
        df_corp_actions['Quantity'] = pd.to_numeric(
            df_corp_actions['Quantity'].astype(str).str.replace(',', ''), 
            errors='coerce'
        ).fillna(0)
        
        df_corp_actions['Proceeds'] = pd.to_numeric(
            df_corp_actions['Proceeds'].astype(str).str.replace(',', ''), 
            errors='coerce'
        ).fillna(0)

        # Vectorized conversion matching "2025-05-28, 20:25:00"
        df_corp_actions['Date/Time'] = pd.to_datetime(
            df_corp_actions['Date/Time'], 
            format='%Y-%m-%d, %H:%M:%S'
        )

        df_corp_actions = df_corp_actions.dropna(subset=['Date/Time'])

        # 2. Compile Regex Patterns
        # Pattern for Splits: "Split 4 for 1"
        split_regex = re.compile(r'Split\s+(\d+(?:\.\d+)?)\s+for\s+(\d+(?:\.\d+)?)')
        
        # Pattern for the "Target" stock in a Rights action.
        # Looks for: "Expire Dividend Right (TARGET, ..."
        # We capture the "TARGET" part.
        rights_target_regex = re.compile(r'Expire Dividend Right\s+\(([A-Za-z0-9]+)[,)]')
        
        # Pattern to grab the symbol at the very start of the string (for Splits)
        start_symbol_regex = re.compile(r'^([A-Za-z0-9]+)(?:\.[A-Za-z0-9]+)?\(')

        # 3. Row-by-Row Parsing (Necessary due to variable logic)
        for _, row in df_corp_actions.iterrows():
            desc = str(row['Description'])
            
            # --- Scenario A: Splits ---
            if "Split" in desc:
                split_match = split_regex.search(desc)
                if split_match:
                    # Calculate Ratio (e.g., 4 for 1 = 4.0)
                    # Note: IBKR descriptions are usually "New for Old". 
                    # If you hold 1 share and it splits 4 for 1, you now have 4.
                    numerator = float(split_match.group(1))
                    denominator = float(split_match.group(2))
                    ratio = numerator / denominator
                    
                    # Extract Symbol from start of string: "IBKR(US...)" -> "IBKR"
                    sym_match = start_symbol_regex.search(desc)
                    raw_symbol = sym_match.group(1) if sym_match else None
                    clean_sym = clean_ibkr_symbol(raw_symbol)
                    
                    if clean_sym:
                        all_events.append({
                            'timestamp': row['Date/Time'],
                            'event_type': 'SPLIT',
                            'symbol': clean_sym,
                            'quantity_change': 0, # Splits don't change quantity via "trade", they change the holding basis.
                            'cash_change_native': 0,
                            'currency': row['Currency'],
                            'split_ratio': ratio 
                        })

            # --- Scenario B: Dividend Rights (Conversion only) ---
            elif "Expire Dividend Right" in desc:
                # Logic: We only care if this expiration results in a STOCK.
                # Example 1 (Ignore): "...(VNA.DRTS, ...)" -> Cleaner sees dot -> Returns None -> Ignored.
                # Example 2 (Keep):   "...(VNAd, ...)"     -> Cleaner sees 'd' -> Returns "VNA" -> Kept.
                
                target_match = rights_target_regex.search(desc)
                if target_match:
                    raw_target = target_match.group(1)
                    clean_target = clean_ibkr_symbol(raw_target)
                    
                    # Only process if we have a valid stock symbol (no dots) and actual quantity change
                    if clean_target and row['Quantity'] != 0:
                        all_events.append({
                            'timestamp': row['Date/Time'],
                            'event_type': 'CORP_ACTION',
                            'symbol': clean_target,
                            'quantity_change': row['Quantity'],
                            'cash_change_native': row['Proceeds'],
                            'currency': row['Currency'],
                            'split_ratio': 1.0 # Default for non-splits
                        })
            
            # "Dividend Rights Issue" is explicitly ignored by not having an 'elif' block here.



# --- Finalize ---
    # Define the strict column order we expect in the output
    expected_cols = [
        'timestamp', 'event_type', 'symbol', 'currency', 
        'quantity_change', 'cash_change_native', 'split_ratio'
    ]

    if not all_events:
        # Return empty DataFrame with the correct schema
        return pd.DataFrame(columns=expected_cols)
    
    df_event_log = pd.DataFrame(all_events)

    # 1. Handle Split Ratio Consistency
    # If no splits occurred, the column might not exist. Create it.
    if 'split_ratio' not in df_event_log.columns:
        df_event_log['split_ratio'] = 1.0
    else:
        # If splits exist, other rows (Trades, Divs) will have NaN. Fill with 1.0.
        df_event_log['split_ratio'] = df_event_log['split_ratio'].fillna(1.0)

    # 2. Type Safety (Ensure numbers are floats, not objects)
    cols_to_numeric = ['quantity_change', 'cash_change_native', 'split_ratio']
    for col in cols_to_numeric:
        df_event_log[col] = pd.to_numeric(df_event_log[col], errors='coerce').fillna(0)

    # 3. Sort and Reorder
    df_event_log = df_event_log.sort_values(by='timestamp').reset_index(drop=True)
    
    # Reindex ensures consistent column order and adds any missing columns (filled with NaN/None)
    return df_event_log.reindex(columns=expected_cols)



# --- MAIN LOADER FUNCTION ---

def load_ibkr_report(filepath):
    """
    Loads an IBKR Activity Report CSV and parses it into key DataFrames
    and extracts the stated report period and base currency.
    """
    try:
        df_raw = pd.read_csv(
            filepath, header=None, dtype=object, 
            on_bad_lines='skip', low_memory=False
        )
        
        # --- Parse Report Period and Base Currency ---
        # default values
        report_start_date, report_end_date = None, None
        base_currency = 'CHF'
        
        try:
            # Parse Period
            period_row = df_raw[(df_raw[0] == 'Statement') & (df_raw[2] == 'Period')]
            period_string = period_row.iloc[0, 3] # e.g., "January 1, 2025 - October 29, 2025"
            start_str, end_str = period_string.split(' - ')
            
            # 1. Parse and Normalize (sets to 00:00:00)
            report_start_date = pd.to_datetime(start_str, format='%B %d, %Y').normalize()
            report_end_date = pd.to_datetime(end_str, format='%B %d, %Y').normalize()

            # 2. Add 24 hours to the end date
            # This turns "2025-10-29 00:00:00" into "2025-10-30 00:00:00"
            report_end_date = report_end_date + pd.Timedelta(days=1)
        except Exception as e:
            print(f"Warning: Could not parse report period string. Will use event log min/max. Error: {e}")

        try:
            # Parse Base Currency
            base_curr_row = df_raw[(df_raw[0] == 'Account Information') & (df_raw[2] == 'Base Currency')]
            base_currency = base_curr_row.iloc[0, 3] # e.g., "CHF"
        except Exception as e:
            print(f"Warning: Could not parse Base Currency. Error: {e}")

        # --- 1. Build Master Symbol Set & Currency Map ---
        master_symbol_set = set()
        symbol_currency_map = {}
        
        # --- Section A: Open Positions ---
        df_open_pos = get_section(df_raw, 'Open Positions')
        if df_open_pos is not None:
            df_open_pos_stocks = df_open_pos[df_open_pos['Asset Category'] == 'Stocks'].copy()
    
            # 1. Fill Currency gaps (IBKR format specific)
            df_open_pos_stocks['Currency'] = df_open_pos_stocks['Currency'].ffill()
    
            # 2. Apply Symbol Cleaning Rules
            df_open_pos_stocks['Symbol'] = df_open_pos_stocks['Symbol'].apply(clean_ibkr_symbol)
    
            # 3. Drop symbols that returned None (entries with dots)
            df_open_pos_stocks = df_open_pos_stocks.dropna(subset=['Symbol'])
    
            # 4. Update Map and Set
            map_from_open_pos = df_open_pos_stocks.set_index('Symbol')['Currency'].to_dict()
            symbol_currency_map.update(map_from_open_pos)
            master_symbol_set.update(df_open_pos_stocks['Symbol'].unique())

        # --- Section B: Trades ---
        df_trades = get_section(df_raw, 'Trades')
        if df_trades is not None:
            df_trade_stocks = df_trades[df_trades['Asset Category'] == 'Stocks'].copy()
    
            # 1. Apply Symbol Cleaning Rules
            df_trade_stocks['Symbol'] = df_trade_stocks['Symbol'].apply(clean_ibkr_symbol)
            df_trade_stocks = df_trade_stocks.dropna(subset=['Symbol'])
    
            # 2. Update Set
            master_symbol_set.update(df_trade_stocks['Symbol'].unique())
    
            # 3. Update Currency Map (only fill missing keys to prefer Open Positions data)
            df_trade_stocks_unique = df_trade_stocks.drop_duplicates(subset=['Symbol'])
            map_from_trades = df_trade_stocks_unique.set_index('Symbol')['Currency'].to_dict()
    
            for symbol, currency in map_from_trades.items():
                if symbol not in symbol_currency_map:
                    symbol_currency_map[symbol] = currency

        # --- Section C: Grant Activity ---
        df_grants = get_section(df_raw, 'Grant Activity')
        if df_grants is not None:
            # 1. Apply Cleaning
            df_grants['Symbol'] = df_grants['Symbol'].apply(clean_ibkr_symbol)
            df_grants = df_grants.dropna(subset=['Symbol'])
            
            # 2. Update Set
            master_symbol_set.update(df_grants['Symbol'].unique())
            
            # 3. Handle default IBKR currency
            if 'IBKR' not in symbol_currency_map:
                symbol_currency_map['IBKR'] = 'USD'

        # --- Section D: Mark-to-Market ---
        df_mtm = get_section(df_raw, 'Mark-to-Market Performance Summary')
        if df_mtm is not None:
            df_mtm_stocks = df_mtm[df_mtm['Asset Category'] == 'Stocks'].copy()
            
            # 1. Numeric Conversion
            df_mtm_stocks['Prior Quantity'] = pd.to_numeric(
                df_mtm_stocks['Prior Quantity'], errors='coerce'
            ).fillna(0)
            
            # 2. Filter first, then Clean (efficiency)
            held_at_start_df = df_mtm_stocks[df_mtm_stocks['Prior Quantity'] > 0].copy()
            held_at_start_df['Symbol'] = held_at_start_df['Symbol'].apply(clean_ibkr_symbol)
            held_at_start_df = held_at_start_df.dropna(subset=['Symbol'])
            
            # 3. Update Set
            master_symbol_set.update(held_at_start_df['Symbol'].unique())
        
        # --- 2. Parse Initial State ---
        df_initial_state = parse_initial_state(df_mtm, symbol_currency_map, master_symbol_set)

        # --- 3. Parse Event Log ---
        df_event_log = parse_event_log(df_raw, df_trades)
        
        # --- 4. Get Financial Info ---
        df_financial_info = get_section(df_raw, 'Financial Instrument Information')
        
        if df_financial_info is not None:
            # 1. Filter for specific columns (Use intersection to avoid KeyErrors)
            target_cols = ['Symbol', 'Security ID', 'Listing Exch']
            available_cols = [c for c in target_cols if c in df_financial_info.columns]
            df_financial_info = df_financial_info[available_cols].copy()
            
            # 2. Clean the Symbol
            # Apply the same logic as Trades/Dividends (e.g., remove 'd' suffix, drop dots)
            if 'Symbol' in df_financial_info.columns:
                df_financial_info['Symbol'] = df_financial_info['Symbol'].apply(clean_ibkr_symbol)
                
                # Drop rows where symbol became None (excluded assets)
                df_financial_info = df_financial_info.dropna(subset=['Symbol'])

            # 3. Rename columns
            # We rename 'Symbol' -> 'symbol' to match the Event Log and Initial State keys
            rename_map = {
                'Symbol': 'symbol',
                'Security ID': 'ISIN',
                'Listing Exch': 'Exchange'
            }
            df_financial_info = df_financial_info.rename(columns=rename_map)
            
            # 4. Deduplicate (Optional but recommended)
            # IBKR often lists the same symbol multiple times (once per exchange). 
            # We usually want unique Symbol -> ISIN mappings.
            df_financial_info = df_financial_info.drop_duplicates(subset=['symbol'])
            
        else:
            # Fallback if section is missing
            df_financial_info = pd.DataFrame(columns=['symbol', 'ISIN', 'Exchange'])
            
        return {
            'initial_state': df_initial_state,
            'events': df_event_log,
            'financial_info': df_financial_info,
            'report_start_date': report_start_date,
            'report_end_date': report_end_date,
            'base_currency': base_currency
        }
        
    except FileNotFoundError:
        print(f"Error: The file was not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"A fatal error occurred during loading: {e}")
        import traceback
        traceback.print_exc()
        return None