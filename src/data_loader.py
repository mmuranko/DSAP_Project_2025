import pandas as pd
import re

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
        
        column_names = df_raw.iloc[header_row_index]
        
        # Find all 'Data' rows for that section
        data_rows = df_raw[
            (df_raw[0] == section_name) & (df_raw[1] == 'Data')
        ]
        
        # Create the new DataFrame
        df_section = pd.DataFrame(data_rows.values, columns=column_names)
        
        # Clean up: Drop all-NaN columns and reset index
        df_section = df_section.dropna(axis=1, how='all').reset_index(drop=True)
        
        # Drop the first two columns (which are just 'Section', 'Data')
        return df_section.iloc[:, 2:]
        
    except IndexError:
        # This is not an error, just a section that doesn't exist
        return None
    except Exception as e:
        print(f"Error parsing section '{section_name}': {e}")
        return None

# --- PARSER FUNCTIONS (PUBLIC) ---

def parse_initial_state(df_mtm, symbol_currency_map, master_symbol_set):
    """
    Parses the Mark-to-Market to build the initial state DataFrame.
    Uses the pre-built symbol_currency_map to assign currencies.
    Uses master_symbol_set to filter out irrelevant "ghost" assets.
    """
    if df_mtm is None:
        print("Warning: 'Mark-to-Market Performance Summary' section not found. Cannot determine initial state.")
        return pd.DataFrame(columns=['symbol', 'asset_category', 'currency', 'quantity', 'value_native'])

    all_assets = []
    
    # Process Stocks
    df_mtm_stocks = df_mtm[df_mtm['Asset Category'] == 'Stocks'].copy()
    df_mtm_stocks['Prior Quantity'] = pd.to_numeric(df_mtm_stocks['Prior Quantity'], errors='coerce').fillna(0)
    df_mtm_stocks['Prior Price'] = pd.to_numeric(df_mtm_stocks['Prior Price'], errors='coerce').fillna(0)
    
    for _, row in df_mtm_stocks.iterrows():
        symbol = row['Symbol']
        
        # --- THIS IS THE FIX ---
        # Only add stocks that are in our master list of relevant assets
        if symbol in master_symbol_set:
            quantity = row['Prior Quantity']
            currency = symbol_currency_map.get(symbol, 'Unknown') 
            
            all_assets.append({
                'symbol': symbol,
                'asset_category': 'Stock',
                'currency': currency,
                'quantity': quantity,
                'value_native': quantity * row['Prior Price']
            })

    # Process Forex (Cash)
    df_mtm_cash = df_mtm[df_mtm['Asset Category'] == 'Forex'].copy()
    df_mtm_cash['Prior Quantity'] = pd.to_numeric(df_mtm_cash['Prior Quantity'], errors='coerce').fillna(0)

    for _, row in df_mtm_cash.iterrows():
        currency_symbol = row['Symbol']
        balance = row['Prior Quantity']
        
        all_assets.append({
            'symbol': currency_symbol,
            'asset_category': 'Cash',
            'currency': currency_symbol,
            'quantity': balance,
            'value_native': balance
        })
        
    df_state = pd.DataFrame(all_assets)
    return df_state[['symbol', 'asset_category', 'currency', 'quantity', 'value_native']]


def parse_event_log(df_raw, df_trades):
    """
    Parses all transactional sections into a single master event log.
    Receives df_trades to avoid loading it twice.
    """
    all_events = []
    
    # --- 1. Parse Trades (Stocks & Forex) ---
    if df_trades is not None:
        df_trades_orders = df_trades[df_trades['DataDiscriminator'] == 'Order'].copy()
        
        # Stock Trades
        df_stock_trades = df_trades_orders[df_trades_orders['Asset Category'] == 'Stocks'].copy()
        
        # --- FIX: Removed the lines that added new columns ---
        
        for _, row in df_stock_trades.iterrows():
            if pd.isna(row['Date/Time']): continue
            
            # --- FIX: Do all conversions and math *inside* the loop ---
            # --- using the original UPPERCASE column names ---
            quantity_change = pd.to_numeric(row['Quantity'].replace(',', ''), errors='coerce')
            proceeds = pd.to_numeric(row['Proceeds'], errors='coerce')
            comm_fee = pd.to_numeric(row['Comm/Fee'], errors='coerce')
            
            all_events.append({
                'timestamp': pd.to_datetime(row['Date/Time'], format='mixed'),
                'event_type': 'TRADE_BUY' if quantity_change > 0 else 'TRADE_SELL',
                'symbol': row['Symbol'],
                'quantity_change': quantity_change,
                'cash_change_native': proceeds + comm_fee,
                'currency': row['Currency']
            })
            
        # --- FIX: Forex Trades (This is the critical fix from last time) ---
        df_fx_trades = df_trades_orders[df_trades_orders['Asset Category'] == 'Forex'].copy()
        
        for _, row in df_fx_trades.iterrows():
            if pd.isna(row['Date/Time']): continue
            
            # Get the two currencies from the symbol, e.g., "EUR.CHF"
            try:
                curr_from, curr_to = row['Symbol'].split('.')
            except ValueError:
                print(f"Warning: Could not parse FX symbol '{row['Symbol']}'. Skipping row.")
                continue

            # The 'Quantity' is the amount of the "from" currency
            cash_change_from = pd.to_numeric(row['Quantity'].replace(',', ''), errors='coerce')
            
            # The 'Proceeds' is the amount of the "to" currency
            cash_change_to = pd.to_numeric(row['Proceeds'], errors='coerce')

            # Create the "FROM" currency event (e.g., selling EUR)
            all_events.append({
                'timestamp': pd.to_datetime(row['Date/Time'], format='mixed'),
                'event_type': 'FX_TRADE',
                'symbol': curr_from, # This is the symbol
                'quantity_change': 0, # It's a cash event, not a share change
                'cash_change_native': cash_change_from,
                'currency': curr_from # The currency of this cash event
            })
            
            # Create the "TO" currency event (e.g., buying CHF)
            all_events.append({
                'timestamp': pd.to_datetime(row['Date/Time'], format='mixed'),
                'event_type': 'FX_TRADE',
                'symbol': curr_to, # This is the symbol
                'quantity_change': 0,
                'cash_change_native': cash_change_to,
                'currency': curr_to # The currency of this cash event
            })

    # --- 2. Parse Dividends, Taxes, Interest ---
    # (This section is correct, no changes)
    for section_name, event_type in [('Dividends', 'DIVIDEND'), 
                                     ('Withholding Tax', 'TAX'),
                                     ('Interest', 'INTEREST')]:
        df_section = get_section(df_raw, section_name)
        if df_section is not None:
            df_section['Amount'] = pd.to_numeric(df_section['Amount'], errors='coerce')
            symbol_regex = re.compile(r'([A-Z]{2,6})\(')
            
            for _, row in df_section.iterrows():
                if pd.isna(row['Date']): continue
                
                description_str = str(row['Description'])
                symbol_match = symbol_regex.search(description_str)
                symbol = symbol_match.group(1) if symbol_match else None
                
                all_events.append({
                    'timestamp': pd.to_datetime(row['Date'], format='mixed'),
                    'event_type': event_type,
                    'symbol': symbol,
                    'quantity_change': 0,
                    'cash_change_native': row['Amount'],
                    'currency': row['Currency']
                })

    # --- 3. Parse Fees ---
    # (This section is correct, no changes)
    df_fees = get_section(df_raw, 'Fees')
    if df_fees is not None:
         for _, row in df_fees.iterrows():
            if pd.isna(row['Date']): continue
                
            amount = pd.to_numeric(row['Amount'], errors='coerce')
            event_type = 'FEE_REBATE' if amount > 0 else 'FEE'
            
            all_events.append({
                'timestamp': pd.to_datetime(row['Date'], format='mixed'),
                'event_type': event_type,
                'symbol': None,
                'quantity_change': 0,
                'cash_change_native': amount,
                'currency': row['Currency']
            })

    # --- 4. Parse Deposits & Withdrawals ---
    # (This section is correct, no changes)
    df_deposits = get_section(df_raw, 'Deposits & Withdrawals')
    if df_deposits is not None:
        for _, row in df_deposits.iterrows():
            if pd.isna(row['Settle Date']): continue
                
            amount = pd.to_numeric(row['Amount'], errors='coerce')
            event_type = 'DEPOSIT' if amount > 0 else 'WITHDRAWAL'
            
            all_events.append({
                'timestamp': pd.to_datetime(row['Settle Date'], format='mixed'),
                'event_type': event_type,
                'symbol': None,
                'quantity_change': 0,
                'cash_change_native': amount,
                'currency': row['Currency']
            })

    # --- 5. Parse Grant Activity (Stock Awards) ---
    # (This section is correct, no changes)
    df_grants = get_section(df_raw, 'Grant Activity')
    if df_grants is not None:
        for _, row in df_grants.iterrows():
            if pd.isna(row['Vesting Date']): continue
                
            all_events.append({
                'timestamp': pd.to_datetime(row['Vesting Date'], format='mixed'),
                'event_type': 'GIFT_VEST',
                'symbol': row['Symbol'],
                'quantity_change': pd.to_numeric(row['Quantity'], errors='coerce'),
                'cash_change_native': 0,
                'currency': 'USD' # IBKR is a USD stock
            })
            
# --- 6. Parse Corporate Actions ---
    # Use get_section() to extract the rows that start with 'Corp Actions' and 'Headers' or 'Data'
    df_corp_actions = get_section(df_raw, 'Corporate Actions')
    # Check if df_corp_actions is empty or not
    if df_corp_actions is not None:
        # interrows() yields a generator that once called returns a tuple (row index, pd Series of the row)
        # The "_" is a placeholder variable for the index
        for _, row in df_corp_actions.iterrows():
            if pd.isna(row['Date/Time']): continue
                
            description_str = str(row['Description'])
            
            # --- New Logic: Determine Event Type ---
            event_type = 'CORP_ACTION' # Default
            split_ratio = None        # Default
            
            if "Split" in description_str:
                event_type = 'SPLIT'
                # Parse the "X for Y" ratio
                split_match = re.search(r'Split (\d+) for (\d+)', description_str)
                if split_match:
                    # e.g., "4 for 1" -> split_ratio = 4 / 1 = 4.0
                    split_ratio = float(split_match.group(1)) / float(split_match.group(2))
            
            elif "Dividend Rights Issue" in description_str:
                event_type = 'RIGHTS_ISSUE'
            
            elif "Expire Dividend Right" in description_str:
                event_type = 'RIGHTS_EXPIRY'
            
            # --- New Filter: Skip the bookkeeping events ---
            # This is a cleaner way to filter than checking the symbol
            if event_type in ['RIGHTS_ISSUE', 'RIGHTS_EXPIRY']:
                continue
                
            # --- Symbol parsing (your correct regex) ---
            symbol_match = re.search(r'\(([^,)]+),', description_str)
            symbol = symbol_match.group(1) if symbol_match else None
            
            # --- Build the Event Dictionary ---
            # This dictionary will be added to the list
            event_data = {
                'timestamp': pd.to_datetime(row['Date/Time'], format='mixed'),
                'event_type': event_type, # Use the new, smart event_type
                'symbol': symbol,
                'currency': row['Currency']
            }
            
            # --- THIS IS THE FIX ---
            # Set quantity/cash based on the event type
            if event_type == 'SPLIT':
                event_data['split_ratio'] = split_ratio
                event_data['quantity_change'] = 0 # Don't log the redundant quantity
                event_data['cash_change_native'] = 0 # Splits are non-cash events
            else:
                # For any other corp action (e.g., VNAd share delivery)
                event_data['split_ratio'] = None # Ensure column exists
                event_data['quantity_change'] = pd.to_numeric(row['Quantity'], errors='coerce')
                event_data['cash_change_native'] = pd.to_numeric(row['Proceeds'], errors='coerce')
            
            all_events.append(event_data)

    # --- Finalize ---
    if not all_events:
        return pd.DataFrame(columns=['timestamp', 'event_type', 'symbol', 'quantity_change', 'cash_change_native', 'currency'])
        
    df_event_log = pd.DataFrame(all_events)
    df_event_log = df_event_log.sort_values(by='timestamp').reset_index(drop=True)
    
    return df_event_log

# --- MAIN LOADER FUNCTION (UPDATED) ---

def load_ibkr_report(filepath):
    """
    Loads an IBKR Activity Report CSV and parses it into two key DataFrames:
    1. df_initial_state: The portfolio holdings and cash at the start.
    2. df_event_log: A chronological log of all transactions and events.
    """
    try:
        # Load the entire file once, without a header
        df_raw = pd.read_csv(
            filepath,
            header=None,
            dtype=object, 
            on_bad_lines='skip',
            low_memory=False
        )
        
        # --- 1. Build Master Symbol Set & Currency Map ---
        master_symbol_set = set()
        symbol_currency_map = {}
        
        # Get from Open Positions (reliable for assets held at END)
        df_open_pos = get_section(df_raw, 'Open Positions')
        if df_open_pos is not None:
            df_open_pos_stocks = df_open_pos[df_open_pos['Asset Category'] == 'Stocks'].copy()
            df_open_pos_stocks['Currency'] = df_open_pos_stocks['Currency'].ffill()
            map_from_open_pos = df_open_pos_stocks.set_index('Symbol')['Currency'].to_dict()
            symbol_currency_map.update(map_from_open_pos)
            master_symbol_set.update(df_open_pos_stocks['Symbol'].unique())

        # Get from Trades (reliable for ALL assets ever traded)
        df_trades = get_section(df_raw, 'Trades')
        if df_trades is not None:
            df_trade_stocks = df_trades[df_trades['Asset Category'] == 'Stocks'].copy()
            master_symbol_set.update(df_trade_stocks['Symbol'].unique())
            
            # Drop duplicates to be efficient for map building
            df_trade_stocks_unique = df_trade_stocks.drop_duplicates(subset=['Symbol'])
            map_from_trades = df_trade_stocks_unique.set_index('Symbol')['Currency'].to_dict()
            
            # Update the main map *only* with new symbols.
            for symbol, currency in map_from_trades.items():
                if symbol not in symbol_currency_map:
                    symbol_currency_map[symbol] = currency

        # Get from Grant Activity
        df_grants = get_section(df_raw, 'Grant Activity')
        if df_grants is not None:
            master_symbol_set.update(df_grants['Symbol'].unique())
            # Add IBKR currency if not already present
            if 'IBKR' not in symbol_currency_map:
                 symbol_currency_map['IBKR'] = 'USD' # IBKR is a USD stock

        # Get from Mark-to-Market (for assets held at START)
        df_mtm = get_section(df_raw, 'Mark-to-Market Performance Summary')
        if df_mtm is not None:
            df_mtm_stocks = df_mtm[df_mtm['Asset Category'] == 'Stocks'].copy()
            df_mtm_stocks['Prior Quantity'] = pd.to_numeric(df_mtm_stocks['Prior Quantity'], errors='coerce').fillna(0)
            
            # Add any stock with a starting quantity > 0 to the master set
            held_at_start = df_mtm_stocks[df_mtm_stocks['Prior Quantity'] > 0]['Symbol'].unique()
            master_symbol_set.update(held_at_start)
        
        # --- 2. Parse Initial State ---
        # Pass the completed map AND the master set to the parser
        df_initial_state = parse_initial_state(df_mtm, symbol_currency_map, master_symbol_set)

# --- 3. Parse Event Log ---
        # Pass df_trades to avoid loading it again
        df_event_log = parse_event_log(df_raw, df_trades)
        
        # --- 4. Get Financial Info ---
        df_financial_info = get_section(df_raw, 'Financial Instrument Information')
        
        return {
            'initial_state': df_initial_state,
            'events': df_event_log,
            'financial_info': df_financial_info # <-- ADD THIS
        }
        
    except FileNotFoundError:
        print(f"Error: The file was not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"A fatal error occurred during loading: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Example of how to use this loader ---
if __name__ == "__main__":
    
    # Use the 'raw string' (r'...') for your Windows path
    # Example: file_path = r'C:\Users\Marvin\data\U13271915_20250101_20251029.csv'
    file_path = r'U13271915_20250101_20251029.csv' # Assumes file is in same directory
    
    # Set pandas display options for better printing
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 150)
    
    report_data = load_ibkr_report(file_path)
    
    if report_data:
        print("--- 1. Initial State DataFrame ---")
        print(report_data['initial_state'])
        print("\n" + "="*80 + "\n")
        
        print("--- 2. Master Event Log (First 20 Events) ---")
        print(report_data['events'].head(20))