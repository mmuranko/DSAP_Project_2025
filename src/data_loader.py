"""
IBKR Activity Report Parser.

Handles ingestion and transformation of Interactive Brokers (IBKR) Activity Flex
Queries (CSV format). Parses the "stacked" CSV structure—where multiple distinct
tables exist within one file—and normalizes the data into a coherent financial
timeline.

Outputs a 'Data Package' containing:
1. Initial State: Portfolio snapshot at report start date.
2. Event Log: Chronological sequence of trades, dividends, fees, and corporate actions.
3. Metadata: Financial instrument details and report period boundaries.

Acts as the ETL (Extract, Transform, Load) layer for the simulation engine.
"""
import pandas as pd
import re
import time
from typing import Any, Optional, Set, Dict, Union

# ==========================================
# SECTION 1: GENERIC UTILITIES
# ==========================================
def _get_section(df_raw: pd.DataFrame, section_name: str) -> Optional[pd.DataFrame]:
    """
    Extracts a specific sub-table from the stacked IBKR CSV based on section headers.

    Locates the specific header row for a given section (e.g., 'Trades', 'Open Positions'),
    extracts the subsequent data rows, cleans empty columns, and resets the index.

    Args:
        df_raw (pd.DataFrame): The full, raw CSV dataframe loaded with no header.
        section_name (str): The specific section title to extract (column 0 in raw data).

    Returns:
        Optional[pd.DataFrame]: Clean DataFrame of the requested section, or None 
        if section does not exist or parsing fails.
    """
    try:
        # Locates the specific row index where column 0 matches the section name 
        # and column 1 identifies it as the 'Header' row.
        header_row_index = df_raw[
            (df_raw[0] == section_name) & (df_raw[1] == 'Header')
        ].index[0]
        
        # Extracts header row values to use as column names for the new DataFrame.
        column_names = df_raw.iloc[header_row_index].values

        # Filters for rows belonging to this section tagged as 'Data'.
        data_rows = df_raw[
            (df_raw[0] == section_name) & (df_raw[1] == 'Data')
        ]
        
        df_section = pd.DataFrame(data_rows.values, columns=column_names)
        
        # Data Cleaning: 
        # 1. dropna(axis=1, how='all') removes columns that are purely empty (spacer cols).
        # 2. reset_index drops the original CSV row numbers to create a clean 0..n index.
        df_section = df_section.dropna(axis=1, how='all').reset_index(drop=True)
        
        # Returns all rows, but skips the first two columns (metadata columns).
        return df_section.iloc[:, 2:]
        
    except IndexError:
        return None
    except Exception as e:
        print(f" [!] Warning: Error parsing section '{section_name}': {e}")
        time.sleep(0.5)
        return None

def _clean_symbol(symbol: Any) -> Optional[str]:
    """
    Standardizes ticker symbols by removing descriptions and validating formatting.

    Handles IBKR quirks, such as symbols formatted as "SYMB(Description)".
    Preserves dot-notation tickers (e.g., 'BRK.B') but filters out invalid entries.

    Args:
        symbol (Any): The raw symbol input (usually a string or NaN).

    Returns:
        Optional[str]: Cleaned ticker string, or None if input was invalid/empty.
    """
    if pd.isna(symbol) or symbol == '':
        return None
    
    s = str(symbol).strip()
    
    # IBKR outputs "SYMB(Description)" in certain sections.
    # Splits on '(' and keeps the preceding part.
    if '(' in s:
        s = s.split('(')[0].strip()
    
    if s and s[-1].islower() and not '.' in s:
        # Heuristic: If string ends in lowercase without a dot, treat as stray char.
        return s[:-1]
        
    return s

def _clean_number(series: pd.Series) -> pd.Series:
    """
    Converts currency strings to floats using vectorized operations.

    Handles removal of thousands separators (commas) and coerces errors 
    (non-numeric strings) to NaN before filling with 0.

    Args:
        series (pd.Series): Pandas Series containing string representations of numbers.

    Returns:
        pd.Series: Numeric pandas Series.
    """
    # Uses .str accessor for vectorization instead of row looping.
    # 'errors="coerce"' transforms unparseable strings into NaN.
    return pd.to_numeric(
        series.astype(str).str.replace(',', '', regex=False), 
        errors='coerce'
    ).fillna(0)

# ==========================================
# SECTION 2: PARSING LOGIC
# ==========================================
def parse_initial_state(
    df_mtm: Optional[pd.DataFrame], 
    symbol_currency_map: Dict[str, str], 
    master_symbol_set: Set[str]
) -> pd.DataFrame:
    """
    Constructs the portfolio snapshot as it existed at the start of the report period.

    Uses 'Mark-to-Market Performance Summary' to determine prior quantities and prices.
    Validates assets against the master symbol set to ensure inclusion of only 
    relevant (traded or held) assets.

    Args:
        df_mtm (Optional[pd.DataFrame]): Parsed 'Mark-to-Market' section dataframe.
        symbol_currency_map (Dict[str, str]): Lookup dictionary for asset currencies.
        master_symbol_set (Set[str]): Set of all valid symbols found in the report.

    Returns:
        pd.DataFrame: DataFrame with columns ['symbol', 'asset_category', 'currency', 
        'quantity', 'value_native']. Returns empty DataFrame on failure.
    """
    if df_mtm is None:
        print(" [!] Warning: 'Mark-to-Market Performance Summary' section missing. Initial state empty.")
        time.sleep(0.5)
        return pd.DataFrame(columns=['symbol', 'asset_category', 'currency', 'quantity', 'value_native'])

    # --- 1. Process Stocks ---
    df_stocks = df_mtm[df_mtm['Asset Category'] == 'Stocks'].copy()
    
    # Applies custom function element-wise because 'clean_symbol' contains 
    # conditional logic not easily vectorized.
    df_stocks['symbol'] = df_stocks['Symbol'].apply(_clean_symbol)
    
    # Checks membership against large set of known symbols efficiently.
    mask_valid = (df_stocks['symbol'].notna()) & (df_stocks['symbol'].isin(master_symbol_set))
    df_stocks = df_stocks[mask_valid].copy()

    df_stocks['quantity'] = pd.to_numeric(df_stocks['Prior Quantity'], errors='coerce').fillna(0)
    prior_price = pd.to_numeric(df_stocks['Prior Price'], errors='coerce').fillna(0)
    
    df_stocks['value_native'] = df_stocks['quantity'] * prior_price
    
    # Uses .map() as vectorized lookup (similar to VLOOKUP).
    # Pulls values from 'symbol_currency_map' using 'symbol' column keys.
    df_stocks['currency'] = df_stocks['symbol'].map(symbol_currency_map).fillna('Unknown')
    df_stocks['asset_category'] = 'Stock'

    cols = ['symbol', 'asset_category', 'currency', 'quantity', 'value_native']
    df_stocks_final = df_stocks[cols]

    # --- 2. Process Forex (Cash) ---
    df_cash = df_mtm[df_mtm['Asset Category'] == 'Forex'].copy()
    
    df_cash['quantity'] = pd.to_numeric(df_cash['Prior Quantity'], errors='coerce').fillna(0)
    df_cash['symbol'] = df_cash['Symbol']
    df_cash['currency'] = df_cash['Symbol']
    df_cash['value_native'] = df_cash['quantity']
    df_cash['asset_category'] = 'Cash'
    
    df_cash_final = df_cash[cols]

    return pd.concat([df_stocks_final, df_cash_final], ignore_index=True)


def parse_event_log(df_raw: pd.DataFrame, df_trades: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregates portfolio activity into a single chronological timeline.

    Extracts and unifies data from multiple CSV sections:
    - Trades (Stocks and Forex)
    - Dividends and Withholding Taxes
    - Interest Income/Expense
    - General and Transaction Fees
    - Deposits and Withdrawals
    - Corporate Actions (Splits, Rights)

    Args:
        df_raw (pd.DataFrame): Full raw CSV data (for accessing non-Trade sections).
        df_trades (Optional[pd.DataFrame]): Specific 'Trades' section dataframe.

    Returns:
        pd.DataFrame: Standardized event log sorted by timestamp, containing
        columns for event type, symbol, cash changes, and quantity changes.
    """
    all_events = []
    
    # --- A. Parse Trades ---
    if df_trades is not None:
        df_trades_orders = df_trades[df_trades['DataDiscriminator'] == 'Order'].copy()
        
        # 1. STOCK TRADES
        df_stock_trades = df_trades_orders[df_trades_orders['Asset Category'] == 'Stocks'].copy()
        df_stock_trades['Symbol'] = df_stock_trades['Symbol'].apply(_clean_symbol)
        df_stock_trades = df_stock_trades.dropna(subset=['Symbol', 'Date/Time'])
        
        # Passes entire Series to _clean_number rather than using .apply().
        df_stock_trades['Quantity'] = _clean_number(df_stock_trades['Quantity'])
        
        df_stock_trades['Proceeds'] = pd.to_numeric(df_stock_trades['Proceeds'], errors='coerce').fillna(0)
        df_stock_trades['Comm/Fee'] = pd.to_numeric(df_stock_trades['Comm/Fee'], errors='coerce').fillna(0)

        # Defines format string explicitly to speed up parsing.
        df_stock_trades['Date/Time'] = pd.to_datetime(
            df_stock_trades['Date/Time'], 
            format='%Y-%m-%d, %H:%M:%S'
        )

        # Iterates via iterrows() to construct standardized list of dictionaries 
        # (schema normalization) from disparate sources.
        for _, row in df_stock_trades.iterrows():
            quantity_change = row['Quantity']
            cash_impact = row['Proceeds'] + row['Comm/Fee']
            
            all_events.append({
                'timestamp': row['Date/Time'],
                'event_type': 'TRADE_BUY' if quantity_change > 0 else 'TRADE_SELL',
                'symbol': row['Symbol'],
                'quantity_change': quantity_change,
                'cash_change_native': cash_impact,
                'currency': row['Currency']
            })

        # 2. FOREX TRADES
        df_fx_trades = df_trades_orders[df_trades_orders['Asset Category'] == 'Forex'].copy()
        
        # Splits string with Expand to transform "EUR.USD" into two separate columns.
        symbol_split = df_fx_trades['Symbol'].astype(str).str.split('.', n=1, expand=True)
        
        if symbol_split.shape[1] == 2:
            df_fx_trades['curr_base'] = symbol_split[0]
            df_fx_trades['curr_quote'] = symbol_split[1]
            df_fx_trades = df_fx_trades.dropna(subset=['curr_base', 'curr_quote', 'Date/Time'])
            
            df_fx_trades['Quantity'] = _clean_number(df_fx_trades['Quantity'])
            df_fx_trades['Proceeds'] = _clean_number(df_fx_trades['Proceeds'])
            
            df_fx_trades['Date/Time'] = pd.to_datetime(df_fx_trades['Date/Time'], format='%Y-%m-%d, %H:%M:%S')

            for _, row in df_fx_trades.iterrows():
                # Event 1: Base Currency impact
                all_events.append({
                    'timestamp': row['Date/Time'],
                    'event_type': 'FX_TRADE',
                    'symbol': row['curr_base'],
                    'quantity_change': 0,
                    'cash_change_native': row['Quantity'], 
                    'currency': row['curr_base']
                })
                
                # Event 2: Quote Currency impact
                all_events.append({
                    'timestamp': row['Date/Time'],
                    'event_type': 'FX_TRADE',
                    'symbol': row['curr_quote'],
                    'quantity_change': 0,
                    'cash_change_native': row['Proceeds'],
                    'currency': row['curr_quote']
                })

    # --- B. Dividends & Tax ---
    extract_symbol_regex = r'^([A-Za-z0-9]+)(?:\.[A-Za-z0-9]+)?\('

    for section_name, event_type in [('Dividends', 'DIVIDEND'), ('Withholding Tax', 'TAX')]:
        df_section = _get_section(df_raw, section_name)
        if df_section is not None and not df_section.empty:
            df_section['Amount'] = pd.to_numeric(df_section['Amount'], errors='coerce').fillna(0)
            first_date = str(df_section['Date'].iloc[0])

            if '-' in first_date:
                df_section['Date'] = pd.to_datetime(df_section['Date'], format='%Y-%m-%d')
            else:
                df_section['Date'] = pd.to_datetime(df_section['Date'], format='%d/%m/%Y')
            df_section = df_section.dropna(subset=['Date'])
            
            # Applies regex to entire column using .str.extract() and returns captured groups.
            df_section['parsed_symbol'] = df_section['Description'].str.extract(extract_symbol_regex)[0]
            df_section['parsed_symbol'] = df_section['parsed_symbol'].apply(_clean_symbol)
                
            for _, row in df_section.iterrows():
                if pd.isna(row['parsed_symbol']): continue

                all_events.append({
                    'timestamp': row['Date'],
                    'event_type': event_type,
                    'symbol': row['parsed_symbol'],
                    'quantity_change': 0,
                    'cash_change_native': row['Amount'],
                    'currency': row['Currency']
                })

    # --- C. Interest ---
    df_interest = _get_section(df_raw, 'Interest')
    if df_interest is not None and not df_interest.empty:
        df_interest['Amount'] = pd.to_numeric(df_interest['Amount'], errors='coerce').fillna(0)

        first_date = str(df_interest['Date'].iloc[0])
        if '-' in first_date:
            df_interest['Date'] = pd.to_datetime(df_interest['Date'], format='%Y-%m-%d')
        else:
            df_interest['Date'] = pd.to_datetime(df_interest['Date'], format='%d/%m/%Y')
        df_interest = df_interest.dropna(subset=['Date'])
            
        for _, row in df_interest.iterrows():
            all_events.append({
                'timestamp': row['Date'],
                'event_type': 'INTEREST',
                'symbol': row['Currency'], 
                'quantity_change': 0,
                'cash_change_native': row['Amount'],
                'currency': row['Currency']
            })

    # --- D. Fees (General & Transaction) ---
    # 1. General Fees
    df_fees = _get_section(df_raw, 'Fees')
    if df_fees is not None and not df_fees.empty:
        df_fees['Amount'] = pd.to_numeric(df_fees['Amount'], errors='coerce').fillna(0)

        first_date = str(df_fees['Date'].iloc[0])
        if '-' in first_date:
            df_fees['Date'] = pd.to_datetime(df_fees['Date'], format='%Y-%m-%d', errors='coerce')
        else:
            df_fees['Date'] = pd.to_datetime(df_fees['Date'], format='%d/%m/%Y', errors='coerce')
        df_fees = df_fees.dropna(subset=['Date'])

        for _, row in df_fees.iterrows():
            event_type = 'FEE_REBATE' if row['Amount'] > 0 else 'FEE'
            all_events.append({
                'timestamp': row['Date'],
                'event_type': event_type,
                'symbol': row['Currency'],
                'quantity_change': 0,
                'cash_change_native': row['Amount'],
                'currency': row['Currency']
            })

    # 2. Transaction Fees
    df_trans_fees = _get_section(df_raw, 'Transaction Fees')
    if df_trans_fees is not None:
        df_trans_fees['Amount'] = pd.to_numeric(df_trans_fees['Amount'], errors='coerce').fillna(0)
        df_trans_fees['Symbol'] = df_trans_fees['Symbol'].apply(_clean_symbol)
        df_trans_fees['Date/Time'] = pd.to_datetime(df_trans_fees['Date/Time'], format='%Y-%m-%d, %H:%M:%S')
        df_trans_fees = df_trans_fees.dropna(subset=['Symbol', 'Date/Time'])

        for _, row in df_trans_fees.iterrows():
            all_events.append({
                'timestamp': row['Date/Time'],
                'event_type': 'FEE' if row['Amount'] < 0 else 'FEE_REBATE',
                'symbol': row['Symbol'],
                'quantity_change': 0,
                'cash_change_native': row['Amount'],
                'currency': row['Currency']
            })

    # --- E. Deposits & Withdrawals ---
    df_deposits = _get_section(df_raw, 'Deposits & Withdrawals')
    if df_deposits is not None and not df_deposits.empty:
        df_deposits['Amount'] = _clean_number(df_deposits['Amount'])
        
        first_date = str(df_deposits['Settle Date'].iloc[0])
        if '-' in first_date:
            df_deposits['Settle Date'] = pd.to_datetime(df_deposits['Settle Date'], format='%Y-%m-%d')
        else:
            df_deposits['Settle Date'] = pd.to_datetime(df_deposits['Settle Date'], format='%d/%m/%Y')
        df_deposits = df_deposits.dropna(subset=['Settle Date'])

        for _, row in df_deposits.iterrows():
            all_events.append({
                'timestamp': row['Settle Date'],
                'event_type': 'DEPOSIT' if row['Amount'] > 0 else 'WITHDRAWAL',
                'symbol': row['Currency'], 
                'quantity_change': 0, 
                'cash_change_native': row['Amount'],
                'currency': row['Currency']
            })

    # --- G. Corporate Actions (Splits/Rights) ---
    df_corp_actions = _get_section(df_raw, 'Corporate Actions')
    if df_corp_actions is not None:
        df_corp_actions['Quantity'] = _clean_number(df_corp_actions['Quantity'])
        df_corp_actions['Proceeds'] = _clean_number(df_corp_actions['Proceeds'])
        
        df_corp_actions['Date/Time'] = pd.to_datetime(df_corp_actions['Date/Time'], format='%Y-%m-%d, %H:%M:%S')
        df_corp_actions = df_corp_actions.dropna(subset=['Date/Time'])

        # Compiles regex patterns to improve efficiency within loop.
        split_regex = re.compile(r'Split\s+(\d+(?:\.\d+)?)\s+for\s+(\d+(?:\.\d+)?)')
        rights_target_regex = re.compile(r'Expire Dividend Right\s+\(([A-Za-z0-9]+)[,)]') 
        start_symbol_regex = re.compile(r'^([A-Za-z0-9]+)(?:\.[A-Za-z0-9]+)?\(')

        for _, row in df_corp_actions.iterrows():
            desc = str(row['Description'])
            
            if "Split" in desc:
                split_match = split_regex.search(desc)
                if split_match:
                    numerator = float(split_match.group(1))
                    denominator = float(split_match.group(2))
                    ratio = numerator / denominator
                    
                    sym_match = start_symbol_regex.search(desc)
                    clean_sym = _clean_symbol(sym_match.group(1) if sym_match else None)
                    
                    if clean_sym:
                        all_events.append({
                            'timestamp': row['Date/Time'],
                            'event_type': 'SPLIT',
                            'symbol': clean_sym,
                            'quantity_change': 0,
                            'cash_change_native': 0,
                            'currency': row['Currency'],
                            'split_ratio': ratio 
                        })

            elif "Expire Dividend Right" in desc:
                target_match = rights_target_regex.search(desc)
                if target_match:
                    clean_target = _clean_symbol(target_match.group(1))
                    if clean_target and row['Quantity'] != 0:
                        all_events.append({
                            'timestamp': row['Date/Time'],
                            'event_type': 'CORP_ACTION',
                            'symbol': clean_target,
                            'quantity_change': row['Quantity'],
                            'cash_change_native': row['Proceeds'],
                            'currency': row['Currency'],
                            'split_ratio': 1.0 
                        })

    # --- Final DataFrame Construction ---
    expected_cols = [
        'timestamp', 'event_type', 'symbol', 'currency', 
        'quantity_change', 'cash_change_native', 'split_ratio'
    ]

    if not all_events:
        return pd.DataFrame(columns=expected_cols)
    
    df_event_log = pd.DataFrame(all_events)
    
    if 'split_ratio' not in df_event_log.columns:
        df_event_log['split_ratio'] = 1.0
    else:
        df_event_log['split_ratio'] = df_event_log['split_ratio'].fillna(1.0)

    # Schema Enforcement: Sorts chronologically and ensures all expected columns exist 
    # (even if no split/corporate action events occurred in this specific file).
    df_event_log = df_event_log.sort_values(by='timestamp').reset_index(drop=True)
    return df_event_log.reindex(columns=expected_cols)

# ==========================================
# SECTION 3: MAIN PART
# ==========================================
def load_ibkr_report(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Loads, parses, and validates an IBKR Activity Report.

    Steps:
    1. Loads CSV file into memory.
    2. Extracts report metadata (Start Date, End Date, Base Currency).
    3. Discovers all unique symbols and their currencies across sections.
    4. Parses Initial State (Portfolio at T=0).
    5. Parses Event Log (Activity from T=0 to T=n).
    6. Extracts Financial Instrument Information (ISINs, Exchanges).
    7. Validates data integrity.

    Args:
        filepath (str): Absolute or relative path to the IBKR CSV file.

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing parsed data package
        keys ('initial_state', 'events', 'financial_info', etc.), or None if 
        parsing fails.
    """
    print(f"\n [>] Reading IBKR report: {filepath}...")
    time.sleep(0.5)

    try:
        # Adds 'names' to force a wide schema and prevent skipping wide rows.
        # Uses 'utf-8-sig' to handle potential BOM characters safely.
        df_raw = pd.read_csv(
            filepath, 
            header=None, 
            names=list(range(100)), # Creates 0..99 columns to fit widest rows
            dtype=object, 
            on_bad_lines='skip', 
            low_memory=False,
            encoding='utf-8-sig'    # Handles BOM (Byte Order Mark) often found in financial CSVs
        )

        # Strips whitespace from the Section Name column (Column 0)
        # to ensure strictly equal matches like df_raw[0] == 'Trades'.
        df_raw[0] = df_raw[0].str.strip()

        report_start_date, report_end_date = None, None
        base_currency = 'CHF' 
        
        try:
            period_row = df_raw[(df_raw[0] == 'Statement') & (df_raw[2] == 'Period')]
            period_string = str(period_row.iloc[0, 3])  # Forces cast to string
            start_str, end_str = period_string.split(' - ')
            
            report_start_date = pd.to_datetime(start_str, format='%B %d, %Y').normalize()
            report_end_date = pd.to_datetime(end_str, format='%B %d, %Y').normalize() + pd.Timedelta(days=1)
        except Exception as e:
            print(f" [!] Warning parsing dates: {e}")
            time.sleep(0.5)

        try:
            base_curr_row = df_raw[(df_raw[0] == 'Account Information') & (df_raw[2] == 'Base Currency')]
            base_currency = base_curr_row.iloc[0, 3]
        except Exception as e:
            print(f" [!] Warning parsing Base Currency: {e}")
            time.sleep(0.5)

        # --- 2. Symbol Discovery Phase ---

        master_symbol_set = set()
        symbol_currency_map = {}
        
        # A. Check Open Positions
        df_open_pos = _get_section(df_raw, 'Open Positions')
        if df_open_pos is not None:
            stocks = df_open_pos[df_open_pos['Asset Category'] == 'Stocks'].copy()
            
            # Forward Fill: Propagates the last valid Currency down, as IBKR reports 
            # often list Currency once until it changes.
            stocks['Currency'] = stocks['Currency'].ffill()
            
            stocks['Symbol'] = stocks['Symbol'].apply(_clean_symbol)
            stocks = stocks.dropna(subset=['Symbol'])
            
            # Efficiently converts 2 columns into a Key:Value dictionary.
            symbol_currency_map.update(stocks.set_index('Symbol')['Currency'].to_dict())
            master_symbol_set.update(stocks['Symbol'].unique())

        # B. Check Trades
        df_trades = _get_section(df_raw, 'Trades')
        if df_trades is not None:
            trade_stocks = df_trades[df_trades['Asset Category'] == 'Stocks'].copy()
            trade_stocks['Symbol'] = trade_stocks['Symbol'].apply(_clean_symbol)
            trade_stocks = trade_stocks.dropna(subset=['Symbol'])
            master_symbol_set.update(trade_stocks['Symbol'].unique())
            
            trade_map = trade_stocks.drop_duplicates(subset=['Symbol']).set_index('Symbol')['Currency'].to_dict()
            for sym, curr in trade_map.items():
                if sym not in symbol_currency_map:
                    symbol_currency_map[sym] = curr

        # C. Check Grants
        df_grants = _get_section(df_raw, 'Grant Activity')
        if df_grants is not None:
            df_grants['Symbol'] = df_grants['Symbol'].apply(_clean_symbol)
            master_symbol_set.update(df_grants['Symbol'].dropna().unique())
            if 'IBKR' not in symbol_currency_map: symbol_currency_map['IBKR'] = 'USD'

        # D. Check MtM
        df_mtm = _get_section(df_raw, 'Mark-to-Market Performance Summary')
        if df_mtm is not None:
            mtm_stocks = df_mtm[df_mtm['Asset Category'] == 'Stocks'].copy()
            mtm_stocks['Prior Quantity'] = pd.to_numeric(mtm_stocks['Prior Quantity'], errors='coerce').fillna(0)
            held = mtm_stocks[mtm_stocks['Prior Quantity'] > 0].copy()
            held['Symbol'] = held['Symbol'].apply(_clean_symbol)
            master_symbol_set.update(held['Symbol'].dropna().unique())
        
        # --- 3. Execution Phase ---
        
        df_initial_state = parse_initial_state(df_mtm, symbol_currency_map, master_symbol_set)
        df_event_log = parse_event_log(df_raw, df_trades)
        
        # Parse Reference Data
        df_financial_info = _get_section(df_raw, 'Financial Instrument Information')
        if df_financial_info is not None:
            target_cols = ['Underlying', 'Security ID', 'Listing Exch']
            cols_present = [c for c in target_cols if c in df_financial_info.columns]
            df_financial_info = df_financial_info[cols_present].copy()
            
            # Apply standard cleaning to the 'Underlying' column
            if 'Underlying' in df_financial_info.columns:
                df_financial_info['Underlying'] = df_financial_info['Underlying'].apply(_clean_symbol)
                df_financial_info = df_financial_info.dropna(subset=['Underlying'])

                # Rename 'Underlying' to 'symbol' to match internal schema.
                df_financial_info = df_financial_info.rename(columns={
                    'Underlying': 'symbol',
                    'Security ID': 'ISIN',
                    'Listing Exch': 'Exchange'
                })
                
                # Deduplicates based on clean symbol.
                df_financial_info = df_financial_info.drop_duplicates(subset=['symbol'])
            else:
                # Fallback if Underlying is missing.
                df_financial_info = pd.DataFrame(columns=['symbol', 'ISIN', 'Exchange'])
        else:
            df_financial_info = pd.DataFrame(columns=['symbol', 'ISIN', 'Exchange'])
        
        # --- 4. Critical Validation ---
        if report_start_date is None or report_end_date is None:
            print(" [!] CRITICAL ERROR: Could not parse Report Period (Start/End dates).")
            print("     The input file format might have changed.")
            time.sleep(0.5)
            return None

        if df_initial_state.empty and df_event_log.empty:
            print(" [!] CRITICAL ERROR: No positions or events found.")
            print("     Check if the CSV file is empty or corrupted.")
            time.sleep(0.5)
            return None

        if report_start_date and report_end_date:
            print(f"     - Period: {report_start_date.date()} to {report_end_date.date()}")
        print(f"     - Initial Positions: {len(df_initial_state)}")
        print(f"     - Total Events: {len(df_event_log)}")
        print(f" [+] Report loaded successfully.\n")
        time.sleep(0.5)

        return {
            'initial_state': df_initial_state,
            'events': df_event_log,
            'financial_info': df_financial_info,
            'report_start_date': report_start_date,
            'report_end_date': report_end_date,
            'base_currency': base_currency
        }
        
    except FileNotFoundError:
        print(f" [!] Error: The file was not found at '{filepath}'")
        time.sleep(0.5)
        return None
    except Exception as e:
        print(f" [!] Fatal error loading report: {e}")
        time.sleep(0.5)
        import traceback
        traceback.print_exc()
        return None