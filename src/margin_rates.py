"""
Margin Rate Data Provider (Hybrid).

Reconstructs the historical daily cost of borrowing capital (margin rates) for
specific target currencies using a hybrid strategy to maximize data coverage:

1. Primary Source (IBKR): Scrapes official historical margin rates directly
   from the Interactive Brokers website for recent months.
2. Secondary Source (FRED): Falls back to Federal Reserve Economic Data (FRED)
   proxies (e.g., Fed Funds Rate, ESTR) adjusted by a fixed spread for dates
   prior to IBKR's public history.
3. Gap Filling: Fills missing data points (weekends, holidays) using forward-filling
   (last known rate) and backward-filling.

Output:
    Normalized DataFrame of daily interest multipliers, formatted for the
    simulation engine.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from dateutil.relativedelta import relativedelta
import time
from datetime import datetime
import io
from typing import Optional
from .config import CURRENCIES_365, TARGET_CURRENCIES, MARGIN_SPREADS, FRED_PROXIES

def get_ibkr_rates_hybrid(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Retrieves margin rates using a prioritized hybrid strategy.

    Combines recent scraped data with historical backfills to produce a
    continuous timeline of interest rates for 'TARGET_CURRENCIES'.

    Data Priority:
    1. Official IBKR Scraped Data.
    2. FRED Proxy Data + Spread (Backfill for older history).
    3. Forward/Backward Fill (Gap bridging for non-business days).

    Args:
        start_date (pd.Timestamp): The beginning of the required data window.
        end_date (pd.Timestamp): The end of the required data window.

    Returns:
        pd.DataFrame: A DataFrame indexed by Date, with columns for each target
        currency. Values represent the *daily* interest rate factor (e.g., 0.00015).
    """
    print(f" [>] Fetching Margin Rates ({start_date.date()} to {end_date.date()})...")
    time.sleep(0.5)
    
    # 1. Scrape Recent Data
    # Attempts to fetch exact historical rates from the broker's public website.
    df_scraped = _get_daily_margin_rates(start_date, end_date)
    
    if df_scraped.empty:
        # If scraping completely fails, set the cutoff date to the future 
        # so the entire period is covered by the backfill.
        min_scraped_date = end_date + pd.Timedelta(days=1)
        print(" [!] Warning: IBKR Scraper returned no data. Switching to full FRED backfill.")
        time.sleep(0.5)
    else:
        # Identify the earliest date for which we have official data.
        min_scraped_date = df_scraped.index.min()
        print(f"     - Scraper returned data from: {min_scraped_date.date()}")
        time.sleep(0.5)

    # 2. Backfill Rest with FRED
    # If the official data doesn't cover the start of the simulation, fill the 
    # preceding period with proxy data (Central Bank rates + spread).
    if min_scraped_date > start_date:
        backfill_end = min_scraped_date - pd.Timedelta(days=1)
        print(f"     - Backfilling history from {start_date.date()} to {backfill_end.date()} using FRED...")
        time.sleep(0.5)
        df_backfill = _get_fred_proxy_rates(start_date, backfill_end)
        
        # Combine: Concatenate backfill (oldest) with scraped data (newest).
        df_final = pd.concat([df_backfill, df_scraped])
    else:
        df_final = df_scraped

    # 3. Gap Filling Strategy
    # The raw data often has gaps (weekends, holidays) or missing columns.
    # We must normalize the DataFrame structure before filling values.
    
    # A. Align Date Index
    # Reindex to a complete daily frequency to expose missing days as NaNs.
    full_date_range = pd.date_range(start_date, end_date, freq='D')
    df_final = df_final.reindex(full_date_range)
    
    # B. Reindex Columns
    # Ensure every target currency exists as a column, filling missing ones with NaN initially.
    df_final = df_final.reindex(columns=TARGET_CURRENCIES)
    
    # C. Fill Gaps
    # Logic: 
    # 1. ffill(): Propagate Friday's rate to Saturday/Sunday.
    # 2. bfill(): If the dataset starts with NaNs, look ahead for the first valid rate.
    # 3. fillna(0.0): Final safety net for currencies with absolutely no data (free borrowing assumed).
    df_final = df_final.ffill().bfill().fillna(0.0)
    
    print(" [+] Margin rates loaded successfully.")
    time.sleep(0.5)

    return df_final.loc[start_date:end_date]


def _get_daily_margin_rates(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Scrapes official historical margin rates from the Interactive Brokers website.

    Iterates through requested months, constructing URLs for the IBKR 'monthlyInterestRates'
    endpoint. Parses returned HTML tables to extract benchmark rates for target currencies.
    
    Includes exponential backoff for network errors or rate limits.

    Args:
        start_date (pd.Timestamp): The start date for the scrape range.
        end_date (pd.Timestamp): The end date for the scrape range.

    Returns:
        pd.DataFrame: A DataFrame containing the scraped rates, pivoted so that
        currencies are columns and dates are the index. Returns an empty DataFrame
        on failure.
    """
    base_url = "https://www.interactivebrokers.com/en/accounts/fees/monthlyInterestRates.php"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    # Generate list of "YYYYMM" strings for the requested range.
    cursor = start_date.replace(day=1)
    months_to_scrape = []
    while cursor <= end_date:
        months_to_scrape.append(cursor.strftime("%Y%m"))
        cursor += relativedelta(months=1)
    
    all_data = []
    
    for month_str in months_to_scrape:
        # Optimization: Don't attempt to scrape future months.
        if month_str > datetime.now().strftime("%Y%m"): continue

        url = f"{base_url}?date={month_str}&ib_entity=llc"
        
        # Network Retry Logic (Exponential Backoff)
        resp = None
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                time.sleep(1 + attempt)
        
        if resp is None or resp.status_code != 200:
            print(f" [!] Warning: Failed to fetch margin data for {month_str}")
            time.sleep(0.5)
            continue

        try:
            # HTML Parsing
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find('table', class_='table table-freeze-col table-bordered table-striped')
            if not table: continue 

            thead = table.find('thead')
            if not thead: continue
            
            # Extract headers to identify column indices.
            headers_list = [th.text.strip().upper() for th in thead.find_all('th')]

            # Locate the 'DATE' column dynamically.
            try:
                date_idx = next(i for i, h in enumerate(headers_list) if 'DATE' in h)
            except StopIteration: continue

            # Detect Table Format:
            # "Wide" format has currencies as columns. "Narrow" format has BENCHMARK columns.
            is_wide = not any('BENCHMARK' in h for h in headers_list)

            tbody = table.find('tbody')
            if not tbody: continue
            rows = tbody.find_all('tr')
            
            if is_wide:
                # Map column indices to currency codes (e.g., Col 2 -> 'USD', Col 3 -> 'EUR')
                col_map = {}
                for i, h in enumerate(headers_list):
                    curr = h.split('(')[0].strip() # Clean header "USD (BM)" -> "USD"
                    if curr in TARGET_CURRENCIES:
                        col_map[i] = curr
                
                # Iterate rows to extract rates
                for row in rows:
                    cols = [td.text.strip() for td in row.find_all('td')]
                    if not cols: continue
                    date_str = cols[date_idx]
                    
                    for c_idx, curr in col_map.items():
                        if c_idx >= len(cols): continue
                        # Parse the specific cell value
                        rate = _parse_ibkr_rate(cols[c_idx], curr)
                        if rate is not None:
                            all_data.append({'date': pd.to_datetime(date_str), 'currency': curr, 'rate': rate})
            
        except Exception as e:
            print(f" [!] Warning: Error parsing HTML for {month_str}: {e}")
            time.sleep(0.5)
            continue
            
    if not all_data: return pd.DataFrame()
    
    # Pivot Data: Transform list of records into a Time-Series DataFrame.
    # Rows = Dates, Columns = Currencies.
    df = pd.DataFrame(all_data)
    return df.pivot_table(index='date', columns='currency', values='rate', aggfunc='mean')


def _get_fred_proxy_rates(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Downloads and normalizes historical benchmark rates from FRED.

    Fallback engine. Fetches proxy rates (defined in config) for each currency,
    such as the Effective Federal Funds Rate for USD.

    Transforms raw benchmark data into a proxy for margin rates by:
    1. Forward-filling monthly/weekly data to daily frequency.
    2. Adding a defined 'Margin Spread' (from config) to approximate broker markup.
    3. Converting annualized percentage rates into daily multipliers based on
       day-count conventions (360 vs 365 days).

    Args:
        start_date (pd.Timestamp): The start date for the backfill.
        end_date (pd.Timestamp): The end date for the backfill.

    Returns:
        pd.DataFrame: A DataFrame of daily proxy margin rates.
    """
    if start_date > end_date: return pd.DataFrame()

    # Initialize container with the target full date range.
    proxy_data = pd.DataFrame(index=pd.date_range(start_date, end_date))
    
    for currency, fred_code in FRED_PROXIES.items():
        # Only process currencies relevant to the user's configuration.
        if currency not in TARGET_CURRENCIES: continue
        
        try:
            # Construct FRED API URL for CSV download.
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_code}"
            
            r = None
            for attempt in range(3):
                try:
                    r = requests.get(url, timeout=10)
                    if r.status_code == 200:
                        break 
                except requests.exceptions.RequestException:
                    time.sleep(1 + attempt)

            if r is None or r.status_code != 200:
                print(f" [!] Warning: Failed to fetch FRED data for {currency}")
                time.sleep(0.5)
                continue

            # Load CSV into Pandas directly from memory buffer.
            series_df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True, na_values='.')
            
            # Align downloaded data to the master date index.
            series = series_df.reindex(proxy_data.index)
            
            # Data Normalization:
            # FRED data (e.g. Fed Funds) often has gaps or is reported monthly.
            # Forward Fill propagates the last valid rate to cover subsequent days.
            series = series.ffill()
            
            # Determine Day-Count Convention (ACT/365 vs ACT/360).
            # GBP/HKD/AUD typically use 365, USD/EUR/CHF use 360.
            divisor = 365.0 if currency in CURRENCIES_365 else 360.0

            # Calculate Spread:
            # Start with the default spread, then check if there is a currency-specific override.
            default_spread = MARGIN_SPREADS.get('DEFAULT', 0.0)
            spread = MARGIN_SPREADS.get(currency, default_spread)

            # Calculation: (Benchmark + Spread) / 100 (to decimal) / DaysPerYear
            daily_rate = (series + spread) / 100.0 / divisor
            proxy_data[currency] = daily_rate
            
        except Exception as e:
            print(f" [!] Warning: Error processing FRED data for {currency}: {e}")
            time.sleep(0.5)
            continue

    return proxy_data

def _parse_ibkr_rate(rate_str: str, currency: str) -> Optional[float]:
    """
    Parses and standardizes a raw rate string from the IBKR HTML table.

    Converts strings like "1.58%" or "(0.50%)" into float representations.
    Applies the broker spread and adjusts for the specific currency's
    day-count convention (ACT/360 or ACT/365) to return a daily multiplier.

    Args:
        rate_str (str): The raw text string from the HTML cell.
        currency (str): The currency code (used to determine day-count logic).

    Returns:
        Optional[float]: The calculated daily interest rate, or None if parsing fails.
    """
    # Clean String: Remove %, parentheses (accounting negative), and whitespace.
    clean = rate_str.replace('%', '').replace('(', '-').replace(')', '').strip()
    
    if not clean or clean == '-': return None
    try:
        val = float(clean)
        
        # Select divisor based on financial convention for the currency.
        divisor = 365.0 if currency in CURRENCIES_365 else 360.0
        
        # Apply broker markup (spread).
        default_spread = MARGIN_SPREADS.get('DEFAULT', 0.0)
        spread = MARGIN_SPREADS.get(currency, default_spread)
        
        # Formula: (Rate% + Spread%) / 100 / Days
        return (val + spread) / 100.0 / divisor
    except:
        return None