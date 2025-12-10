"""
Margin Rate Data Provider (Hybrid).

This module is responsible for reconstructing the historical daily cost of borrowing
capital (margin rates) for specific target currencies. It employs a resilient
"Hybrid" strategy to ensure complete data coverage:

1. Primary Source (IBKR): It attempts to scrape precise, official historical 
   margin rates directly from the Interactive Brokers website for recent months.
2. Secondary Source (FRED): For dates prior to IBKR's public history, or if scraping 
   fails, it falls back to Federal Reserve Economic Data (FRED) proxies (e.g., 
   Fed Funds Rate, ESTR) adjusted by a fixed spread.
3. Gap Filling: Any remaining missing data points (weekends, holidays) are filled 
   using forward-filling (holding the last known rate) and backward-filling 
   (safety net).

The output is a normalized DataFrame of daily interest multipliers, ready for 
calculating margin expenses in the simulation engine.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from dateutil.relativedelta import relativedelta
import time
from datetime import datetime
import io
from typing import Optional, Dict, Any, Union
from .config import CURRENCIES_365, TARGET_CURRENCIES, MARGIN_SPREADS, FRED_PROXIES

def get_ibkr_rates_hybrid(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Orchestrates the retrieval of margin rates using a prioritized hybrid strategy.

    This function combines recent scraped data with deep historical backfills to
    produce a continuous timeline of interest rates for 'TARGET_CURRENCIES'.
    
    The hierarchy of data authority is:
    1. Official IBKR Scraped Data (Highest Priority).
    2. FRED Proxy Data + Spread (Backfill for older history).
    3. Forward/Backward Fill (Gap bridging for non-business days).

    Args:
        start_date (pd.Timestamp): The beginning of the required data window.
        end_date (pd.Timestamp): The end of the required data window.

    Returns:
        pd.DataFrame: A DataFrame indexed by Date, with columns for each target 
        currency. Values represent the *daily* interest rate factor (e.g., 0.00015).
    """
    print(f"   [>] Fetching Margin Rates ({start_date.date()} to {end_date.date()})...")
    time.sleep(0.5)
    
    # 1. Scrape Recent Data
    df_scraped = get_daily_margin_rates_scraper(start_date, end_date)
    
    if df_scraped.empty:
        min_scraped_date = end_date 
        print("       [!] Warning: IBKR Scraper returned no data. Switching to full FRED backfill.")
    else:
        min_scraped_date = df_scraped.index.min()
        print(f"       - Scraper returned data from: {min_scraped_date.date()}")

    # 2. Backfill Deep History with FRED
    if min_scraped_date > start_date:
        backfill_end = min_scraped_date - pd.Timedelta(days=1)
        print(f"       - Backfilling history from {start_date.date()} to {backfill_end.date()} using FRED...")
        df_backfill = get_fred_proxy_rates(start_date, backfill_end)
        
        # Combine: Priority to Scraped Data, then FRED
        df_final = pd.concat([df_backfill, df_scraped])
    else:
        df_final = df_scraped

    # 3. Gap Filling Strategy
    df_final = df_final.sort_index()
    
    # A. Forward Fill: Fills weekends/holidays
    df_final = df_final.ffill()
    
    # B. Backward Fill: The "Safety Net"
    df_final = df_final.bfill()
    
    # 4. Strict Formatting & Filtering
    df_final = df_final.reindex(columns=TARGET_CURRENCIES)
    
    print(f"   [+] Margin rates loaded successfully.")
    print()
    time.sleep(0.5)

    return df_final.loc[start_date:end_date]


def get_daily_margin_rates_scraper(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Scrapes official historical margin rates from the Interactive Brokers website.

    Iterates through the requested months, constructing URLs for the IBKR 
    'monthlyInterestRates' endpoint. It parses the returned HTML tables to extract 
    benchmark rates for target currencies.
    
    Includes retry logic (exponential backoff) to handle transient network errors 
    or rate limits.

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
    
    cursor = start_date.replace(day=1)
    months_to_scrape = []
    while cursor <= end_date:
        months_to_scrape.append(cursor.strftime("%Y%m"))
        cursor += relativedelta(months=1)
    
    all_data = []
    
    for month_str in months_to_scrape:
        if month_str > datetime.now().strftime("%Y%m"): continue

        url = f"{base_url}?date={month_str}&ib_entity=llc"
        # Retry Logic
        resp = None
        for attempt in range(3): # Try up to 3 times
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                # Wait 1s, 2s, etc. before retrying
                time.sleep(1 + attempt)
        
        # If resp is still None or failed after 3 tries, skip this month
        if resp is None or resp.status_code != 200:
            print(f"       [!] Warning: Failed to fetch margin data for {month_str}")
            continue

        try:
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find('table', class_='table table-freeze-col table-bordered table-striped')
            if not table: continue 

            headers_list = [th.text.strip().upper() for th in table.find('thead').find_all('th')]
            try:
                date_idx = next(i for i, h in enumerate(headers_list) if 'DATE' in h)
            except StopIteration: continue

            is_wide = not any('BENCHMARK' in h for h in headers_list)
            rows = table.find('tbody').find_all('tr')
            
            if is_wide:
                col_map = {}
                for i, h in enumerate(headers_list):
                    curr = h.split('(')[0].strip()
                    if curr in TARGET_CURRENCIES:
                        col_map[i] = curr
                
                for row in rows:
                    cols = [td.text.strip() for td in row.find_all('td')]
                    if not cols: continue
                    date_str = cols[date_idx]
                    
                    for c_idx, curr in col_map.items():
                        if c_idx >= len(cols): continue
                        rate = parse_ibkr_rate(cols[c_idx], curr)
                        if rate is not None:
                            all_data.append({'date': pd.to_datetime(date_str), 'currency': curr, 'rate': rate})
            
        except Exception as e:
            print(f"       [!] Warning: Error parsing HTML for {month_str}: {e}")
            continue
            
    if not all_data: return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    return df.pivot_table(index='date', columns='currency', values='rate', aggfunc='mean')


def get_fred_proxy_rates(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Downloads and normalizes historical benchmark rates from FRED.

    This acts as the fallback engine. It fetches proxy rates (defined in config)
    for each currency, such as the Effective Federal Funds Rate for USD.

    Crucially, this function transforms the raw benchmark data into a proxy for 
    margin rates by:
    1. Forward-filling monthly/weekly data to daily frequency.
    2. Adding a defined 'Margin Spread' (from config) to approximate the markup 
       charged by the broker.
    3. Converting annualized percentage rates into daily multipliers based on 
       day-count conventions (360 vs 365 days).

    Args:
        start_date (pd.Timestamp): The start date for the backfill.
        end_date (pd.Timestamp): The end date for the backfill.

    Returns:
        pd.DataFrame: A DataFrame of daily proxy margin rates.
    """
    if start_date > end_date: return pd.DataFrame()

    # Create empty container for the date range
    proxy_data = pd.DataFrame(index=pd.date_range(start_date, end_date))
    
    for currency, fred_code in FRED_PROXIES.items():
        if currency not in TARGET_CURRENCIES: continue
        
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_code}"
            
            # Retry Logic
            r = None
            for attempt in range(3): # Try up to 3 times
                try:
                    r = requests.get(url, timeout=10)
                    if r.status_code == 200:
                        break 
                except requests.exceptions.RequestException:
                    # Wait 1s, 2s, etc. before retrying
                    time.sleep(1 + attempt)

            # If resp is still None or failed after 3 tries, skip this month
            if r is None or r.status_code != 200:
                print(f"       [!] Warning: Failed to fetch FRED data for {currency}")
                continue

            series_df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True, na_values='.')
            
            # Align to our dates
            series = series_df.reindex(proxy_data.index)
            
            # Forward fill: Crucial for OECD data which is often Monthly (dates are 2025-01-01, 2025-02-01)
            # This propagates the Jan 1st rate to Jan 31st.
            series = series.ffill()
            
            divisor = 365.0 if currency in CURRENCIES_365 else 360.0

            # Looks up specific currency spread, defaults to 'DEFAULT' key
            spread = MARGIN_SPREADS.get(currency, MARGIN_SPREADS.get('DEFAULT'))

            daily_rate = (series + spread) / 100.0 / divisor
            proxy_data[currency] = daily_rate
            
        except Exception as e:
            print(f"       [!] Warning: Error processing FRED data for {currency}: {e}")
            continue

    return proxy_data

def parse_ibkr_rate(rate_str: str, currency: str) -> Optional[float]:
    """
    Parses and standardizes a raw rate string from the IBKR HTML table.

    Converts strings like "1.58%" or "(0.50%)" into float representations.
    It automatically applies the broker spread and adjusts for the specific 
    currency's day-count convention (ACT/360 or ACT/365) to return a daily multiplier.

    Args:
        rate_str (str): The raw text string from the HTML cell.
        currency (str): The currency code (used to determine day-count logic).

    Returns:
        Optional[float]: The calculated daily interest rate, or None if parsing fails.
    """
    clean = rate_str.replace('%', '').replace('(', '-').replace(')', '').strip()
    if not clean or clean == '-': return None
    try:
        val = float(clean)
        divisor = 365.0 if currency in CURRENCIES_365 else 360.0
        spread = MARGIN_SPREADS.get(currency, MARGIN_SPREADS.get('DEFAULT'))
        return (val + spread) / 100.0 / divisor
    except:
        return None