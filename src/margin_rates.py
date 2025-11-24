import requests
from bs4 import BeautifulSoup
import pandas as pd
from dateutil.relativedelta import relativedelta
import time
from datetime import datetime

# --- 1. USER SETTINGS (STRICT) ---
TARGET_CURRENCIES = [
    'AUD', 'CAD', 'GBP', 'HKD', 'KRW', 'ILS', 'INR', 'NZD', 'SGD', # 365 Day Divisor
    'USD', 'EUR', 'CHF', 'CZK', 'JPY', 'SEK', 'NOK', 'DKK', 'MXN'  # 360 Day Divisor
]

CURRENCIES_365 = ['AUD', 'CAD', 'GBP', 'HKD', 'KRW', 'ILS', 'INR', 'NZD', 'SGD']

# --- 2. FRED MAPPINGS (Expanded for Developed Nations) ---
# Source: OECD "Immediate Rates: Less than 24 Hours: Call Money/Interbank Rate"
FRED_PROXIES = {
    # Majors (Daily Data usually available)
    'USD': 'DFF',               # Fed Funds Effective Rate
    'EUR': 'ECBDFR',            # ECB Deposit Facility Rate
    'GBP': 'IUDSOIA',           # SONIA
    'JPY': 'IRSTCI01JPM156N',   # Japan Overnight Call Rate
    'CHF': 'IRSTCI01CHM156N',   # Swiss SARON
    'CAD': 'IRSTCI01CAM156N',   # Canada Overnight
    'AUD': 'RBAACTRBDAL',       # RBA Cash Rate Target
    'MXN': 'IRSTCI01MXM156N',   # Mexico Overnight
    
    # Developed / OECD Nations (Monthly Averages used as Daily Proxy)
    'NOK': 'IRSTCI01NOM156N',   # Norway Interbank Rate
    'SEK': 'IRSTCI01SEM156N',   # Sweden Interbank Rate
    'DKK': 'IRSTCI01DKM156N',   # Denmark Interbank Rate
    'CZK': 'IRSTCI01CZM156N',   # Czech Rep Interbank Rate
    'HUF': 'IRSTCI01HUM156N',   # Hungary Interbank Rate
    'ILS': 'IRSTCI01ILM156N',   # Israel Interbank Rate
    'NZD': 'IRSTCI01NZM156N',   # New Zealand Interbank Rate
    'KRW': 'IRSTCI01KRM156N',   # Korea Overnight
    'ZAR': 'IRSTCI01ZAM156N',   # South Africa Interbank Rate
    
    # SGD is tricky on FRED (often missing). We rely on scraper + bfill for SGD if FRED fails.
    'SGD': 'IRSTCI01SGM156N',   
}

def get_ibkr_rates_hybrid(start_date, end_date):
    """
    Returns margin rates strictly for TARGET_CURRENCIES.
    1. Scrapes IBKR (Recent).
    2. Backfills using FRED/OECD data (Deep History).
    3. Backfills remaining gaps using the last known rate.
    """
    print(f"--- Fetching Margin Rates ({start_date.date()} to {end_date.date()}) ---")
    
    # 1. Scrape Recent Data
    df_scraped = get_daily_margin_rates_scraper(start_date, end_date)
    
    if df_scraped.empty:
        min_scraped_date = end_date 
    else:
        min_scraped_date = df_scraped.index.min()
    
    print(f"Scraper returned data from: {min_scraped_date.date()}")

    # 2. Backfill Deep History with FRED
    if min_scraped_date > start_date:
        backfill_end = min_scraped_date - pd.Timedelta(days=1)
        print(f"Backfilling history from {start_date.date()} to {backfill_end.date()} using FRED...")
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
    # If SGD is missing in FRED, this takes the first scraped SGD rate 
    # and extends it backwards to the start.
    df_final = df_final.bfill()
    
    # 4. Strict Formatting & Filtering
    df_final = df_final.reindex(columns=TARGET_CURRENCIES)
    
    return df_final.loc[start_date:end_date]


def get_daily_margin_rates_scraper(start_date, end_date):
    """ Scrapes IBKR Website """
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
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code != 200: continue
            
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
            
        except Exception:
            continue 
            
    if not all_data: return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    return df.pivot_table(index='date', columns='currency', values='rate', aggfunc='mean')


def get_fred_proxy_rates(start_date, end_date):
    """ Downloads FRED data via CSV """
    if start_date > end_date: return pd.DataFrame()

    # Create empty container for the date range
    proxy_data = pd.DataFrame(index=pd.date_range(start_date, end_date))
    
    for currency, fred_code in FRED_PROXIES.items():
        if currency not in TARGET_CURRENCIES: continue
        
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_code}"
            series_df = pd.read_csv(url, index_col=0, parse_dates=True, na_values='.')
            
            # Align to our dates
            series = series_df.reindex(proxy_data.index)
            
            # Forward fill: Crucial for OECD data which is often Monthly (dates are 2025-01-01, 2025-02-01)
            # This propagates the Jan 1st rate to Jan 31st.
            series = series.ffill()
            
            divisor = 365.0 if currency in CURRENCIES_365 else 360.0
            
            daily_rate = (series + 1.50) / 100.0 / divisor
            proxy_data[currency] = daily_rate
            
        except Exception:
            # If download fails, column stays NaN, will be caught by bfill later
            continue 
            
    return proxy_data


def parse_ibkr_rate(rate_str, currency):
    clean = rate_str.replace('%', '').replace('(', '-').replace(')', '').strip()
    if not clean or clean == '-': return None
    try:
        val = float(clean)
        divisor = 365.0 if currency in CURRENCIES_365 else 360.0
        return (val + 1.50) / 100.0 / divisor
    except:
        return None