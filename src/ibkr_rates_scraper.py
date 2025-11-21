import requests
from bs4 import BeautifulSoup
import pandas as pd
from dateutil.relativedelta import relativedelta
import time

def get_daily_margin_rates(start_date, end_date, target_currencies=['USD', 'CHF', 'EUR', 'GBP']):
    """
    Scrapes IBKR's Monthly Benchmark Rates.
    Handles both LONG format (Currency|Date|Rate) and WIDE format (Date|USD|CHF|...).
    """
    base_url = "https://www.interactivebrokers.com/en/accounts/fees/monthlyInterestRates.php"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0"}
    
    # Generate list of months
    current_cursor = start_date.replace(day=1) - relativedelta(months=1)
    months_to_scrape = []
    while current_cursor <= end_date:
        months_to_scrape.append(current_cursor.strftime("%Y%m"))
        current_cursor += relativedelta(months=1)
    
    all_data = []
    print(f"Scraping IBKR rates for {len(months_to_scrape)} months...")

    for month_str in months_to_scrape:
        url = f"{base_url}?date={month_str}&ib_entity=llc"
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200: continue

            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find('table', class_='table table-freeze-col table-bordered table-striped')
            
            if not table:
                print(f" - Warning: Table not found for {month_str}")
                continue

            # Extract Headers
            headers_list = [th.text.strip() for th in table.find('thead').find_all('th')]
            
            # --- DETECT FORMAT ---
            # Wide Format usually has 'Date' but NO 'Benchmark' column header
            # Long Format has 'Currency', 'Date', 'Benchmark'
            is_wide_format = False
            date_idx = -1
            
            try:
                # Find Date Column (Case Insensitive)
                date_idx = next(i for i, h in enumerate(headers_list) if 'DATE' in h.upper())
                
                # If "Benchmark" is missing, it is likely Wide Format
                if not any('BENCHMARK' in h.upper() for h in headers_list):
                    is_wide_format = True
            except StopIteration:
                # If we can't even find a Date column, skip
                print(f" - Warning: Could not identify Date column for {month_str}")
                continue

            rows = table.find('tbody').find_all('tr')

            if is_wide_format:
                # --- WIDE FORMAT LOGIC (Date | AED | AUD | ... | USD | ...) ---
                # Map header index to currency code
                col_map = {}
                for idx, h in enumerate(headers_list):
                    # clean header (e.g. "CHF(%)" -> "CHF")
                    clean_h = h.split('(')[0].strip().upper()
                    if clean_h in target_currencies:
                        col_map[idx] = clean_h

                for row in rows:
                    cols = [td.text.strip() for td in row.find_all('td')]
                    if not cols: continue
                    if date_idx >= len(cols): continue
                    
                    date_str = cols[date_idx]
                    
                    for col_idx, currency in col_map.items():
                        if col_idx >= len(cols): continue
                        rate_str = cols[col_idx]
                        
                        # Clean Rate
                        clean_rate = rate_str.replace('%', '').replace('(', '-').replace(')', '').strip()
                        if not clean_rate: continue

                        try:
                            rate_val = float(clean_rate)
                            date_val = pd.to_datetime(date_str)
                            
                            # Math
                            divisor = 365.0 if currency in ['GBP', 'AUD', 'HKD'] else 360.0
                            daily_rate = (rate_val + 1.50) / 100.0 / divisor
                            
                            all_data.append({
                                'date': date_val,
                                'currency': currency,
                                'daily_rate': daily_rate
                            })
                        except ValueError:
                            continue

            else:
                # --- LONG FORMAT LOGIC (Currency | Date | Benchmark) ---
                try:
                    curr_idx = next(i for i, h in enumerate(headers_list) if 'CURRENCY' in h.upper())
                    rate_idx = next(i for i, h in enumerate(headers_list) if 'BENCHMARK' in h.upper())
                    
                    for row in rows:
                        cols = [td.text.strip() for td in row.find_all('td')]
                        if not cols: continue
                        
                        currency = cols[curr_idx]
                        if currency not in target_currencies: continue
                        
                        date_str = cols[date_idx]
                        rate_str = cols[rate_idx]
                        clean_rate = rate_str.replace('%', '').replace('(', '-').replace(')', '').strip()
                        
                        try:
                            rate_val = float(clean_rate)
                            date_val = pd.to_datetime(date_str)
                            
                            divisor = 365.0 if currency in ['GBP', 'AUD', 'HKD'] else 360.0
                            daily_rate = (rate_val + 1.50) / 100.0 / divisor
                            
                            all_data.append({
                                'date': date_val,
                                'currency': currency,
                                'daily_rate': daily_rate
                            })
                        except ValueError:
                            continue
                except StopIteration:
                    pass
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f" - Error scraping {month_str}: {e}")

    if not all_data:
        print("Warning: No rate data found. Returning fallback.")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Pivot to shape: Index=Date, Cols=Currencies
    # Handle duplicates (sometimes IBKR lists same date twice) by taking mean
    df_pivot = df.pivot_table(index='date', columns='currency', values='daily_rate', aggfunc='mean')
    
    # Fill missing days (weekends)
    df_daily_rates = df_pivot.resample('D').ffill()
    
    return df_daily_rates