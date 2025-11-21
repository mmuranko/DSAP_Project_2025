import pandas as pd
import random

class MonteCarloEngine:
    def __init__(self, data_package, market_data_map, daily_margin_rates_df):
        self.original_initial_state = data_package['initial_state'].copy()
        self.original_events = data_package['events'].copy()
        self.base_currency = data_package['base_currency']
        self.market_data = market_data_map
        self.daily_rates = daily_margin_rates_df
        
        self.report_start = data_package['report_start_date']
        self.report_end = data_package['report_end_date']
        
        self.raw_price_map = {}
        self.dividend_map = {}
        self.split_map = {}
        
        self._prepare_market_data()
        
        # Universe definition
        self.stock_universe = self.original_initial_state[
            self.original_initial_state['asset_category'] == 'Stock'
        ]['symbol'].unique().tolist()
        
        self.sym_curr_map = self.original_initial_state.set_index('symbol')['currency'].to_dict()

    def _prepare_market_data(self):
        """
        Prepares 'Raw' price history and action maps.
        INCLUDES: Aggressive Dividend Projection & Strict TZ Normalization.
        """
        # Normalize Simulation Window to Naive
        sim_start = self.report_start.replace(tzinfo=None).normalize()
        sim_end = self.report_end.replace(tzinfo=None).normalize()
        
        for ticker, df in self.market_data.items():
            # --- 1. GLOBAL TZ STRIPPING ---
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()

            # --- 2. SPLITS ---
            if 'Stock Splits' in df.columns:
                self.split_map[ticker] = df[df['Stock Splits'] != 0]['Stock Splits']
            else:
                self.split_map[ticker] = pd.Series(dtype=float)

            # --- 3. DIVIDENDS (With Aggressive Projection) ---
            if 'Dividends' in df.columns:
                raw_divs = df[df['Dividends'] != 0]['Dividends']
                
                # Strict Check: Do we have dividends IN the simulation window?
                divs_in_window = raw_divs[(raw_divs.index >= sim_start) & (raw_divs.index <= sim_end)]
                
                if divs_in_window.empty and not raw_divs.empty:
                    # Window is empty. Force Projection from last year.
                    lookback_start = sim_start - pd.Timedelta(days=370) 
                    lookback_end = sim_end - pd.Timedelta(days=360)   
                    
                    past_divs = raw_divs[(raw_divs.index >= lookback_start) & (raw_divs.index <= lookback_end)]
                    
                    if not past_divs.empty:
                        projected_dates = past_divs.index + pd.Timedelta(days=365)
                        projected_divs = pd.Series(past_divs.values, index=projected_dates)
                        self.dividend_map[ticker] = pd.concat([raw_divs, projected_divs]).sort_index()
                    else:
                        self.dividend_map[ticker] = raw_divs
                else:
                    self.dividend_map[ticker] = raw_divs
            else:
                self.dividend_map[ticker] = pd.Series(dtype=float)

            # --- 4. RAW PRICE REVERSE ENGINEERING ---
            if not self.split_map[ticker].empty:
                split_series = df['Stock Splits'].replace(0, 1.0)
                cum_split_factor = split_series.sort_index(ascending=False).cumprod().sort_index(ascending=True).shift(-1).fillna(1.0)
                self.raw_price_map[ticker] = df['Close'] * cum_split_factor
            else:
                self.raw_price_map[ticker] = df['Close']

    def _get_fx_rate(self, from_curr, to_curr, date):
        if from_curr == to_curr: return 1.0
        date = pd.to_datetime(date).replace(tzinfo=None).normalize()

        def get_price(ticker, dt):
            if ticker in self.market_data:
                series = self.market_data[ticker]['Close']
                idx = series.index.get_indexer([dt], method='nearest')[0]
                found_date = series.index[idx]
                if abs((found_date - dt).days) < 7:
                    return series.iloc[idx]
            return 1.0

        rate_to_base = 1.0
        if from_curr != self.base_currency:
            t1, t2 = f"{from_curr}{self.base_currency}=X", f"{self.base_currency}{from_curr}=X"
            if t1 in self.market_data: rate_to_base = get_price(t1, date)
            elif t2 in self.market_data: rate_to_base = 1.0 / get_price(t2, date)

        rate_from_base = 1.0
        if to_curr != self.base_currency:
            t1, t2 = f"{to_curr}{self.base_currency}=X", f"{self.base_currency}{to_curr}=X"
            if t1 in self.market_data: rate_from_base = 1.0 / get_price(t1, date)
            elif t2 in self.market_data: rate_from_base = get_price(t2, date)

        return rate_to_base * rate_from_base

    def _get_raw_price(self, symbol, date):
        """Helper to fetch raw price for quantity calculation."""
        price = 1.0
        if symbol in self.raw_price_map:
            series = self.raw_price_map[symbol]
            try:
                ts = date.replace(tzinfo=None) if hasattr(date, 'tzinfo') else date
                idx = series.index.get_indexer([ts], method='nearest')[0]
                price = series.iloc[idx]
            except: pass
        return price if price > 0 else 1.0

    def generate_scenario(self, num_paths=100):
        results = []
        print(f"Generating {num_paths} Inventory-Aware paths...")
        
        for i in range(num_paths):
            # Seed the random generator differently for each path if needed, 
            # but Python's random handles this naturally.
            sim_events = self._run_single_path()
            
            sim_package = {
                'initial_state': self.original_initial_state.copy(),
                'events': sim_events,
                'report_start_date': self.report_start,
                'report_end_date': self.report_end,
                'base_currency': self.base_currency,
                'financial_info': pd.DataFrame()
            }
            results.append(sim_package)
            # if (i+1) % 10 == 0: print(f"Path {i+1}/{num_paths} done.")

        return results

    def _run_single_path(self):
        """
        Runs a simulation where SELL trades are strictly validated against current inventory.
        """
        # 1. Initialize State
        cash_balances = {}
        current_holdings = {}
        
        for _, row in self.original_initial_state.iterrows():
            if row['asset_category'] == 'Cash':
                cash_balances[row['currency']] = row['quantity']
            elif row['asset_category'] == 'Stock':
                current_holdings[row['symbol']] = row['quantity']

        # 2. Build Timeline (Just-In-Time Logic)
        timeline = []
        
        # We iterate through original events chronologically
        sorted_events = self.original_events.sort_values('timestamp')
        
        # Group by date so we can process Dividends/Splits/Interest daily
        sorted_events['day'] = sorted_events['timestamp'].dt.normalize()
        grouped_events = sorted_events.groupby('day')
        date_range = pd.date_range(self.report_start, self.report_end, freq='D')

        for d in date_range:
            d_naive = d.replace(tzinfo=None).normalize()
            
            # --- A. Corporate Actions (Start of Day) ---
            # These happen BEFORE trades, so inventory is updated first.
            for sym, qty in list(current_holdings.items()):
                if abs(qty) < 1e-9: continue
                
                # Dividends
                if sym in self.dividend_map:
                    if d_naive in self.dividend_map[sym].index:
                        div_amt = self.dividend_map[sym].loc[d_naive]
                        total_div = div_amt * qty
                        curr = self.sym_curr_map.get(sym, 'USD')
                        cash_balances[curr] = cash_balances.get(curr, 0) + total_div
                        timeline.append({
                            'timestamp': d, 'event_type': 'DIVIDEND',
                            'symbol': sym, 'quantity_change': 0,
                            'cash_change_native': total_div, 'currency': curr
                        })

                # Splits
                if sym in self.split_map:
                    if d_naive in self.split_map[sym].index:
                        ratio = self.split_map[sym].loc[d_naive]
                        current_holdings[sym] = qty * ratio
                        timeline.append({
                            'timestamp': d, 'event_type': 'SPLIT',
                            'symbol': sym, 'quantity_change': 0, 'cash_change_native': 0,
                            'currency': self.sym_curr_map.get(sym, 'USD'), 'split_ratio': ratio
                        })

            # --- B. Process Trades (Inventory Aware) ---
            if d_naive in grouped_events.groups:
                days_events = grouped_events.get_group(d_naive)
                
                for _, row in days_events.iterrows():
                    # Skip original actions
                    if row['event_type'] in ['DIVIDEND', 'SPLIT', 'TAX']: continue
                    
                    # Pass-through non-trade events
                    if row['event_type'] not in ['TRADE_BUY', 'TRADE_SELL']:
                        timeline.append(row.to_dict())
                        # Update cash for fees/deposits
                        if row['currency']:
                            cash_balances[row['currency']] = cash_balances.get(row['currency'], 0) + row['cash_change_native']
                        continue
                    
                    # --- TRADE LOGIC ---
                    is_buy = (row['event_type'] == 'TRADE_BUY')
                    orig_val = abs(row['cash_change_native'])
                    
                    # 1. Select New Symbol
                    if is_buy:
                        # BUY: Can buy anything from the universe
                        new_sym = random.choice(self.stock_universe)
                    else:
                        # SELL: Must own it.
                        # Filter holdings for stocks with value > 0
                        # (Simple heuristic: qty > 0.01. For strict value check, we'd need price, but qty is usually sufficient)
                        valid_candidates = [s for s, q in current_holdings.items() if q > 0.0001]
                        
                        if not valid_candidates:
                            # CORNER CASE: We want to sell, but own NOTHING.
                            # Skip trade? Or Buy something to Sell later?
                            # Decision: Skip trade to maintain integrity.
                            # print(f"Skipped SELL trade on {d_naive.date()} - No Inventory.")
                            continue
                        
                        new_sym = random.choice(valid_candidates)

                    # 2. Calculate Details
                    new_curr = self.sym_curr_map.get(new_sym, self.base_currency)
                    fx = self._get_fx_rate(row['currency'], new_curr, row['timestamp'])
                    new_val = orig_val * fx
                    
                    raw_price = self._get_raw_price(new_sym, row['timestamp'])
                    
                    direction = 1 if is_buy else -1
                    new_qty = direction * (new_val / raw_price)
                    
                    # 3. STRICT SELL CHECK: Do we have enough?
                    # Even if we picked a valid candidate, the specific $ amount might be too high.
                    # e.g. We hold $500 of MSFT, but the trade tries to sell $10,000.
                    if not is_buy:
                        current_qty = current_holdings.get(new_sym, 0)
                        # If we try to sell more than we have...
                        if abs(new_qty) > current_qty:
                            # Capping Logic: Sell ONLY what we have.
                            new_qty = -current_qty
                            # Recalculate cash value based on the capped quantity
                            new_val = abs(new_qty) * raw_price
                    
                    # 4. Execute
                    new_event = row.copy()
                    new_event['symbol'] = new_sym
                    new_event['currency'] = new_curr
                    new_event['quantity_change'] = new_qty
                    new_event['cash_change_native'] = -1 * direction * new_val
                    
                    timeline.append(new_event.to_dict())
                    
                    # Update Inventory
                    current_holdings[new_sym] = current_holdings.get(new_sym, 0) + new_qty
                    cash_balances[new_curr] = cash_balances.get(new_curr, 0) + (-1 * direction * new_val)

            # --- C. Interest (End of Day) ---
            for curr, bal in cash_balances.items():
                if bal < -0.01:
                    rate = 0.00018
                    if curr in self.daily_rates.columns:
                        try:
                            idx_r = self.daily_rates.index.get_indexer([d_naive], method='ffill')[0]
                            if idx_r != -1: rate = self.daily_rates.iloc[idx_r][curr]
                        except: pass
                    
                    interest = bal * rate
                    cash_balances[curr] += interest
                    timeline.append({
                        'timestamp': d + pd.Timedelta(hours=23, minutes=59),
                        'event_type': 'INTEREST_MARGIN',
                        'symbol': curr, 'quantity_change': 0,
                        'cash_change_native': interest, 'currency': curr
                    })

        return pd.DataFrame(timeline)