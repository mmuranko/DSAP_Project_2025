import pandas as pd
import random

class MonteCarloEngine:
    def __init__(self, data_package, market_data_map, daily_margin_rates_df):
        # 1. Unpack Data Package
        self.original_initial_state = data_package['initial_state'].copy()
        self.original_events = data_package['events'].copy()
        self.base_currency = data_package['base_currency']
        self.financial_info = data_package['financial_info'] 
        
        self.report_start = data_package['report_start_date']
        self.report_end = data_package['report_end_date']
        
        self.market_data = market_data_map
        self.daily_rates = daily_margin_rates_df
        
        self.raw_price_map = {}
        self.dividend_map = {}
        self.split_map = {}
        
        # 2. Build Maps
        if 'Exchange' in self.financial_info.columns:
            self.sym_exchange_map = self.financial_info.set_index('symbol')['Exchange'].to_dict()
        else:
            self.sym_exchange_map = {}

        self.sym_curr_map = self.original_initial_state.set_index('symbol')['currency'].to_dict()
        
        # 3. Prepare Data
        self._prepare_market_data()
        
        # 4. Define Universe
        self.stock_universe = self.original_initial_state[
            self.original_initial_state['asset_category'] == 'Stock'
        ]['symbol'].unique().tolist()

        # ==========================================
        # CONFIG A: TRANSACTION FEE SCHEDULE
        # ==========================================
        # 'fee': Fixed cost in Asset Currency
        # 'tax_buy': Percentage tax on PURCHASE only (e.g. Stamp Duty/FTT)
        self.FEE_SCHEDULE = {
            # US (1 USD)
            'NASDAQ': {'fee': 1.0, 'tax_buy': 0.0},
            'NYSE':   {'fee': 1.0, 'tax_buy': 0.0},
            'AMEX':   {'fee': 1.0, 'tax_buy': 0.0},
            'ARCA':   {'fee': 1.0, 'tax_buy': 0.0},
            'PINK':   {'fee': 1.0, 'tax_buy': 0.0},
            # CH (5 CHF)
            'EBS':    {'fee': 5.0, 'tax_buy': 0.0},
            'VIRTX':  {'fee': 5.0, 'tax_buy': 0.0},
            # FR (3 EUR + 0.4% FTT)
            'SBF':    {'fee': 3.0, 'tax_buy': 0.004}, 
            # UK (3 GBP + 0.5% Stamp Duty)
            'LSE':    {'fee': 3.0, 'tax_buy': 0.005}, 
            # Rest of Europe (3 EUR)
            'IBIS':   {'fee': 3.0, 'tax_buy': 0.0}, # DE
            'FWB':    {'fee': 3.0, 'tax_buy': 0.0}, # DE
            'AEB':    {'fee': 3.0, 'tax_buy': 0.0}, # NL
            'BM':     {'fee': 3.0, 'tax_buy': 0.0}, # ES
            'BVME':   {'fee': 3.0, 'tax_buy': 0.0}, # IT
            'PL':     {'fee': 3.0, 'tax_buy': 0.0}, # PT
            'EBR':    {'fee': 3.0, 'tax_buy': 0.0}, # BE
            # Default
            'DEFAULT': {'fee': 4.0, 'tax_buy': 0.0} 
        }

        # ==========================================
        # CONFIG B: WITHHOLDING TAX SCHEDULE
        # ==========================================
        # Maps Exchange -> Dividend Withholding Tax Rate
        # Note: Capital Gains Tax is assumed 0%
        self.DIV_TAX_RATES = {
            # US: 15% (Treaty Rate)
            'NASDAQ': 0.15, 'NYSE': 0.15, 'AMEX': 0.15, 'ARCA': 0.15, 'PINK': 0.15,
            
            # Switzerland: 35% (Verrechnungssteuer)
            'EBS': 0.35, 'VIRTX': 0.35,
            
            # France: 25%
            'SBF': 0.25,
            
            # Germany: 26.375% (Kapitalertragsteuer + Soli)
            'IBIS': 0.26375, 'FWB': 0.26375,
            
            # Netherlands: 17% (Dividendbelasting) - Note: Often 15% via treaty, keeping 17 as requested
            'AEB': 0.17,
            
            # Spain: 19%
            'BM': 0.19,
            
            # Italy: 26%
            'BVME': 0.26,
            
            # UK: 0%
            'LSE': 0.00,
            
            # Fallback
            'DEFAULT': 0.15
        }

    def _prepare_market_data(self):
        """ Prepares 'Raw' price history, Dividend projections, and TZ normalization. """
        sim_start = self.report_start.replace(tzinfo=None).normalize()
        sim_end = self.report_end.replace(tzinfo=None).normalize()
        
        for ticker, df in self.market_data.items():
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()

            if 'Stock Splits' in df.columns:
                self.split_map[ticker] = df[df['Stock Splits'] != 0]['Stock Splits']
            else:
                self.split_map[ticker] = pd.Series(dtype=float)

            if 'Dividends' in df.columns:
                raw_divs = df[df['Dividends'] != 0]['Dividends']
                divs_in_window = raw_divs[(raw_divs.index >= sim_start) & (raw_divs.index <= sim_end)]
                
                if divs_in_window.empty and not raw_divs.empty:
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
                return series.iloc[idx]
            return 1.0

        rate = 1.0
        if from_curr != self.base_currency:
            t1, t2 = f"{from_curr}{self.base_currency}=X", f"{self.base_currency}{from_curr}=X"
            if t1 in self.market_data: rate *= get_price(t1, date)
            elif t2 in self.market_data: rate *= (1.0 / get_price(t2, date))

        if to_curr != self.base_currency:
            t1, t2 = f"{to_curr}{self.base_currency}=X", f"{self.base_currency}{to_curr}=X"
            if t1 in self.market_data: rate *= (1.0 / get_price(t1, date))
            elif t2 in self.market_data: rate *= get_price(t2, date)

        return rate

    def _get_raw_price(self, symbol, date):
        price = 1.0
        if symbol in self.raw_price_map:
            series = self.raw_price_map[symbol]
            try:
                ts = date.replace(tzinfo=None) if hasattr(date, 'tzinfo') else date
                idx = series.index.get_indexer([ts], method='nearest')[0]
                price = series.iloc[idx]
            except: pass
        return price if price > 0 else 1.0

    def _calculate_buy_quantity(self, symbol, available_cash, price):
        exchange = self.sym_exchange_map.get(symbol, 'DEFAULT')
        rule = self.FEE_SCHEDULE.get(exchange, self.FEE_SCHEDULE['DEFAULT'])
        fee = rule['fee']
        tax_rate = rule['tax_buy']
        
        if price <= 0: return 0
        cash_pre_tax = available_cash / (1 + tax_rate)
        cash_for_stock = cash_pre_tax - fee
        if cash_for_stock <= 0: return 0 
        return cash_for_stock / price

    def _calculate_sell_proceeds(self, symbol, quantity, price):
        exchange = self.sym_exchange_map.get(symbol, 'DEFAULT')
        rule = self.FEE_SCHEDULE.get(exchange, self.FEE_SCHEDULE['DEFAULT'])
        gross_val = abs(quantity) * price
        net_val = gross_val - rule['fee']
        return max(0.0, net_val)

    def generate_scenario(self, num_paths=100):
        results = []
        print(f"Generating {num_paths} Tax & Fee-Aware paths...")
        for i in range(num_paths):
            sim_events = self._run_single_path()
            sim_package = {
                'initial_state': self.original_initial_state.copy(),
                'events': sim_events,
                'report_start_date': self.report_start,
                'report_end_date': self.report_end,
                'base_currency': self.base_currency,
                'financial_info': self.financial_info 
            }
            results.append(sim_package)
        return results

    def _run_single_path(self):
        cash_balances = {}
        current_holdings = {}
        
        for _, row in self.original_initial_state.iterrows():
            if row['asset_category'] == 'Cash':
                cash_balances[row['currency']] = row['quantity']
            elif row['asset_category'] == 'Stock':
                current_holdings[row['symbol']] = row['quantity']

        timeline = []
        sorted_events = self.original_events.sort_values('timestamp')
        sorted_events['day'] = sorted_events['timestamp'].dt.normalize()
        grouped_events = sorted_events.groupby('day')
        date_range = pd.date_range(self.report_start, self.report_end, freq='D')

        for d in date_range:
            d_naive = d.replace(tzinfo=None).normalize()
            
            # --- A. Corporate Actions ---
            for sym, qty in list(current_holdings.items()):
                if abs(qty) < 1e-9: continue
                
                # 1. Dividends with Tax Logic
                if sym in self.dividend_map:
                    if d_naive in self.dividend_map[sym].index:
                        # Get Gross Amount
                        div_per_share = self.dividend_map[sym].loc[d_naive]
                        gross_div_total = div_per_share * qty
                        
                        # Determine Currency & Exchange
                        curr = self.sym_curr_map.get(sym, 'USD')
                        exchange = self.sym_exchange_map.get(sym, 'DEFAULT')
                        
                        # Calculate Withholding Tax
                        tax_rate = self.DIV_TAX_RATES.get(exchange, self.DIV_TAX_RATES['DEFAULT'])
                        tax_amount = gross_div_total * tax_rate
                        
                        # Apply Cash Flows (Gross In, Tax Out)
                        cash_balances[curr] = cash_balances.get(curr, 0) + gross_div_total - tax_amount
                        
                        # Record Gross Payment
                        timeline.append({
                            'timestamp': d, 'event_type': 'DIVIDEND',
                            'symbol': sym, 'quantity_change': 0,
                            'cash_change_native': gross_div_total, 'currency': curr
                        })
                        
                        # Record Tax Deduction
                        if tax_amount > 0:
                            timeline.append({
                                'timestamp': d, 'event_type': 'TAX',
                                'symbol': sym, 'quantity_change': 0,
                                'cash_change_native': -tax_amount, 'currency': curr
                            })

                # 2. Splits
                if sym in self.split_map:
                    if d_naive in self.split_map[sym].index:
                        ratio = self.split_map[sym].loc[d_naive]
                        current_holdings[sym] = qty * ratio
                        timeline.append({
                            'timestamp': d, 'event_type': 'SPLIT',
                            'symbol': sym, 'quantity_change': 0, 'cash_change_native': 0,
                            'currency': self.sym_curr_map.get(sym, 'USD'), 'split_ratio': ratio
                        })

            # --- B. Process Trades ---
            if d_naive in grouped_events.groups:
                days_events = grouped_events.get_group(d_naive)
                
                for _, row in days_events.iterrows():
                    if row['event_type'] in ['DIVIDEND', 'SPLIT', 'TAX']: continue
                    
                    if row['event_type'] not in ['TRADE_BUY', 'TRADE_SELL']:
                        timeline.append(row.to_dict())
                        if row['currency']:
                            cash_balances[row['currency']] = cash_balances.get(row['currency'], 0) + row['cash_change_native']
                        continue
                    
                    # --- TRADE LOGIC ---
                    is_buy = (row['event_type'] == 'TRADE_BUY')
                    orig_val = abs(row['cash_change_native'])
                    
                    if is_buy:
                        new_sym = random.choice(self.stock_universe)
                    else:
                        valid_candidates = [s for s, q in current_holdings.items() if q > 0.0001]
                        if not valid_candidates: continue
                        new_sym = random.choice(valid_candidates)

                    new_curr = self.sym_curr_map.get(new_sym, self.base_currency)
                    fx = self._get_fx_rate(row['currency'], new_curr, row['timestamp'])
                    available_cash_new_curr = orig_val * fx
                    raw_price = self._get_raw_price(new_sym, row['timestamp'])
                    
                    if is_buy:
                        new_qty = self._calculate_buy_quantity(new_sym, available_cash_new_curr, raw_price)
                        cash_impact = -1 * available_cash_new_curr
                        direction = 1
                    else:
                        current_qty = current_holdings.get(new_sym, 0)
                        nominal_qty = available_cash_new_curr / raw_price
                        qty_to_sell = min(nominal_qty, current_qty)
                        net_proceeds = self._calculate_sell_proceeds(new_sym, qty_to_sell, raw_price)
                        new_qty = -1 * qty_to_sell
                        cash_impact = net_proceeds
                        direction = -1

                    new_event = row.copy()
                    new_event['symbol'] = new_sym
                    new_event['currency'] = new_curr
                    new_event['quantity_change'] = new_qty
                    new_event['cash_change_native'] = cash_impact
                    
                    timeline.append(new_event.to_dict())
                    
                    current_holdings[new_sym] = current_holdings.get(new_sym, 0) + new_qty
                    cash_balances[new_curr] = cash_balances.get(new_curr, 0) + cash_impact

            # --- C. Interest ---
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

    def run_actual_with_sim_logic(self):
        """
        Re-runs the EXACT historical trades through the Simulation's Fee/Tax engine.
        This creates a 'Shadow Benchmark' that is perfectly comparable to the random paths.
        """
        print("Generating 'Shadow' Actual Portfolio (Fee-Normalized)...")
        
        # 1. Initialize State (Same as random path)
        cash_balances = {}
        current_holdings = {}
        
        for _, row in self.original_initial_state.iterrows():
            if row['asset_category'] == 'Cash':
                cash_balances[row['currency']] = row['quantity']
            elif row['asset_category'] == 'Stock':
                current_holdings[row['symbol']] = row['quantity']

        timeline = []
        sorted_events = self.original_events.sort_values('timestamp')
        sorted_events['day'] = sorted_events['timestamp'].dt.normalize()
        grouped_events = sorted_events.groupby('day')
        date_range = pd.date_range(self.report_start, self.report_end, freq='D')

        for d in date_range:
            d_naive = d.replace(tzinfo=None).normalize()
            
            # --- A. Corporate Actions ---
            for sym, qty in list(current_holdings.items()):
                if abs(qty) < 1e-9: continue
                
                # Dividends (Using SIMULATED Tax Logic)
                if sym in self.dividend_map:
                    if d_naive in self.dividend_map[sym].index:
                        div_per_share = self.dividend_map[sym].loc[d_naive]
                        gross_div_total = div_per_share * qty
                        
                        curr = self.sym_curr_map.get(sym, 'USD')
                        exchange = self.sym_exchange_map.get(sym, 'DEFAULT')
                        
                        # Apply SIMULATED Tax Rate (e.g. 15%, 35%)
                        tax_rate = self.DIV_TAX_RATES.get(exchange, self.DIV_TAX_RATES['DEFAULT'])
                        tax_amount = gross_div_total * tax_rate
                        
                        cash_balances[curr] = cash_balances.get(curr, 0) + gross_div_total - tax_amount
                        
                        timeline.append({
                            'timestamp': d, 'event_type': 'DIVIDEND', 'symbol': sym, 
                            'quantity_change': 0, 'cash_change_native': gross_div_total, 'currency': curr
                        })
                        if tax_amount > 0:
                            timeline.append({
                                'timestamp': d, 'event_type': 'TAX', 'symbol': sym, 
                                'quantity_change': 0, 'cash_change_native': -tax_amount, 'currency': curr
                            })

                # Splits
                if sym in self.split_map:
                    if d_naive in self.split_map[sym].index:
                        ratio = self.split_map[sym].loc[d_naive]
                        current_holdings[sym] = qty * ratio
                        timeline.append({
                            'timestamp': d, 'event_type': 'SPLIT', 'symbol': sym, 
                            'quantity_change': 0, 'cash_change_native': 0,
                            'currency': self.sym_curr_map.get(sym, 'USD'), 'split_ratio': ratio
                        })

            # --- B. Process Trades (Using Original Symbols but Sim Math) ---
            if d_naive in grouped_events.groups:
                days_events = grouped_events.get_group(d_naive)
                
                for _, row in days_events.iterrows():
                    if row['event_type'] in ['DIVIDEND', 'SPLIT', 'TAX']: continue
                    
                    if row['event_type'] not in ['TRADE_BUY', 'TRADE_SELL']:
                        timeline.append(row.to_dict())
                        if row['currency']:
                            cash_balances[row['currency']] = cash_balances.get(row['currency'], 0) + row['cash_change_native']
                        continue
                    
                    # --- TRADE LOGIC ---
                    is_buy = (row['event_type'] == 'TRADE_BUY')
                    orig_val = abs(row['cash_change_native'])
                    
                    # KEY DIFFERENCE: FORCE ORIGINAL SYMBOL
                    new_sym = row['symbol']
                    
                    # Validation: Ensure we track it in our maps (safety check)
                    if new_sym not in self.sym_curr_map:
                        # If it's a new stock we bought during the year, map it now
                        self.sym_curr_map[new_sym] = row['currency']

                    # Recalculate Logic (Same as Random Path)
                    new_curr = self.sym_curr_map.get(new_sym, self.base_currency)
                    fx = self._get_fx_rate(row['currency'], new_curr, row['timestamp'])
                    available_cash_new_curr = orig_val * fx
                    raw_price = self._get_raw_price(new_sym, row['timestamp'])
                    
                    if is_buy:
                        # Apply SIMULATED Fees to Actual Trade
                        new_qty = self._calculate_buy_quantity(new_sym, available_cash_new_curr, raw_price)
                        cash_impact = -1 * available_cash_new_curr
                        direction = 1
                    else:
                        # Apply SIMULATED Selling Logic
                        current_qty = current_holdings.get(new_sym, 0)
                        nominal_qty = available_cash_new_curr / raw_price
                        
                        # Cap at inventory (should match perfectly if data is clean, but safer to keep)
                        qty_to_sell = min(nominal_qty, current_qty)
                        
                        # Apply SIMULATED Fees
                        net_proceeds = self._calculate_sell_proceeds(new_sym, qty_to_sell, raw_price)
                        new_qty = -1 * qty_to_sell
                        cash_impact = net_proceeds
                        direction = -1

                    # Record Event
                    new_event = row.copy()
                    new_event['symbol'] = new_sym
                    new_event['currency'] = new_curr
                    new_event['quantity_change'] = new_qty
                    new_event['cash_change_native'] = cash_impact
                    
                    timeline.append(new_event.to_dict())
                    
                    current_holdings[new_sym] = current_holdings.get(new_sym, 0) + new_qty
                    cash_balances[new_curr] = cash_balances.get(new_curr, 0) + cash_impact

            # --- C. Interest (Same Logic) ---
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
                        'symbol': curr, 'quantity_change': 0, 'cash_change_native': interest, 'currency': curr
                    })

        # Return structured as a single-item list to match 'scenarios' format
        return [ {
            'initial_state': self.original_initial_state.copy(),
            'events': pd.DataFrame(timeline),
            'report_start_date': self.report_start,
            'report_end_date': self.report_end,
            'base_currency': self.base_currency,
            'financial_info': self.financial_info 
        } ]