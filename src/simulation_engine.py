import pandas as pd
import random
import time
from .config import FEE_SCHEDULE, DIV_TAX_RATES

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
        
        self.dividend_map = {}
        
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

        # 5. Load Fee Schedule and Dividend Tax Rates
        self.FEE_SCHEDULE = FEE_SCHEDULE
        self.DIV_TAX_RATES = DIV_TAX_RATES

    def _prepare_market_data(self):
        """ Prepares price history and dividend projections. """
        sim_start = self.report_start.replace(tzinfo=None).normalize()
        sim_end = self.report_end.replace(tzinfo=None).normalize()
        
        self.price_cache = {}

        for ticker, df in self.market_data.items():
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()

            # --- Dividends ---
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
                        combined_divs = pd.concat([raw_divs, projected_divs]).sort_index()
                        combined_divs.index = combined_divs.index.normalize()
                        self.dividend_map[ticker] = combined_divs
                    else:
                        self.dividend_map[ticker] = raw_divs
                else:
                    self.dividend_map[ticker] = raw_divs
            else:
                self.dividend_map[ticker] = pd.Series(dtype=float)

            # --- Prices ---
            # market_data_loader has already resampled to 'D' and ffilled.
            # We simply load the Close column into the cache.
            self.price_cache[ticker] = df['Close'].to_dict()

    def _get_fx_rate(self, from_curr, to_curr, date):
        if from_curr == to_curr: return 1.0
        date = pd.to_datetime(date).replace(tzinfo=None).normalize()

        def get_price(ticker, dt):
            if ticker in self.price_cache:
                return self.price_cache[ticker].get(dt, 1.0)
            return 1.0

        rate = 1.0
        if from_curr != self.base_currency:
            t1, t2 = f"{from_curr}{self.base_currency}=X", f"{self.base_currency}{from_curr}=X"
            if t1 in self.price_cache: rate *= get_price(t1, date)
            elif t2 in self.price_cache: rate *= (1.0 / get_price(t2, date))

        if to_curr != self.base_currency:
            t1, t2 = f"{to_curr}{self.base_currency}=X", f"{self.base_currency}{to_curr}=X"
            if t1 in self.price_cache: rate *= (1.0 / get_price(t1, date))
            elif t2 in self.price_cache: rate *= get_price(t2, date)

        return rate

    def _get_raw_price(self, symbol, date):
        if symbol in self.price_cache:
             return self.price_cache[symbol].get(date, 1.0)
        return 1.0

    def _calculate_buy_quantity(self, symbol, available_cash, price):
        """
        Solves X = k * s * (1 + t) + f for k.
        k = (X - f) / (s * (1 + t))
        """
        exchange = self.sym_exchange_map.get(symbol, 'DEFAULT')
        rule = self.FEE_SCHEDULE.get(exchange, self.FEE_SCHEDULE['DEFAULT'])
        fee = rule['fee']
        tax_rate = rule['tax_buy']
        
        if price <= 0: return 0
        
        numerator = available_cash - fee
        if numerator <= 0: return 0
        
        denominator = price * (1 + tax_rate)
        
        return numerator / denominator

    def _calculate_sell_proceeds(self, symbol, quantity, price):
        exchange = self.sym_exchange_map.get(symbol, 'DEFAULT')
        rule = self.FEE_SCHEDULE.get(exchange, self.FEE_SCHEDULE['DEFAULT'])
        gross_val = abs(quantity) * price
        net_val = gross_val - rule['fee']
        return max(0.0, net_val)

    def _run_single_path(self, override_trade_symbol=False):
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
        # Use inclusive='left' to ensure the loop stops on the true last day,
        # matching the logic in portfolio_reconstructor.
        date_range = pd.date_range(self.report_start, self.report_end, freq='D', inclusive='left')

        # Define skip list once (Optimization & Consistency)
        events_to_skip = {'DIVIDEND', 'SPLIT', 'TAX', 'INTEREST', 'FEE', 'FEE_REBATE', 'FX_TRADE'}

        for d in date_range:
            d_naive = d.replace(tzinfo=None).normalize()
            
            # --- A. Simulate Dividends ---
            # This block runs for ALL scenarios (random and actual) to apply a normalized tax rate.
            for sym, qty in list(current_holdings.items()):
                if abs(qty) < 1e-9: continue
                
                if sym in self.dividend_map:
                    if d_naive in self.dividend_map[sym].index:
                        div_per_share = self.dividend_map[sym].loc[d_naive]
                        gross_div_total = div_per_share * qty
                        
                        curr = self.sym_curr_map.get(sym, 'USD')
                        exchange = self.sym_exchange_map.get(sym, 'DEFAULT')
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

            # --- B. Simulate Trades ---
            if d_naive in grouped_events.groups:
                days_events = grouped_events.get_group(d_naive)
                
                for _, row in days_events.iterrows():
                    # Use the pre-defined set for faster lookup
                    if row['event_type'] in events_to_skip:
                        continue
                    
                    # Process non-trade events that are NOT skipped (e.g., DEPOSIT, WITHDRAWAL, GIFT_VEST).
                    if row['event_type'] not in ['TRADE_BUY', 'TRADE_SELL']: 
                        timeline.append(row.to_dict())

                        # Update cash for events like DEPOSIT, WITHDRAWAL
                        if row['cash_change_native'] != 0 and pd.notna(row['currency']):
                            cash_balances[row['currency']] = cash_balances.get(row['currency'], 0) + row['cash_change_native']
                        
                        # Update holdings for events like GIFT_VEST
                        if row['quantity_change'] != 0 and pd.notna(row['symbol']):
                            current_holdings[row['symbol']] = current_holdings.get(row['symbol'], 0) + row['quantity_change']

                        continue
                    
                    # --- TRADE LOGIC ---
                    is_buy = (row['event_type'] == 'TRADE_BUY')                    
                    orig_val = abs(row['cash_change_native']) # The monetary value of the original trade.
                    
                    if override_trade_symbol:
                        new_sym = row['symbol']
                        if new_sym not in self.sym_curr_map: self.sym_curr_map[new_sym] = row['currency']
                    else:
                        if is_buy:
                            new_sym = random.choice(self.stock_universe)
                        else:
                            valid_candidates = [s for s, q in current_holdings.items() if q > 0.0001]
                            if not valid_candidates: continue
                            new_sym = random.choice(valid_candidates)

                    new_curr = self.sym_curr_map.get(new_sym, self.base_currency)
                    fx = self._get_fx_rate(row['currency'], new_curr, d_naive)
                    available_cash_new_curr = orig_val * fx
                    raw_price = self._get_raw_price(new_sym, d_naive)
                    
                    if is_buy:
                        new_qty = self._calculate_buy_quantity(new_sym, available_cash_new_curr, raw_price)
                        
                        if new_qty > 0:
                             exchange = self.sym_exchange_map.get(new_sym, 'DEFAULT')
                             rule = self.FEE_SCHEDULE.get(exchange, self.FEE_SCHEDULE['DEFAULT'])
                             actual_cost = new_qty * raw_price * (1 + rule['tax_buy']) + rule['fee']
                             cash_impact = -1 * actual_cost
                        else:
                             cash_impact = 0
                    else:
                        current_qty = current_holdings.get(new_sym, 0)
                        nominal_qty = available_cash_new_curr / raw_price
                        qty_to_sell = min(nominal_qty, current_qty)
                        
                        if qty_to_sell > 0:
                             net_proceeds = self._calculate_sell_proceeds(new_sym, qty_to_sell, raw_price)
                             new_qty = -1 * qty_to_sell
                             cash_impact = net_proceeds
                        else:
                             new_qty = 0
                             cash_impact = 0

                    new_event = row.copy()
                    new_event['symbol'] = new_sym
                    new_event['currency'] = new_curr
                    new_event['quantity_change'] = new_qty
                    new_event['cash_change_native'] = cash_impact
                    
                    timeline.append(new_event.to_dict())
                    
                    current_holdings[new_sym] = current_holdings.get(new_sym, 0) + new_qty
                    cash_balances[new_curr] = cash_balances.get(new_curr, 0) + cash_impact

            # --- C. Simulate Margin Interest ---
            # This block runs for ALL scenarios to apply a normalized interest model.
            for curr, bal in cash_balances.items():
                if bal < -0.01:
                    rate = 0.00018 # Fallback rate
                    if curr in self.daily_rates.columns:
                        try:
                            idx_r = self.daily_rates.index.get_indexer([d_naive], method='ffill')[0]
                            if idx_r != -1: rate = self.daily_rates.iloc[idx_r][curr]
                        except: pass
                    
                    interest = bal * rate
                    cash_balances[curr] += interest
                    timeline.append({
                        'timestamp': d + pd.Timedelta(hours=23, minutes=59),
                        'event_type': 'INTEREST_MARGIN', 'symbol': curr, 'quantity_change': 0,
                        'cash_change_native': interest, 'currency': curr
                    })

        return pd.DataFrame(timeline)
    
    def generate_scenario(self, num_paths=100, verbose=False):

        results = []
        if verbose:
            print(f"   [>] Generating {num_paths} simulated paths...")
            time.sleep(0.5)

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
        if verbose:
            print(f"   [+] Batch of {num_paths} paths generated.")
            time.sleep(0.5)

        return results

    def run_actual_with_sim_logic(self):
        """ Runs the actual historical trades through the simulation engine. """

        print("   [>] Generating Shadow Benchmark (Fee-Normalized)...")
        time.sleep(0.5)

        sim_events = self._run_single_path(override_trade_symbol=True)

        print("   [+] Shadow Benchmark generated.")
        time.sleep(0.5)

        return [ {
            'initial_state': self.original_initial_state.copy(),
            'events': sim_events,
            'report_start_date': self.report_start,
            'report_end_date': self.report_end,
            'base_currency': self.base_currency,
            'financial_info': self.financial_info 
        } ]