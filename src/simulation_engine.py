"""
Monte Carlo Simulation Engine Module.

This module encapsulates the core stochastic logic required to generate counterfactual
investment scenarios. It takes an investor's historical trade timing and capital
allocation decisions and applies them to a randomized selection of assets from their
'Competence Universe'.

The engine enforces strict 'Apple-to-Apple' comparability by applying standardized
transaction fees, tax models, and margin logic to both the Control Portfolio and
the Simulated Paths.
"""
import pandas as pd
import random
import time
from typing import Any
from .config import FEE_SCHEDULE, DIV_TAX_RATES

class MonteCarloEngine:
    def __init__(self, data_package: dict[str, Any], market_data_map: dict[str, pd.DataFrame], daily_margin_rates_df: pd.DataFrame) -> None:
        # 1. Unpack Data Package
        # The initial state and event logs are deep-copied to prevent mutation 
        # of the source data during recursive simulation runs.
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
        # Lookups are converted into dictionaries to ensure O(1) access speed during 
        # the simulation loops, which is critical for performance when running 10k+ paths.
        # This avoids repeated DataFrame indexing overhead.
        if 'Exchange' in self.financial_info.columns:
            self.sym_exchange_map = self.financial_info.set_index('symbol')['Exchange'].to_dict()
        else:
            self.sym_exchange_map = {}

        self.sym_curr_map = self.original_initial_state.set_index('symbol')['currency'].to_dict()
        
        # 3. Prepare Data
        # Invokes internal logic to normalize timezones and fill data gaps.
        self._prepare_market_data()
        
        # 4. Define Universe
        # The 'Competence Universe' is restricted to stocks actually held or traded 
        # in the original portfolio. This acknowledges survivorship bias but ensures 
        # the simulation picks assets the investor was actually aware of, rather than
        # picking obscure assets from the global set.
        self.stock_universe = self.original_initial_state[
            self.original_initial_state['asset_category'] == 'Stock'
        ]['symbol'].unique().tolist()

        # 5. Load Fee Schedule and Dividend Tax Rates
        # These are imported from configuration to maintain consistency across modules.
        self.FEE_SCHEDULE = FEE_SCHEDULE
        self.DIV_TAX_RATES = DIV_TAX_RATES

    def _prepare_market_data(self) -> None:
        """
        Pre-processes market data to optimise lookup speed during simulation.

        Price histories are normalised to UTC midnight, and dividend projections are
        generated for missing forward data points to ensure accurate yield modelling.
        """
        # Timezones are stripped to ensure strict equality comparisons between 
        # trade dates and market data dates. Times are normalised to midnight.
        sim_start = self.report_start.replace(tzinfo=None).normalize()
        sim_end = self.report_end.replace(tzinfo=None).normalize()
        
        self.price_cache = {}

        for ticker, df in self.market_data.items():
            # Explicitly cast to DatetimeIndex so Pylance knows .tz and .normalize exist
            dt_idx = pd.DatetimeIndex(df.index)

            if dt_idx.tz is not None:
                dt_idx = dt_idx.tz_localize(None)
            
            df.index = dt_idx.normalize()
            """
            # --- Dividends ---
            # Dividend data is extracted directly from the market data.
            # We assume the loader has successfully retrieved all relevant corporate actions.
            if 'Dividends' in df.columns:
                # Only keep non-zero dividend entries to save memory.
                self.dividend_map[ticker] = df[df['Dividends'] != 0]['Dividends']
            else:
                self.dividend_map[ticker] = pd.Series(dtype=float)
            """
            # --- Dividends ---
            # Dividend data is often sparse. If a simulation path picks a stock that 
            # pays dividends, but the data source is incomplete for the current year,
            # past dividends are projected forward to avoid artificially penalising the stock.
            if 'Dividends' in df.columns:
                raw_divs = df[df['Dividends'] != 0]['Dividends']
                divs_in_window = raw_divs[(raw_divs.index >= sim_start) & (raw_divs.index <= sim_end)]
                
                # If no dividends exist in the simulation window, but historical dividends do exist,
                # we attempt to project the previous year's payments forward.
                if divs_in_window.empty and not raw_divs.empty:
                    lookback_start = sim_start - pd.Timedelta(days=370) 
                    lookback_end = sim_end - pd.Timedelta(days=360)   
                    past_divs = raw_divs[(raw_divs.index >= lookback_start) & (raw_divs.index <= lookback_end)]
                    
                    # Shift the old dates forward by exactly one year to create proxy data.
                    if not past_divs.empty:
                        projected_dates = past_divs.index + pd.Timedelta(days=365)
                        projected_divs = pd.Series(past_divs.values, index=projected_dates)
                        combined_divs = pd.concat([raw_divs, projected_divs]).sort_index()
                        combined_divs.index = pd.DatetimeIndex(combined_divs.index).normalize()
                        self.dividend_map[ticker] = combined_divs
                    else:
                        self.dividend_map[ticker] = raw_divs
                else:
                    self.dividend_map[ticker] = raw_divs
            else:
                self.dividend_map[ticker] = pd.Series(dtype=float)

            # --- Prices ---
            # The market_data_loader has already handled resampling (gap filling).
            # The 'Close' column is converted to a dictionary for O(1) retrieval speed,
            # replacing slower DataFrame indexing (.loc) in the main loop.
            self.price_cache[ticker] = df['Close'].to_dict()

    def _get_fx_rate(self, from_curr: str, to_curr: str, date: pd.Timestamp) -> float:
        """
        Retrieves or synthesises the cross-rate between two currencies for a specific date.

        Args:
            from_curr (str): The currency to convert from.
            to_curr (str): The currency to convert to.
            date (pd.Timestamp): The date of the transaction.

        Returns:
            float: The exchange rate multiplier.
        """
        if from_curr == to_curr: return 1.0
        date = pd.to_datetime(date).replace(tzinfo=None).normalize()

        def get_price(ticker, dt):
            # Safe retrieval from cache; defaults to 1.0 to prevent crash on missing data.
            if ticker in self.price_cache:
                return self.price_cache[ticker].get(dt, 1.0)
            return 1.0
        
        # Cross-rates are calculated via the base currency (triangulation).
        # E.g., converting EUR -> JPY is performed as EUR -> Base -> JPY.
        rate = 1.0

        # Convert 'from_curr' to Base
        if from_curr != self.base_currency:
            t1, t2 = f"{from_curr}{self.base_currency}=X", f"{self.base_currency}{from_curr}=X"
            if t1 in self.price_cache: rate *= get_price(t1, date)
            elif t2 in self.price_cache: rate *= (1.0 / get_price(t2, date))

        # Convert Base to 'to_curr'
        if to_curr != self.base_currency:
            t1, t2 = f"{to_curr}{self.base_currency}=X", f"{self.base_currency}{to_curr}=X"
            if t1 in self.price_cache: rate *= (1.0 / get_price(t1, date))
            elif t2 in self.price_cache: rate *= get_price(t2, date)

        return rate

    def _get_raw_price(self, symbol: str, date: pd.Timestamp) -> float:
        """
        Retrieves the closing price of an asset for a specific date from the cache.

        Args:
            symbol (str): The asset ticker symbol.
            date (pd.Timestamp): The date of lookup.

        Returns:
            float: The closing price (defaults to 1.0 if missing/Cash).
        """
        if symbol in self.price_cache:
             return self.price_cache[symbol].get(date, 1.0)
        return 1.0

    def _calculate_buy_quantity(self, symbol: str, available_cash: float, price: float) -> float:
        """
        Calculates the number of shares that can be purchased for a given cash amount, 
        accounting for fees.

        This method solves for the quantity of shares such that the total cost (including
        transaction fees and taxes) matches the available cash allocated for the trade.

        Args:
            symbol (str): The ticker symbol of the asset being purchased.
            available_cash (float): The total capital allocated to this transaction.
            price (float): The current unit price of the asset.

        Returns:
            float: The maximum number of shares that can be purchased.
        """
        exchange = self.sym_exchange_map.get(symbol, 'DEFAULT')
        rule = self.FEE_SCHEDULE.get(exchange, self.FEE_SCHEDULE['DEFAULT'])
        fee = rule['fee']
        tax_rate = rule['tax_buy']
        
        if price <= 0: return 0

        # Algebra: Total Cost = (Price * Qty) * (1 + Tax) + Fee = Available Cash
        # Therefore: Qty = (Available Cash - Fee) / (Price * (1 + Tax))
        numerator = available_cash - fee
        if numerator <= 0: return 0
        denominator = price * (1 + tax_rate)
    
        return numerator / denominator

    def _calculate_sell_proceeds(self, symbol: str, quantity: float, price: float) -> float:
        """
        Calculates the net cash proceeds from selling a position, net of fees.

        Args:
            symbol (str): The ticker symbol of the asset being sold.
            quantity (float): The number of shares being sold (absolute value).
            price (float): The current unit price of the asset.

        Returns:
            float: The net cash resulting from the sale (floored at 0.0).
        """
        exchange = self.sym_exchange_map.get(symbol, 'DEFAULT')
        rule = self.FEE_SCHEDULE.get(exchange, self.FEE_SCHEDULE['DEFAULT'])

        # Net Proceeds = (Price * Qty) - Fixed_Fee
        # Note: Tax on sell (Capital Gains) is ignored as private investors in Switzerland are 
        # exempt from it. This would need to be adapted for professional or non-Swiss investors.
        gross_val = abs(quantity) * price
        net_val = gross_val - rule['fee']
        return max(0.0, net_val)

    def _run_single_path(self, override_trade_symbol: bool = False) -> pd.DataFrame:
        """
        Executes a single simulation iteration (one full timeline reconstruction).

        The sorted event log is iterated through sequentially. For every trade event,
        either a random asset is selected from the universe (Simulation) or the
        actual traded asset is used (Control), while applying consistent fee and margin logic.

        Args:
            override_trade_symbol (bool): If True, the actual historical symbol is used
                (Control Portfolio). If False, a random symbol is selected (Simulation).
                Defaults to False.

        Returns:
            pd.DataFrame: A generated event log representing the counterfactual timeline.
        """
        cash_balances = {}
        current_holdings = {}

        # Initialise simulation state from the initial state of the real portfolio.
        for _, row in self.original_initial_state.iterrows():
            if row['asset_category'] == 'Cash':
                cash_balances[row['currency']] = row['quantity']
            elif row['asset_category'] == 'Stock':
                current_holdings[row['symbol']] = row['quantity']

        timeline = []

        # Organise events by day to process them chronologically.
        sorted_events = self.original_events.sort_values('timestamp')
        sorted_events['day'] = sorted_events['timestamp'].dt.normalize()
        grouped_events = sorted_events.groupby('day')

        # Use inclusive='left' to ensure the loop stops on the true last day,
        # just like in the module portfolio_reconstructor.
        date_range = pd.date_range(self.report_start, self.report_end, freq='D', inclusive='left')

        # The following events should not be used in the simulation as they are linked to the events 
        # of the real portfolio history. FX_TRADE is usually triggered by a purchase in 
        # another currency and would favour the control portfolio path (must be ignored as well).
        events_to_skip = {'DIVIDEND', 'SPLIT', 'TAX', 'INTEREST', 'FEE', 'FEE_REBATE', 'FX_TRADE'}

        for d in date_range:
            d_naive = d.replace(tzinfo=None).normalize()
            
            # --- A. Simulate Dividends ---
            for sym, qty in list(current_holdings.items()):
                if abs(qty) < 1e-9: continue
                
                if sym in self.dividend_map:
                    if d_naive in self.dividend_map[sym].index:
                        # Force scalar extraction
                        div_val = self.dividend_map[sym].loc[d_naive]
                        if isinstance(div_val, pd.Series):
                            div_val = div_val.iloc[0]
                        div_per_share = float(div_val)
                        div_per_share = self.dividend_map[sym].loc[d_naive]
                        gross_div_total = div_per_share * qty
                        
                        curr = self.sym_curr_map.get(sym, 'USD')
                        exchange = self.sym_exchange_map.get(sym, 'DEFAULT')
                        tax_rate = self.DIV_TAX_RATES.get(exchange, self.DIV_TAX_RATES['DEFAULT'])
                        tax_amount = gross_div_total * tax_rate
                        
                        # Apply net dividend to cash balance.
                        cash_balances[curr] = cash_balances.get(curr, 0) + gross_div_total - tax_amount
                        
                        # Log the events for the report.
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
                    # Skip events that are not relevant.
                    if row['event_type'] in events_to_skip:
                        continue
                    
                    # Process Non-TRADE events, not in the events_to_skip list.
                    if row['event_type'] not in ['TRADE_BUY', 'TRADE_SELL']: 
                        timeline.append(row.to_dict())

                        # Non-TRADE events that change cash balances (DEPOSIT, WITHDRAWAL)
                        if row['cash_change_native'] != 0 and pd.notna(row['currency']):
                            cash_balances[row['currency']] = cash_balances.get(row['currency'], 0) + row['cash_change_native']
                        
                        # Non-TRADE events that affect holding quantities
                        if row['quantity_change'] != 0 and pd.notna(row['symbol']):
                            current_holdings[row['symbol']] = current_holdings.get(row['symbol'], 0) + row['quantity_change']

                        continue
                    
                    # --- TRADE LOGIC ---
                    # Defines the core counterfactual logic.
                    is_buy = (row['event_type'] == 'TRADE_BUY')                    
                    orig_val = abs(row['cash_change_native']) # The monetary value of the original trade.
                    
                    new_sym = None

                    if override_trade_symbol:
                        # CONTROL PATH: Force the simulation to use the actual historical asset to build the 
                        # event log of the real portfolio with the same restrictions as for the simulated paths.
                        new_sym = row['symbol']
                        if new_sym not in self.sym_curr_map: self.sym_curr_map[new_sym] = row['currency']
                    else:
                        # RANDOM PATH: Select a random asset from the Competence Universe.
                        if is_buy:
                            # --- STRATEGY 3 SAFEGUARD ---
                            # Attempt to find a valid asset that actually existed on this date.
                            # We limit retries to prevent infinite loops on holidays or empty data days.
                            max_retries = 50
                            found_valid = False
                            
                            for _ in range(max_retries):
                                candidate = random.choice(self.stock_universe)
                                
                                # 1. Check if we have any data map for this ticker
                                if candidate not in self.market_data:
                                    continue
                                
                                prices = self.market_data[candidate]
                                
                                # 2. Check if the specific date exists in the index
                                # Using d_naive because market data index is typically normalized/naive
                                if d_naive in prices.index:
                                    val = prices.loc[d_naive]['Close']
                                    if isinstance(val, pd.Series):
                                        val = val.iloc[0] # Handle rare duplicates
                                        
                                    price_check = float(val) # Explicit cast to float
                                    
                                    # 3. Check for strictly positive price
                                    if price_check > 0:
                                        new_sym = candidate
                                        found_valid = True
                                        break
                            
                            if not found_valid:
                                continue # Skip this trade entirely
                                
                        else:
                            # For sells, assets from the current simulated holdings must be chosen
                            valid_candidates = [s for s, q in current_holdings.items() if q > 0.0001]
                            if not valid_candidates: continue
                            new_sym = random.choice(valid_candidates)

                    # Safety Check: If new_sym is still None, skip processing
                    if new_sym is None:
                        continue

                    new_curr = self.sym_curr_map.get(new_sym, self.base_currency)

                    # Convert the original capital amount into the currency of the new asset.
                    fx = self._get_fx_rate(row['currency'], new_curr, d_naive)
                    available_cash_new_curr = orig_val * fx
                    raw_price = self._get_raw_price(new_sym, d_naive)
                    
                    if is_buy:
                        # Determine how many shares of the random stock can be bought with the capital.
                        new_qty = self._calculate_buy_quantity(new_sym, available_cash_new_curr, raw_price)
                        
                        if new_qty > 0:
                             exchange = self.sym_exchange_map.get(new_sym, 'DEFAULT')
                             rule = self.FEE_SCHEDULE.get(exchange, self.FEE_SCHEDULE['DEFAULT'])
                             # Recalculate cost with fees to update cash balance accurately.
                             actual_cost = new_qty * raw_price * (1 + rule['tax_buy']) + rule['fee']
                             cash_impact = -1 * actual_cost
                        else:
                             cash_impact = 0
                    else:
                        # For sells, calculate proceeds based on current simulation holding.
                        current_qty = current_holdings.get(new_sym, 0)
                        # Determine the nominal percentage of the portfolio to sell based on the original trade value.
                        nominal_qty = available_cash_new_curr / raw_price
                        qty_to_sell = min(nominal_qty, current_qty)
                        
                        if qty_to_sell > 0:
                             net_proceeds = self._calculate_sell_proceeds(new_sym, qty_to_sell, raw_price)
                             new_qty = -1 * qty_to_sell
                             cash_impact = net_proceeds
                        else:
                             new_qty = 0
                             cash_impact = 0

                    # Construct the new simulated event.
                    new_event = row.copy()
                    new_event['symbol'] = new_sym
                    new_event['currency'] = new_curr
                    new_event['quantity_change'] = new_qty
                    new_event['cash_change_native'] = cash_impact
                    
                    timeline.append(new_event.to_dict())
                    
                    # Update simulation state.
                    current_holdings[new_sym] = current_holdings.get(new_sym, 0) + new_qty
                    cash_balances[new_curr] = cash_balances.get(new_curr, 0) + cash_impact

            # --- C. Simulate Margin Interest ---
            # Negative cash balances incur margin interest expenses, which are approximated using historical 
            # benchmark rates from IBKR and interest rates from FRED in case of missing data
            for curr, bal in cash_balances.items():
                if bal < -0.01:
                    rate = 0.00018 # Fallback daily rate if data is missing.
                    if curr in self.daily_rates.columns:
                        try:
                            # Forward-fill lookup for the closest available interest rate.
                            target_idx = pd.Index([d_naive])
                            idx_r = self.daily_rates.index.get_indexer(target_idx, method='ffill')[0]
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
    
    def generate_scenario(self, num_paths: int = 100, verbose: bool = False) -> list[dict[str, Any]]:
        """
        Produces N stochastic portfolio paths for analysis.

        Args:
            num_paths (int): The number of independent simulations to run. Defaults to 100.
            verbose (bool): If True, progress is printed to the console. Defaults to False.

        Returns:
            list[dict[str, Any]]: A list of data packages, where each package contains
            the event log for a single simulated path.
        """
        results = []
        if verbose:
            print(f" [>] Generating {num_paths} simulated paths...")
            time.sleep(0.5)

        for i in range(num_paths):
            # Executes the path generation with random asset selection (default).
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
            print(f" [+] Batch of {num_paths} paths generated.")
            time.sleep(0.5)

        return results

    def run_real_with_sim_logic(self) -> list[dict[str, Any]]:
        """
        Processes the actual historical trades through the simulation engine.

        This generates the 'Control Portfolio'—a scientific baseline that shares the
        exact same fee structure, margin calculations, and execution assumptions as
        the random paths, ensuring an 'Apple-to-Apple' comparison.

        Returns:
            list[dict[str, Any]]: A list containing the single Control Portfolio data package.
        """
        print(" [>] Generating Control Portfolio...")
        time.sleep(0.5)

        # Runs the path generation but forces the use of historical symbols (override_trade_symbol=True).
        sim_events = self._run_single_path(override_trade_symbol=True)

        print(" [+] Control Portfolio generated.")
        time.sleep(0.5)

        return [ {
            'initial_state': self.original_initial_state.copy(),
            'events': sim_events,
            'report_start_date': self.report_start,
            'report_end_date': self.report_end,
            'base_currency': self.base_currency,
            'financial_info': self.financial_info 
        } ]