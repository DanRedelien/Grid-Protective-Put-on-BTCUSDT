# backtest_engine.py
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from options import HedgingManager  # Импорт из options.py

class Trade:
    def __init__(self, symbol, entry_price, size, entry_time, tag="GRID"):
        self.symbol = symbol
        self.entry_price = entry_price
        self.size = size
        self.entry_time = entry_time
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0.0
        self.status = 'OPEN' 
        self.tag = tag

class BacktestEngine:
    def __init__(self, config, data):
        self.cfg = config
        self.data = data
        self.symbol = self.cfg.SYMBOLS[0] # Primary logic for BTC
        
        # Проверка наличия данных
        if self.symbol not in data or data[self.symbol].empty:
            raise ValueError(f"No data found for {self.symbol}")

        self.timeline = data[self.symbol].index
        
        # --- CASH & EQUITY MANAGEMENT ---
        self.cash = self.cfg.INITIAL_CAPITAL
        # Для "Грязного" NAV (без хеджа) мы виртуально возвращаем премии обратно
        self.cash_unhedged = self.cfg.INITIAL_CAPITAL 
        
        self.positions = [] # List of Trade objects
        
        # --- UNITIZATION (Fund Accounting) ---
        # Strategy (Hedged / Net)
        self.units_strat = self.cfg.INITIAL_CAPITAL 
        self.nav_strat = 1.0
        
        # Strategy (Unhedged / Gross) - для анализа стоимости страховки
        self.units_gross = self.cfg.INITIAL_CAPITAL
        self.nav_gross = 1.0
        
        # Benchmark (Unitized Buy & Hold)
        start_price = self.data[self.symbol].iloc[0]['close']
        self.bench_btc_held = 0.0 
        self.units_bench = self.cfg.INITIAL_CAPITAL
        self.nav_bench = 1.0
        
        # Initial Benchmark Purchase
        self.bench_btc_held = self.cfg.INITIAL_CAPITAL / start_price
        
        # --- CASH FLOWS FOR MWR ---
        self.flows_strat = [(self.timeline[0], -self.cfg.INITIAL_CAPITAL)]
        self.flows_bench = [(self.timeline[0], -self.cfg.INITIAL_CAPITAL)]
        self.invested_capital = self.cfg.INITIAL_CAPITAL

        self.trade_id_counter = 0
        self.last_grid_ref_price = 0.0  
        self.closed_trades = []         

        # --- HEDGING & OPTIONS ---
        self.hedger = HedgingManager(self.cfg.RISK_FREE_RATE, self.cfg.VOL_RISK_PREMIUM) if self.cfg.ENABLE_HEDGING else None
        
        self.last_mtm_date = None
        self.cached_option_val = 0.0
        self._last_hedged_qty = 0.0

        # --- GRID STATE ---
        self.pending_orders = [] 
        self.history = []

    def _update_options_mtm(self, current_time, spot_price):
        if not self.hedger or not self.hedger.active_option:
            return 0.0
            
        if 'expiry_date' in self.hedger.active_option:
            expiry_ts = self.hedger.active_option['expiry_date']
            seconds_left = (expiry_ts - current_time).total_seconds()
            if seconds_left <= 0: return 0.0
            days_passed = (current_time - pd.to_datetime(self.hedger.active_option['open_time'])).total_seconds() / 86400.0
        else:
            days_passed = 0.0

        unit_val = self.hedger.get_mtm(spot_price, days_passed)
        return unit_val * self._last_hedged_qty

    def _get_equity(self, current_price, current_time):
        """Считаем Equity внутри бара для динамического сайзинга"""
        btc_val = sum(t['size'] for t in self.positions) * current_price
        opt_val = self._update_options_mtm(current_time, current_price)
        return self.cash + btc_val + opt_val

    def run(self):
        print(f"Starting Institutional Simulation ({self.cfg.TIMEFRAME})...")
        
        # Pre-fetching data arrays for speed
        df = self.data[self.symbol]
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        atrs = df['atr'].values
        vols = df['rolling_vol_annual'].values
        
        prev_month = None
        
        for i, ts in enumerate(self.timeline):
            if i < 100: continue # Skip warmup
            
            p_open = opens[i]
            p_high = highs[i]
            p_low = lows[i]
            p_close = closes[i]
            p_atr = atrs[i]
            
            # ----------------------------------------------
            # 1. DEPOSITS & FUND ACCOUNTING (Start of Bar)
            # ----------------------------------------------
            if prev_month != ts.month:
                if prev_month is not None:
                    deposit = self.cfg.MONTHLY_DEPOSIT
                    
                    # A. Strategy Deposit
                    new_units_strat = deposit / self.nav_strat
                    self.units_strat += new_units_strat
                    self.cash += deposit
                    
                    # Shadow Unhedged Strategy
                    new_units_gross = deposit / self.nav_gross
                    self.units_gross += new_units_gross
                    self.cash_unhedged += deposit
                    
                    self.flows_strat.append((ts, -deposit))
                    
                    # B. Benchmark Deposit
                    new_btc_bench = deposit / p_open
                    self.bench_btc_held += new_btc_bench
                    new_units_bench = deposit / self.nav_bench
                    self.units_bench += new_units_bench
                    
                    self.flows_bench.append((ts, -deposit))
                    self.invested_capital += deposit

                    # C. HEDGING ROLL (Monthly)
                    if self.hedger:
                        # 1. Settle Old
                        payoff = self.hedger.settle_hedge(p_open)
                        if payoff > 0:
                            self.cash += payoff * self._last_hedged_qty
                        
                        # 2. Open New
                        total_btc_pos = sum(t['size'] for t in self.positions)
                        if total_btc_pos > 0.001:
                            next_month_date = ts + relativedelta(months=1)
                            next_month_date = next_month_date.replace(day=1, hour=0, minute=0, second=0)
                            
                            days_to_expiry = (next_month_date - ts).days
                            days_to_expiry = max(days_to_expiry, 1)
                            
                            vol = vols[i-1] if not np.isnan(vols[i-1]) else 0.6
                            
                            opt = self.hedger.open_hedge('BTC', p_open, vol, days_to_expiry, self.cfg.HEDGE_STRIKE_PCT)
                            
                            self.hedger.active_option['open_time'] = ts
                            self.hedger.active_option['expiry_date'] = next_month_date 
                            
                            cost = opt['premium'] * total_btc_pos
                            self.cash -= cost
                            self._last_hedged_qty = total_btc_pos
                            self.last_mtm_date = None

                prev_month = ts.month

            # ----------------------------------------------
            # 2. STRICT GRID EXECUTION
            # ----------------------------------------------
            # A. Take Profit
            remaining_positions = []
            for pos in self.positions:
                tp_price = pos['entry'] * (1 + self.cfg.TAKE_PROFIT_PCT)
                if p_high >= tp_price:
                    revenue = pos['size'] * tp_price * (1 - self.cfg.FEE_RATE)
                    self.cash += revenue
                    self.cash_unhedged += revenue
                    
                    self.closed_trades.append({
                        'ts': ts, 'entry': pos['entry'], 'exit': tp_price, 
                        'pnl': revenue - (pos['size'] * pos['entry'])
                    })
                else:
                    remaining_positions.append(pos)
            self.positions = remaining_positions

            # B. Pending Buy
            active_orders = []
            for order in self.pending_orders:
                if p_low <= order['price']:
                    cost = order['size'] * order['price']
                    if self.cash >= cost:
                        qty_after_fee = order['size'] * (1 - self.cfg.FEE_RATE)
                        trade = {
                            'symbol': 'BTC', 
                            'entry': order['price'], 
                            'size': qty_after_fee, 
                            'ts': ts,
                            'id': self.trade_id_counter
                        }
                        self.trade_id_counter += 1
                        self.positions.append(trade)
                        self.cash -= cost
                        self.cash_unhedged -= cost
                else:
                    active_orders.append(order)
            self.pending_orders = active_orders

            # ----------------------------------------------
            # 3. STRATEGY LOGIC (Dynamic Grid)
            # ----------------------------------------------
            current_equity = self._get_equity(p_close, ts)
            
            current_pos_count = len(self.positions)
            target_levels = self.cfg.GRID_LEVELS
            if current_pos_count >= self.cfg.GRID_LEVELS:
                target_levels = min(self.cfg.GRID_LEVELS + 1, self.cfg.GRID_LEVELS_MAX)
            
            spacing = p_atr * self.cfg.GRID_SPACING_ATR
            
            max_pending_price = max([o['price'] for o in self.pending_orders]) if self.pending_orders else 0
            
            # Logic: Update pending orders if price moves too far or slots available
            slots_needed = target_levels - current_pos_count
            
            if slots_needed > 0:
                self.pending_orders = [] # Reset trailing
                start_price = p_close 
                self.last_grid_ref_price = start_price

                order_size_usd = current_equity * self.cfg.GRID_SIZE_PCT
                
                for lvl in range(1, slots_needed + 1):
                    price_lvl = start_price - (spacing * lvl)
                    if price_lvl > 0:
                        qty = order_size_usd / price_lvl
                        self.pending_orders.append({
                            'price': price_lvl,
                            'size': qty,
                            'created_at': ts
                        })

            # ----------------------------------------------
            # 4. VALUATION & NAV
            # ----------------------------------------------
            btc_val = sum(t['size'] * p_close for t in self.positions)
            opt_val = self._update_options_mtm(ts, p_close)
            
            total_equity_hedged = self.cash + btc_val + opt_val
            total_equity_gross = self.cash_unhedged + btc_val 
            bench_equity = self.bench_btc_held * p_close 

            self.nav_strat = total_equity_hedged / self.units_strat
            self.nav_gross = total_equity_gross / self.units_gross
            self.nav_bench = bench_equity / self.units_bench
            
            self.history.append({
                'timestamp': ts,
                'nav_net': self.nav_strat,
                'nav_gross': self.nav_gross,
                'nav_bench': self.nav_bench,
                'equity_net': total_equity_hedged,
                'equity_bench': bench_equity,
                'equity_gross': total_equity_gross,
                'cash': self.cash,
                'option_val': opt_val,
                'invested_capital': self.invested_capital
            })
            
            if i % 500 == 0:
                print(f"Simulating {ts} | NAV: {self.nav_strat:.4f}", end='\r')

        final_eq = self.history[-1]['equity_net']
        self.flows_strat.append((self.timeline[-1], final_eq))
        
        final_bench_eq = self.history[-1]['equity_bench']
        self.flows_bench.append((self.timeline[-1], final_bench_eq))
        
        return pd.DataFrame(self.history).set_index('timestamp'), self.flows_strat, self.flows_bench