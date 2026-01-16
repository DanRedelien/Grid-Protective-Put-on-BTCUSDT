# advanced_benchmarks.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dateutil.relativedelta import relativedelta

try:
    from options import HedgingManager
except ImportError:
    HedgingManager = None

class AdvancedBenchmarks:
    def __init__(self, config, data_df):
        self.cfg = config
        self.df = data_df.copy()
        
        # Подготовка данных (Intraday)
        self.df['ret'] = self.df['close'].pct_change().fillna(0)
        self.df['ma_trend'] = self.df['close'].rolling(window=self.cfg.BENCH_MA_PERIOD).mean()
        # Годовая волатильность для 30м таймфрейма: sqrt(365 * 48)
        # 48 получасовок в сутках
        self.df['realized_vol'] = self.df['ret'].rolling(window=self.cfg.BENCH_VOL_LOOKBACK).std() * np.sqrt(365 * 48) 

    def run_vol_targeting(self):
        """ Target Volatility Strategy """
        print("Calculating Benchmark: Volatility Targeting...")
        target_vol = self.cfg.BENCH_VOL_TARGET
        
        weights = target_vol / self.df['realized_vol'].shift(1)
        weights = weights.replace([np.inf, -np.inf], 0).fillna(0)
        weights = np.clip(weights, 0.0, 1.0)
        
        strat_ret = weights * self.df['ret']
        nav = (1 + strat_ret).cumprod()
        return nav

    def run_dynamic_exposure(self):
        """ Trend Following Strategy """
        print("Calculating Benchmark: Dynamic Exposure (Trend)...")
        signal = np.where(self.df['close'].shift(1) > self.df['ma_trend'].shift(1), 1.0, 0.0)
        strat_ret = signal * self.df['ret']
        nav = (1 + strat_ret).cumprod()
        return nav

    def run_put_spread(self):
        """ Collar Strategy (Long Put + Short Put) """
        print("Calculating Benchmark: BTC + Put Spread...")
        
        if not HedgingManager:
            print("Options module not found, skipping Put Spread.")
            return pd.Series(1.0, index=self.df.index)

        hedger_long = HedgingManager(self.cfg.RISK_FREE_RATE, self.cfg.VOL_RISK_PREMIUM)
        hedger_short = HedgingManager(self.cfg.RISK_FREE_RATE, self.cfg.VOL_RISK_PREMIUM)
        
        cash = 1.0 
        btc_units = 1.0 / self.df['open'].iloc[0]
        
        nav_history = []
        timeline = self.df.index
        prev_month = None
        
        for i, ts in enumerate(timeline):
            price = self.df['close'].iloc[i]
            
            # --- MONTHLY ROLL ---
            if prev_month != ts.month:
                if prev_month is not None:
                    # Settle
                    payoff_long = hedger_long.settle_hedge(price) 
                    payoff_short = hedger_short.settle_hedge(price) 
                    net_payoff = payoff_long - payoff_short
                    cash += net_payoff * btc_units 
                    
                    # Open New
                    vol = self.df['rolling_vol_annual'].iloc[i-1]
                    if np.isnan(vol): vol = 0.6
                    
                    next_month_date = ts + relativedelta(months=1)
                    days = max((next_month_date - ts).days, 1)
                    
                    # Long Put
                    opt_long = hedger_long.open_hedge('BTC', price, vol, days, self.cfg.BENCH_PUT_SPREAD_LONG_PCT)
                    hedger_long.active_option['open_time'] = ts
                    cost_long = opt_long['premium']
                    
                    # Short Put
                    opt_short = hedger_short.open_hedge('BTC', price, vol, days, self.cfg.BENCH_PUT_SPREAD_SHORT_PCT)
                    hedger_short.active_option['open_time'] = ts
                    premium_received = opt_short['premium']
                    
                    cash -= (cost_long - premium_received) * btc_units
                
                prev_month = ts.month
            
            # --- VALUATION ---
            days_passed = 0
            if hedger_long.active_option and 'open_time' in hedger_long.active_option:
                days_passed = (ts - pd.to_datetime(hedger_long.active_option['open_time'])).total_seconds() / 86400.0
                mtm_long = hedger_long.get_mtm(price, days_passed)
            else: mtm_long = 0
                
            if hedger_short.active_option and 'open_time' in hedger_short.active_option:
                mtm_short = hedger_short.get_mtm(price, days_passed)
            else: mtm_short = 0
            
            total_equity = cash + (btc_units * price) + (mtm_long * btc_units) - (mtm_short * btc_units)
            nav_history.append({'ts': ts, 'nav': total_equity})
            
        res_df = pd.DataFrame(nav_history).set_index('ts')
        if not res_df.empty:
            return res_df['nav'] / res_df['nav'].iloc[0]
        else:
            return pd.Series()

    def run_all(self):
        results = {}
        if len(self.df) > 100:
            results['Vol Target'] = self.run_vol_targeting()
            results['Trend Follow'] = self.run_dynamic_exposure()
            try:
                results['Put Spread'] = self.run_put_spread()
            except Exception as e:
                print(f"Error calculating Put Spread benchmark: {e}")
        return pd.DataFrame(results)

    # --- REPORTING & ANALYTICS ---
    
    def _calculate_metrics(self, nav_series):
        """ 
        Calculates comprehensive risk metrics using DAILY resampling.
        This ensures consistency with the main logger report.
        """
        if nav_series.empty: 
            return [0]*10
            
        # FIX: Force Resample to 1 Day to align with Institutional Standards
        # and match the logic of logger_and_analytics.py
        nav_daily = nav_series.resample('1D').last().dropna()
        
        rets = nav_daily.pct_change().dropna()
        if rets.empty: return [0]*10

        # 1. Basic
        days = (nav_daily.index[-1] - nav_daily.index[0]).days
        cagr = (nav_daily.iloc[-1] / nav_daily.iloc[0]) ** (365/days) - 1 if days > 0 else 0
        vol = rets.std() * np.sqrt(365)
        dd = (nav_daily / nav_daily.cummax() - 1).min()
        sharpe = (rets.mean() / rets.std()) * np.sqrt(365) if rets.std() > 0 else 0
        
        # 2. Distributional (Tails)
        skew = stats.skew(rets)
        kurt = stats.kurtosis(rets)
        
        # VaR & CVaR
        var_95 = np.percentile(rets, 5)
        cvar_95 = rets[rets <= var_95].mean()
        
        # Scaled
        scale_m = np.sqrt(30)
        var_95_m = var_95 * scale_m
        cvar_95_m = cvar_95 * scale_m
        
        return cagr, vol, dd, sharpe, skew, kurt, var_95, var_95_m, cvar_95_m

    def print_performance_report(self, main_nav, bench_df, dca_nav=None):
        """ Prints detailed comparison table """
        print("\n" + "="*140)
        print(f"{'ADVANCED STRATEGY COMPARISON & TAIL RISK REPORT (DAILY BASIS)':^140}")
        print("="*140)
        
        # Headers
        headers = f"{'Strategy':<20} | {'CAGR':<8} | {'Vol':<8} | {'MaxDD':<8} | {'Sharpe':<6} | {'Skew':<6} | {'Kurt':<6} | {'D.VaR':<8} | {'M.VaR':<8} | {'M.CVaR':<8}"
        print(headers)
        print("-" * 140)

        # 1. Align Data
        strategies = {'Your Strategy': main_nav}
        if dca_nav is not None:
            strategies['Benchmark (DCA)'] = dca_nav
        for col in bench_df.columns:
            strategies[col] = bench_df[col]

        # Use 30m index for alignment first
        common_index = main_nav.index
        for name, series in strategies.items():
            if not series.empty:
                common_index = common_index.intersection(series.index)
        
        # 2. Calculate & Print (Metrics will resample internally)
        for name, series in strategies.items():
            # Align raw data first
            aligned_series = series.loc[common_index]
            
            # Calculate metrics (Correctly resampled to Daily)
            m = self._calculate_metrics(aligned_series)
            
            prefix = ">> " if name == "Your Strategy" else "   "
            
            row = (f"{prefix + name:<20} | {m[0]:>8.2%} | {m[1]:>8.2%} | {m[2]:>8.2%} | {m[3]:>6.2f} | "
                   f"{m[4]:>6.2f} | {m[5]:>6.2f} | {m[6]:>8.2%} | {m[7]:>8.2%} | {m[8]:>8.2%}")
            print(row)
        
        print("-" * 140)

    @staticmethod
    def plot_comparative_dashboard(main_nav, benchmarks_df, dca_nav=None):
        """
        Plots NAV and Drawdown comparison with soft colors
        """
        # Soft Color Palette
        colors = {
            'Your Strategy': 'blue', 
            'Benchmark (DCA)': 'darkgray', 
            'Vol Target': 'mediumorchid', # Soft Purple
            'Trend Follow': 'sandybrown',  # Soft Orange
            'Put Spread': 'mediumseagreen' # Soft Green
        }
        
        # Create Subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # --- 1. NAV PLOT ---
        # Main
        ax1.plot(main_nav.index, main_nav, label='YOUR STRATEGY', color=colors['Your Strategy'], linewidth=2.5, zorder=10)
        
        # DCA (Context)
        if dca_nav is not None:
            common_idx = main_nav.index.intersection(dca_nav.index)
            ax1.plot(common_idx, dca_nav.loc[common_idx], label='Benchmark (DCA)', 
                     color=colors['Benchmark (DCA)'], linestyle='--', linewidth=1.5, alpha=0.8)

        # Benchmarks
        for name, series in benchmarks_df.items():
            common_idx = main_nav.index.intersection(series.index)
            color = colors.get(name, 'black')
            ax1.plot(common_idx, series.loc[common_idx], label=f'{name}', color=color, linewidth=1.5, alpha=0.9)
            
        ax1.set_title("COMPARATIVE PERFORMANCE: Growth of $1")
        ax1.set_ylabel("NAV Multiplier")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.2)

        # --- 2. DRAWDOWN PLOT ---
        def calc_dd(series):
            # Resample for visual consistency with metrics (Optional, but cleaner)
            # Keeping high res for charts is okay, but daily looks smoother
            return series / series.cummax() - 1

        # Main DD
        dd_main = calc_dd(main_nav)
        ax2.fill_between(dd_main.index, dd_main, 0, color=colors['Your Strategy'], alpha=0.2, label='Your Strategy DD')
        ax2.plot(dd_main.index, dd_main, color=colors['Your Strategy'], linewidth=1)

        # DCA DD
        if dca_nav is not None:
            dd_dca = calc_dd(dca_nav.loc[main_nav.index.intersection(dca_nav.index)])
            ax2.plot(dd_dca.index, dd_dca, color=colors['Benchmark (DCA)'], linestyle='--', linewidth=1, alpha=0.6)

        # Benchmarks DD
        for name, series in benchmarks_df.items():
            common_idx = main_nav.index.intersection(series.index)
            dd_bench = calc_dd(series.loc[common_idx])
            color = colors.get(name, 'black')
            ax2.plot(dd_bench.index, dd_bench, color=color, linewidth=1, alpha=0.8)

        ax2.set_title("COMPARATIVE RISK: Drawdowns")
        ax2.set_ylabel("% Drawdown")
        ax2.grid(True, alpha=0.2)
        
        plt.tight_layout()