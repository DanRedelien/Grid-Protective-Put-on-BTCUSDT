import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
import warnings

class RiskResearch:
    
    @staticmethod
    def calculate_xirr(cashflows):
        """ 
        Robust Newton-Raphson XIRR calculation.
        Uses precise daily compounding logic matching institutional standards.
        """
        if not cashflows: return 0.0
        
        # Сортировка потоков по дате обязательна
        cf = sorted(cashflows, key=lambda x: x[0])
        start_date = cf[0][0]
        
        # Перевод дат в доли лет
        years = [(c[0] - start_date).days / 365.0 for c in cf]
        amounts = [c[1] for c in cf]
        
        def npv(rate):
            # Защита от деления на ноль и бессмысленных ставок
            if rate <= -0.99: return 1e9 
            return sum(a / ((1 + rate) ** y) for a, y in zip(amounts, years))
        
        try:
            return optimize.newton(npv, 0.1, maxiter=50)
        except:
            # Fallback если метод Ньютона не сходится
            return 0.0

    @staticmethod
    def calculate_rolling_var(returns, window=90, confidence=0.95):
        """ Calculates Historical Rolling VaR (Value at Risk) """
        return returns.rolling(window=window).quantile(1 - confidence)

    @staticmethod
    def analyze(df, flows_strat, flows_bench):
        print("\n\n" + "="*40)
        print("   INSTITUTIONAL RISK AUDIT REPORT")
        print("="*40)
        
        # 1. DATA PREPARATION
        # Собираем абсолютно все колонки, которые могут понадобиться для графиков и метрик
        target_cols = [
            'nav_net', 'nav_bench', 'nav_gross', 
            'equity_net', 'equity_bench', 'equity_gross', 
            'invested_capital', 'cash'
        ]
        
        # Фильтруем только те, что реально есть в DataFrame
        existing_cols = [c for c in target_cols if c in df.columns]
        
        # Ресемплинг на 1 день (берем последнее значение дня)
        daily = df[existing_cols].resample('1D').last().dropna()
        
        # Расчет доходностей (Returns) строго по NAV
        rets_net = daily['nav_net'].pct_change().dropna()
        
        if 'nav_bench' in daily.columns:
            rets_bench = daily['nav_bench'].pct_change().dropna()
        else:
            rets_bench = pd.Series(dtype=float)
            
        # 2. RETURN METRICS
        print(f"\n[1] RETURN METRICS (Annualized)")
        
        # MWR (Money Weighted Return) - XIRR
        mwr_strat = RiskResearch.calculate_xirr(flows_strat)
        mwr_bench = RiskResearch.calculate_xirr(flows_bench)
        
        # CAGR (Time Weighted Return) - Based on NAV
        days_total = (daily.index[-1] - daily.index[0]).days
        if days_total > 0:
            cagr_strat = (daily['nav_net'].iloc[-1] / daily['nav_net'].iloc[0]) ** (365/days_total) - 1
            if 'nav_bench' in daily.columns:
                cagr_bench = (daily['nav_bench'].iloc[-1] / daily['nav_bench'].iloc[0]) ** (365/days_total) - 1
            else:
                cagr_bench = 0.0
        else:
            cagr_strat = 0.0
            cagr_bench = 0.0
            
        print(f"{'Metric':<20} | {'Strategy (Net)':<15} | {'Benchmark (DCA)':<15}")
        print("-" * 55)
        print(f"{'MWR (XIRR)':<20} | {mwr_strat:.2%}          | {mwr_bench:.2%}")
        print(f"{'CAGR (TWR)':<20} | {cagr_strat:.2%}          | {cagr_bench:.2%}")
        
        # 3. DISTRIBUTIONAL RISK (Detailed Statistics from Old Version)
        print(f"\n[2] DISTRIBUTIONAL RISK")
        
        skew = stats.skew(rets_net)
        kurt = stats.kurtosis(rets_net)
        
        # Daily VaR (95%)
        var_95 = np.percentile(rets_net, 5)
        
        # Expected Shortfall (CVaR) - средний убыток в хвосте 5%
        cvar_95 = rets_net[rets_net <= var_95].mean()
        
        # Scaling to Monthly (для сравнения с традиционными фондами)
        scale_m = np.sqrt(30)
        var_95_m = var_95 * scale_m
        cvar_95_m = cvar_95 * scale_m
        
        print(f"Skewness:            {skew:.2f} (Pos=Right Tail, Neg=Crash Risk)")
        print(f"Kurtosis:            {kurt:.2f} (>3 = Fat Tails)")
        print(f"Daily VaR (95%):     {var_95:.2%}")
        print(f"Monthly VaR (95%):   {var_95_m:.2%} (Scaled)")
        print(f"Monthly CVaR (95%):  {cvar_95_m:.2%} (Expected Shortfall)")
        
        # 4. EFFICIENCY & RATIOS
        print(f"\n[3] EFFICIENCY & RATIOS")
        
        # Annualized Volatility
        vol_strat = rets_net.std() * np.sqrt(365)
        vol_bench = rets_bench.std() * np.sqrt(365) if not rets_bench.empty else 0.0
        
        # Max Drawdown (Calculated on NAV)
        dd_strat = (daily['nav_net'] / daily['nav_net'].cummax() - 1).min()
        dd_bench = 0.0
        if 'nav_bench' in daily.columns:
            dd_bench = (daily['nav_bench'] / daily['nav_bench'].cummax() - 1).min()
            
        # Sharpe Ratio (Assuming Rf approx 0 for simplicity in crypto context)
        if rets_net.std() > 0:
            sharpe = (rets_net.mean() / rets_net.std()) * np.sqrt(365)
        else:
            sharpe = 0.0
            
        # Sortino Ratio
        downside_rets = rets_net[rets_net < 0]
        if len(downside_rets) > 0 and downside_rets.std() > 0:
            sortino = (rets_net.mean() / downside_rets.std()) * np.sqrt(365)
        else:
            sortino = 0.0
            
        # MAR Ratio (CAGR / MaxDD)
        if dd_strat != 0:
            mar = cagr_strat / abs(dd_strat)
        else:
            mar = 0.0
            
        # Benchmark Sharpe
        if not rets_bench.empty and rets_bench.std() > 0:
            sharpe_bench = (rets_bench.mean() / rets_bench.std()) * np.sqrt(365)
        else:
            sharpe_bench = 0.0
            
        # Benchmark MAR
        if dd_bench != 0:
            mar_bench = cagr_bench / abs(dd_bench)
        else:
            mar_bench = 0.0

        print(f"{'Metric':<20} | {'Strategy':<15} | {'Benchmark':<15}")
        print("-" * 55)
        print(f"{'Volatility (Ann)':<20} | {vol_strat:.2%}          | {vol_bench:.2%}")
        print(f"{'Max Drawdown':<20} | {dd_strat:.2%}          | {dd_bench:.2%}")
        print(f"{'Sharpe Ratio':<20} | {sharpe:.2f}            | {sharpe_bench:.2f}")
        print(f"{'Sortino Ratio':<20} | {sortino:.2f}            | - ")
        print(f"{'MAR Ratio':<20} | {mar:.2f}            | {mar_bench:.2f}")

        # Возвращаем словарь метрик для использования в графиках
        return {
            'daily_stats': daily,
            'daily_rets': rets_net
        }

    @staticmethod
    def plot_dashboard(df_res, metrics):
        """
        Generates a 6-panel institutional dashboard.
        Includes both NAV-based performance and Absolute Wealth analysis.
        """
        daily = metrics['daily_stats']
        r_strat = metrics['daily_rets']
        
        # Настройка фигуры: 3 ряда, 2 колонки
        fig = plt.figure(figsize=(18, 14)) # Чуть выше высота, чтобы графики дышали
        layout = (3, 2)
        
        # =========================================================
        # GRAPH 1 (Top Left): SKILL (NAV Growth)
        # Classic comparison of strategy efficiency vs Benchmark
        # =========================================================
        ax1 = plt.subplot2grid(layout, (0, 0))
        
        # Main Strategy Line
        ax1.plot(daily.index, daily['nav_net'], label='Strategy NAV (Net)', color='blue', linewidth=2)
        
        # Unhedged Line (Optional - если данные есть)
        if 'nav_gross' in daily.columns:
            ax1.plot(daily.index, daily['nav_gross'], label='Strategy NAV (Gross)', 
                     color='green', linestyle=':', alpha=0.5, linewidth=1)
            
        # Benchmark Line
        if 'nav_bench' in daily.columns:
            ax1.plot(daily.index, daily['nav_bench'], label='Benchmark NAV (DCA)', 
                     color='gray', linestyle='--', alpha=0.8, linewidth=1.5)
        
        ax1.set_title("1. SKILL: Growth of $1 (NAV Basis)")
        ax1.set_ylabel("NAV Multiplier")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.2)
        
        # =========================================================
        # GRAPH 2 (Top Right): WEALTH (Absolute $)
        # "Burning Money Check": Equity vs Invested Capital
        # =========================================================
        ax2 = plt.subplot2grid(layout, (0, 1))
        
        # Красная линия: Сколько денег мы положили (Invested Capital)
        ax2.step(daily.index, daily['invested_capital'], where='post', 
                 label='Invested Capital', color='red', linestyle='--', linewidth=2)
        
        # Синяя линия: Сколько денег у нас есть (Total Equity)
        ax2.plot(daily.index, daily['equity_net'], label='Total Equity ($)', color='blue', linewidth=1.5)
        
        # Зоны: Зеленая = Работаем в плюс от депозита, Красная = Сжигаем депозит
        ax2.fill_between(daily.index, daily['invested_capital'], daily['equity_net'],
                         where=(daily['equity_net'] >= daily['invested_capital']),
                         color='green', alpha=0.1)
        ax2.fill_between(daily.index, daily['invested_capital'], daily['equity_net'],
                         where=(daily['equity_net'] < daily['invested_capital']),
                         color='red', alpha=0.2, label='BURNING MONEY ZONE')
        
        ax2.set_title("2. WEALTH: Equity vs Deposits ($)")
        ax2.set_ylabel("Account Balance ($)")
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.2)
        
        # =========================================================
        # GRAPH 3 (Mid Left): DRAWDOWN (NAV Based)
        # =========================================================
        ax3 = plt.subplot2grid(layout, (1, 0))
        
        dd_strat = daily['nav_net'] / daily['nav_net'].cummax() - 1
        ax3.fill_between(dd_strat.index, dd_strat, 0, color='blue', alpha=0.3, label='Strategy DD')
        
        if 'nav_bench' in daily.columns:
            dd_bench = daily['nav_bench'] / daily['nav_bench'].cummax() - 1
            ax3.plot(dd_bench.index, dd_bench, color='gray', alpha=0.5, linewidth=1, label='Benchmark DD')
            
        ax3.set_title("3. RISK: Drawdown from Peak (NAV)")
        ax3.set_ylabel("% Drawdown")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # =========================================================
        # GRAPH 4 (Mid Right): ROLLING VOLATILITY
        # =========================================================
        ax4 = plt.subplot2grid(layout, (1, 1))
        
        rolling_vol = r_strat.rolling(30).std() * np.sqrt(365)
        ax4.plot(rolling_vol.index, rolling_vol, color='orange', label='30D Rolling Vol (Ann)')
        
        ax4.set_title("4. RISK: Rolling Volatility (Stability)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # =========================================================
        # GRAPH 5 (Bot Left): RETURNS DISTRIBUTION
        # =========================================================
        ax5 = plt.subplot2grid(layout, (2, 0))
        
        ax5.hist(r_strat, bins=50, density=True, alpha=0.6, color='purple', label='Daily Returns')
        
        # Fitting Normal Distribution for comparison
        if len(r_strat) > 10:
            mu, std = stats.norm.fit(r_strat)
            xmin, xmax = ax5.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, std)
            ax5.plot(x, p, 'k', linewidth=2, label='Normal Dist')
            
        ax5.set_title("5. DISTRIBUTION: Fat Tails Analysis")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # =========================================================
        # GRAPH 6 (Bot Right): ROLLING VaR
        # =========================================================
        ax6 = plt.subplot2grid(layout, (2, 1))
        
        # Calculate Rolling 95% VaR
        roll_var = r_strat.rolling(window=30).quantile(0.05)
        
        ax6.plot(roll_var.index, roll_var, color='red', label='95% Rolling VaR (30d)')
        ax6.set_title("6. TAIL RISK: Rolling Historical VaR")
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()