# main.py
import warnings
import matplotlib.pyplot as plt
from config import Config
from data_fetcher import DataEngine
from backtest_engine import BacktestEngine
from logger_and_analytics import RiskResearch

try:
    from advanced_benchmarks import AdvancedBenchmarks
except ImportError:
    AdvancedBenchmarks = None
    print("Warning: advanced_benchmarks.py not found.")

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # 1. Initialize Configuration
    cfg = Config()
    
    # 2. Data Preparation
    eng = DataEngine(cfg)
    data = eng.prepare()
    
    if 'BTCUSDT' in data and not data['BTCUSDT'].empty:
        # --- MAIN STRATEGY RUN ---
        bt = BacktestEngine(cfg, data)
        df_res, flows_strat, flows_bench = bt.run()
        
        # --- ANALYTICS WINDOW 1 (Main Report) ---
        met = RiskResearch.analyze(df_res, flows_strat, flows_bench)
        RiskResearch.plot_dashboard(df_res, met)
        
        # --- ADVANCED BENCHMARKS WINDOW 2 & LOG ---
        if cfg.ENABLE_ADVANCED_BENCHMARKS and AdvancedBenchmarks:
            print("\n" + "="*40)
            print("   RUNNING ADVANCED BENCHMARKS...")
            print("="*40)
            
            adv_bench = AdvancedBenchmarks(cfg, data['BTCUSDT'])
            bench_results = adv_bench.run_all()
            
            # Извлекаем DCA NAV
            dca_nav = df_res['nav_bench'] if 'nav_bench' in df_res else None
            
            # 1. Полный отчет в терминал
            adv_bench.print_performance_report(df_res['nav_net'], bench_results, dca_nav)
            
            # 2. График сравнения (теперь с DCA и Drawdowns)
            AdvancedBenchmarks.plot_comparative_dashboard(
                df_res['nav_net'], 
                bench_results, 
                dca_nav
            )

        plt.show()
        
    else:
        print("Simulation aborted: No data available for BTCUSDT.")