# config.py
class Config:
    # ==========================================
    # CONFIGURATION (Research Grade)
    # ==========================================
    
    # --- Data ---
    START_DATE = "2018-01-01 00:00:00"
    END_DATE = "2026-01-15 00:00:00"
    TIMEFRAME = '30m'
    
    # --- Capital ---
    INITIAL_CAPITAL = 100.0
    MONTHLY_DEPOSIT = 300.0
    DEPOSIT_DAY = 9
    
    # --- Symbols ---
    SYMBOLS = ['BTCUSDT'] 
    
    # --- GRID CONFIGURATION ---
    GRID_TYPE = 'ATR' 
    GRID_LEVELS = 4
    GRID_LEVELS_MAX = 100       # Максимум (включая аварийный уровень)          
    GRID_SPACING_ATR = 1.5   
    GRID_SIZE_PCT = 0.2        # 20% от Equity на один ордер (в коде было 0.2)
    TAKE_PROFIT_PCT = 0.024    # Тейк профит на каждую сделку
    
    # --- OPTIONS HEDGING CONFIGURATION ---
    ENABLE_HEDGING = True
    HEDGE_STRIKE_PCT = 0.93    # Страховка от падения >10%
    VOL_LOOKBACK = 30          # Дней для расчета исторической волатильности
    VOL_RISK_PREMIUM = 1.15    # VRP
    RISK_FREE_RATE = 0.04    
    
    # --- Execution Reality ---
    FEE_RATE = 0.001           # 0.1% Taker/Maker avg
    SLIPPAGE_FACTOR = 0.05     # Влияние волатильности на цену исполнения

    # ==========================================
    # EXTRA BENCHMARKS CONFIGURATION
    # ==========================================
    ENABLE_ADVANCED_BENCHMARKS = True  # Включить расчет и доп. окно графиков
    
    # 1. Volatility Targeting
    BENCH_VOL_TARGET = 0.50            # Целевая волатильность (50% годовых)
    BENCH_VOL_LOOKBACK = 20            # Окно расчета волатильности
    
    # 2. Dynamic Exposure (Trend)
    BENCH_MA_PERIOD = 480              # Период скользящей средней (напр. 10 дней для 30м)
    
    # 3. Put Spread Structure
    # Мы покупаем Put @ STRIKE_LONG (дорогой) и продаем Put @ STRIKE_SHORT (дешевый)
    BENCH_PUT_SPREAD_LONG_PCT = 0.95   # Страйк покупки защиты (95% от цены)
    BENCH_PUT_SPREAD_SHORT_PCT = 0.80  # Страйк продажи защиты (80% от цены)