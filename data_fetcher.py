# data_fetcher.py
import ccxt
import pandas as pd
import numpy as np
import time

class DataEngine:
    def __init__(self, config):
        self.cfg = config
        self.exchange = ccxt.binance()

    def fetch_data(self, symbol):
        print(f"[{symbol}] Fetching data...")
        
        since = self.exchange.parse8601(self.cfg.START_DATE)
        end_ts = self.exchange.parse8601(self.cfg.END_DATE)
        
        all_candles = []
        
        # Standard CCXT Loop
        while since < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, self.cfg.TIMEFRAME, since, limit=1000)
                if not ohlcv: 
                    break
                since = ohlcv[-1][0] + 1
                all_candles.extend(ohlcv)
                # Небольшая пауза, чтобы не получить бан API
                time.sleep(0.1)
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
                
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Risk Metrics Calculation
            df['returns'] = df['close'].pct_change()
            df['tr'] = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)))
            df['atr'] = df['tr'].rolling(window=14).mean()
            
            # Annualized Rolling Volatility (for Options Pricing)
            # 365 days for crypto
            df['rolling_vol_annual'] = df['returns'].rolling(window=self.cfg.VOL_LOOKBACK).std() * np.sqrt(365)
            
            # Удаляем дубликаты и обрезаем по даты конфига
            df = df[~df.index.duplicated(keep='first')]
            return df[(df.index >= self.cfg.START_DATE) & (df.index <= self.cfg.END_DATE)]
        else:
            print("Warning: Empty DataFrame returned.")
            return pd.DataFrame()

    def prepare(self):
        data = {}
        for sym in self.cfg.SYMBOLS:
            df = self.fetch_data(sym)
            data[sym] = df
        return data