import numpy as np
from scipy.stats import norm

class BlackScholes:
    """
    Minimalist Black-Scholes-Merton pricer for European Options.
    Assumptions for Research:
    1. No dividends (Crypto spot).
    2. Constant volatility over the option life (Model simplification).
    3. Risk-free rate is constant.
    """
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        if sigma <= 0 or T <= 0: return 0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S, K, T, r, sigma):
        if sigma <= 0 or T <= 0: return 0
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def price_put(S, K, T, r, sigma):
        """
        Calculates the theoretical price of a European Put.
        S: Spot Price
        K: Strike Price
        T: Time to maturity (in years)
        r: Risk-free rate (decimal)
        sigma: Implied Volatility (decimal)
        """
        # Boundary checks
        if T <= 0: return max(K - S, 0.0)
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        put_price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
        return max(put_price, 0.0)

class HedgingManager:
    """
    Manages the lifecycle of a monthly rolling hedge.
    Strategy: Protective Put (Long Put).
    """
    def __init__(self, risk_free_rate=0.04, vol_risk_premium=1.1):
        self.rf = risk_free_rate
        # VRP: Market Makers charge more than realized vol. 
        # 1.1 means we pay 10% over realized volatility for the option.
        self.vol_risk_premium = vol_risk_premium 
        
        self.active_option = None
        self.cumulative_cost = 0.0
        self.cumulative_payoff = 0.0

    def open_hedge(self, symbol, spot_price, annualized_vol, days_to_expiry=30, strike_pct=0.95):
        """
        Opens a new Put Option.
        strike_pct: 0.95 means 5% OTM Put.
        """
        # Proxies IV as Realized Vol * Risk Premium
        implied_vol_proxy = annualized_vol * self.vol_risk_premium
        
        strike_price = spot_price * strike_pct
        t_years = days_to_expiry / 365.0
        
        premium_per_unit = BlackScholes.price_put(
            S=spot_price,
            K=strike_price,
            T=t_years,
            r=self.rf,
            sigma=implied_vol_proxy
        )
        
        self.active_option = {
            'symbol': symbol,
            'strike': strike_price,
            'premium': premium_per_unit,
            'open_price': spot_price,
            'vol_used': implied_vol_proxy,
            'expiry_days': days_to_expiry
        }
        
        self.cumulative_cost += premium_per_unit
        return self.active_option

    def settle_hedge(self, current_spot_price):
        """
        Calculates payoff at expiry: Max(Strike - Spot, 0)
        """
        if not self.active_option:
            return 0.0
            
        strike = self.active_option['strike']
        payoff = max(strike - current_spot_price, 0.0)
        
        self.cumulative_payoff += payoff
        self.active_option = None # Hedge is now closed
        
        return payoff

    def get_mtm(self, current_spot, time_passed_days):
        """
        Mark-to-Market valuation for mid-month reporting.
        """
        if not self.active_option:
            return 0.0
            
        remaining_days = self.active_option['expiry_days'] - time_passed_days
        if remaining_days <= 0:
            return max(self.active_option['strike'] - current_spot, 0.0)
            
        t_years = remaining_days / 365.0
        
        return BlackScholes.price_put(
            S=current_spot,
            K=self.active_option['strike'],
            T=t_years,
            r=self.rf,
            sigma=self.active_option['vol_used']
        )