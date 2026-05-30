"""
Data Manager for Volatility Engine
Handles data fetching from yfinance and Bloomberg (stub)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class DataManager:
    """
    Unified interface for market data from multiple sources
    """
    
    def __init__(self, source: str = 'yfinance'):
        """
        Initialize data manager
        
        Args:
            source: 'yfinance' or 'bloomberg'
        """
        self.source = source.lower()
        if self.source not in ['yfinance', 'bloomberg']:
            raise ValueError("Source must be 'yfinance' or 'bloomberg'")
    
    def get_spot_price(self, ticker: str) -> float:
        """
        Get current spot price for underlying
        
        Args:
            ticker: Asset ticker (e.g., 'AAPL', 'EURUSD=X')
        
        Returns:
            Current spot price
        """
        if self.source == 'yfinance':
            return self._yf_get_spot(ticker)
        else:
            return self._bbg_get_spot(ticker)
    
    def get_historical_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical OHLCV data
        
        Args:
            ticker: Asset ticker
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        if self.source == 'yfinance':
            return self._yf_get_historical(ticker, start_date, end_date)
        else:
            return self._bbg_get_historical(ticker, start_date, end_date)
    
    def get_option_chain(self, ticker: str, expiry_date: Optional[str] = None) -> Dict:
        """
        Get option chain for specific expiry or all available expiries
        
        Args:
            ticker: Asset ticker
            expiry_date: Optional specific expiry date 'YYYY-MM-DD'
        
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames
        """
        if self.source == 'yfinance':
            return self._yf_get_options(ticker, expiry_date)
        else:
            return self._bbg_get_options(ticker, expiry_date)
    
    def calculate_implied_volatility_surface(self, ticker: str) -> pd.DataFrame:
        """
        Build implied volatility surface from option prices
        
        Returns:
            DataFrame with columns: Strike, Maturity, IV_Call, IV_Put, Mid_IV
        """
        if self.source == 'yfinance':
            return self._yf_build_iv_surface(ticker)
        else:
            return self._bbg_build_iv_surface(ticker)
    
    # ========== yfinance Implementation ==========
    
    def _yf_get_spot(self, ticker: str) -> float:
        """Get spot price from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            # Get most recent price from history
            hist = stock.history(period='1d')
            if hist.empty:
                raise ValueError(f"No data available for {ticker}")
            return hist['Close'].iloc[-1]
        except Exception as e:
            raise RuntimeError(f"Error fetching spot price for {ticker}: {str(e)}")
    
    def _yf_get_historical(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical data from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No historical data for {ticker}")
            return data[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            raise RuntimeError(f"Error fetching historical data: {str(e)}")
    
    def _yf_get_options(self, ticker: str, expiry_date: Optional[str] = None) -> Dict:
        """Get option chain from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            
            if not expirations:
                raise ValueError(f"No options available for {ticker}")
            
            if expiry_date:
                if expiry_date not in expirations:
                    raise ValueError(f"Expiry {expiry_date} not available. Available: {expirations}")
                opt = stock.option_chain(expiry_date)
            else:
                # Get nearest expiry
                opt = stock.option_chain(expirations)
            
            return {
                'calls': opt.calls,
                'puts': opt.puts,
                'expiry': expiry_date or expirations,
                'all_expiries': list(expirations)
            }
        except Exception as e:
            raise RuntimeError(f"Error fetching options: {str(e)}")
    
    def _yf_build_iv_surface(self, ticker: str) -> pd.DataFrame:
        """
        Build IV surface from all available option expiries
        Uses Black-Scholes implied volatility calculation
        """
        from scipy.stats import norm
        from scipy.optimize import brentq
        
        stock = yf.Ticker(ticker)
        S0 = self._yf_get_spot(ticker)
        expirations = stock.options
        
        surface_data = []
        
        # Risk-free rate approximation (use US 3-month Treasury as proxy)
        # In production, fetch actual risk-free rate
        r = 0.045  # ~4.5% as of 2024
        
        for expiry in expirations[:6]:  # Limit to first 6 expiries for speed
            try:
                opt = stock.option_chain(expiry)
                expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
                T = (expiry_date - datetime.now()).days / 365.0
                
                if T <= 0:
                    continue
                
                # Process calls
                for _, row in opt.calls.iterrows():
                    K = row['strike']
                    price = row['lastPrice']
                    
                    if price > 0.01 and K > 0:  # Valid price
                        try:
                            iv = self._calculate_iv(S0, K, T, r, price, 'call')
                            surface_data.append({
                                'Strike': K,
                                'Maturity': T,
                                'Type': 'Call',
                                'IV': iv,
                                'Price': price
                            })
                        except:
                            pass
                
                # Process puts
                for _, row in opt.puts.iterrows():
                    K = row['strike']
                    price = row['lastPrice']
                    
                    if price > 0.01 and K > 0:
                        try:
                            iv = self._calculate_iv(S0, K, T, r, price, 'put')
                            surface_data.append({
                                'Strike': K,
                                'Maturity': T,
                                'Type': 'Put',
                                'IV': iv,
                                'Price': price
                            })
                        except:
                            pass
            except:
                continue
        
        return pd.DataFrame(surface_data)
    
    def _calculate_iv(self, S: float, K: float, T: float, r: float, 
                      price: float, option_type: str) -> float:
        """
        Calculate implied volatility using Black-Scholes
        
        Black-Scholes Formula:
        C = S*N(d1) - K*exp(-r*T)*N(d2)  [Call]
        P = K*exp(-r*T)*N(-d2) - S*N(-d1)  [Put]
        
        where:
        d1 = [ln(S/K) + (r + σ²/2)*T] / (σ*√T)
        d2 = d1 - σ*√T
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            price: Observed option price
            option_type: 'call' or 'put'
        
        Returns:
            Implied volatility (annualized)
        """
        from scipy.stats import norm
        from scipy.optimize import brentq
        
        def bs_price(sigma):
            """Black-Scholes pricing formula"""
            if sigma <= 0 or T <= 0:
                return 0
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        # Objective function: difference between market and model price
        objective = lambda sigma: bs_price(sigma) - price
        
        try:
            # Search for IV between 0.01% and 500%
            iv = brentq(objective, 0.0001, 5.0, maxiter=100)
            return iv
        except:
            # If brentq fails, return NaN
            return np.nan
    
    # ========== Bloomberg Stub Implementation ==========
    
    def _bbg_get_spot(self, ticker: str) -> float:
        """
        Bloomberg stub - replace with actual Bloomberg API implementation
        
        To implement:
        1. Install: pip install blpapi
        2. Import: from blpapi import Session, SessionOptions
        3. Connect to Bloomberg Terminal or B-PIPE
        4. Use refDataService to fetch PX_LAST field
        """
        raise NotImplementedError(
            "Bloomberg integration requires blpapi package and active Bloomberg subscription.\n"
            "Replace this method with actual Bloomberg API calls.\n"
            "Example placeholder: return bbg_session.bdp(ticker, 'PX_LAST')"
        )
    
    def _bbg_get_historical(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Bloomberg historical data stub"""
        raise NotImplementedError(
            "Implement using Bloomberg bdh() function to fetch OHLC data"
        )
    
    def _bbg_get_options(self, ticker: str, expiry_date: Optional[str] = None) -> Dict:
        """Bloomberg options chain stub"""
        raise NotImplementedError(
            "Implement using Bloomberg OMON<GO> function or API equivalent"
        )
    
    def _bbg_build_iv_surface(self, ticker: str) -> pd.DataFrame:
        """Bloomberg IV surface stub"""
        raise NotImplementedError(
            "Implement using Bloomberg OVME<GO> function or fetch from IVOL field"
        )

# ========== Risk-Free Rate Helper ==========

def get_risk_free_rate(currency: str = 'USD') -> float:
    """
    Get risk-free rate for given currency
    
    In production, fetch from:
    - USD: US Treasury rates
    - EUR: EURIBOR
    - SGD: Singapore Government Securities
    - JPY: JGB rates
    
    Args:
        currency: Currency code
    
    Returns:
        Annualized risk-free rate
    """
    # Placeholder rates (update with real-time data in production)
    rates = {
        'USD': 0.045,  # ~4.5% US Treasury
        'EUR': 0.035,  # ~3.5% EURIBOR
        'SGD': 0.038,  # ~3.8% SGS
        'JPY': 0.001,  # ~0.1% JGB
        'GBP': 0.048,  # ~4.8% Gilts
    }
    
    return rates.get(currency.upper(), 0.03)  # Default 3%

# ========== Utility Functions ==========

def get_ticker_info(ticker: str) -> Dict:
    """
    Get asset information
    
    Returns:
        Dictionary with asset type, currency, etc.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    return {
        'name': info.get('longName', ticker),
        'currency': info.get('currency', 'USD'),
        'asset_type': 'FX' if '=X' in ticker else 'Equity',
        'exchange': info.get('exchange', 'Unknown')
    }
