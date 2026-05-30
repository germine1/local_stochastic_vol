"""
Dupire Local Volatility Model

The Dupire equation provides local volatility σ_L(K,T) that fits the market
implied volatility surface exactly.

Dupire's Formula:
σ_L²(K,T) = [∂C/∂T + r*K*∂C/∂K] / [0.5*K²*∂²C/∂K²]

where C(K,T) is the call option price for strike K and maturity T
"""

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional

class DupireLocalVol:
    """
    Dupire Local Volatility Model
    
    Attributes:
        S0: Initial spot price
        r: Risk-free rate
        q: Dividend yield
        local_vol_surface: 2D grid of local volatilities σ_L(K,T)
    """
    
    def __init__(self, S0: float, r: float, q: float = 0.0):
        """
        Initialize Dupire model
        
        Args:
            S0: Current spot price
            r: Risk-free rate (annualized)
            q: Dividend yield (annualized)
        """
        self.S0 = S0
        self.r = r
        self.q = q
        self.local_vol_surface = None
        self.strikes = None
        self.maturities = None
        
    def calibrate_from_iv_surface(self, iv_surface: pd.DataFrame,
                                   smooth: bool = True) -> np.ndarray:
        """
        Calibrate local volatility from implied volatility surface
        
        Dupire Formula Derivation:
        Starting from option pricing PDE and market call prices C(K,T),
        local variance is:
        
        σ_L²(K,T) = 2*[∂C/∂T + (r-q)*K*∂C/∂K + q*C] / [K²*∂²C/∂K²]
        
        Numerically computed using finite differences on call price surface
        
        Args:
            iv_surface: DataFrame with columns ['Strike', 'Maturity', 'IV']
            smooth: Apply Gaussian smoothing to reduce numerical noise
        
        Returns:
            2D array of local volatilities
        """
        # Create regular grid
        strikes = np.sort(iv_surface['Strike'].unique())
        maturities = np.sort(iv_surface['Maturity'].unique())
        
        # Build call price surface from Black-Scholes using implied vols
        call_surface = np.zeros((len(maturities), len(strikes)))
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                # Find IV for this (K, T) point
                iv_point = iv_surface[
                    (iv_surface['Strike'] == K) & 
                    (iv_surface['Maturity'] == T)
                ]
                
                if not iv_point.empty:
                    iv = iv_point['IV'].iloc
                    # Calculate Black-Scholes call price
                    call_surface[i, j] = self._black_scholes_call(
                        self.S0, K, T, self.r, self.q, iv
                    )
                else:
                    # Interpolate if missing
                    call_surface[i, j] = np.nan
        
        # Interpolate missing values
        call_surface = self._interpolate_missing(call_surface)
        
        if smooth:
            # Apply Gaussian filter to smooth surface (reduces finite difference noise)
            call_surface = gaussian_filter(call_surface, sigma=1.0)
        
        # Compute derivatives using central finite differences
        # ∂C/∂T: derivative with respect to maturity
        dC_dT = np.gradient(call_surface, maturities, axis=0)
        
        # ∂C/∂K: first derivative with respect to strike
        dC_dK = np.gradient(call_surface, strikes, axis=1)
        
        # ∂²C/∂K²: second derivative with respect to strike
        d2C_dK2 = np.gradient(dC_dK, strikes, axis=1)
        
        # Dupire formula: σ_L²(K,T)
        numerator = 2 * (dC_dT + (self.r - self.q) * strikes[np.newaxis, :] * dC_dK + 
                        self.q * call_surface)
        denominator = strikes[np.newaxis, :]**2 * d2C_dK2
        
        # Avoid division by zero and negative local variance
        local_var = np.where(
            denominator > 1e-10,
            numerator / denominator,
            0.0
        )
        
        # Local variance must be non-negative
        local_var = np.maximum(local_var, 1e-6)
        
        # Local volatility is square root of local variance
        self.local_vol_surface = np.sqrt(local_var)
        self.strikes = strikes
        self.maturities = maturities
        
        return self.local_vol_surface
    
    def get_local_vol(self, K: float, T: float) -> float:
        """
        Get local volatility for specific strike and maturity
        
        Uses bilinear interpolation on calibrated surface
        
        Args:
            K: Strike price
            T: Time to maturity (years)
        
        Returns:
            Local volatility σ_L(K,T)
        """
        if self.local_vol_surface is None:
            raise ValueError("Model not calibrated. Run calibrate_from_iv_surface first.")
        
        # Create interpolator
        interpolator = RectBivariateSpline(
            self.maturities, self.strikes, self.local_vol_surface
        )
        
        return float(interpolator(T, K)[0, 0])
    
    def simulate_path(self, T: float, n_steps: int, 
                     n_paths: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate price paths using local volatility
        
        SDE: dS_t = (r - q)*S_t*dt + σ_L(S_t, t)*S_t*dW_t
        
        Discretization (Euler scheme):
        S_{t+dt} = S_t * exp[(r - q - 0.5*σ_L²)*dt + σ_L*sqrt(dt)*Z]
        
        where Z ~ N(0,1)
        
        Args:
            T: Final time (years)
            n_steps: Number of time steps
            n_paths: Number of simulation paths
        
        Returns:
            times: Array of time points
            paths: Array of simulated prices (n_paths × n_steps)
        """
        if self.local_vol_surface is None:
            raise ValueError("Model not calibrated.")
        
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        for i in range(n_steps):
            t = times[i]
            S = paths[:, i]
            
            # Get local vol for current (S, t) - vectorized
            local_vols = np.array([
                self.get_local_vol(s, max(t, 1e-6)) for s in S
            ])
            
            # Generate random shocks
            Z = np.random.standard_normal(n_paths)
            
            # Euler discretization with log-normal correction
            drift = (self.r - self.q - 0.5 * local_vols**2) * dt
            diffusion = local_vols * np.sqrt(dt) * Z
            
            paths[:, i + 1] = S * np.exp(drift + diffusion)
        
        return times, paths
    
    def _black_scholes_call(self, S: float, K: float, T: float, 
                           r: float, q: float, sigma: float) -> float:
        """
        Black-Scholes call option price
        
        Formula:
        C = S*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)
        
        d1 = [ln(S/K) + (r - q + σ²/2)*T] / (σ*√T)
        d2 = d1 - σ*√T
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility
        
        Returns:
            Call option price
        """
        from scipy.stats import norm
        
        if T <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        return call
    
    def _interpolate_missing(self, surface: np.ndarray) -> np.ndarray:
        """Fill missing values using linear interpolation"""
        from scipy.interpolate import griddata
        
        # Get valid points
        valid = ~np.isnan(surface)
        coords = np.array(np.where(valid)).T
        values = surface[valid]
        
        # Get all points
        all_coords = np.array(np.where(np.ones_like(surface))).T
        
        # Interpolate
        filled = griddata(coords, values, all_coords, method='linear')
        
        # Reshape back
        result = filled.reshape(surface.shape)
        
        # Fill any remaining NaNs with nearest neighbor
        if np.any(np.isnan(result)):
            filled = griddata(coords, values, all_coords, method='nearest')
            result = filled.reshape(surface.shape)
        
        return result
