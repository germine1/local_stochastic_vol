"""
Carr-Madan Fast Fourier Transform (FFT) Method for Option Pricing

The Carr-Madan method (1999) uses FFT to efficiently price European options
when the characteristic function of the log-asset price is known.

Key idea:
Transform option pricing into Fourier space, where convolution becomes multiplication,
then use FFT for fast computation.

Advantages:
- Extremely fast: O(N log N) for N strikes
- High accuracy
- Works for any model with known characteristic function

Applications:
- Heston, SABR, Jump Diffusion, Variance Gamma, etc.
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Callable, Dict, Tuple, Optional
import warnings

class CarrMadanFFT:
    """
    Carr-Madan FFT pricing engine
    
    Prices options using Fourier transform and FFT algorithm
    """
    
    def __init__(self, N: int = 4096, B: float = 500, alpha: float = 1.5):
        """
        Initialize Carr-Madan engine
        
        Args:
            N: Number of FFT points (power of 2, e.g., 2048, 4096, 8192)
               Larger N → more strikes, better accuracy
            B: Upper limit for log-strike integration
               Covers strikes in range [S*e^(-B), S*e^B]
            alpha: Damping parameter
               Must satisfy conditions for Fourier transform to exist
               Typical values: 0.75 - 2.5
        """
        if N & (N - 1) != 0:
            raise ValueError(f"N must be power of 2, got {N}")
        
        self.N = N
        self.B = B
        self.alpha = alpha
        
        # Grid parameters
        self.lambda_ = 2 * B / N  # Spacing in log-strike
        self.eta = 2 * np.pi / (N * self.lambda_)  # Spacing in Fourier domain
        
        # Log-strike grid
        self.k = -B + self.lambda_ * np.arange(N)
        
        # Frequency grid
        self.v = self.eta * np.arange(N)
    
    def price_european(self, S0: float, K: float, T: float, r: float,
                      char_func: Callable, option_type: str = 'call') -> float:
        """
        Price European option using Carr-Madan method
        
        Carr-Madan Formula:
        C(K) = e^(-α*k) / π * ∫[0,∞] e^(-i*v*k) * Ψ(v) dv
        
        where:
        k = ln(K/S)
        Ψ(v) = [e^(-r*T) * φ(v - (α+1)i)] / [α² + α - v² + i*(2α+1)*v]
        φ(u) = E[e^(i*u*ln(S_T))] is the characteristic function
        
        Using FFT:
        C(k_j) ≈ e^(-α*k_j) / π * Σ e^(-i*v_u*k_j) * Ψ(v_u) * η
        
        Args:
            S0: Spot price
            K: Strike price (single value or array)
            T: Time to maturity
            r: Risk-free rate
            char_func: Characteristic function φ(u, T)
                       Takes complex argument u, returns complex value
            option_type: 'call' or 'put'
        
        Returns:
            Option price (single value or array matching K)
        """
        # Modified characteristic function
        psi = np.zeros(self.N, dtype=np.complex128)
        
        for j in range(self.N):
            u = self.v[j] - (self.alpha + 1) * 1j
            
            # Get characteristic function value
            phi = char_func(u, T)
            
            # Carr-Madan damping
            numerator = np.exp(-r * T) * phi
            denominator = self.alpha**2 + self.alpha - self.v[j]**2 + \
                         1j * (2 * self.alpha + 1) * self.v[j]
            
            psi[j] = numerator / denominator
        
        # Apply Simpson's rule weights for better accuracy
        # w = [1, 4, 2, 4, 2, ..., 4, 1] / 3
        simpson_weights = np.ones(self.N)
        simpson_weights[1:-1:2] = 4
        simpson_weights[2:-1:2] = 2
        simpson_weights = 1
        simpson_weights[-1] = 1
        simpson_weights = simpson_weights / 3
        
        # Apply weights
        psi = psi * simpson_weights * self.eta
        
        # FFT
        fft_result = np.fft.fft(psi)
        
        # Extract call prices for all strikes
        call_prices = np.real(np.exp(-self.alpha * self.k) / np.pi * fft_result)
        
        # Interpolate to get price at desired strike(s)
        strike_grid = S0 * np.exp(self.k)
        interpolator = interp1d(strike_grid, call_prices, 
                               kind='cubic', fill_value='extrapolate')
        
        if np.isscalar(K):
            call_price = float(interpolator(K))
        else:
            call_price = interpolator(K)
        
        # Convert to put if needed (Put-Call parity)
        if option_type.lower() == 'put':
            if np.isscalar(K):
                put_price = call_price - S0 + K * np.exp(-r * T)
            else:
                put_price = call_price - S0 + K * np.exp(-r * T)
            return put_price
        
        return call_price
    
    def price_surface(self, S0: float, strikes: np.ndarray, T: float, 
                     r: float, char_func: Callable,
                     option_type: str = 'call') -> np.ndarray:
        """
        Price options for multiple strikes simultaneously
        
        This is where FFT shines: compute all strikes at once!
        
        Args:
            S0: Spot price
            strikes: Array of strike prices
            T: Time to maturity
            r: Risk-free rate
            char_func: Characteristic function
            option_type: 'call' or 'put'
        
        Returns:
            Array of option prices matching strikes
        """
        # Modified characteristic function
        psi = np.zeros(self.N, dtype=np.complex128)
        
        for j in range(self.N):
            u = self.v[j] - (self.alpha + 1) * 1j
            phi = char_func(u, T)
            
            numerator = np.exp(-r * T) * phi
            denominator = self.alpha**2 + self.alpha - self.v[j]**2 + \
                         1j * (2 * self.alpha + 1) * self.v[j]
            
            psi[j] = numerator / denominator
        
        # Simpson weights
        simpson_weights = np.ones(self.N)
        simpson_weights[1:-1:2] = 4
        simpson_weights[2:-1:2] = 2
        simpson_weights = 1
        simpson_weights[-1] = 1
        simpson_weights = simpson_weights / 3
        
        psi = psi * simpson_weights * self.eta
        
        # FFT
        fft_result = np.fft.fft(psi)
        call_prices = np.real(np.exp(-self.alpha * self.k) / np.pi * fft_result)
        
        # Interpolate
        strike_grid = S0 * np.exp(self.k)
        interpolator = interp1d(strike_grid, call_prices, 
                               kind='cubic', fill_value='extrapolate')
        
        prices = interpolator(strikes)
        
        if option_type.lower() == 'put':
            prices = prices - S0 + strikes * np.exp(-r * T)
        
        return prices

# ========== Characteristic Functions for Different Models ==========

def black_scholes_char_func(u: complex, S0: float, r: float, q: float, 
                           sigma: float, T: float) -> complex:
    """
    Black-Scholes characteristic function
    
    φ(u) = exp{i*u*[ln(S0) + (r-q-σ²/2)*T] - 0.5*σ²*u²*T}
    
    Args:
        u: Complex frequency
        S0: Spot price
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        T: Time to maturity
    
    Returns:
        Complex characteristic function value
    """
    drift = np.log(S0) + (r - q - 0.5 * sigma**2) * T
    return np.exp(1j * u * drift - 0.5 * sigma**2 * u**2 * T)

def heston_char_func(u: complex, S0: float, v0: float, r: float, q: float,
                    kappa: float, theta: float, sigma: float, rho: float,
                    T: float) -> complex:
    """
    Heston model characteristic function
    
    φ(u) = exp{C(T,u) + D(T,u)*v0 + i*u*ln(S0)}
    
    See Heston model documentation for full formula
    
    Args:
        u: Complex frequency
        S0: Spot price
        v0: Initial variance
        r: Risk-free rate
        q: Dividend yield
        kappa: Mean reversion speed
        theta: Long-run variance
        sigma: Vol-of-vol
        rho: Correlation
        T: Time to maturity
    
    Returns:
        Complex characteristic function value
    """
    # Using formulation from Heston (1993)
    d = np.sqrt((rho * sigma * 1j * u - kappa)**2 - 
                sigma**2 * (-1j * u - u**2))
    
    g = (kappa - rho * sigma * 1j * u - d) / \
        (kappa - rho * sigma * 1j * u + d)
    
    exp_dT = np.exp(d * T)
    
    C = (r - q) * 1j * u * T + \
        (kappa * theta / sigma**2) * \
        ((kappa - rho * sigma * 1j * u - d) * T - 
         2 * np.log((1 - g * exp_dT) / (1 - g)))
    
    D = ((kappa - rho * sigma * 1j * u - d) / sigma**2) * \
        ((1 - exp_dT) / (1 - g * exp_dT))
    
    return np.exp(C + D * v0 + 1j * u * np.log(S0))

def merton_jump_char_func(u: complex, S0: float, r: float, q: float,
                         sigma: float, lambda_: float, mu_J: float, 
                         sigma_J: float, T: float) -> complex:
    """
    Merton Jump Diffusion characteristic function
    
    φ(u) = exp{i*u*[(r-q-λ*k-σ²/2)*T] - 0.5*σ²*u²*T + 
                λ*T*[exp(i*u*μ_J - 0.5*σ_J²*u²) - 1]}
    
    where k = exp(μ_J + σ_J²/2) - 1
    
    Args:
        u: Complex frequency
        S0: Spot price
        r: Risk-free rate
        q: Dividend yield
        sigma: Diffusion volatility
        lambda_: Jump intensity
        mu_J: Mean log-jump size
        sigma_J: Std of log-jump size
        T: Time to maturity
    
    
