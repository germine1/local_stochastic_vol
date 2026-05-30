"""
Carr-Madan Fast Fourier Transform (FFT) Method for Option Pricing

The Carr-Madan method (1999) uses FFT to efficiently price European options
when the characteristic function of the log-asset price is known.

Key idea:
Transform option pricing into Fourier space, where convolution becomes multiplication,
then use FFT for fast computation.

Mathematical Foundation:
======================
Call option price can be written as:
C(k) = e^(-α*k) / π * ∫[0,∞] e^(-i*v*k) * Ψ(v) dv

where:
- k = ln(K/S): log-strike
- α > 0: damping parameter (ensures integrability)
- Ψ(v): modified characteristic function

Modified characteristic function:
Ψ(v) = [e^(-r*T) * φ(v - (α+1)i)] / [α² + α - v² + i*(2α+1)*v]

where φ(u) = E[e^(i*u*ln(S_T))] is the characteristic function

Using FFT:
- Discretize integral using trapezoidal/Simpson's rule
- Apply FFT to compute all strikes simultaneously
- Complexity: O(N log N) instead of O(N²)

Advantages:
- Extremely fast: Price entire surface in milliseconds
- High accuracy
- Works for any model with known characteristic function (Heston, VG, CGMY, etc.)

Applications:
- Heston stochastic volatility
- Jump diffusion models (Merton, Kou)
- Variance Gamma
- CGMY (Carr-Geman-Madan-Yor)
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Callable, Dict, Tuple, Optional, List
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
            N: Number of FFT points (must be power of 2, e.g., 2048, 4096, 8192)
               Larger N → more strikes covered, better resolution
               Typical values: 2048 - 8192
               
            B: Upper limit for log-strike integration
               Covers strikes in range [S*e^(-B), S*e^B]
               Typical values: 200 - 1000
               Larger B → wider strike range
               
            alpha: Damping parameter (α in Carr-Madan paper)
               Must satisfy: α > 0 for calls, α < 0 for puts
               Ensures the Fourier transform exists
               Typical values: 0.75 - 2.5
               
               Physical interpretation:
               - Controls exponential decay of integrand
               - Too small → numerical instability
               - Too large → loss of accuracy
               - Optimal: α ≈ 1.5 for calls, α ≈ -1.5 for puts
        """
        # Validate N is power of 2
        if N & (N - 1) != 0:
            raise ValueError(f"N must be power of 2 for FFT, got {N}")
        
        self.N = N
        self.B = B
        self.alpha = alpha
        
        # Grid spacing parameters
        # λ: spacing in log-strike domain
        self.lambda_ = 2 * B / N
        
        # η: spacing in frequency domain
        # Related by: η * λ = 2π / N (Nyquist condition)
        self.eta = 2 * np.pi / (N * self.lambda_)
        
        # Log-strike grid: k = ln(K/S)
        # Ranges from -B to +B
        self.k = -B + self.lambda_ * np.arange(N)
        
        # Frequency grid: v
        # Ranges from 0 to N*η
        self.v = self.eta * np.arange(N)
        
        print(f"Carr-Madan FFT initialized:")
        print(f"  N = {N} points")
        print(f"  Log-strike range: [{-B:.2f}, {B:.2f}]")
        print(f"  Log-strike spacing: λ = {self.lambda_:.6f}")
        print(f"  Frequency spacing: η = {self.eta:.6f}")
    
    def price_european(self, S0: float, K: float, T: float, r: float,
                      char_func: Callable[[complex, float], complex], 
                      option_type: str = 'call',
                      q: float = 0.0) -> float:
        """
        Price European option using Carr-Madan method
        
        Carr-Madan Formula (1999):
        ==========================
        C(K) = e^(-α*k) / π * ∫[0,∞] e^(-i*v*k) * Ψ(v) dv
        
        where:
        k = ln(K/S): log-moneyness
        
        Ψ(v) = [e^(-r*T) * φ(v - (α+1)i)] / [α² + α - v² + i*(2α+1)*v]
        
        φ(u) = E[e^(i*u*ln(S_T))]: characteristic function
        
        Discretization:
        C(k_j) ≈ e^(-α*k_j) / π * Σ[u=0 to N-1] e^(-i*v_u*k_j) * Ψ(v_u) * η * w_u
        
        where w_u are Simpson's rule weights
        
        FFT formula:
        C(k_j) = e^(-α*k_j) / π * Re[FFT{Ψ(v_u) * η * w_u}]
        
        Args:
            S0: Current spot price
            K: Strike price (single value or array)
            T: Time to maturity (years)
            r: Risk-free rate (annualized)
            char_func: Characteristic function φ(u, T)
                       Function signature: char_func(u: complex, T: float) -> complex
                       Returns E[exp(i*u*ln(S_T))]
            option_type: 'call' or 'put'
            q: Dividend yield (annualized, default 0)
        
        Returns:
            Option price (float if K is scalar, array if K is array)
        
        Example:
            >>> # Black-Scholes characteristic function
            >>> def bs_char_func(u, T):
            ...     return np.exp(1j*u*np.log(S0) + 
            ...                   1j*u*(r-q-0.5*sigma**2)*T - 
            ...                   0.5*sigma**2*u**2*T)
            >>> 
            >>> cm = CarrMadanFFT(N=4096)
            >>> price = cm.price_european(100, 100, 1.0, 0.05, bs_char_func)
        """
        # Step 1: Compute modified characteristic function Ψ(v)
        psi = np.zeros(self.N, dtype=np.complex128)
        
        for j in range(self.N):
            v_j = self.v[j]
            
            # Argument for characteristic function: u = v - (α+1)i
            u = v_j - (self.alpha + 1) * 1j
            
            # Get characteristic function value: φ(u, T)
            phi = char_func(u, T)
            
            # Modified characteristic function numerator
            # e^(-r*T) * φ(v - (α+1)i)
            numerator = np.exp(-r * T) * phi
            
            # Modified characteristic function denominator
            # α² + α - v² + i*(2α+1)*v
            denominator = (self.alpha**2 + self.alpha - v_j**2 + 
                          1j * (2 * self.alpha + 1) * v_j)
            
            # Avoid division by zero
            if abs(denominator) < 1e-10:
                psi[j] = 0
            else:
                psi[j] = numerator / denominator
        
        # Step 2: Apply Simpson's rule weights for numerical integration
        # Simpson's rule: w = [1, 4, 2, 4, 2, ..., 4, 2, 4, 1] / 3
        # Improves accuracy from O(h²) to O(h⁴)
        simpson_weights = np.ones(self.N)
        simpson_weights[1:-1:2] = 4  # Odd indices (except first/last)
        simpson_weights[2:-1:2] = 2  # Even indices (except first/last)
        simpson_weights = 1       # First point
        simpson_weights[-1] = 1      # Last point
        simpson_weights = simpson_weights / 3
        
        # Apply weights and spacing
        psi = psi * simpson_weights * self.eta
        
        # Step 3: Compute FFT
        # FFT computes: X[k] = Σ[n=0 to N-1] x[n] * e^(-2πikn/N)
        # In our case: Σ e^(-i*v_u*k_j) * Ψ(v_u) * η * w_u
        fft_result = np.fft.fft(psi)
        
        # Step 4: Extract call prices
        # C(k_j) = e^(-α*k_j) / π * Re[FFT_result]
        call_prices = np.real(np.exp(-self.alpha * self.k) / np.pi * fft_result)
        
        # Step 5: Interpolate to get price at desired strike(s)
        # Convert log-strikes to actual strikes
        strike_grid = S0 * np.exp(self.k)
        
        # Create cubic spline interpolator
        interpolator = interp1d(
            strike_grid, 
            call_prices, 
            kind='cubic',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # Interpolate at desired strike
        if np.isscalar(K):
            call_price = float(interpolator(K))
        else:
            call_price = interpolator(np.array(K))
        
        # Step 6: Convert to put if needed using put-call parity
        # Put-Call Parity: P = C - S*e^(-q*T) + K*e^(-r*T)
        if option_type.lower() == 'put':
            if np.isscalar(K):
                put_price = call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T)
            else:
                put_price = call_price - S0 * np.exp(-q * T) + np.array(K) * np.exp(-r * T)
            return put_price
        
        return call_price
    
    def price_surface(self, S0: float, strikes: np.ndarray, T: float, 
                     r: float, char_func: Callable[[complex, float], complex],
                     option_type: str = 'call',
                     q: float = 0.0) -> np.ndarray:
        """
        Price options for multiple strikes simultaneously
        
        This is where FFT truly shines: compute prices for all strikes at once
        with O(N log N) complexity instead of O(N²) for individual pricing.
        
        Perfect for:
        - Building volatility surfaces
        - Risk management (pricing entire portfolio)
        - Calibration (need many strikes)
        
        Args:
            S0: Spot price
            strikes: Array of strike prices (can be hundreds or thousands)
            T: Time to maturity
            r: Risk-free rate
            char_func: Characteristic function
            option_type: 'call' or 'put'
            q: Dividend yield
        
        Returns:
            Array of option prices matching strikes array
            
        Example:
            >>> strikes = np.linspace(80, 120, 100)  # 100 strikes
            >>> prices = cm.price_surface(100, strikes, 1.0, 0.05, char_func)
            >>> # Computed 100 prices in milliseconds!
        """
        # Compute modified characteristic function
        psi = np.zeros(self.N, dtype=np.complex128)
        
        for j in range(self.N):
            v_j = self.v[j]
            u = v_j - (self.alpha + 1) * 1j
            phi = char_func(u, T)
            
            numerator = np.exp(-r * T) * phi
            denominator = (self.alpha**2 + self.alpha - v_j**2 + 
                          1j * (2 * self.alpha + 1) * v_j)
            
            if abs(denominator) > 1e-10:
                psi[j] = numerator / denominator
        
        # Simpson's rule weights
        simpson_weights = np.ones(self.N)
        simpson_weights[1:-1:2] = 4
        simpson_weights[2:-1:2] = 2
        simpson_weights = 1
        simpson_weights[-1] = 1
        simpson_weights = simpson_weights / 3
        
        psi = psi * simpson_weights * self.eta
        
        # FFT
        fft_result = np.fft.fft(psi)
        call_prices_grid = np.real(np.exp(-self.alpha * self.k) / np.pi * fft_result)
        
        # Interpolate
        strike_grid = S0 * np.exp(self.k)
        interpolator = interp1d(
            strike_grid, 
            call_prices_grid, 
            kind='cubic',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        prices = interpolator(strikes)
        
        # Convert to put if needed
        if option_type.lower() == 'put':
            prices = prices - S0 * np.exp(-q * T) + strikes * np.exp(-r * T)
        
        return prices
    
    def implied_volatility_surface(self, S0: float, strikes: np.ndarray,
                                   maturities: np.ndarray, r: float,
                                   char_func_factory: Callable,
                                   q: float = 0.0) -> np.ndarray:
        """
        Generate implied volatility surface from model
        
        Args:
            S0: Spot price
            strikes: Array of strikes
            maturities: Array of maturities
            r: Risk-free rate
            char_func_factory: Function that takes (T) and returns char_func(u, T)
            q: Dividend yield
        
        Returns:
            2D array of implied volatilities [maturities × strikes]
        """
        from scipy.stats import norm
        from scipy.optimize import brentq
        
        iv_surface = np.zeros((len(maturities), len(strikes)))
        
        for i, T in enumerate(maturities):
            # Get characteristic function for this maturity
            char_func = lambda u, t=T: char_func_factory(t)(u, t)
            
            # Price all strikes for this maturity
            prices = self.price_surface(S0, strikes, T, r, char_func, 'call', q)
            
            # Convert to implied volatilities
            for j, (K, price) in enumerate(zip(strikes, prices)):
                try:
                    # Newton-Raphson to find IV
                    def bs_price_diff(sigma):
                        if sigma <= 0:
                            return 1e10
                        d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                        d2 = d1 - sigma*np.sqrt(T)
                        bs = S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                        return bs - price
                    
                    iv = brentq(bs_price_diff, 0.001, 5.0, maxiter=100)
                    iv_surface[i, j] = iv
                except:
                    iv_surface[i, j] = np.nan
        
        return iv_surface

# ========== Characteristic Functions for Different Models ==========

def black_scholes_char_func(u: complex, S0: float, r: float, q: float, 
                           sigma: float, T: float) -> complex:
    """
    Black-Scholes characteristic function
    
    Under risk-neutral measure:
    ln(S_T) = ln(S_0) + (r - q - σ²/2)*T + σ*√T*Z
    
    where Z ~ N(0,1)
    
    Characteristic function of normal random variable:
    If X ~ N(μ, σ²), then E[e^(iuX)] = exp(iuμ - 0.5σ²u²)
    
    Therefore:
    φ(u) = E[exp(i*u*ln(S_T))]
         = exp{i*u*[ln(S_0) + (r-q-σ²/2)*T] - 0.5*σ²*u²*T}
    
    Args:
        u: Complex frequency parameter
        S0: Spot price
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        T: Time to maturity
    
    Returns:
        Complex characteristic function value
    """
    drift = np.log(S0) + (r - q - 0.5 * sigma**2) * T
    diffusion_var = sigma**2 * T
    
    return np.exp(1j * u * drift - 0.5 * diffusion_var * u**2)

def heston_char_func(u: complex, S0: float, v0: float, r: float, q: float,
                    kappa: float, theta: float, sigma: float, rho: float,
                    T: float) -> complex:
    """
    Heston model characteristic function
    
    Heston model dynamics:
    dS_t = (r-q)*S_t*dt + √v_t*S_t*dW_S
    dv_t = κ(θ - v_t)*dt + σ*√v_t*dW_v
    
    where Corr(dW_S, dW_v) = ρ*dt
    
    Characteristic function (Heston 1993):
    φ(u) = exp{C(T,u) + D(T,u)*v_0 + i*u*ln(S_0)}
    
    where C and D solve complex-valued Riccati ODEs:
    
    d = √[(ρσiu - κ)² - σ²(-iu - u²)]
    g = (κ - ρσiu - d) / (κ - ρσiu + d)
    
    C(T,u) = (r-q)*i*u*T + (κθ/σ²)*[(κ - ρσiu - d)*T - 2*ln((1 - g*e^(dT))/(1-g))]
    D(T,u) = [(κ - ρσiu - d)/σ²] * [(1 - e^(dT))/(1 - g*e^(dT))]
    
    Args:
        u: Complex frequency
        S0: Spot price
        v0: Initial variance (not volatility!)
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
    # Complex square root with correct branch
    d = np.sqrt((rho * sigma * 1j * u - kappa)**2 - 
                sigma**2 * (-1j * u - u**2))
    
    # Ratio g
    g = (kappa - rho * sigma * 1j * u - d) / \
        (kappa - rho * sigma * 1j * u + d)
    
    # exp(d*T)
    exp_dT = np.exp(d * T)
    
    # C(T, u) - affects option price level
    C = ((r - q) * 1j * u * T + 
         (kappa * theta / sigma**2) * 
         ((kappa - rho * sigma * 1j * u - d) * T - 
          2 * np.log((1 - g * exp_dT) / (1 - g))))
    
    # D(T, u) - affects variance dynamics
    D = ((kappa - rho * sigma * 1j * u - d) / sigma**2) * \
        ((1 - exp_dT) / (1 - g * exp_dT))
    
    # Full characteristic function
    return np.exp(C + D * v0 + 1j * u * np.log(S0))

def merton_jump_char_func(u: complex, S0: float, r: float, q: float,
                         sigma: float, lambda_: float, mu_J: float, 
                         sigma_J: float, T: float) -> complex:
    """
    Merton Jump Diffusion characteristic function
    
    Merton model:
    dS_t/S_t = (r-q)*dt + σ*dW_t + dJ_t
    
    where J_t = Σ(i=1 to N_t) (Y_i - 1)
    N_t ~ Poisson(λt)
    ln(Y_i) ~ N(μ_J, σ_J²)
    
    Characteristic function:
    φ(u) = exp{i*u*[ln(S_0) + (r-q-λk-σ²/2)*T] - 0.5*σ²*u²*T + 
                λ*T*[exp(i*u*μ_J - 0.5*σ_J²*u²) - 1]}
    
    where k = E[Y-1] = exp(μ_J + σ_J²/2) - 1
    
    Components:
    1. Diffusion: -0.5*σ²*u²*T (Brownian motion)
    2. Jump: λ*T*[E[e^(iu*ln(Y))] - 1] (compound Poisson)
    
    Args:
        u: Complex frequency
        S0: Spot price
        r: Risk-free rate
        q: Dividend yield
        sigma: Diffusion volatility
        lambda_: Jump intensity (jumps per year)
        mu_J: Mean of log-jump size
        sigma_J: Std of log-jump size
        T: Time to maturity
    
    Returns:
        Complex characteristic function value
    """
    # Mean jump size
    k = np.exp(mu_J + 0.5 * sigma_J**2) - 1
    
    # Drift term (risk-neutral adjustment)
    drift_term = 1j * u * (np.log(S0) + (r - q - lambda_ * k - 0.5 * sigma**2) * T)
    
    # Diffusion term (Brownian motion)
    diffusion_term = -0.5 * sigma**2 * u**2 * T
    
    # Jump term (characteristic function of lognormal jump)
    # E[e^(iu*ln(Y))] where ln(Y) ~ N(μ_J, σ_J²)
    jump_char = np.exp(1j * u * mu_J - 0.5 * sigma_J**2 * u**2)
    jump_term = lambda_ * T * (jump_char - 1)
    
    return np.exp(drift_term + diffusion_term + jump_term)

def kou_jump_char_func(u: complex, S0: float, r: float, q: float,
                      sigma: float, lambda_: float, p: float,
                      eta1: float, eta2: float, T: float) -> complex:
    """
    Kou Double Exponential Jump Diffusion characteristic function
    
    Kou model with asymmetric double exponential jumps:
    Jump size Y-1 has density:
    f(x) = p*η₁*e^(-η₁*x)*I(x≥0) + (1-p)*η₂*e^(η₂*x)*I(x<0)
    
    Characteristic function:
    φ(u) = exp{i*u*[ln(S_0) + (r-q-λζ-σ²/2)*T] - 0.5*σ²*u²*T +
                λ*T*[Π(u) - 1]}
    
    where:
    Π(u) = p*η₁/(η₁-iu) + (1-p)*η₂/(η₂+iu)
    ζ = E[Y-1] = p/(η₁-1) - (1-p)/(η₂+1)
    
    Args:
        u: Complex frequency
        S0: Spot price
        r: Risk-free rate
        q: Dividend yield
        sigma: Diffusion volatility
        lambda_: Jump intensity
        p: Probability of upward jump
        eta1: Decay rate for upward jumps (must be > 1)
        eta2: Decay rate for downward jumps (must be > 0)
        T: Time to maturity
    
    Returns:
        Complex characteristic function value
    """
    # Mean jump size
    zeta = p / (eta1 - 1) - (1 - p) / (eta2 + 1)
    
    # Drift term
    drift_term = 1j * u * (np.log(S0) + (r - q - lambda_ * zeta - 0.5 * sigma**2) * T)
    
    # Diffusion term
    diffusion_term = -0.5 * sigma**2 * u**2 * T
    
    # Jump term with double exponential
    # Characteristic function of double exponential jump
    pi_up = p * eta1 / (eta1 - 1j * u)
    pi_down = (1 - p) * eta2 / (eta2 + 1j * u)
    pi_u = pi_up + pi_down
    
    jump_term = lambda_ * T * (pi_u - 1)
    
    return np.exp(drift_term + diffusion_term + jump_term)

# ========== Example Usage and Validation ==========

def example_carr_madan_heston():
    """
    Example: Price Heston model options using Carr-Madan
    
    Demonstrates complete workflow
    """
    print("\n" + "="*70)
    print("Carr-Madan FFT Example: Heston Model")
    print("="*70)
    
    # Market parameters
    S0 = 100.0
    r = 0.05
    q = 0.02
    T = 1.0
    
    # Heston parameters
    v0 = 0.04       # Initial variance
    kappa = 2.0     # Mean reversion speed
    theta = 0.04    # Long-run variance
    sigma = 0.3     # Vol-of-vol
    rho = -0.7      # Correlation
    
    print(f"\nMarket Parameters:")
    print(f"  S0 = {S0}, r = {r}, q = {q}, T = {T}")
    print(f"\nHeston Parameters:")
    print(f"  v0 = {v0}, κ = {kappa}, θ = {theta}, σ = {sigma}, ρ = {rho}")
    
    # Initialize Carr-Madan engine
    cm = CarrMadanFFT(N=4096, B=500, alpha=1.5)
    
    # Define characteristic function
    def char_func(u, T_val):
        return heston_char_func(u, S0, v0, r, q, kappa, theta, sigma, rho, T_val)
    
    # Price single option
    K = 100
    call_price = cm.price_european(S0, K, T, r, char_func, 'call', q)
    put_price = cm.price_european(S0, K, T, r, char_func, 'put', q)
    
    print(f"\n ATM Option Prices (K={K}):")
    print(f"  Call: {call_price:.4f}")
    print(f"  Put:  {put_price:.4f}")
    
    # Verify put-call parity
    parity_check = call_price - put_price - (S0 * np.exp(-q*T) - K * np.exp(-r*T))
    print(f"  Put-Call Parity Error: {abs(parity_check):.2e}")
    
    # Price multiple strikes
    strikes = np.linspace(80, 120, 41)
    call_prices = cm.price_surface(S0, strikes, T, r, char_func, 'call', q)
    
    print(f"\nPriced {len(strikes)} strikes in one FFT call")
    print(f"  Strike range: [{strikes:.0f}, {strikes[-1]:.0f}]")
    print(f"  Price range: [{call_prices.min():.4f}, {call_prices.max():.4f}]")
    
    return strikes, call_prices

if __name__ == "__main__":
    # Run example
    strikes, prices = example_carr_madan_heston()
