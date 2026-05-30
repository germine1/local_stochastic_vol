"""
Heston Stochastic Volatility Model

The Heston model describes asset price dynamics with stochastic volatility:

dS_t = μ*S_t*dt + √v_t*S_t*dW_t^S
dv_t = κ(θ - v_t)*dt + σ*√v_t*dW_t^v

where:
- S_t: asset price
- v_t: instantaneous variance (volatility squared)
- μ: drift (risk-free rate under risk-neutral measure)
- κ: mean reversion speed
- θ: long-run variance
- σ: volatility of volatility (vol-of-vol)
- ρ: correlation between price and variance shocks (dW^S · dW^v = ρ*dt)

Feller Condition: 2κθ > σ² ensures v_t > 0
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
from typing import Tuple, Dict, Optional
import warnings

class HestonModel:
    """
    Heston Stochastic Volatility Model
    
    Attributes:
        S0: Initial spot price
        v0: Initial variance
        r: Risk-free rate
        q: Dividend yield
        kappa: Mean reversion speed
        theta: Long-run variance
        sigma: Volatility of volatility
        rho: Correlation between asset and variance
    """
    
    def __init__(self, S0: float, v0: float, r: float, q: float = 0.0,
                 kappa: float = 2.0, theta: float = 0.04, 
                 sigma: float = 0.3, rho: float = -0.7):
        """
        Initialize Heston model
        
        Args:
            S0: Current spot price
            v0: Initial variance (not volatility!)
            r: Risk-free rate (annualized)
            q: Dividend yield (annualized)
            kappa: Mean reversion speed (typically 0.5 - 5.0)
            theta: Long-run variance (typically 0.01 - 0.09)
            sigma: Vol-of-vol (typically 0.1 - 1.0)
            rho: Correlation (typically -0.8 to -0.3 for equities, leverage effect)
        """
        self.S0 = S0
        self.v0 = v0
        self.r = r
        self.q = q
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        
        # Check Feller condition
        if 2 * kappa * theta <= sigma**2:
            warnings.warn(
                f"Feller condition violated: 2κθ = {2*kappa*theta:.4f} <= σ² = {sigma**2:.4f}. "
                "Variance can reach zero in simulations."
            )
    
    def calibrate(self, market_data: pd.DataFrame, 
                  method: str = 'global') -> Dict[str, float]:
        """
        Calibrate Heston parameters to market option prices
        
        Minimizes sum of squared errors between market and model prices
        
        Loss function:
        L(κ, θ, σ, ρ, v0) = Σ [Price_market - Price_Heston(K, T)]²
        
        Args:
            market_data: DataFrame with columns ['Strike', 'Maturity', 'Price', 'Type']
            method: 'global' (differential evolution) or 'local' (L-BFGS-B)
        
        Returns:
            Dictionary of calibrated parameters
        """
        def objective(params):
            """
            Objective function for calibration
            
            params = [kappa, theta, sigma, rho, v0]
            """
            kappa, theta, sigma, rho, v0 = params
            
            # Check constraints
            if kappa <= 0 or theta <= 0 or sigma <= 0 or v0 <= 0:
                return 1e10
            if abs(rho) >= 1:
                return 1e10
            
            # Temporary parameter assignment
            self.kappa = kappa
            self.theta = theta
            self.sigma = sigma
            self.rho = rho
            self.v0 = v0
            
            # Calculate model prices for all options
            total_error = 0
            
            for _, row in market_data.iterrows():
                K = row['Strike']
                T = row['Maturity']
                market_price = row['Price']
                option_type = row['Type'].lower()
                
                try:
                    # Price using Heston semi-analytical formula
                    model_price = self.price_european(K, T, option_type)
                    
                    # Squared error (can weight by vega or other)
                    error = (market_price - model_price)**2
                    total_error += error
                    
                except:
                    # If pricing fails, add large penalty
                    total_error += 1e8
            
            return total_error
        
        # Parameter bounds: [kappa, theta, sigma, rho, v0]
        bounds = [
            (0.1, 10.0),    # kappa: mean reversion speed
            (0.01, 0.5),    # theta: long-run variance
            (0.05, 2.0),    # sigma: vol-of-vol
            (-0.99, -0.01), # rho: correlation (negative for equities)
            (0.01, 0.5)     # v0: initial variance
        ]
        
        # Initial guess
        x0 = [self.kappa, self.theta, self.sigma, self.rho, self.v0]
        
        print("Starting Heston calibration...")
        
        if method == 'global':
            # Global optimization using differential evolution
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=100,
                popsize=15,
                seed=42,
                polish=True,
                workers=-1,  # Use all CPU cores
                updating='deferred'
            )
        else:
            # Local optimization
            result = minimize(
                objective,
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500}
            )
        
        # Update parameters with calibrated values
        self.kappa, self.theta, self.sigma, self.rho, self.v0 = result.x
        
        calibrated_params = {
            'kappa': self.kappa,
            'theta': self.theta,
            'sigma': self.sigma,
            'rho': self.rho,
            'v0': self.v0,
            'rmse': np.sqrt(result.fun / len(market_data))
        }
        
        print(f"Calibration complete. RMSE: {calibrated_params['rmse']:.6f}")
        print(f"Parameters: κ={self.kappa:.4f}, θ={self.theta:.4f}, "
              f"σ={self.sigma:.4f}, ρ={self.rho:.4f}, v0={self.v0:.4f}")
        
        return calibrated_params
    
    def price_european(self, K: float, T: float, 
                       option_type: str = 'call') -> float:
        """
        Price European option using Heston semi-analytical formula
        
        Uses characteristic function and Fourier inversion:
        
        C(S, v, t) = S*P1 - K*exp(-r*T)*P2
        
        where P1 and P2 are probabilities computed via:
        Pj = 1/2 + (1/π) * ∫[0,∞] Re[e^(-iφ*ln(K)) * fj(φ) / (iφ)] dφ
        
        fj(φ) is the characteristic function of ln(S_T)
        
        Args:
            K: Strike price
            T: Time to maturity (years)
            option_type: 'call' or 'put'
        
        Returns:
            Option price
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return max(self.S0 - K, 0)
            else:
                return max(K - self.S0, 0)
        
        # Call price using characteristic function method
        P1 = self._heston_P(1, K, T)
        P2 = self._heston_P(2, K, T)
        
        call_price = self.S0 * np.exp(-self.q * T) * P1 - K * np.exp(-self.r * T) * P2
        
        if option_type.lower() == 'call':
            return max(call_price, 0)
        else:
            # Put-call parity: P = C - S*exp(-q*T) + K*exp(-r*T)
            put_price = call_price - self.S0 * np.exp(-self.q * T) + K * np.exp(-self.r * T)
            return max(put_price, 0)
    
    def _heston_P(self, j: int, K: float, T: float) -> float:
        """
        Compute probability Pj using Fourier inversion
        
        Pj = 1/2 + (1/π) * ∫[0,∞] Re[e^(-iφ*ln(K)) * fj(φ) / (iφ)] dφ
        
        Args:
            j: 1 or 2 (different formulations)
            K: Strike price
            T: Time to maturity
        
        Returns:
            Probability Pj
        """
        # Numerical integration
        integrand = lambda phi: self._heston_integrand(phi, j, K, T)
        
        # Integrate from 0 to 100 (effectively infinity for most cases)
        integral, _ = quad(integrand, 0, 100, limit=100)
        
        P = 0.5 + (1 / np.pi) * integral
        
        return np.clip(P, 0, 1)  # Probability must be in [0, 1]
    
    def _heston_integrand(self, phi: float, j: int, K: float, T: float) -> float:
        """
        Integrand for Heston probability calculation
        
        Re[e^(-iφ*ln(K)) * fj(φ) / (iφ)]
        
        where fj(φ) is the characteristic function
        """
        if phi == 0:
            return 0
        
        # Characteristic function
        f = self._heston_characteristic_function(phi, j, T)
        
        # e^(-iφ*ln(K)) / (iφ)
        numerator = np.exp(-1j * phi * np.log(K)) * f
        denominator = 1j * phi
        
        return np.real(numerator / denominator)
    
    def _heston_characteristic_function(self, phi: float, j: int, T: float) -> complex:
        """
        Heston characteristic function
        
        f_j(φ) = exp(C(T,φ) + D(T,φ)*v_t + iφ*ln(S_t))
        
        where C and D are complex-valued functions solving Riccati equations
        
        Args:
            phi: Frequency parameter
            j: 1 or 2 (different parameter sets)
            T: Time to maturity
        
        Returns:
            Complex-valued characteristic function
        """
        # Parameters depend on j
        if j == 1:
            u = 0.5
            b = self.kappa - self.rho * self.sigma
        else:
            u = -0.5
            b = self.kappa
        
        # Complex intermediate calculations
        d = np.sqrt((self.rho * self.sigma * 1j * phi - b)**2 - 
                   self.sigma**2 * (2 * u * 1j * phi - phi**2))
        
        g = (b - self.rho * self.sigma * 1j * phi + d) / \
            (b - self.rho * self.sigma * 1j * phi - d)
        
        # Functions C(T,φ) and D(T,φ)
        exp_dT = np.exp(d * T)
        
        C = (self.r - self.q) * 1j * phi * T + \
            (self.kappa * self.theta / self.sigma**2) * \
            ((b - self.rho * self.sigma * 1j * phi + d) * T - 
             2 * np.log((1 - g * exp_dT) / (1 - g)))
        
        D = ((b - self.rho * self.sigma * 1j * phi + d) / self.sigma**2) * \
            ((1 - exp_dT) / (1 - g * exp_dT))
        
        # Characteristic function
        f = np.exp(C + D * self.v0 + 1j * phi * np.log(self.S0))
        
        return f
    
    def simulate_paths(self, T: float, n_steps: int, 
                      n_paths: int = 1000,
                      scheme: str = 'euler') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate price and variance paths using Monte Carlo
        
        Euler Scheme (simple but may violate positivity):
        S_{t+dt} = S_t + (r-q)*S_t*dt + √v_t*S_t*√dt*Z_S
        v_{t+dt} = v_t + κ(θ-v_t)*dt + σ*√v_t*√dt*Z_v
        
        Full Truncation Scheme (ensures v > 0):
        v_{t+dt} = v_t + κ(θ-v_t^+)*dt + σ*√(v_t^+)*√dt*Z_v
        where v^+ = max(v, 0)
        
        Correlation: Z_v = ρ*Z_S + √(1-ρ²)*Z_indep
        
        Args:
            T: Final time (years)
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            scheme: 'euler' or 'milstein' or 'truncation'
        
        Returns:
            times: Array of time points
            S_paths: Simulated price paths (n_paths × n_steps)
            v_paths: Simulated variance paths (n_paths × n_steps)
        """
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        
        # Initialize arrays
        S_paths = np.zeros((n_paths, n_steps + 1))
        v_paths = np.zeros((n_paths, n_steps + 1))
        
        S_paths[:, 0] = self.S0
        v_paths[:, 0] = self.v0
        
        for i in range(n_steps):
            # Generate correlated random variables
            Z1 = np.random.standard_normal(n_paths)  # For asset price
            Z2 = np.random.standard_normal(n_paths)  # Independent
            
            # Correlated Brownian motion for variance
            Z_v = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            S = S_paths[:, i]
            v = v_paths[:, i]
            
            if scheme == 'truncation' or scheme == 'euler':
                # Full truncation scheme (Lord et al. 2010)
                # Ensures variance stays positive
                v_plus = np.maximum(v, 0)
                
                # Variance process
                v_paths[:, i + 1] = v + self.kappa * (self.theta - v_plus) * dt + \
                                   self.sigma * np.sqrt(v_plus * dt) * Z_v
                
                # Asset price process (log-normal discretization)
                S_paths[:, i + 1] = S * np.exp(
                    (self.r - self.q - 0.5 * v_plus) * dt + 
                    np.sqrt(v_plus * dt) * Z1
                )
                
            elif scheme == 'milstein':
                # Milstein scheme for better accuracy
                v_plus = np.maximum(v, 0)
                
                # Variance with Milstein correction
                v_paths[:, i + 1] = v + self.kappa * (self.theta - v_plus) * dt + \
                                   self.sigma * np.sqrt(v_plus * dt) * Z_v + \
                                   0.25 * self.sigma**2 * dt * (Z_v**2 - 1)
                
                # Asset price
                S_paths[:, i + 1] = S * np.exp(
                    (self.r - self.q - 0.5 * v_plus) * dt + 
                    np.sqrt(v_plus * dt) * Z1
                )
            
            else:
                raise ValueError(f"Unknown scheme: {scheme}")
            
            # Ensure variance stays non-negative (reflection at zero)
            v_paths[:, i + 1] = np.maximum(v_paths[:, i + 1], 0)
        
        return times, S_paths, v_paths
    
    def implied_volatility(self, K: float, T: float, 
                          option_type: str = 'call') -> float:
        """
        Calculate implied volatility from Heston model price
        
        Uses Newton-Raphson to invert Black-Scholes formula
        
        Args:
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
        
        Returns:
            Implied volatility
        """
        from scipy.stats import norm
        
        # Get Heston price
        heston_price = self.price_european(K, T, option_type)
        
        # Newton-Raphson iteration
        sigma = np.sqrt(self.v0)  # Initial guess
        
        for _ in range(100):
            # Black-Scholes price and vega
            d1 = (np.log(self.S0 / K) + (self.r - self.q + 0.5 * sigma**2) * T) / \
                 (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                bs_price = self.S0 * np.exp(-self.q * T) * norm.cdf(d1) - \
                          K * np.exp(-self.r * T) * norm.cdf(d2)
            else:
                bs_price = K * np.exp(-self.r * T) * norm.cdf(-d2) - \
                          self.S0 * np.exp(-self.q * T) * norm.cdf(-d1)
            
            # Vega (∂Price/∂σ)
            vega = self.S0 * np.exp(-self.q * T) * norm.pdf(d1) * np.sqrt(T)
            
            # Price difference
            diff = bs_price - heston_price
            
            if abs(diff) < 1e-6 or vega < 1e-10:
                break
            
            # Newton-Raphson update
            sigma = sigma - diff / vega
            sigma = max(sigma, 0.001)  # Keep positive
        
        return sigma
    
    def generate_iv_surface(self, strikes: np.ndarray, 
                           maturities: np.ndarray) -> pd.DataFrame:
        """
        Generate implied volatility surface from Heston model
        
        Args:
            strikes: Array of strike prices
            maturities: Array of maturities (years)
        
        Returns:
            DataFrame with columns ['Strike', 'Maturity', 'IV_Call', 'IV_Put']
        """
        surface_data = []
        
        for T in maturities:
            for K in strikes:
                iv_call = self.implied_volatility(K, T, 'call')
                iv_put = self.implied_volatility(K, T, 'put')
                
                surface_data.append({
                    'Strike': K,
                    'Maturity': T,
                    'Moneyness': K / self.S0,
                    'IV_Call': iv_call,
                    'IV_Put': iv_put,
                    'IV_Mid': (iv_call + iv_put) / 2
                })
        
        return pd.DataFrame(surface_data)
