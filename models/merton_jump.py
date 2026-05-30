"""
Merton Jump Diffusion Model

The Merton model extends Black-Scholes by adding discrete jumps to the price process:

dS_t/S_t = μ*dt + σ*dW_t + dJ_t

where:
- μ: drift (r - q under risk-neutral measure)
- σ: continuous diffusion volatility
- dW_t: standard Brownian motion
- dJ_t: compound Poisson jump process

Jump process:
J_t = Σ(i=1 to N_t) (Y_i - 1)

where:
- N_t: Poisson process with intensity λ (average jumps per year)
- Y_i: jump size, Y_i ~ Lognormal(μ_J, σ_J)
  - ln(Y_i) ~ N(μ_J, σ_J²)
- Jump mean: E[Y_i - 1] = exp(μ_J + σ_J²/2) - 1 ≡ k

Under risk-neutral measure:
μ = r - q - λ*k - 0.5*σ²

This ensures no arbitrage.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial
from typing import Tuple, Dict, Optional
import warnings

class MertonJumpDiffusion:
    """
    Merton Jump Diffusion Model
    
    Attributes:
        S0: Initial spot price
        r: Risk-free rate
        q: Dividend yield
        sigma: Diffusion volatility (continuous part)
        lambda_: Jump intensity (average jumps per year)
        mu_J: Mean of log-jump size
        sigma_J: Std deviation of log-jump size
    """
    
    def __init__(self, S0: float, r: float, q: float = 0.0,
                 sigma: float = 0.2, lambda_: float = 0.5,
                 mu_J: float = -0.1, sigma_J: float = 0.15):
        """
        Initialize Merton Jump Diffusion model
        
        Args:
            S0: Current spot price
            r: Risk-free rate (annualized)
            q: Dividend yield (annualized)
            sigma: Diffusion volatility (typically 0.15 - 0.35)
            lambda_: Jump intensity, λ (typically 0.1 - 2.0 jumps/year)
            mu_J: Mean of log-jump size (typically -0.2 to 0)
                  Negative → downward jumps (crashes)
            sigma_J: Std of log-jump size (typically 0.1 - 0.3)
        """
        self.S0 = S0
        self.r = r
        self.q = q
        self.sigma = sigma
        self.lambda_ = lambda_
        self.mu_J = mu_J
        self.sigma_J = sigma_J
        
        # Validate parameters
        if sigma <= 0 or lambda_ < 0 or sigma_J <= 0:
            raise ValueError("sigma, lambda, and sigma_J must be positive")
        
        # Calculate mean jump size: k = E[Y - 1] = exp(μ_J + σ_J²/2) - 1
        self.k = np.exp(mu_J + 0.5 * sigma_J**2) - 1
    
    def price_european_analytical(self, K: float, T: float, 
                                  option_type: str = 'call',
                                  n_terms: int = 50) -> float:
        """
        Price European option using Merton's analytical formula
        
        Merton (1976) derived a series expansion:
        
        C = Σ(n=0 to ∞) [e^(-λ'*T) * (λ'*T)^n / n!] * BS(S, K, T, r_n, σ_n)
        
        where each term is a Black-Scholes price with adjusted parameters:
        - λ' = λ * (1 + k): adjusted jump intensity
        - r_n = r - λ*k + n*γ/T: adjusted risk-free rate
        - σ_n² = σ² + n*δ²/T: adjusted variance
        - γ = ln(1 + k) = μ_J + σ_J²/2
        - δ² = σ_J²
        
        The series converges rapidly (usually 20-50 terms sufficient)
        
        Args:
            K: Strike price
            T: Time to maturity (years)
            option_type: 'call' or 'put'
            n_terms: Number of terms in the series (more = more accurate)
        
        Returns:
            Option price
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return max(self.S0 - K, 0)
            else:
                return max(K - self.S0, 0)
        
        # Adjusted jump intensity
        lambda_prime = self.lambda_ * (1 + self.k)
        
        # Jump mean and variance on log scale
        gamma = self.mu_J + 0.5 * self.sigma_J**2  # E[ln(Y)]
        delta_sq = self.sigma_J**2  # Var[ln(Y)]
        
        # Sum over jump scenarios (n = number of jumps)
        option_price = 0.0
        
        for n in range(n_terms):
            # Poisson probability of n jumps
            poisson_prob = np.exp(-lambda_prime * T) * (lambda_prime * T)**n / factorial(n)
            
            # Adjusted parameters for this term
            r_n = self.r - self.lambda_ * self.k + n * gamma / T
            sigma_n_sq = self.sigma**2 + n * delta_sq / T
            sigma_n = np.sqrt(sigma_n_sq)
            
            # Black-Scholes price with adjusted parameters
            bs_price = self._black_scholes(self.S0, K, T, r_n, self.q, 
                                          sigma_n, option_type)
            
            # Add weighted term
            option_price += poisson_prob * bs_price
            
            # Check convergence
            if n > 10 and poisson_prob < 1e-10:
                break
        
        return option_price
    
    def _black_scholes(self, S: float, K: float, T: float, r: float, 
                       q: float, sigma: float, option_type: str) -> float:
        """
        Black-Scholes formula
        
        Call: C = S*e^(-q*T)*N(d1) - K*e^(-r*T)*N(d2)
        Put: P = K*e^(-r*T)*N(-d2) - S*e^(-q*T)*N(-d1)
        
        d1 = [ln(S/K) + (r - q + σ²/2)*T] / (σ√T)
        d2 = d1 - σ√T
        """
        if sigma <= 0 or T <= 0:
            if option_type.lower() == 'call':
                return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
            else:
                return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        return max(price, 0)
    
    def simulate_paths(self, T: float, n_steps: int, 
                      n_paths: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate price paths using Monte Carlo
        
        Over small interval dt:
        
        1. Diffusion component (Brownian):
           S_t → S_t * exp[(r - q - λ*k - 0.5*σ²)*dt + σ*√dt*Z]
        
        2. Jump component (Poisson):
           Number of jumps: N ~ Poisson(λ*dt)
           For each jump: J_i ~ Lognormal(μ_J, σ_J)
           S_t → S_t * Π(j=1 to N) J_j
        
        Combined:
        S_{t+dt} = S_t * exp[drift*dt + σ*√dt*Z] * Π(jumps)
        
        Args:
            T: Final time (years)
            n_steps: Number of time steps
            n_paths: Number of simulation paths
        
        Returns:
            times: Array of time points
            paths: Simulated price paths (n_paths × n_steps)
            jump_times: List of lists containing jump times for each path
        """
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        
        # Initialize
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        # Track jumps for visualization
        jump_times = [[] for _ in range(n_paths)]
        
        # Drift adjustment for risk-neutral measure
        drift = self.r - self.q - self.lambda_ * self.k - 0.5 * self.sigma**2
        
        for i in range(n_steps):
            # Diffusion component
            Z = np.random.standard_normal(n_paths)
            diffusion = np.exp(drift * dt + self.sigma * np.sqrt(dt) * Z)
            
            # Jump component
            # Number of jumps in this time step (for each path)
            n_jumps = np.random.poisson(self.lambda_ * dt, size=n_paths)
            
            # Initialize jump multiplier
            jump_product = np.ones(n_paths)
            
            for path_idx in range(n_paths):
                if n_jumps[path_idx] > 0:
                    # Generate jump sizes: Y ~ Lognormal(μ_J, σ_J)
                    # ln(Y) ~ N(μ_J, σ_J²)
                    log_jumps = np.random.normal(
                        self.mu_J, self.sigma_J, size=n_jumps[path_idx]
                    )
                    jumps = np.exp(log_jumps)
                    
                    # Multiply all jumps together
                    jump_product[path_idx] = np.prod(jumps)
                    
                    # Record jump time
                    jump_times[path_idx].append(times[i])
            
            # Update price: diffusion × jumps
            paths[:, i + 1] = paths[:, i] * diffusion * jump_product
        
        return times, paths, jump_times
    
    def calibrate(self, market_data: pd.DataFrame,
                  method: str = 'global') -> Dict[str, float]:
        """
        Calibrate Merton model parameters to market option prices
        
        Minimizes sum of squared errors:
        L(σ, λ, μ_J, σ_J) = Σ [Price_market - Price_Merton(K, T)]²
        
        Note: High-dimensional optimization (4 parameters) can have multiple
        local minima. Global optimization recommended.
        
        Args:
            market_data: DataFrame with ['Strike', 'Maturity', 'Price', 'Type']
            method: 'global' (differential evolution) or 'local' (L-BFGS-B)
        
        Returns:
            Dictionary of calibrated parameters
        """
        def objective(params):
            """
            Objective function
            params = [sigma, lambda_, mu_J, sigma_J]
            """
            sigma, lambda_, mu_J, sigma_J = params
            
            # Validate constraints
            if sigma <= 0 or lambda_ < 0 or sigma_J <= 0:
                return 1e10
            
            # Temporarily set parameters
            self.sigma = sigma
            self.lambda_ = lambda_
            self.mu_J = mu_J
            self.sigma_J = sigma_J
            self.k = np.exp(mu_J + 0.5 * sigma_J**2) - 1
            
            total_error = 0
            
            for _, row in market_data.iterrows():
                K = row['Strike']
                T = row['Maturity']
                market_price = row['Price']
                option_type = row['Type'].lower()
                
                try:
                    model_price = self.price_european_analytical(
                        K, T, option_type, n_terms=40
                    )
                    error = (market_price - model_price)**2
                    total_error += error
                except:
                    total_error += 1e8
            
            return total_error
        
        # Parameter bounds: [sigma, lambda_, mu_J, sigma_J]
        bounds = [
            (0.01, 0.8),    # sigma: diffusion vol
            (0.0, 5.0),     # lambda_: jump intensity (0-5 jumps/year)
            (-0.5, 0.2),    # mu_J: log-jump mean (negative for crashes)
            (0.01, 0.5)     # sigma_J: log-jump std
        ]
        
        # Initial guess
        x0 = [self.sigma, self.lambda_, self.mu_J, self.sigma_J]
        
        print("Calibrating Merton Jump Diffusion model...")
        
        if method == 'global':
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=150,
                popsize=15,
                seed=42,
                polish=True,
                workers=-1,
                updating='deferred'
            )
        else:
            result = minimize(
                objective,
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500}
            )
        
        # Update parameters
        self.sigma, self.lambda_, self.mu_J, self.sigma_J = result.x
        self.k = np.exp(self.mu_J + 0.5 * self.sigma_J**2) - 1
        
        calibrated_params = {
            'sigma': self.sigma,
            'lambda': self.lambda_,
            'mu_J': self.mu_J,
            'sigma_J': self.sigma_J,
            'k': self.k,
            'rmse': np.sqrt(result.fun / len(market_data))
        }
        
        print(f"Calibration complete. RMSE: {calibrated_params['rmse']:.6f}")
        print(f"Parameters: σ={self.sigma:.4f}, λ={self.lambda_:.4f}, "
              f"μ_J={self.mu_J:.4f}, σ_J={self.sigma_J:.4f}")
        print(f"Expected jump size: {100*self.k:.2f}%")
        
        return calibrated_params
    
    def implied_volatility(self, K: float, T: float, 
                          option_type: str = 'call') -> float:
        """
        Calculate Black-Scholes implied volatility from Merton price
        
        Uses Newton-Raphson to invert BS formula
        
        Args:
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
        
        Returns:
            Implied volatility
        """
        # Get Merton price
        merton_price = self.price_european_analytical(K, T, option_type)
        
        # Newton-Raphson iteration
        sigma_iv = 0.3  # Initial guess
        
        for _ in range(100):
            # BS price and vega
            bs_price = self._black_scholes(
                self.S0, K, T, self.r, self.q, sigma_iv, option_type
            )
            
            # Vega: ∂Price/∂σ = S*e^(-q*T)*φ(d1)*√T
            d1 = (np.log(self.S0 / K) + (self.r - self.q + 0.5 * sigma_iv**2) * T) / \
                 (sigma_iv * np.sqrt(T))
            vega = self.S0 * np.exp(-self.q * T) * norm.pdf(d1) * np.sqrt(T)
            
            # Price difference
            diff = bs_price - merton_price
            
            if abs(diff) < 1e-6 or vega < 1e-10:
                break
            
            # Newton update
            sigma_iv = sigma_iv - diff / vega
            sigma_iv = max(sigma_iv, 0.001)
        
        return sigma_iv
    
    def generate_iv_surface(self, strikes: np.ndarray,
                           maturities: np.ndarray) -> pd.DataFrame:
        """
        Generate implied volatility surface from Merton model
        
        Jump diffusion naturally produces:
        - Volatility smile (especially for short maturities)
        - Negative skew if μ_J < 0 (downward jumps)
        - Fat tails in return distribution
        
        Args:
            strikes: Array of strike prices
            maturities: Array of maturities (years)
        
        Returns:
            DataFrame with IV surface
        """
        surface_data = []
        
        for T in maturities:
            for K in strikes:
                try:
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
                except:
                    pass
        
        return pd.DataFrame(surface_data)
    
    def total_variance(self, T: float) -> float:
        """
        Calculate total variance over time T
        
        Total variance = σ²*T + λ*T*(σ_J² + μ_J²)
        
        Components:
        1. Diffusion variance: σ²*T
        2. Jump variance: λ*T*E[(ln Y)²] = λ*T*(σ_J² + μ_J²)
        
        Args:
            T: Time horizon
        
        Returns:
            Total variance
        """
        diffusion_var = self.sigma**2 * T
        jump_var = self.lambda_ * T * (self.sigma_J**2 + self.mu_J**2)
        
        return diffusion_var + jump_var
    
    def skewness(self, T: float) -> float:
        """
        Calculate skewness of log-returns over time T
        
        Jump component introduces skewness (crashes → negative skew)
        
        Skewness ∝ λ*T*μ_J*(σ_J² + 3*μ_J²) / (total_var)^(3/2)
        
        Args:
            T: Time horizon
        
        Returns:
            Skewness
        """
        total_var = self.total_variance(T)
        
        # Third central moment from jumps
        third_moment = self.lambda_ * T * self.mu_J * \
                      (self.sigma_J**2 + 3 * self.mu_J**2)
        
        skew = third_moment / (total_var ** 1.5)
        
        return skew
    
    def kurtosis(self, T: float) -> float:
        """
        Calculate excess kurtosis of log-returns over time T
        
        Jump diffusion produces fat tails (excess kurtosis > 0)
        
        Args:
            T: Time horizon
        
        Returns:
            Excess kurtosis (0 for normal distribution)
        """
        total_var = self.total_variance(T)
        
        # Fourth central moment from jumps
        fourth_moment = self.lambda_ * T * \
                       (self.mu_J**4 + 6*self.mu_J**2*self.sigma_J**2 + 3*self.sigma_J**4)
        
        # Excess kurtosis
        excess_kurt = fourth_moment / (total_var**2)
        
        return excess_kurt

# ========== Utility Functions ==========

def fit_merton_to_historical(returns: np.ndarray, 
                             dt: float = 1/252) -> Dict[str, float]:
    """
    Estimate Merton parameters from historical returns using MLE
    
    This is a simplified estimation. In practice, use:
    - EM algorithm (Expectation-Maximization)
    - Method of moments
    - Bayesian estimation
    
    Args:
        returns: Array of log-returns
        dt: Time step (default: daily = 1/252)
    
    Returns:
        Dictionary of estimated parameters
    """
    # Annualize statistics
    mean_return = np.mean(returns) / dt
    total_var = np.var(returns) / dt
    skew = pd.Series(returns).skew()
    kurt = pd.Series(returns).kurtosis()
    
    # Simple moment matching (rough estimates)
    # Assumes jumps are rare and can be identified as outliers
    
    # Estimate jump intensity from large moves
    threshold = 3 * np.std(returns)
    jumps = returns[np.abs(returns) > threshold]
    lambda_est = len(jumps) / (len(returns) * dt)
    
    if len(jumps) > 0:
        mu_J_est = np.mean(jumps) / lambda_est
        sigma_J_est = np.std(jumps) / np.sqrt(lambda_est)
    else:
        mu_J_est = -0.1
        sigma_J_est = 0.15
    
    # Diffusion volatility (residual after removing jump variance)
    jump_var = lambda_est * (sigma_J_est**2 + mu_J_est**2)
    sigma_est = np.sqrt(max(total_var - jump_var, 0.01))
    
    return {
        'sigma': sigma_est,
        'lambda': max(lambda_est, 0.1),
        'mu_J': mu_J_est,
        'sigma_J': max(sigma_J_est, 0.05)
    }
