"""
Kou Double Exponential Jump Diffusion Model

The Kou model (2002) extends Merton by using asymmetric double exponential 
jump sizes, allowing different behavior for upward vs downward jumps.

Asset price dynamics:
dS_t/S_t = μ*dt + σ*dW_t + d(Σ(V_i - 1))

where:
- μ: drift (r - q under risk-neutral measure)
- σ: diffusion volatility
- dW_t: Brownian motion
- V_i: jump sizes with asymmetric double exponential distribution

Jump size distribution:
V - 1 has double exponential density:

f(x) = p * η₁ * e^(-η₁*x) * 1_{x≥0} + q * η₂ * e^(η₂*x) * 1_{x<0}

where:
- p: probability of upward jump (0 < p < 1)
- q = 1 - p: probability of downward jump
- η₁ > 1: rate of exponential decay for upward jumps (controls right tail)
- η₂ > 0: rate of exponential decay for downward jumps (controls left tail)

Mean jump size:
E[V - 1] = p/(η₁ - 1) - q/(η₂ + 1) ≡ ζ

Key advantages over Merton:
1. Tractable analytical formulas (Laplace transform)
2. Captures asymmetric jumps (crashes vs rallies)
3. Heavy tails on both sides
4. Better fit to empirical return distributions
"""

import numpy as np
import pandas as pd
from scipy.stats import expon
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
from typing import Tuple, Dict, Optional, List
import warnings

class KouJumpDiffusion:
    """
    Kou Double Exponential Jump Diffusion Model
    
    Attributes:
        S0: Initial spot price
        r: Risk-free rate
        q: Dividend yield
        sigma: Diffusion volatility
        lambda_: Jump intensity (jumps per year)
        p: Probability of upward jump
        eta1: Decay rate for upward jumps (η₁ > 1)
        eta2: Decay rate for downward jumps (η₂ > 0)
    """
    
    def __init__(self, S0: float, r: float, q: float = 0.0,
                 sigma: float = 0.2, lambda_: float = 1.0,
                 p: float = 0.4, eta1: float = 3.0, eta2: float = 4.0):
        """
        Initialize Kou Jump Diffusion model
        
        Args:
            S0: Current spot price
            r: Risk-free rate (annualized)
            q: Dividend yield (annualized)
            sigma: Diffusion volatility (typically 0.1 - 0.3)
            lambda_: Jump intensity (typically 0.5 - 3.0 jumps/year)
            p: Probability of upward jump (typically 0.3 - 0.5)
            eta1: Upward jump decay rate (must be > 1, typically 2 - 10)
                  Smaller η₁ → fatter right tail
            eta2: Downward jump decay rate (must be > 0, typically 3 - 10)
                  Smaller η₂ → fatter left tail (more severe crashes)
        """
        self.S0 = S0
        self.r = r
        self.q = q
        self.sigma = sigma
        self.lambda_ = lambda_
        self.p = p
        self.eta1 = eta1
        self.eta2 = eta2
        
        # Validate parameters
        if not 0 < p < 1:
            raise ValueError(f"p must be in (0, 1), got {p}")
        if eta1 <= 1:
            raise ValueError(f"eta1 must be > 1 (for finite expectation), got {eta1}")
        if eta2 <= 0:
            raise ValueError(f"eta2 must be > 0, got {eta2}")
        if sigma <= 0 or lambda_ < 0:
            raise ValueError("sigma must be positive, lambda non-negative")
        
        # Calculate mean jump size: ζ = E[V - 1]
        self.zeta = p / (eta1 - 1) - (1 - p) / (eta2 + 1)
        
        # Risk-neutral drift adjustment
        # μ = r - q - λ*ζ - 0.5*σ²
        self.drift = r - q - lambda_ * self.zeta - 0.5 * sigma**2
    
    def simulate_paths(self, T: float, n_steps: int, 
                      n_paths: int = 10000) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Simulate price paths using Monte Carlo
        
        Discretization over interval dt:
        
        1. Diffusion component:
           S_t → S_t * exp[(r - q - λ*ζ - 0.5*σ²)*dt + σ*√dt*Z]
        
        2. Jump component:
           - Generate number of jumps: N ~ Poisson(λ*dt)
           - For each jump i:
             * With probability p: upward jump, size ~ Exp(η₁)
             * With probability q: downward jump, size ~ -Exp(η₂)
           - Apply: S_t → S_t * exp(Σ jump_sizes)
        
        Args:
            T: Final time (years)
            n_steps: Number of time steps
            n_paths: Number of simulation paths
        
        Returns:
            times: Array of time points
            paths: Simulated price paths (n_paths × n_steps)
            jump_info: List of dicts with jump information per path
        """
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        
        # Initialize
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        # Track jump information
        jump_info = [{
            'times': [],
            'sizes': [],
            'directions': []  # 'up' or 'down'
        } for _ in range(n_paths)]
        
        for i in range(n_steps):
            # 1. Diffusion component
            Z = np.random.standard_normal(n_paths)
            diffusion = np.exp(self.drift * dt + self.sigma * np.sqrt(dt) * Z)
            
            # 2. Jump component
            # Number of jumps in this time step
            n_jumps = np.random.poisson(self.lambda_ * dt, size=n_paths)
            
            # Initialize jump factor
            jump_factor = np.ones(n_paths)
            
            for path_idx in range(n_paths):
                if n_jumps[path_idx] > 0:
                    total_jump = 0
                    
                    for _ in range(n_jumps[path_idx]):
                        # Determine jump direction
                        if np.random.random() < self.p:
                            # Upward jump: Y ~ Exponential(η₁)
                            jump_size = np.random.exponential(1 / self.eta1)
                            direction = 'up'
                        else:
                            # Downward jump: Y ~ -Exponential(η₂)
                            jump_size = -np.random.exponential(1 / self.eta2)
                            direction = 'down'
                        
                        total_jump += jump_size
                        
                        # Record jump
                        jump_info[path_idx]['times'].append(times[i])
                        jump_info[path_idx]['sizes'].append(jump_size)
                        jump_info[path_idx]['directions'].append(direction)
                    
                    # Apply total jump: multiply by exp(total_jump)
                    jump_factor[path_idx] = np.exp(total_jump)
            
            # Update price
            paths[:, i + 1] = paths[:, i] * diffusion * jump_factor
        
        return times, paths, jump_info
    
    def price_european_laplace(self, K: float, T: float, 
                               option_type: str = 'call',
                               n_points: int = 2048) -> float:
        """
        Price European option using Laplace transform inversion
        
        Kou (2002) showed that option prices can be computed via:
        
        C(S, K, T) = e^(-r*T) * E[max(S_T - K, 0)]
        
        Using the Laplace transform of the density of ln(S_T) and
        numerical inversion (e.g., Fast Fourier Transform or quadrature)
        
        This is similar to Carr-Madan method but uses characteristic function
        of the Kou model:
        
        φ(u) = E[e^(i*u*ln(S_T))] = exp{Ψ(u)*T}
        
        where Ψ(u) is the characteristic exponent:
        Ψ(u) = i*u*(r - q - 0.5*σ²) - 0.5*σ²*u² + λ*[Π(u) - 1]
        Π(u) = p*η₁/(η₁ - i*u) + (1-p)*η₂/(η₂ + i*u)
        
        Args:
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
            n_points: Number of points for numerical integration
        
        Returns:
            Option price
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return max(self.S0 - K, 0)
            else:
                return max(K - self.S0, 0)
        
        # Use Carr-Madan approach with Kou characteristic function
        alpha = 1.5  # Damping parameter
        
        # Define modified characteristic function
        def char_func(v):
            """
            Characteristic function of ln(S_T)
            
            φ(u) = exp{Ψ(u)*T}
            where Ψ(u) = i*u*drift - 0.5*σ²*u² + λ*[Π(u) - 1]
            """
            u = v - (alpha + 1) * 1j
            
            # Diffusion component
            psi_diffusion = 1j * u * self.drift - 0.5 * self.sigma**2 * u**2
            
            # Jump component: λ*[Π(u) - 1]
            # Π(u) = p*η₁/(η₁ - i*u) + q*η₂/(η₂ + i*u)
            pi_up = self.p * self.eta1 / (self.eta1 - 1j * u)
            pi_down = (1 - self.p) * self.eta2 / (self.eta2 + 1j * u)
            pi_u = pi_up + pi_down
            
            psi_jump = self.lambda_ * (pi_u - 1)
            
            # Total characteristic exponent
            psi = psi_diffusion + psi_jump
            
            return np.exp(psi * T)
        
        # Carr-Madan pricing formula
        k = np.log(K / self.S0)
        
        def integrand(v):
            """Integrand for Carr-Madan formula"""
            cf = char_func(v)
            numerator = np.exp(-1j * v * k) * cf
            denominator = alpha**2 + alpha - v**2 + 1j * (2*alpha + 1) * v
            return (numerator / denominator).real
        
        # Numerical integration
        integral, _ = quad(integrand, 0, 100, limit=n_points)
        
        # Call price
        call_price = self.S0 * np.exp(-alpha * k) / np.pi * integral
        call_price = call_price * np.exp(-self.r * T)
        
        if option_type.lower() == 'call':
            return max(call_price, 0)
        else:
            # Put-Call parity
            put_price = call_price - self.S0 * np.exp(-self.q * T) + K * np.exp(-self.r * T)
            return max(put_price, 0)
    
    def price_european_mc(self, K: float, T: float, 
                         option_type: str = 'call',
                         n_paths: int = 100000) -> Tuple[float, float]:
        """
        Price European option using Monte Carlo simulation
        
        Faster than Laplace transform for single option,
        but less accurate for building entire surface
        
        Args:
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
            n_paths: Number of Monte Carlo paths
        
        Returns:
            (price, standard_error)
        """
        n_steps = max(int(T * 252), 50)
        
        # Simulate paths
        _, paths, _ = self.simulate_paths(T, n_steps, n_paths)
        
        # Terminal prices
        S_T = paths[:, -1]
        
        # Payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        # Discount to present value
        price = np.exp(-self.r * T) * np.mean(payoffs)
        std_error = np.exp(-self.r * T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, std_error
    
    def calibrate(self, market_data: pd.DataFrame,
                  method: str = 'global',
                  fix_eta1: bool = False) -> Dict[str, float]:
        """
        Calibrate Kou model parameters to market option prices
        
        Minimizes weighted sum of squared errors:
        L(σ, λ, p, η₁, η₂) = Σ [Price_market - Price_Kou]²
        
        Note: 5-parameter optimization is challenging. Consider:
        - Fixing β parameters based on empirical distributions
        - Multi-stage calibration
        - Adding regularization
        
        Args:
            market_data: DataFrame with ['Strike', 'Maturity', 'Price', 'Type']
            method: 'global' or 'local' optimization
            fix_eta1: If True, fix η₁ (reduces dimensionality)
        
        Returns:
            Dictionary of calibrated parameters
        """
        def objective(params):
            """
            Objective function
            params = [sigma, lambda_, p, eta1, eta2] or
                    [sigma, lambda_, p, eta2] if fix_eta1=True
            """
            if fix_eta1:
                sigma, lambda_, p, eta2 = params
                eta1 = self.eta1  # Keep current value
            else:
                sigma, lambda_, p, eta1, eta2 = params
            
            # Validate constraints
            if sigma <= 0 or lambda_ < 0:
                return 1e10
            if not 0.01 < p < 0.99:
                return 1e10
            if eta1 <= 1.01 or eta2 <= 0.01:  # Small buffer for stability
                return 1e10
            
            # Temporarily set parameters
            self.sigma = sigma
            self.lambda_ = lambda_
            self.p = p
            self.eta1 = eta1
            self.eta2 = eta2
            self.zeta = p / (eta1 - 1) - (1 - p) / (eta2 + 1)
            self.drift = self.r - self.q - lambda_ * self.zeta - 0.5 * sigma**2
            
            total_error = 0
            
            for _, row in market_data.iterrows():
                K = row['Strike']
                T = row['Maturity']
                market_price = row['Price']
                option_type = row['Type'].lower()
                
                try:
                    # Price using Laplace transform (faster than MC for calibration)
                    model_price = self.price_european_laplace(K, T, option_type)
                    error = (market_price - model_price)**2
                    total_error += error
                except:
                    total_error += 1e8
            
            return total_error
        
        # Parameter bounds
        if fix_eta1:
            # [sigma, lambda_, p, eta2]
            bounds = [
                (0.01, 0.8),   # sigma
                (0.0, 10.0),   # lambda_
                (0.01, 0.99),  # p
                (0.5, 20.0)    # eta2
            ]
            x0 = [self.sigma, self.lambda_, self.p, self.eta2]
        else:
            # [sigma, lambda_, p, eta1, eta2]
            bounds = [
                (0.01, 0.8),   # sigma
                (0.0, 10.0),   # lambda_
                (0.01, 0.99),  # p
                (1.1, 20.0),   # eta1 (must be > 1)
                (0.5, 20.0)    # eta2
            ]
            x0 = [self.sigma, self.lambda_, self.p, self.eta1, self.eta2]
        
        print(f"Calibrating Kou Jump Diffusion model (eta1 {'fixed' if fix_eta1 else 'free'})...")
        
        if method == 'global':
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=100,
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
        if fix_eta1:
            self.sigma, self.lambda_, self.p, self.eta2 = result.x
        else:
            self.sigma, self.lambda_, self.p, self.eta1, self.eta2 = result.x
        
        self.zeta = self.p / (self.eta1 - 1) - (1 - self.p) / (self.eta2 + 1)
        self.drift = self.r - self.q - self.lambda_ * self.zeta - 0.5 * self.sigma**2
        
        calibrated_params = {
            'sigma': self.sigma,
            'lambda': self.lambda_,
            'p': self.p,
            'eta1': self.eta1,
            'eta2': self.eta2,
            'zeta': self.zeta,
            'rmse': np.sqrt(result.fun / len(market_data))
        }
        
        print(f"Calibration complete. RMSE: {calibrated_params['rmse']:.6f}")
        print(f"Parameters: σ={self.sigma:.4f}, λ={self.lambda_:.4f}, "
              f"p={self.p:.4f}, η₁={self.eta1:.4f}, η₂={self.eta2:.4f}")
        print(f"Mean jump size: {100*self.zeta:.2f}%")
        
        return calibrated_params
    
    def implied_volatility(self, K: float, T: float, 
                          option_type: str = 'call') -> float:
        """
        Calculate Black-Scholes implied volatility from Kou price
        
        Args:
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
        
        Returns:
            Implied volatility
        """
        from scipy.stats import norm
        
        # Get Kou price
        kou_price = self.price_european_laplace(K, T, option_type)
        
        # Newton-Raphson
        sigma_iv = 0.3
        
        for _ in range(100):
            d1 = (np.log(self.S0 / K) + (self.r - self.q + 0.5 * sigma_iv**2) * T) / \
                 (sigma_iv * np.sqrt(T))
            d2 = d1 - sigma_iv * np.sqrt(T)
            
            if option_type.lower() == 'call':
                bs_price = self.S0 * np.exp(-self.q * T) * norm.cdf(d1) - \
                          K * np.exp(-self.r * T) * norm.cdf(d2)
            else:
                bs_price = K * np.exp(-self.r * T) * norm.cdf(-d2) - \
                          self.S0 * np.exp(-self.q * T) * norm.cdf(-d1)
            
            vega = self.S0 * np.exp(-self.q * T) * norm.pdf(d1) * np.sqrt(T)
            
            diff = bs_price - kou_price
            
            if abs(diff) < 1e-6 or vega < 1e-10:
                break
            
            sigma_iv = sigma_iv - diff / vega
            sigma_iv = max(sigma_iv, 0.001)
        
        return sigma_iv
    
    def generate_iv_surface(self, strikes: np.ndarray,
                           maturities: np.ndarray) -> pd.DataFrame:
        """
        Generate implied volatility surface
        
        Kou model produces:
        - Asymmetric smiles (due to p ≠ 0.5)
        - Pronounced skew for short maturities
        - Heavy tails (fatter than lognormal)
        
        Args:
            strikes: Array of strike prices
            maturities: Array of maturities
        
        Returns:
            DataFrame with IV surface
        """
        surface_data = []
        
        for T in maturities:
            for K in strikes:
                try:
                    iv = self.implied_volatility(K, T, 'call')
                    
                    surface_data.append({
                        'Strike': K,
                        'Maturity': T,
                        'Moneyness': K / self.S0,
                        'Log_Moneyness': np.log(K / self.S0),
                        'IV': iv
                    })
                except:
                    pass
        
        return pd.DataFrame(surface_data)
    
    def jump_statistics(self) -> Dict[str, float]:
        """
        Calculate statistics of the jump size distribution
        
        Returns:
            Dictionary with jump moments and characteristics
        """
        # Mean jump size (already computed)
        mean_jump = self.zeta
        
        # Variance of jump size
        # Var[Y] = p*E[Y_up²] + q*E[Y_down²] - E[Y]²
        # For exponential: E[X²] = 2/η²
        var_up = self.p * 2 / (self.eta1 - 1)**2 / (self.eta1 - 2) if self.eta1 > 2 else np.inf
        var_down = (1 - self.p) * 2 / (self.eta2 + 1)**2 / (self.eta2 + 2)
        var_jump = var_up + var_down - mean_jump**2
        
        # Skewness (third moment)
        # More negative η₂ → larger negative skew
        skew_approx = (self.p / self.eta1**3 - (1 - self.p) / self.eta2**3) / (var_jump**1.5)
        
        return {
            'mean_jump': mean_jump,
            'variance_jump': var_jump,
            'std_jump': np.sqrt(var_jump) if var_jump > 0 else np.nan,
            'skewness_approx': skew_approx,
            'prob_up': self.p,
            'prob_down': 1 - self.p,
            'avg_up_size': 1 / (self.eta1 - 1) if self.eta1 > 1 else np.inf,
            'avg_down_size': -1 / (self.eta2 + 1),
            'tail_index_up': self.eta1,
            'tail_index_down': self.eta2
        }

# ========== Utility Functions ==========

def fit_kou_to_returns(returns: np.ndarray, 
                       dt: float = 1/252) -> Dict[str, float]:
    """
    Estimate Kou parameters from historical returns
    
    Uses method of moments matching:
    1. Diffusion volatility from small returns
    2. Jump parameters from tail behavior
    
    Args:
        returns: Array of log-returns
        dt: Time step (default: daily)
    
    Returns:
        Dictionary of estimated parameters
    """
    # Separate large moves (likely jumps) from normal diffusion
    threshold = 2.5 * np.std(returns)
    
    normal_returns = returns[np.abs(returns) < threshold]
    large_returns = returns[np.abs(returns) >= threshold]
    
    # Estimate diffusion volatility from normal returns
    sigma_est = np.std(normal_returns) / np.sqrt(dt)
    
    # Estimate jump parameters from large returns
    if len(large_returns) > 10:
        # Jump intensity
        lambda_est = len(large_returns) / (len(returns) * dt)
        
        # Separate up and down jumps
        up_jumps = large_returns[large_returns > 0]
        down_jumps = -large_returns[large_returns < 0]  # Make positive
        
        # Probability of upward jump
        p_est = len(up_jumps) / len(large_returns) if len(large_returns) > 0 else 0.4
        
        # Fit exponential to tails
        if len(up_jumps) > 0:
            eta1_est = 1 / np.mean(up_jumps) + 1  # Must be > 1
        else:
            eta1_est = 3.0
        
        if len(down_jumps) > 0:
            eta2_est = 1 / np.mean(down_jumps)
        else:
            eta2_est = 4.0
    else:
        # Default values if not enough jump data
        lambda_est = 0.5
        p_est = 0.4
        eta1_est = 3.0
        eta2_est = 4.0
    
    return {
        'sigma': max(sigma_est, 0.05),
        'lambda': max(lambda_est, 0.1),
        'p': np.clip(p_est, 0.1, 0.9),
        'eta1': max(eta1_est, 1.5),
        'eta2': max(eta2_est, 1.0)
    }

def compare_jump_models(S0: float, K: float, T: float, r: float,
                       merton_params: Dict, kou_params: Dict) -> pd.DataFrame:
    """
    Compare Merton vs Kou model prices
    
    Args:
        S0: Spot price
        K: Strike price
        T: Maturity
        r: Risk-free rate
        merton_params: Merton model parameters
        kou_params: Kou model parameters
    
    Returns:
        Comparison DataFrame
    """
    # Initialize models
    merton = MertonJumpDiffusion(S0, r, **merton_params)
    kou = KouJumpDiffusion(S0, r, **kou_params)
    
    # Price options
    merton_call = merton.price_european_analytical(K, T, 'call')
    kou_call = kou.price_european_laplace(K, T, 'call')
    
    # Get IVs
    merton_iv = merton.implied_volatility(K, T, 'call')
    kou_iv = kou.implied_volatility(K, T, 'call')
    
    comparison = pd.DataFrame({
        'Model': ['Merton', 'Kou'],
        'Call_Price': [merton_call, kou_call],
        'Implied_Vol': [merton_iv, kou_iv],
        'Price_Diff': [0, kou_call - merton_call],
        'IV_Diff': [0, kou_iv - merton_iv]
    })
    
    return comparison
