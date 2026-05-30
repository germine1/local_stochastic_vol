"""
Monte Carlo Simulation Methods for Option Pricing

Monte Carlo simulation prices derivatives by:
1. Simulating many paths of the underlying asset
2. Computing payoff for each path
3. Taking the average and discounting to present value

Key advantages:
- Handles complex payoffs (path-dependent, multi-asset)
- Easy to implement
- Naturally parallelizable

Key challenges:
- Slow convergence: Error ∝ 1/√N
- Inefficient for Greeks (finite difference bumping)
- Variance reduction techniques needed for accuracy
"""

import numpy as np
from typing import Tuple, Dict, Callable, Optional, List
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

class MonteCarloEngine:
    """
    Generic Monte Carlo engine for option pricing
    
    Supports:
    - European, American (Longstaff-Schwartz), Asian, Barrier options
    - Variance reduction: antithetic variates, control variates
    - Parallel computation
    """
    
    def __init__(self, n_simulations: int = 100000, 
                 n_steps: int = 252,
                 use_antithetic: bool = True,
                 use_control_variate: bool = False,
                 parallel: bool = False,
                 seed: Optional[int] = None):
        """
        Initialize Monte Carlo engine
        
        Args:
            n_simulations: Number of Monte Carlo paths
            n_steps: Number of time steps per path
            use_antithetic: Use antithetic variates for variance reduction
            use_control_variate: Use control variate technique
            parallel: Use parallel processing
            seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.use_antithetic = use_antithetic
        self.use_control_variate = use_control_variate
        self.parallel = parallel
        
        if seed is not None:
            np.random.seed(seed)
    
    def price_european(self, S0: float, K: float, T: float, r: float, 
                       sigma: float, q: float = 0.0, 
                       option_type: str = 'call') -> Dict[str, float]:
        """
        Price European option using Monte Carlo
        
        Geometric Brownian Motion:
        S_T = S_0 * exp[(r - q - σ²/2)*T + σ*√T*Z]
        
        where Z ~ N(0,1)
        
        Option value:
        V = e^(-r*T) * E[Payoff(S_T)]
        
        Standard error (CLT):
        SE = σ_payoff / √N
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            option_type: 'call' or 'put'
        
        Returns:
            Dictionary with price, standard_error, confidence_interval
        """
        # Determine effective number of paths (double if antithetic)
        n_paths = self.n_simulations // 2 if self.use_antithetic else self.n_simulations
        
        # Generate random normals
        Z = np.random.standard_normal(n_paths)
        
        # Terminal stock prices using GBM
        drift = (r - q - 0.5 * sigma**2) * T
        diffusion = sigma * np.sqrt(T)
        
        S_T = S0 * np.exp(drift + diffusion * Z)
        
        # Antithetic variates: use -Z as well
        if self.use_antithetic:
            S_T_anti = S0 * np.exp(drift + diffusion * (-Z))
            S_T = np.concatenate([S_T, S_T_anti])
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S_T - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - S_T, 0)
        else:
            raise ValueError(f"Unknown option type: {option_type}")
        
        # Control variate (use analytical BS price as control)
        if self.use_control_variate:
            from scipy.stats import norm
            
            # Black-Scholes price (known exact value)
            d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                bs_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                control_payoff = S_T - K  # Linear in S_T
                control_exact = S0 * np.exp(-q * T) - K * np.exp(-r * T)
            else:
                bs_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
                control_payoff = K - S_T
                control_exact = K * np.exp(-r * T) - S0 * np.exp(-q * T)
            
            # Optimal beta (regression coefficient)
            cov = np.cov(payoffs, control_payoff)[0, 1]
            var_control = np.var(control_payoff)
            beta = cov / var_control if var_control > 0 else 0
            
            # Adjusted payoffs
            payoffs_adjusted = payoffs - beta * (control_payoff - control_exact)
            payoffs = payoffs_adjusted
        
        # Discount to present value
        discount = np.exp(-r * T)
        price = discount * np.mean(payoffs)
        
        # Standard error and confidence interval
        std_payoff = np.std(payoffs)
        se = discount * std_payoff / np.sqrt(len(payoffs))
        ci_95 = (price - 1.96 * se, price + 1.96 * se)
        
        return {
            'price': price,
            'standard_error': se,
            'confidence_interval_95': ci_95,
            'n_paths': len(payoffs)
        }
    
    def price_asian(self, S0: float, K: float, T: float, r: float,
                   sigma: float, q: float = 0.0,
                   option_type: str = 'call',
                   average_type: str = 'arithmetic') -> Dict[str, float]:
        """
        Price Asian option using Monte Carlo
        
        Asian option payoff depends on average price:
        - Arithmetic average: Ā = (1/n) * Σ S_i
        - Geometric average: Ḡ = (Π S_i)^(1/n)
        
        Call payoff: max(Ā - K, 0)
        Put payoff: max(K - Ā, 0)
        
        Note: Geometric Asian has closed-form solution
        Arithmetic Asian requires numerical methods
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            option_type: 'call' or 'put'
            average_type: 'arithmetic' or 'geometric'
        
        Returns:
            Dictionary with pricing results
        """
        dt = T / self.n_steps
        n_paths = self.n_simulations // 2 if self.use_antithetic else self.n_simulations
        
        # Initialize paths
        paths = np.zeros((n_paths, self.n_steps + 1))
        paths[:, 0] = S0
        
        # Generate correlated random numbers
        Z = np.random.standard_normal((n_paths, self.n_steps))
        
        # Simulate GBM paths
        drift = (r - q - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        for i in range(self.n_steps):
            paths[:, i + 1] = paths[:, i] * np.exp(drift + diffusion * Z[:, i])
        
        # Antithetic paths
        if self.use_antithetic:
            paths_anti = np.zeros((n_paths, self.n_steps + 1))
            paths_anti[:, 0] = S0
            for i in range(self.n_steps):
                paths_anti[:, i + 1] = paths_anti[:, i] * np.exp(drift + diffusion * (-Z[:, i]))
            paths = np.vstack([paths, paths_anti])
        
        # Calculate average
        if average_type == 'arithmetic':
            averages = np.mean(paths, axis=1)
        elif average_type == 'geometric':
            # Geometric mean = exp(mean of logs)
            averages = np.exp(np.mean(np.log(paths), axis=1))
        else:
            raise ValueError(f"Unknown average type: {average_type}")
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(averages - K, 0)
        else:
            payoffs = np.maximum(K - averages, 0)
        
        # Discount and compute statistics
        discount = np.exp(-r * T)
        price = discount * np.mean(payoffs)
        se = discount * np.std(payoffs) / np.sqrt(len(payoffs))
        ci_95 = (price - 1.96 * se, price + 1.96 * se)
        
        return {
            'price': price,
            'standard_error': se,
            'confidence_interval_95': ci_95,
            'n_paths': len(payoffs)
        }
    
    def price_barrier(self, S0: float, K: float, B: float, T: float, 
                     r: float, sigma: float, q: float = 0.0,
                     option_type: str = 'call',
                     barrier_type: str = 'down-and-out') -> Dict[str, float]:
        """
        Price barrier option using Monte Carlo
        
        Barrier types:
        - 'down-and-out': knocked out if S < B
        - 'down-and-in': activated if S < B
        - 'up-and-out': knocked out if S > B
        - 'up-and-in': activated if S > B
        
        Continuous monitoring approximation:
        Check barrier at each time step
        
        Args:
            S0: Initial stock price
            K: Strike price
            B: Barrier level
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            option_type: 'call' or 'put'
            barrier_type: Type of barrier
        
        Returns:
            Dictionary with pricing results
        """
        dt = T / self.n_steps
        n_paths = self.n_simulations
        
        # Simulate paths
        paths = np.zeros((n_paths, self.n_steps + 1))
        paths[:, 0] = S0
        
        drift = (r - q - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        Z = np.random.standard_normal((n_paths, self.n_steps))
        
        for i in range(self.n_steps):
            paths[:, i + 1] = paths[:, i] * np.exp(drift + diffusion * Z[:, i])
        
        # Check barrier crossing
        if 'down' in barrier_type:
            barrier_crossed = np.any(paths <= B, axis=1)
        else:  # 'up'
            barrier_crossed = np.any(paths >= B, axis=1)
        
        # Terminal payoff
        S_T = paths[:, -1]
        if option_type.lower() == 'call':
            terminal_payoff = np.maximum(S_T - K, 0)
        else:
            terminal_payoff = np.maximum(K - S_T, 0)
        
        # Apply barrier condition
        if 'out' in barrier_type:
            # Knocked out → payoff = 0 if barrier crossed
            payoffs = np.where(barrier_crossed, 0, terminal_payoff)
        else:  # 'in'
            # Knocked in → payoff = terminal_payoff if barrier crossed
            payoffs = np.where(barrier_crossed, terminal_payoff, 0)
        
        # Discount and statistics
        discount = np.exp(-r * T)
        price = discount * np.mean(payoffs)
        se = discount * np.std(payoffs) / np.sqrt(n_paths)
        ci_95 = (price - 1.96 * se, price + 1.96 * se)
        
        return {
            'price': price,
            'standard_error': se,
            'confidence_interval_95': ci_95,
            'barrier_hit_prob': np.mean(barrier_crossed),
            'n_paths': n_paths
        }
    
    def price_american_lsm(self, S0: float, K: float, T: float, r: float,
                          sigma: float, q: float = 0.0,
                          option_type: str = 'put',
                          basis_functions: int = 3) -> Dict[str, float]:
        """
        Price American option using Longstaff-Schwartz (LSM) algorithm
        
        LSM Algorithm:
        1. Simulate forward paths
        2. Backward induction from maturity
        3. At each time t:
           - Compute continuation value using regression
           - Exercise if immediate payoff > continuation value
        
        Regression:
        C(S_t) ≈ Σ β_j * φ_j(S_t)
        
        where φ_j are basis functions (e.g., Laguerre polynomials, powers)
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            option_type: 'call' or 'put'
            basis_functions: Number of basis functions (polynomial degree)
        
        Returns:
            Dictionary with pricing results
        """
        dt = T / self.n_steps
        discount = np.exp(-r * dt)
        n_paths = self.n_simulations
        
        # Simulate paths
        paths = np.zeros((n_paths, self.n_steps + 1))
        paths[:, 0] = S0
        
        drift = (r - q - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        Z = np.random.standard_normal((n_paths, self.n_steps))
        
        for i in range(self.n_steps):
            paths[:, i + 1] = paths[:, i] * np.exp(drift + diffusion * Z[:, i])
        
        # Initialize cash flows (terminal payoff)
        if option_type.lower() == 'put':
            cash_flows = np.maximum(K - paths[:, -1], 0)
        else:
            cash_flows = np.maximum(paths[:, -1] - K, 0)
        
        # Backward induction
        for t in range(self.n_steps - 1, 0, -1):
            S_t = paths[:, t]
            
            # Immediate exercise value
            if option_type.lower() == 'put':
                exercise_value = np.maximum(K - S_t, 0)
            else:
                exercise_value = np.maximum(S_t - K, 0)
            
            # Only consider in-the-money paths for regression
            itm = exercise_value > 0
            
            if np.sum(itm) > 0:
                # Regression to estimate continuation value
                # Use polynomial basis: 1, x, x², ..., x^n
                X = S_t[itm]
                Y = discount * cash_flows[itm]
                
                # Build design matrix with polynomial basis
                design_matrix = np.column_stack([X**i for i in range(basis_functions + 1)])
                
                # Least squares regression
                try:
                    coeffs = np.linalg.lstsq(design_matrix, Y, rcond=None)
                    continuation_value = design_matrix @ coeffs
                except:
                    continuation_value = Y  # Fallback
                
                # Exercise decision: exercise if immediate > continuation
                exercise = exercise_value[itm] > continuation_value
                
                # Update cash flows
                cash_flows[itm] = np.where(exercise, exercise_value[itm], discount * cash_flows[itm])
                cash_flows[~itm] = discount * cash_flows[~itm]
            else:
                # No ITM paths, just discount
                cash_flows = discount * cash_flows
        
        # Discount from t=1 to t=0
        price = discount * np.mean(cash_flows)
        se = discount * np.std(cash_flows) / np.sqrt(n_paths)
        ci_95 = (price - 1.96 * se, price + 1.96 * se)
        
        return {
            'price': price,
            'standard_error': se,
            'confidence_interval_95': ci_95,
            'n_paths': n_paths
        }
    
    def calculate_greeks(self, S0: float, K: float, T: float, r: float,
                        sigma: float, q: float = 0.0,
                        option_type: str = 'call',
                        greek: str = 'delta',
                        bump_size: float = 0.01) -> Dict[str, float]:
        """
        Calculate Greeks using finite difference bumping
        
        Greeks:
        - Delta: ∂V/∂S ≈ [V(S+h) - V(S-h)] / (2h)
        - Gamma: ∂²V/∂S² ≈ [V(S+h) - 2V(S) + V(S-h)] / h²
        - Vega: ∂V/∂σ ≈ [V(σ+h) - V(σ-h)] / (2h)
        - Theta: ∂V/∂T ≈ [V(T-h) - V(T)] / h
        - Rho: ∂V/∂r ≈ [V(r+h) - V(r-h)] / (2h)
        
        Args:
            S0: Stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            option_type: 'call' or 'put'
            greek: 'delta', 'gamma', 'vega', 'theta', 'rho'
            bump_size: Finite difference step size
        
        Returns:
            Dictionary with greek value and standard error
        """
        base_price = self.price_european(S0, K, T, r, sigma, q, option_type)['price']
        
        if greek == 'delta':
            up = self.price_european(S0 * (1 + bump_size), K, T, r, sigma, q, option_type)['price']
            down = self.price_european(S0 * (1 - bump_size), K, T, r, sigma, q, option_type)['price']
            value = (up - down) / (2 * S0 * bump_size)
            
        elif greek == 'gamma':
            up = self.price_european(S0 * (1 + bump_size), K, T, r, sigma, q, option_type)['price']
            down = self.price_european(S0 * (1 - bump_size), K, T, r, sigma, q, option_type)['price']
            value = (up - 2 * base_price + down) / (S0 * bump_size)**2
            
        elif greek == 'vega':
            up = self.price_european(S0, K, T, r, sigma + bump_size, q, option_type)['price']
            down = self.price_european(S0, K, T, r, sigma - bump_size, q, option_type)['price']
            value = (up - down) / (2 * bump_size)
            
        elif greek == 'theta':
            down = self.price_european(S0, K, T - bump_size, r, sigma, q, option_type)['price']
            value = (down - base_price) / bump_size  # Negative time direction
            
        elif greek == 'rho':
            up = self.price_european(S0, K, T, r + bump_size, sigma, q, option_type)['price']
            down = self.price_european(S0, K, T, r - bump_size, sigma, q, option_type)['price']
            value = (up - down) / (2 * bump_size)
            
        else:
            raise ValueError(f"Unknown greek: {greek}")
        
        return {
            'greek': greek,
            'value': value,
            'base_price': base_price
        }
