"""
SABR (Stochastic Alpha Beta Rho) Model

The SABR model is the market standard for modeling implied volatility smiles,
especially in interest rate derivatives markets.

Forward price dynamics:
dF_t = α_t * F_t^β * dW_t^F
dα_t = ν * α_t * dW_t^α

where:
- F_t: forward price
- α_t: stochastic volatility (alpha)
- β: elasticity parameter (0 ≤ β ≤ 1)
  - β = 0: Normal (absolute) volatility model
  - β = 0.5: CIR-like model
  - β = 1: Lognormal model
- ν: volatility of volatility (vol-of-vol)
- ρ: correlation between forward price and volatility (dW^F · dW^α = ρ*dt)

Hagan et al. (2002) derived an asymptotic expansion for implied volatility
that allows fast calibration and pricing.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
from typing import Tuple, Dict, Optional, List
import warnings

class SABRModel:
    """
    SABR Stochastic Alpha Beta Rho Model
    
    Attributes:
        F0: Initial forward price
        alpha: Initial volatility (ATM volatility scaled)
        beta: Elasticity parameter
        rho: Correlation
        nu: Vol-of-vol
        T: Time to expiry
    """
    
    def __init__(self, F0: float, alpha: float = 0.2, beta: float = 0.5,
                 rho: float = -0.3, nu: float = 0.4):
        """
        Initialize SABR model
        
        Args:
            F0: Initial forward price (or spot if r = q)
            alpha: Initial volatility parameter (typically 0.1 - 0.5)
            beta: CEV exponent (0 = normal, 1 = lognormal, typically 0.5)
            rho: Correlation (-1 < ρ < 1, typically -0.5 to 0)
            nu: Vol-of-vol (typically 0.2 - 1.0)
        """
        self.F0 = F0
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        
        # Validate parameters
        if not 0 <= beta <= 1:
            raise ValueError(f"Beta must be in [0, 1], got {beta}")
        if not -1 < rho < 1:
            raise ValueError(f"Rho must be in (-1, 1), got {rho}")
        if alpha <= 0 or nu <= 0:
            raise ValueError("Alpha and nu must be positive")
    
    def implied_volatility(self, K: float, T: float) -> float:
        """
        Calculate implied Black volatility using Hagan's formula
        
        Hagan et al. (2002) asymptotic expansion:
        
        σ_Black(K, T) = α / [(FK)^((1-β)/2) * {1 + (1-β)²/24 * ln²(F/K) + ...}]
                        * {z/x(z)} * {1 + [(1-β)²/24 * α²/(FK)^(1-β) 
                        + 1/4 * ραν/(FK)^((1-β)/2) + (2-3ρ²)/24 * ν²] * T}
        
        where:
        z = (ν/α) * (FK)^((1-β)/2) * ln(F/K)
        x(z) = ln{[√(1 - 2ρz + z²) + z - ρ] / (1 - ρ)}
        
        Args:
            K: Strike price
            T: Time to maturity (years)
        
        Returns:
            Implied Black volatility
        """
        F = self.F0
        
        # ATM case (K ≈ F)
        if abs(K - F) < 1e-10:
            return self._atm_volatility(T)
        
        # General case
        try:
            # Calculate intermediate values
            FK_mid = (F * K) ** ((1 - self.beta) / 2)
            log_FK = np.log(F / K)
            
            # Calculate z (dimensionless moneyness scaled by vol-of-vol)
            z = (self.nu / self.alpha) * FK_mid * log_FK
            
            # Calculate x(z) - handles small z carefully
            if abs(z) < 1e-7:
                # Taylor expansion for small z: x(z) ≈ 1 - ρz/2 + ...
                x_z = 1 - self.rho * z / 2
            else:
                # Full formula
                sqrt_term = np.sqrt(1 - 2 * self.rho * z + z**2)
                numerator = sqrt_term + z - self.rho
                denominator = 1 - self.rho
                
                if numerator <= 0 or denominator <= 0:
                    # Fallback to approximation
                    x_z = 1
                else:
                    x_z = np.log(numerator / denominator) / z
            
            # First term: α / (FK_mid * {1 + ...})
            gamma1 = (1 - self.beta)**2 / 24 * log_FK**2
            gamma2 = (1 - self.beta)**4 / 1920 * log_FK**4
            denominator = FK_mid * (1 + gamma1 + gamma2)
            
            first_term = self.alpha / denominator
            
            # Second term: z / x(z)
            second_term = z / x_z if abs(x_z) > 1e-10 else 1
            
            # Third term: {1 + ... * T}
            term1 = (1 - self.beta)**2 / 24 * self.alpha**2 / (FK_mid**2)
            term2 = 0.25 * self.rho * self.nu * self.alpha / FK_mid
            term3 = (2 - 3 * self.rho**2) / 24 * self.nu**2
            
            third_term = 1 + (term1 + term2 + term3) * T
            
            # Combine all terms
            sigma_black = first_term * second_term * third_term
            
            return max(sigma_black, 1e-6)  # Ensure positive
            
        except Exception as e:
            warnings.warn(f"SABR IV calculation failed: {e}. Using ATM vol.")
            return self._atm_volatility(T)
    
    def _atm_volatility(self, T: float) -> float:
        """
        ATM implied volatility (K = F)
        
        Simplified Hagan formula:
        σ_ATM = α / F^(1-β) * {1 + [(1-β)²/24 * α²/F^(2-2β) 
                                  + 1/4 * ραν/F^(1-β) 
                                  + (2-3ρ²)/24 * ν²] * T}
        
        Args:
            T: Time to maturity
        
        Returns:
            ATM implied volatility
        """
        F = self.F0
        
        # First term
        first_term = self.alpha / (F ** (1 - self.beta))
        
        # Correction terms
        term1 = (1 - self.beta)**2 / 24 * self.alpha**2 / (F ** (2 - 2*self.beta))
        term2 = 0.25 * self.rho * self.nu * self.alpha / (F ** (1 - self.beta))
        term3 = (2 - 3 * self.rho**2) / 24 * self.nu**2
        
        correction = 1 + (term1 + term2 + term3) * T
        
        return first_term * correction
    
    def calibrate(self, market_data: pd.DataFrame, 
                  fix_beta: bool = True,
                  beta_value: float = 0.5) -> Dict[str, float]:
        """
        Calibrate SABR parameters to market implied volatilities
        
        Minimizes weighted sum of squared errors:
        L(α, ρ, ν) = Σ w_i * [IV_market(K_i) - IV_SABR(K_i)]²
        
        Weights can be vega-based or equal
        
        Args:
            market_data: DataFrame with ['Strike', 'Maturity', 'IV']
            fix_beta: If True, beta is fixed; if False, beta is calibrated
            beta_value: Value of beta if fixed
        
        Returns:
            Dictionary of calibrated parameters
        """
        # Get unique maturity (SABR calibrates per maturity slice)
        maturities = market_data['Maturity'].unique()
        
        if len(maturities) > 1:
            warnings.warn(
                f"Multiple maturities found. Calibrating to first maturity T={maturities:.3f}"
            )
        
        T = maturities
        slice_data = market_data[market_data['Maturity'] == T].copy()
        
        def objective(params):
            """
            Objective function for calibration
            
            params = [alpha, rho, nu] if fix_beta=True
            params = [alpha, beta, rho, nu] if fix_beta=False
            """
            if fix_beta:
                alpha, rho, nu = params
                beta = beta_value
            else:
                alpha, beta, rho, nu = params
            
            # Validate constraints
            if alpha <= 0 or nu <= 0:
                return 1e10
            if not -0.999 < rho < 0.999:
                return 1e10
            if not 0 <= beta <= 1:
                return 1e10
            
            # Temporarily set parameters
            self.alpha = alpha
            self.beta = beta
            self.rho = rho
            self.nu = nu
            
            # Calculate model IVs and errors
            errors = []
            for _, row in slice_data.iterrows():
                K = row['Strike']
                market_iv = row['IV']
                
                try:
                    model_iv = self.implied_volatility(K, T)
                    error = (market_iv - model_iv)**2
                    errors.append(error)
                except:
                    errors.append(1e6)  # Large penalty for failure
            
            return np.sum(errors)
        
        # Initial guess
        if fix_beta:
            x0 = [0.2, -0.3, 0.4]  # [alpha, rho, nu]
            bounds = [(0.01, 2.0), (-0.999, 0.999), (0.01, 3.0)]
        else:
            x0 = [0.2, beta_value, -0.3, 0.4]  # [alpha, beta, rho, nu]
            bounds = [(0.01, 2.0), (0.0, 1.0), (-0.999, 0.999), (0.01, 3.0)]
        
        print(f"Calibrating SABR model (beta {'fixed' if fix_beta else 'free'})...")
        
        # Optimize
        result = minimize(
            objective,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        
        # Extract calibrated parameters
        if fix_beta:
            alpha, rho, nu = result.x
            beta = beta_value
        else:
            alpha, beta, rho, nu = result.x
        
        # Update model
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        
        # Calculate RMSE
        rmse = np.sqrt(result.fun / len(slice_data))
        
        calibrated_params = {
            'alpha': alpha,
            'beta': beta,
            'rho': rho,
            'nu': nu,
            'rmse': rmse
        }
        
        print(f"Calibration complete. RMSE: {rmse:.6f}")
        print(f"Parameters: α={alpha:.4f}, β={beta:.4f}, ρ={rho:.4f}, ν={nu:.4f}")
        
        return calibrated_params
    
    def simulate_paths(self, T: float, n_steps: int, 
                      n_paths: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate forward price and volatility paths using Euler scheme
        
        Discretized SDEs:
        F_{t+dt} = F_t + α_t * F_t^β * √dt * Z_F
        α_{t+dt} = α_t + ν * α_t * √dt * Z_α
        
        where Z_α = ρ * Z_F + √(1-ρ²) * Z_indep
        
        Note: For β < 1, absorption at F=0 is possible. Use reflecting boundary.
        
        Args:
            T: Final time (years)
            n_steps: Number of time steps
            n_paths: Number of simulation paths
        
        Returns:
            times: Array of time points
            F_paths: Simulated forward price paths (n_paths × n_steps)
            alpha_paths: Simulated volatility paths (n_paths × n_steps)
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0, T, n_steps + 1)
        
        # Initialize
        F_paths = np.zeros((n_paths, n_steps + 1))
        alpha_paths = np.zeros((n_paths, n_steps + 1))
        
        F_paths[:, 0] = self.F0
        alpha_paths[:, 0] = self.alpha
        
        for i in range(n_steps):
            # Generate correlated random variables
            Z_F = np.random.standard_normal(n_paths)
            Z_indep = np.random.standard_normal(n_paths)
            Z_alpha = self.rho * Z_F + np.sqrt(1 - self.rho**2) * Z_indep
            
            F = F_paths[:, i]
            alpha = alpha_paths[:, i]
            
            # Ensure positivity
            F_pos = np.maximum(F, 1e-10)
            alpha_pos = np.maximum(alpha, 1e-10)
            
            # Euler scheme for forward price
            # For better stability, use log-normal discretization when β ≈ 1
            if self.beta > 0.99:
                # Log-normal approximation
                F_paths[:, i + 1] = F_pos * np.exp(
                    -0.5 * alpha_pos**2 * dt + alpha_pos * sqrt_dt * Z_F
                )
            else:
                # General CEV process
                F_paths[:, i + 1] = F + alpha_pos * (F_pos ** self.beta) * sqrt_dt * Z_F
            
            # Euler scheme for volatility
            alpha_paths[:, i + 1] = alpha + self.nu * alpha_pos * sqrt_dt * Z_alpha
            
            # Reflecting boundary at zero
            F_paths[:, i + 1] = np.maximum(F_paths[:, i + 1], 0)
            alpha_paths[:, i + 1] = np.maximum(alpha_paths[:, i + 1], 0)
        
        return times, F_paths, alpha_paths
    
    def price_european_mc(self, K: float, T: float, option_type: str = 'call',
                         n_paths: int = 100000) -> float:
        """
        Price European option using Monte Carlo simulation
        
        Payoff:
        Call: E[max(F_T - K, 0)]
        Put: E[max(K - F_T, 0)]
        
        No discounting needed since F is a forward price
        
        Args:
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
            n_paths: Number of Monte Carlo paths
        
        Returns:
            Option price
        """
        n_steps = max(int(T * 252), 50)  # Daily steps
        
        # Simulate paths
        _, F_paths, _ = self.simulate_paths(T, n_steps, n_paths)
        
        # Terminal forward prices
        F_T = F_paths[:, -1]
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(F_T - K, 0)
        else:
            payoffs = np.maximum(K - F_T, 0)
        
        # Expected value
        price = np.mean(payoffs)
        
        return price
    
    def generate_smile(self, T: float, strikes: np.ndarray) -> pd.DataFrame:
        """
        Generate implied volatility smile for given maturity
        
        Args:
            T: Time to maturity
            strikes: Array of strike prices
        
        Returns:
            DataFrame with ['Strike', 'Moneyness', 'IV']
        """
        smile_data = []
        
        for K in strikes:
            iv = self.implied_volatility(K, T)
            smile_data.append({
                'Strike': K,
                'Moneyness': K / self.F0,
                'Log_Moneyness': np.log(K / self.F0),
                'IV': iv
            })
        
        return pd.DataFrame(smile_data)
    
    def backbone_slope(self) -> float:
        """
        Calculate the "backbone" or skew of the volatility smile
        
        Backbone ≈ ρ * ν / α (for small maturities)
        
        Interpretation:
        - Negative backbone: downward sloping smile (typical for equities)
        - Positive backbone: upward sloping smile
        
        Returns:
            Backbone slope parameter
        """
        return self.rho * self.nu / self.alpha
    
    def wing_curvature(self) -> float:
        """
        Calculate the curvature/convexity of volatility smile wings
        
        Curvature ≈ ν² (vol-of-vol controls wings)
        
        High ν → pronounced smile wings
        
        Returns:
            Wing curvature parameter
        """
        return self.nu**2

# ========== Helper Functions ==========

def sabr_surface_calibration(market_data: pd.DataFrame,
                             F0: float,
                             fix_beta: bool = True,
                             beta_value: float = 0.5) -> Dict[float, SABRModel]:
    """
    Calibrate SABR model for each maturity slice independently
    
    Args:
        market_data: DataFrame with ['Strike', 'Maturity', 'IV']
        F0: Forward price
        fix_beta: Whether to fix beta parameter
        beta_value: Value of beta if fixed
    
    Returns:
        Dictionary mapping maturity → calibrated SABRModel
    """
    maturities = sorted(market_data['Maturity'].unique())
    calibrated_models = {}
    
    for T in maturities:
        print(f"\n{'='*60}")
        print(f"Calibrating maturity T = {T:.3f} years")
        print(f"{'='*60}")
        
        slice_data = market_data[market_data['Maturity'] == T]
        
        # Initialize model
        model = SABRModel(F0=F0, beta=beta_value)
        
        # Calibrate
        params = model.calibrate(slice_data, fix_beta=fix_beta, beta_value=beta_value)
        
        calibrated_models[T] = model
    
    return calibrated_models
