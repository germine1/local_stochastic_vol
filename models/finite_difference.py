"""
Finite Difference Methods for Solving PDEs in Option Pricing

Finite difference methods solve the Black-Scholes PDE numerically:

∂V/∂t + 0.5*σ²*S²*∂²V/∂S² + (r-q)*S*∂V/∂S - r*V = 0

Discretization methods:
1. Explicit (Forward Euler): Simple but conditionally stable
2. Implicit (Backward Euler): Unconditionally stable, requires solving linear system
3. Crank-Nicolson: Best accuracy, O(Δt²) + O(ΔS²), unconditionally stable

Boundary conditions:
- European call: V(0,t) = 0, V(S_max,t) = S_max - K*e^(-r*(T-t))
- European put: V(0,t) = K*e^(-r*(T-t)), V(S_max,t) = 0
- American: Free boundary problem, check early exercise

Grid:
- S ∈ [0, S_max], discretized into M points
- t ∈ [0, T], discretized into N points
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
from typing import Tuple, Dict, Optional, Callable
import warnings

class FiniteDifferencePDE:
    """
    Finite Difference solver for option pricing PDEs
    
    Supports:
    - European, American options
    - Explicit, Implicit, Crank-Nicolson schemes
    - Variable coefficients (for local volatility)
    """
    
    def __init__(self, S_max: float = 300, M: int = 200, N: int = 1000,
                 scheme: str = 'crank-nicolson'):
        """
        Initialize finite difference solver
        
        Args:
            S_max: Maximum stock price for grid
            M: Number of stock price grid points
            N: Number of time steps
            scheme: 'explicit', 'implicit', or 'crank-nicolson'
        """
        self.S_max = S_max
        self.M = M
        self.N = N
        self.scheme = scheme.lower()
        
        if self.scheme not in ['explicit', 'implicit', 'crank-nicolson']:
            raise ValueError(f"Unknown scheme: {scheme}")
        
        # Will be initialized in price methods
        self.S_grid = None
        self.t_grid = None
        self.dS = None
        self.dt = None
        self.grid = None
    
    def price_european(self, S0: float, K: float, T: float, r: float,
                      sigma: float, q: float = 0.0,
                      option_type: str = 'call') -> Dict[str, float]:
        """
        Price European option using finite difference
        
        Black-Scholes PDE (backwards in time τ = T - t):
        ∂V/∂τ = 0.5*σ²*S²*∂²V/∂S² + (r-q)*S*∂V/∂S - r*V
        
        Discretization:
        V[i,j] ≈ V(i*ΔS, j*Δt)
        
        Derivatives:
        ∂V/∂S ≈ [V[i+1,j] - V[i-1,j]] / (2*ΔS)           [central difference]
        ∂²V/∂S² ≈ [V[i+1,j] - 2*V[i,j] + V[i-1,j]] / ΔS²  [central difference]
        ∂V/∂t ≈ [V[i,j+1] - V[i,j]] / Δt                  [forward difference]
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            option_type: 'call' or 'put'
        
        Returns:
            Dictionary with price and grid information
        """
        # Setup grid
        self.dS = self.S_max / self.M
        self.dt = T / self.N
        
        self.S_grid = np.linspace(0, self.S_max, self.M + 1)
        self.t_grid = np.linspace(0, T, self.N + 1)
        
        # Initialize solution grid
        self.grid = np.zeros((self.M + 1, self.N + 1))
        
        # Terminal condition (payoff at maturity)
        if option_type.lower() == 'call':
            self.grid[:, -1] = np.maximum(self.S_grid - K, 0)
        else:
            self.grid[:, -1] = np.maximum(K - self.S_grid, 0)
        
        # Boundary conditions
        for j in range(self.N + 1):
            t = self.t_grid[j]
            tau = T - t  # Time to maturity
            
            if option_type.lower() == 'call':
                self.grid[0, j] = 0  # V(0, t) = 0
                self.grid[-1, j] = self.S_max - K * np.exp(-r * tau)  # V(S_max, t)
            else:
                self.grid[0, j] = K * np.exp(-r * tau)  # V(0, t) = K*e^(-r*τ)
                self.grid[-1, j] = 0  # V(S_max, t) = 0
        
        # Solve using selected scheme
        if self.scheme == 'explicit':
            self._solve_explicit(r, sigma, q)
        elif self.scheme == 'implicit':
            self._solve_implicit(r, sigma, q)
        else:  # crank-nicolson
            self._solve_crank_nicolson(r, sigma, q)
        
        # Interpolate to get price at S0
        interpolator = interp1d(self.S_grid, self.grid[:, 0], kind='cubic')
        price = float(interpolator(S0))
        
        return {
            'price': price,
            'grid': self.grid,
            'S_grid': self.S_grid,
            't_grid': self.t_grid
        }
    
    def _solve_explicit(self, r: float, sigma: float, q: float):
        """
        Explicit (Forward Euler) scheme
        
        V[i,j] = V[i,j+1] + Δt*[a[i]*V[i-1,j+1] + b[i]*V[i,j+1] + c[i]*V[i+1,j+1]]
        
        where:
        a[i] = 0.5*(r-q)*(i*Δt) - 0.5*σ²*i²*Δt
        b[i] = 1 + σ²*i²*Δt + r*Δt
        c[i] = -0.5*(r-q)*i*Δt - 0.5*σ²*i²*Δt
        
        Stability condition (CFL):
        Δt ≤ ΔS² / (σ²*S_max²)
        
        This method is conditionally stable!
        """
        # Check stability condition
        dt_stable = self.dS**2 / (sigma**2 * self.S_max**2)
        if self.dt > dt_stable:
            warnings.warn(
                f"Explicit scheme may be unstable: dt={self.dt:.6f} > dt_stable={dt_stable:.6f}"
            )
        
        # Backward in time (from j=N to j=0)
        for j in range(self.N - 1, -1, -1):
            for i in range(1, self.M):  # Interior points only
                S = self.S_grid[i]
                
                # Coefficients for explicit scheme
                a = 0.5 * self.dt * ((r - q) * i - sigma**2 * i**2)
                b = 1 - self.dt * (sigma**2 * i**2 + r)
                c = 0.5 * self.dt * ((r - q) * i + sigma**2 * i**2)
                
                # Update formula
                self.grid[i, j] = (a * self.grid[i - 1, j + 1] +
                                  b * self.grid[i, j + 1] +
                                  c * self.grid[i + 1, j + 1])
    
    def _solve_implicit(self, r: float, sigma: float, q: float):
        """
        Implicit (Backward Euler) scheme
        
        Solve linear system at each time step:
        A * V[j] = V[j+1]
        
        where A is tridiagonal matrix:
        A[i,i-1] = -a[i]
        A[i,i]   = 1 + b[i]
        A[i,i+1] = -c[i]
        
        Unconditionally stable!
        """
        # Build coefficient matrix (tridiagonal)
        # This matrix is constant over time if σ, r, q are constant
        
        # Diagonals
        lower = np.zeros(self.M - 1)  # Sub-diagonal
        main = np.zeros(self.M - 1)   # Main diagonal
        upper = np.zeros(self.M - 1)  # Super-diagonal
        
        for i in range(1, self.M):
            a = 0.5 * self.dt * ((r - q) * i - sigma**2 * i**2)
            b = 1 + self.dt * (sigma**2 * i**2 + r)
            c = 0.5 * self.dt * ((r - q) * i + sigma**2 * i**2)
            
            if i > 1:
                lower[i - 2] = -a
            main[i - 1] = b
            if i < self.M - 1:
                upper[i - 1] = -c
        
        # Create sparse matrix
        A = diags([lower, main, upper], [-1, 0, 1], format='csr')
        
        # Backward in time
        for j in range(self.N - 1, -1, -1):
            # Right-hand side (previous time step)
            rhs = self.grid[1:self.M, j + 1].copy()
            
            # Adjust for boundary conditions
            rhs += 0.5 * self.dt * ((r - q) - sigma**2) * self.grid[0, j]
            rhs[-1] += 0.5 * self.dt * ((r - q) * (self.M - 1) + 
                                        sigma**2 * (self.M - 1)**2) * self.grid[self.M, j]
            
            # Solve linear system
            self.grid[1:self.M, j] = spsolve(A, rhs)
    
    def _solve_crank_nicolson(self, r: float, sigma: float, q: float):
        """
        Crank-Nicolson scheme (θ = 0.5)
        
        Average of explicit and implicit:
        V[i,j] = 0.5*[Explicit update] + 0.5*[Implicit update]
        
        Solve:
        (I + 0.5*Δt*L)*V[j] = (I - 0.5*Δt*L)*V[j+1]
        
        where L is the differential operator
        
        Benefits:
        - O(Δt²) accuracy (second-order in time)
        - Unconditionally stable
        - Best accuracy-stability tradeoff
        """
        # Build matrices for Crank-Nicolson
        lower = np.zeros(self.M - 1)
        main = np.zeros(self.M - 1)
        upper = np.zeros(self.M - 1)
        
        # Left-hand side matrix (implicit part)
        for i in range(1, self.M):
            a = 0.25 * self.dt * ((r - q) * i - sigma**2 * i**2)
            b = 1 + 0.5 * self.dt * (sigma**2 * i**2 + r)
            c = 0.25 * self.dt * ((r - q) * i + sigma**2 * i**2)
            
            if i > 1:
                lower[i - 2] = -a
            main[i - 1] = b
            if i < self.M - 1:
                upper[i - 1] = -c
        
        A_implicit = diags([lower, main, upper], [-1, 0, 1], format='csr')
        
        # Right-hand side matrix (explicit part)
        lower_rhs = np.zeros(self.M - 1)
        main_rhs = np.zeros(self.M - 1)
        upper_rhs = np.zeros(self.M - 1)
        
        for i in range(1, self.M):
            a = 0.25 * self.dt * ((r - q) * i - sigma**2 * i**2)
            b = 1 - 0.5 * self.dt * (sigma**2 * i**2 + r)
            c = 0.25 * self.dt * ((r - q) * i + sigma**2 * i**2)
            
            if i > 1:
                lower_rhs[i - 2] = a
            main_rhs[i - 1] = b
            if i < self.M - 1:
                upper_rhs[i - 1] = c
        
        A_explicit = diags([lower_rhs, main_rhs, upper_rhs], [-1, 0, 1], format='csr')
        
        # Backward in time
        for j in range(self.N - 1, -1, -1):
            # Right-hand side
            rhs = A_explicit.dot(self.grid[1:self.M, j + 1])
            
            # Boundary condition adjustments
            # Lower boundary
            a0 = 0.25 * self.dt * ((r - q) - sigma**2)
            rhs += a0 * (self.grid[0, j] + self.grid[0, j + 1])
            
            # Upper boundary
            aM = 0.25 * self.dt * ((r - q) * (self.M - 1) + 
                                   sigma**2 * (self.M - 1)**2)
            rhs[-1] += aM * (self.grid[self.M, j] + self.grid[self.M, j + 1])
            
            # Solve
            self.grid[1:self.M, j] = spsolve(A_implicit, rhs)
    
    def price_american(self, S0: float, K: float, T: float, r: float,
                      sigma: float, q: float = 0.0,
                      option_type: str = 'put') -> Dict[str, float]:
        """
        Price American option using finite difference with early exercise
        
        At each time step, after computing continuation value, check:
        V[i,j] = max(V_continuation[i,j], V_exercise[i,j])
        
        where:
        V_exercise = max(K - S, 0) for put
        V_exercise = max(S - K, 0) for call
        
        This creates a free boundary problem
        
        Args:
            S0: Stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            option_type: 'call' or 'put'
        
        Returns:
            Dictionary with price and early exercise boundary
        """
        # Setup grid
        self.dS = self.S_max / self.M
        self.dt = T / self.N
        
        self.S_grid = np.linspace(0, self.S_max, self.M + 1)
        self.t_grid = np.linspace(0, T, self.N + 1)
        
        # Initialize solution grid
        self.grid = np.zeros((self.M + 1, self.N + 1))
        
        # Terminal condition
        if option_type.lower() == 'call':
            self.grid[:, -1] = np.maximum(self.S_grid - K, 0)
        else:
            self.grid[:, -1] = np.maximum(K - self.S_grid, 0)
        
        # Track early exercise boundary
        exercise_boundary = np.zeros(self.N + 1)
        exercise_boundary[-1] = K if option_type.lower() == 'put' else np.inf
        
        # Boundary conditions
        for j in range(self.N + 1):
            t = self.t_grid[j]
            tau = T - t
            
            if option_type.lower() == 'call':
                self.grid[0, j] = 0
                self.grid[-1, j] = max(self.S_max - K, self.S_max - K * np.exp(-r * tau))
            else:
                self.grid[0, j] = K  # Always exercise immediately for S=0
                self.grid[-1, j] = 0
        
        # Use Crank-Nicolson with early exercise check
        # Build matrices
        lower = np.zeros(self.M - 1)
        main = np.zeros(self.M - 1)
        upper = np.zeros(self.M - 1)
        
        for i in range(1, self.M):
            a = 0.25 * self.dt * ((r - q) * i - sigma**2 * i**2)
            b = 1 + 0.5 * self.dt * (sigma**2 * i**2 + r)
            c = 0.25 * self.dt * ((r - q) * i + sigma**2 * i**2)
            
            if i > 1:
                lower[i - 2] = -a
            main[i - 1] = b
            if i < self.M - 1:
                upper[i - 1] = -c
        
        A_implicit = diags([lower, main, upper], [-1, 0, 1], format='csr')
        
        lower_rhs = np.zeros(self.M - 1)
        main_rhs = np.zeros(self.M - 1)
        upper_rhs = np.zeros(self.M - 1)
        
        for i in range(1, self.M):
            a = 0.25 * self.dt * ((r - q) * i - sigma**2 * i**2)
            b = 1 - 0.5 * self.dt * (sigma**2 * i**2 + r)
            c = 0.25 * self.dt * ((r - q) * i + sigma**2 * i**2)
            
            if i > 1:
                lower_rhs[i - 2] = a
            main_rhs[i - 1] = b
            if i < self.M - 1:
                upper_rhs[i - 1] = c
        
        A_explicit = diags([lower_rhs, main_rhs, upper_rhs], [-1, 0, 1], format='csr')
        
        # Backward in time
        for j in range(self.N - 1, -1, -1):
            # Continuation value (European step)
            rhs = A_explicit.dot(self.grid[1:self.M, j + 1])
            
            # Boundary adjustments
            a0 = 0.25 * self.dt * ((r - q) - sigma**2)
            rhs += a0 * (self.grid[0, j] + self.grid[0, j + 1])
            
            aM = 0.25 * self.dt * ((r - q) * (self.M - 1) + 
                                   sigma**2 * (self.M - 1)**2)
            rhs[-1] += aM * (self.grid[self.M, j] + self.grid[self.M, j + 1])
            
            # Solve for continuation value
            V_continuation = spsolve(A_implicit, rhs)
            
            # Exercise value
            if option_type.lower() == 'call':
                V_exercise = np.maximum(self.S_grid[1:self.M] - K, 0)
            else:
                V_exercise = np.maximum(K - self.S_grid[1:self.M], 0)
            
            # American constraint: max(continuation, exercise)
            self.grid[1:self.M, j] = np.maximum(V_continuation, V_exercise)
            
            # Find exercise boundary (first point where exercise > continuation)
            if option_type.lower() == 'put':
                exercise_points = np.where(V_exercise > V_continuation + 1e-10)
                if len(exercise_points) > 0:
                    exercise_boundary[j] = self.S_grid[exercise_points[-1] + 1]
                else:
                    exercise_boundary[j] = 0
        
        # Interpolate price
        interpolator = interp1d(self.S_grid, self.grid[:, 0], kind='cubic')
        price = float(interpolator(S0))
        
        return {
            'price': price,
            'grid': self.grid,
            'S_grid': self.S_grid,
            't_grid': self.t_grid,
            'exercise_boundary': exercise_boundary
        }
    
    def calculate_greeks(self, S0: float, K: float, T: float, r: float,
                        sigma: float, q: float = 0.0,
                        option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate Greeks using finite difference grid
        
        Greeks from PDE solution:
        - Delta: ∂V/∂S at S=S0, t=0
        - Gamma: ∂²V/∂S² at S=S0, t=0
        - Theta: ∂V/∂t at S=S0, t=0
        
        Args:
            S0: Stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            option_type: 'call' or 'put'
        
        Returns:
            Dictionary with Greeks
        """
        # Price option to get grid
        result = self.price_european(S0, K, T, r, sigma, q, option_type)
        
        # Find grid point closest to S0
        idx = np.argmin(np.abs(self.S_grid - S0))
        
        # Delta: ∂V/∂S using central difference
        if idx > 0 and idx < self.M:
            delta = (self.grid[idx + 1, 0] - self.grid[idx - 1, 0]) / (2 * self.dS)
        else:
            delta = 0
        
        # Gamma: ∂²V/∂S² using central difference
        if idx > 0 and idx < self.M:
            gamma = (self.grid[idx + 1, 0] - 2 * self.grid[idx, 0] + 
                    self.grid[idx - 1, 0]) / (self.dS**2)
        else:
            gamma = 0
        
        # Theta: ∂V/∂t using forward difference
        theta = -(self.grid[idx, 1] - self.grid[idx, 0]) / self.dt
        
        return {
            'price': result['price'],
            'delta': delta,
            'gamma': gamma,
            'theta': theta
        }

# ========== Utility Functions ==========

def compare_fd_schemes(S0: float, K: float, T: float, r: float, sigma: float,
                      option_type: str = 'call') -> Dict[str, Dict]:
    """
    Compare different finite difference schemes
    
    Returns:
        Dictionary with results from each scheme
    """
    schemes = ['explicit', 'implicit', 'crank-nicolson']
    results = {}
    
    for scheme in schemes:
        solver = FiniteDifferencePDE(S_max=2*K, M=200, N=1000, scheme=scheme)
        result = solver.price_european(S0, K, T, r, sigma, option_type=option_type)
        results[scheme] = {
            'price': result['price'],
            'scheme': scheme
        }
    
    return results

def convergence_test(S0: float, K: float, T: float, r: float, sigma: float,
                    grid_sizes: list = [50, 100, 200, 400]) -> Dict:
    """
    Test convergence of finite difference as grid is refined
    
    Args:
        S0: Stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility
        grid_sizes: List of M values to test
    
    Returns:
        Dictionary with convergence results
    """
    from scipy.stats import norm
    
    # Analytical Black-Scholes price
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    results = []
    
    for M in grid_sizes:
        N = M * 5  # Keep time steps proportional
        solver = FiniteDifferencePDE(S_max=3*K, M=M, N=N, scheme='crank-nicolson')
        result = solver.price_european(S0, K, T, r, sigma, option_type='call')
        
        error = abs(result['price'] - bs_price)
        
        results.append({
            'M': M,
            'N': N,
            'price': result['price'],
            'bs_price': bs_price,
            'error': error,
            'relative_error': error / bs_price
        })
    
    return {
        'results': results,
        'bs_price': bs_price
    }
