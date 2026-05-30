"""
Main Engine for Local & Stochastic Volatility Framework

Features:
- User input handling
- Data fetching toggle (yfinance / Bloomberg stub)
- Model selection and initialization
- Numerical method selection
- Calibration engine interface
- Pricing and Greeks calculation
- Output visualization (placeholder integration)
- Modular, extendable for future components

Usage:
    python main_engine.py --data_source yfinance --model heston --method carr_madan --mode calibrate --output price
    
Modes:
    simulate    - Simulate price paths
    calibrate   - Calibrate model parameters
    price       - Price options
    greeks      - Compute Greeks

Models:
    dupire
    heston
    sabr
    merton_jump
    kou_jump

Methods:
    monte_carlo
    carr_madan
    finite_difference

Example:
    python main_engine.py --data_source yfinance --model heston --method carr_madan --mode price --output price --ticker AAPL --expiry 2024-12-20 --strike 150 --n_paths 100000
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import modules (assumed in project structure)
from data.data_manager import DataManager
from models.local_vol import DupireLocalVol
from models.heston import HestonModel
from models.sabr import SABRModel
from models.merton_jump import MertonJumpDiffusion
from models.kou_jump import KouJumpDiffusion
from methods.monte_carlo import MonteCarloEngine
from methods.carr_madan import CarrMadanFFT
from methods.finite_difference import FiniteDifferencePDE
from greeks.greeks_calculator import GreeksCalculator

class VolatilityEngine:
    def __init__(self, args):
        """
        Initialize main engine with user arguments
        """
        self.data_source = args.data_source
        self.model_name = args.model.lower()
        self.method_name = args.method.lower()
        self.mode = args.mode.lower()
        self.output = args.output.lower()
        
        # Instantiate data manager
        self.data_manager = DataManager(source=self.data_source)
        
        # Asset and option parameters
        self.ticker = args.ticker
        self.expiry = args.expiry
        self.strike = args.strike
        self.n_paths = args.n_paths
        
        # Placeholder for model, method instances
        self.model = None
        self.method = None
        self.greeks_calc = GreeksCalculator()
    
    def run(self):
        """
        Run the engine based on mode
        """
        print(f"Running mode: {self.mode}, model: {self.model_name}, method: {self.method_name}")

        # Step 1: Fetch data
        spot = self.data_manager.get_spot_price(self.ticker)
        print(f"Spot price for {self.ticker}: {spot}")
        
        # For simplicity, only implementing yfinance option chain and implied vols
        if self.data_source == 'yfinance':
            options = self.data_manager.get_option_chain(self.ticker, self.expiry)
            iv_surface = self.data_manager.calculate_implied_volatility_surface(self.ticker)
            print(f"Fetched option chain data for expiry {self.expiry}")
        else:
            # Bloomberg stub
            options = None
            iv_surface = None
            print(f"Bloomberg integration not implemented.")
        
        # Step 2: Initialize model with default/example parms or calibrated parms
        self.initialize_model(spot)
        
        # Step 3: Execute mode-specific workflow
        if self.mode == 'simulate':
            self.simulate()
        elif self.mode == 'calibrate':
            self.calibrate(iv_surface)
        elif self.mode == 'price':
            self.price()
        elif self.mode == 'greeks':
            self.calculate_greeks()
        else:
            print(f"Unknown mode: {self.mode}")
    
    def initialize_model(self, spot):
        """
        Create model instance based on selection
        
        For demonstration, uses default parameters or simple estimation.
        Calibration modifies these later.
        """
        if self.model_name == 'dupire':
            self.model = DupireLocalVol(S0=spot, r=0.03, q=0.01)
        elif self.model_name == 'heston':
            self.model = HestonModel(S0=spot, v0=0.04, r=0.03, q=0.01)
        elif self.model_name == 'sabr':
            self.model = SABRModel(F0=spot, alpha=0.2, beta=0.5, rho=-0.4, nu=0.5)
        elif self.model_name == 'merton_jump':
            self.model = MertonJumpDiffusion(S0=spot, r=0.03, q=0.01)
        elif self.model_name == 'kou_jump':
            self.model = KouJumpDiffusion(S0=spot, r=0.03, q=0.01)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Initialize numerical method
        if self.method_name == 'monte_carlo':
            self.method = MonteCarloEngine(n_simulations=self.n_paths or 100000)
        elif self.method_name == 'carr_madan':
            self.method = CarrMadanFFT()
        elif self.method_name == 'finite_difference':
            self.method = FiniteDifferencePDE()
        else:
            raise ValueError(f"Unknown method: {self.method_name}")
    
    def simulate(self):
        """
        Simulate price paths using model and method
        """
        print("Starting simulation...")
        # Currently implementing basic simulation calls per model
        
        if self.model_name == 'dupire':
            T = 1.0
            times, paths = self.model.simulate_path(T, n_steps=252, n_paths=10)
            print(f"Simulated {paths.shape} paths with {paths.shape} steps.")
            # Plot sample paths
            for i in range(min(5, paths.shape)):
                plt.plot(times, paths[i], label=f'Path {i+1}')
            plt.title('Local Volatility Simulation Paths')
            plt.xlabel('Time (years)')
            plt.ylabel('Price')
            plt.legend()
            plt.show()
        
        elif self.model_name == 'heston':
            T = 1.0
            times, S_paths, v_paths = self.model.simulate_paths(T, n_steps=252, n_paths=10)
            print(f"Simulated {S_paths.shape} price paths and variance paths.")
            for i in range(min(5, S_paths.shape)):
                plt.plot(times, S_paths[i], label=f'Price Path {i+1}')
            plt.title('Heston Model Price Paths')
            plt.xlabel('Time (years)')
            plt.ylabel('Price')
            plt.legend()
            plt.show()
        
        # Add other models accordingly
        
    def calibrate(self, iv_surface: pd.DataFrame):
        """
        Calibrate model to implied volatility surface
        """
        if iv_surface is None:
            print("No implied volatility surface data available for calibration.")
            return
        
        print("Starting calibration...")
        if self.model_name == 'heston':
            market_data = iv_surface.copy()
            market_data['Price'] = 0  # Placeholder, needs actual option prices
            
            # For demo, calibrate to IVs converted to prices using BS or approximation
            
            # TODO: Populate market_data['Price'] from IVs or market quotes
            
            calibrated_params = self.model.calibrate(market_data)
            print("Calibrated Heston parameters:", calibrated_params)
        
        elif self.model_name == 'sabr':
            calibrated_models = SABRModel.sabr_surface_calibration(iv_surface, self.model.F0)
            print("Calibrated SABR models for maturities:", list(calibrated_models.keys()))
        
        else:
            print(f"Calibration not yet implemented for model: {self.model_name}")
    
    def price(self):
        """
        Price option based on model, method, and input parameters
        """
        print("Starting pricing...")
        T = self.calc_time_to_expiry(self.expiry)
        K = self.strike
        
        if self.method_name == 'monte_carlo':
            if self.model_name == 'heston':
                price_info = self.method.price_european(S0=self.model.S0,
                                                       K=K, T=T, r=self.model.r,
                                                       sigma=np.sqrt(self.model.v0),
                                                       q=self.model.q,
                                                       option_type='call')
                print(f"Monte Carlo price: {price_info['price']:.4f} ± {1.96*price_info['standard_error']:.4f}")
            else:
                print(f"Pricing for model-method combination ({self.model_name}-{self.method_name}) is a work in progress.")
        
        elif self.method_name == 'carr_madan':
            # Example: price single strike for Heston
            if self.model_name == 'heston':
                def heston_cf(u, T_val):
                    return self.model._heston_characteristic_function(u, T_val)
                
                cm = CarrMadanFFT()
                price = cm.price_european(self.model.S0, K, T, self.model.r, heston_cf, 'call', self.model.q)
                print(f"Carr-Madan FFT price: {price:.4f}")
            else:
                print("Carr-Madan pricing for other models is under development.")
        
        elif self.method_name == 'finite_difference':
            solver = FiniteDifferencePDE()
            result = solver.price_european(self.model.S0, K, T, self.model.r, sigma=0.2)
            print(f"Finite Difference price: {result['price']:.4f}")
        
        else:
            print(f"Method {self.method_name} not supported yet.")
    
    def calculate_greeks(self):
        """
        Calculate Greeks for specified option parameters
        """
        T = self.calc_time_to_expiry(self.expiry)
        K = self.strike
        S = self.model.S0
        
        print("Calculating Greeks...")
        
        # Use GreeksCalculator with BS approximation if no model analytical available
        
        calc = GreeksCalculator()
        greeks = calc.black_scholes_greeks(S, K, T, self.model.r, 0.2, self.model.q)
        
        dashboard = calc.greeks_dashboard(S, K, T, self.model.r, 0.2, self.model.q)
        print(dashboard)
    
    @staticmethod
    def calc_time_to_expiry(expiry_str):
        """
        Calculate time to expiry in years from today
        """
        from datetime import datetime
        
        today = datetime.today()
        expiry = datetime.strptime(expiry_str, '%Y-%m-%d')
        delta = expiry - today
        return max(delta.days / 365.0, 0.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local and Stochastic Volatility Engine")
    parser.add_argument("--data_source", type=str, default="yfinance",
                        help="Data source: yfinance or bloomberg")
    parser.add_argument("--model", type=str, required=True,
                        choices=['dupire', 'heston', 'sabr', 'merton_jump', 'kou_jump'],
                        help="Model to use")
    parser.add_argument("--method", type=str, required=True,
                        choices=['monte_carlo', 'carr_madan', 'finite_difference'],
                        help="Numerical method")
    parser.add_argument("--mode", type=str, required=True,
                        choices=['simulate', 'calibrate', 'price', 'greeks'],
                        help="Operation mode")
    parser.add_argument("--output", type=str, default="price",
                        choices=['price', 'paths', 'surface', 'greeks'],
                        help="Desired output")
    parser.add_argument("--ticker", type=str, required=True,
                        help="Asset ticker symbol")
    parser.add_argument("--expiry", type=str, required=True,
                        help="Option expiry date (YYYY-MM-DD)")
    parser.add_argument("--strike", type=float, required=True,
                        help="Option strike price")
    parser.add_argument("--n_paths", type=int, default=100000,
                        help="Number of Monte Carlo simulation paths")
    
    args = parser.parse_args()
    
    engine = VolatilityEngine(args)
    engine.run()
