"""
Visualization Module for Volatility Models

Creates interactive 3D plots and charts for:
- Implied volatility surfaces
- Price path simulations
- Greeks heatmaps
- Calibration diagnostics
- Model comparison

Uses Plotly for interactive, web-ready visualizations (perfect for Vercel deployment)
Also supports Matplotlib for static plots

All plots can be exported as HTML for GitHub Pages or Vercel hosting
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple
import warnings

class VolatilityVisualizer:
    """
    Comprehensive visualization engine for volatility models
    
    Features:
    - Interactive 3D surfaces (Plotly)
    - Animated simulations
    - Greeks dashboards
    - Export to HTML for web deployment
    """
    
    def __init__(self, theme: str = 'plotly_dark', export_path: str = './exports/'):
        """
        Initialize visualizer
        
        Args:
            theme: Plotly theme ('plotly', 'plotly_dark', 'plotly_white', 'ggplot2')
            export_path: Directory to save exported HTML files
        """
        self.theme = theme
        self.export_path = export_path
        
        # Create export directory if doesn't exist
        import os
        os.makedirs(export_path, exist_ok=True)
    
    # ========== Volatility Surfaces ==========
    
    def plot_iv_surface_3d(self, strikes: np.ndarray, maturities: np.ndarray,
                          iv_surface: np.ndarray, 
                          title: str = "Implied Volatility Surface",
                          spot: Optional[float] = None,
                          save_html: bool = True) -> go.Figure:
        """
        Plot 3D implied volatility surface
        
        Creates interactive 3D surface plot showing volatility smile evolution
        across strikes and maturities
        
        Args:
            strikes: Array of strike prices (X-axis)
            maturities: Array of maturities in years (Y-axis)
            iv_surface: 2D array of implied vols [len(maturities) × len(strikes)]
            title: Plot title
            spot: Current spot price (to mark ATM)
            save_html: Save as interactive HTML file
        
        Returns:
            Plotly Figure object
        
        Example:
            >>> strikes = np.linspace(80, 120, 50)
            >>> maturities = np.array([0.25, 0.5, 1.0, 2.0])
            >>> iv_surface = model.generate_iv_surface(strikes, maturities)
            >>> viz.plot_iv_surface_3d(strikes, maturities, iv_surface)
        """
        # Create meshgrid for surface plot
        K_mesh, T_mesh = np.meshgrid(strikes, maturities)
        
        # Create 3D surface
        fig = go.Figure(data=[
            go.Surface(
                x=K_mesh,
                y=T_mesh,
                z=iv_surface * 100,  # Convert to percentage
                colorscale='Viridis',
                colorbar=dict(
                    title="IV (%)",
                    titleside="right",
                    tickmode="linear",
                    tick0=0,
                    dtick=2
                ),
                hovertemplate='Strike: %{x:.2f}<br>Maturity: %{y:.2f}y<br>IV: %{z:.2f}%<extra></extra>'
            )
        ])
        
        # Add ATM line if spot provided
        if spot is not None:
            atm_line = go.Scatter3d(
                x=[spot] * len(maturities),
                y=maturities,
                z=iv_surface[:, np.argmin(np.abs(strikes - spot))] * 100,
                mode='lines',
                line=dict(color='red', width=5),
                name='ATM',
                hovertemplate='ATM Line<extra></extra>'
            )
            fig.add_trace(atm_line)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title='Strike Price',
                yaxis_title='Time to Maturity (years)',
                zaxis_title='Implied Volatility (%)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            template=self.theme,
            width=1000,
            height=700
        )
        
        # Save to HTML
        if save_html:
            filename = f"{self.export_path}/iv_surface_3d.html"
            fig.write_html(filename)
            print(f"Saved interactive 3D surface to {filename}")
        
        return fig
    
    def plot_volatility_smile(self, strikes: np.ndarray, ivs: np.ndarray,
                             maturity: float, spot: Optional[float] = None,
                             title: Optional[str] = None,
                             save_html: bool = True) -> go.Figure:
        """
        Plot volatility smile for single maturity
        
        Shows how implied volatility varies across strikes (the "smile")
        
        Args:
            strikes: Array of strike prices
            ivs: Implied volatilities (same length as strikes)
            maturity: Time to maturity in years
            spot: Current spot price
            title: Custom title
            save_html: Save as HTML
        
        Returns:
            Plotly Figure
        """
        if title is None:
            title = f"Volatility Smile (T={maturity:.2f} years)"
        
        # Calculate moneyness
        if spot is not None:
            moneyness = strikes / spot
        else:
            moneyness = strikes / np.median(strikes)
        
        # Create figure with secondary x-axis
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": False}]]
        )
        
        # Main smile curve
        fig.add_trace(
            go.Scatter(
                x=strikes,
                y=ivs * 100,
                mode='lines+markers',
                name='Volatility Smile',
                line=dict(color='cyan', width=3),
                marker=dict(size=8),
                hovertemplate='Strike: %{x:.2f}<br>IV: %{y:.2f}%<extra></extra>'
            )
        )
        
        # Mark ATM
        if spot is not None:
            atm_idx = np.argmin(np.abs(strikes - spot))
            fig.add_trace(
                go.Scatter(
                    x=[strikes[atm_idx]],
                    y=[ivs[atm_idx] * 100],
                    mode='markers',
                    name='ATM',
                    marker=dict(color='red', size=15, symbol='star'),
                    hovertemplate='ATM<br>Strike: %{x:.2f}<br>IV: %{y:.2f}%<extra></extra>'
                )
            )
            
            # Add vertical line at ATM
            fig.add_vline(
                x=spot,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                annotation_text="ATM"
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Strike Price',
            yaxis_title='Implied Volatility (%)',
            template=self.theme,
            hovermode='x unified',
            width=900,
            height=500
        )
        
        if save_html:
            filename = f"{self.export_path}/volatility_smile_T{maturity:.2f}.html"
            fig.write_html(filename)
            print(f"Saved volatility smile to {filename}")
        
        return fig
    
    def plot_term_structure(self, maturities: np.ndarray, atm_vols: np.ndarray,
                           title: str = "Volatility Term Structure",
                           save_html: bool = True) -> go.Figure:
        """
        Plot volatility term structure (ATM volatility across maturities)
        
        Shows time evolution of volatility
        
        Args:
            maturities: Time to maturities (years)
            atm_vols: ATM implied volatilities
            title: Plot title
            save_html: Save as HTML
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=maturities,
                y=atm_vols * 100,
                mode='lines+markers',
                name='ATM Volatility',
                line=dict(color='orange', width=3),
                marker=dict(size=10),
                hovertemplate='Maturity: %{x:.2f}y<br>ATM IV: %{y:.2f}%<extra></extra>'
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Time to Maturity (years)',
            yaxis_title='ATM Implied Volatility (%)',
            template=self.theme,
            width=900,
            height=500
        )
        
        if save_html:
            filename = f"{self.export_path}/term_structure.html"
            fig.write_html(filename)
            print(f"Saved term structure to {filename}")
        
        return fig
    
    # ========== Price Path Simulations ==========
    
    def plot_price_paths(self, times: np.ndarray, paths: np.ndarray,
                        title: str = "Simulated Price Paths",
                        n_display: int = 50,
                        spot: Optional[float] = None,
                        save_html: bool = True) -> go.Figure:
        """
        Plot simulated price paths
        
        Args:
            times: Time grid (years)
            paths: 2D array [n_paths × n_steps]
            title: Plot title
            n_display: Number of paths to display (for clarity)
            spot: Initial spot price (for reference line)
            save_html: Save as HTML
        
        Returns:
            Plotly Figure
        """
        n_paths = paths.shape
        n_display = min(n_display, n_paths)
        
        fig = go.Figure()
        
        # Plot sample paths
        for i in range(n_display):
            opacity = 0.3 if i > 10 else 0.7
            show_legend = i == 0
            
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=paths[i],
                    mode='lines',
                    name='Simulated Path' if show_legend else '',
                    line=dict(width=1),
                    opacity=opacity,
                    showlegend=show_legend,
                    hovertemplate='Time: %{x:.2f}<br>Price: %{y:.2f}<extra></extra>'
                )
            )
        
        # Add mean path
        mean_path = np.mean(paths, axis=0)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=mean_path,
                mode='lines',
                name='Mean Path',
                line=dict(color='red', width=3),
                hovertemplate='Time: %{x:.2f}<br>Mean Price: %{y:.2f}<extra></extra>'
            )
        )
        
        # Add confidence bands (5th and 95th percentiles)
        lower_bound = np.percentile(paths, 5, axis=0)
        upper_bound = np.percentile(paths, 95, axis=0)
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=upper_bound,
                mode='lines',
                name='95th Percentile',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=lower_bound,
                mode='lines',
                name='90% Confidence Band',
                line=dict(width=0),
                fillcolor='rgba(68, 68, 68, 0.2)',
                fill='tonexty',
                hovertemplate='Time: %{x:.2f}<br>5th-95th: [%{y:.2f}, ' + 
                             f'{upper_bound:.2f}]<extra></extra>'
            )
        )
        
        # Add initial spot line
        if spot is not None:
            fig.add_hline(
                y=spot,
                line_dash="dash",
                line_color="yellow",
                opacity=0.5,
                annotation_text=f"S0={spot:.2f}"
            )
        
        fig.update_layout(
            title=title,
            xaxis_title='Time (years)',
            yaxis_title='Asset Price',
            template=self.theme,
            width=1000,
            height=600,
            hovermode='x unified'
        )
        
        if save_html:
            filename = f"{self.export_path}/price_paths.html"
            fig.write_html(filename)
            print(f"Saved price paths to {filename}")
        
        return fig
    
    def plot_path_with_jumps(self, times: np.ndarray, path: np.ndarray,
                            jump_times: List[float],
                            title: str = "Price Path with Jumps",
                            save_html: bool = True) -> go.Figure:
        """
        Plot single price path highlighting jump events
        
        For jump diffusion models (Merton, Kou)
        
        Args:
            times: Time grid
            path: Single price path
            jump_times: List of times when jumps occurred
            title: Plot title
            save_html: Save as HTML
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # Price path
        fig.add_trace(
            go.Scatter(
                x=times,
                y=path,
                mode='lines',
                name='Price Path',
                line=dict(color='cyan', width=2),
                hovertemplate='Time: %{x:.2f}<br>Price: %{y:.2f}<extra></extra>'
            )
        )
        
        # Mark jumps
        if len(jump_times) > 0:
            # Find prices at jump times
            jump_indices = [np.argmin(np.abs(times - t)) for t in jump_times]
            jump_prices = [path[idx] for idx in jump_indices]
            
            fig.add_trace(
                go.Scatter(
                    x=jump_times,
                    y=jump_prices,
                    mode='markers',
                    name='Jumps',
                    marker=dict(
                        color='red',
                        size=12,
                        symbol='star',
                        line=dict(color='white', width=2)
                    ),
                    hovertemplate='Jump at t=%{x:.2f}<br>Price: %{y:.2f}<extra></extra>'
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title='Time (years)',
            yaxis_title='Asset Price',
            template=self.theme,
            width=1000,
            height=600
        )
        
        if save_html:
            filename = f"{self.export_path}/path_with_jumps.html"
            fig.write_html(filename)
            print(f"Saved jump path to {filename}")
        
        return fig
    
    # ========== Greeks Visualization ==========
    
    def plot_greeks_heatmap(self, S_range: np.ndarray, sigma_range: np.ndarray,
                           greek_values: np.ndarray, greek_name: str,
                           K: float, T: float,
                           title: Optional[str] = None,
                           save_html: bool = True) -> go.Figure:
        """
        Plot Greek as heatmap across spot and volatility
        
        Visualizes how a Greek changes with spot price and volatility
        
        Args:
            S_range: Range of spot prices
            sigma_range: Range of volatilities
            greek_values: 2D array of Greek values [len(S_range) × len(sigma_range)]
            greek_name: Name of Greek (Delta, Gamma, etc.)
            K: Strike price (for reference)
            T: Time to maturity
            title: Custom title
            save_html: Save as HTML
        
        Returns:
            Plotly Figure
        """
        if title is None:
            title = f"{greek_name} Heatmap (K={K:.2f}, T={T:.2f}y)"
        
        fig = go.Figure(data=
            go.Heatmap(
                x=sigma_range * 100,  # Convert to percentage
                y=S_range,
                z=greek_values,
                colorscale='RdYlGn',
                colorbar=dict(title=greek_name),
                hovertemplate='Spot: %{y:.2f}<br>Vol: %{x:.2f}%<br>' + 
                             f'{greek_name}: ' + '%{z:.4f}<extra></extra>'
            )
        )
        
        # Mark ATM
        fig.add_hline(
            y=K,
            line_dash="dash",
            line_color="white",
            annotation_text=f"Strike K={K:.2f}"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Volatility (%)',
            yaxis_title='Spot Price',
            template=self.theme,
            width=800,
            height=600
        )
        
        if save_html:
            filename = f"{self.export_path}/greek_heatmap_{greek_name.lower()}.html"
            fig.write_html(filename)
            print(f"Saved {greek_name} heatmap to {filename}")
        
        return fig
    
    def plot_greeks_profile(self, S_range: np.ndarray, greeks_dict: Dict[str, np.ndarray],
                           K: float, title: str = "Greeks Profile",
                           save_html: bool = True) -> go.Figure:
        """
        Plot multiple Greeks across spot price range
        
        Shows Delta, Gamma, Vega, Theta profiles simultaneously
        
        Args:
            S_range: Range of spot prices
            greeks_dict: Dictionary mapping Greek name to values array
            K: Strike price
            title: Plot title
            save_html: Save as HTML
        
        Returns:
            Plotly Figure with subplots
        """
        n_greeks = len(greeks_dict)
        
        fig = make_subplots(
            rows=n_greeks, cols=1,
            subplot_titles=list(greeks_dict.keys()),
            vertical_spacing=0.08
        )
        
        colors = px.colors.qualitative.Plotly
        
        for idx, (greek_name, values) in enumerate(greeks_dict.items(), 1):
            fig.add_trace(
                go.Scatter(
                    x=S_range,
                    y=values,
                    mode='lines',
                    name=greek_name,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    hovertemplate=f'{greek_name}: ' + '%{y:.4f}<extra></extra>'
                ),
                row=idx, col=1
            )
            
            # Mark strike
            fig.add_vline(
                x=K,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                row=idx, col=1
            )
            
            # Update y-axis title
            fig.update_yaxes(title_text=greek_name, row=idx, col=1)
        
        # Update x-axis title for last subplot
        fig.update_xaxes(title_text="Spot Price", row=n_greeks, col=1)
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=200 * n_greeks,
            width=900,
            showlegend=False
        )
        
        if save_html:
            filename = f"{self.export_path}/greeks_profile.html"
            fig.write_html(filename)
            print(f"Saved Greeks profile to {filename}")
        
        return fig
    
    # ========== Calibration Diagnostics ==========
    
    def plot_calibration_fit(self, market_ivs: np.ndarray, model_ivs: np.ndarray,
                            strikes: np.ndarray, maturity: float,
                            title: Optional[str] = None,
                            save_html: bool = True) -> go.Figure:
        """
        Compare market vs model implied volatilities
        
        Diagnostic plot for calibration quality
        
        Args:
            market_ivs: Market implied volatilities
            model_ivs: Model-generated implied volatilities
            strikes: Strike prices
            maturity: Time to maturity
            title: Custom title
            save_html: Save as HTML
        
        Returns:
            Plotly Figure
        """
        if title is None:
            title = f"Calibration Fit (T={maturity:.2f}y)"
        
        fig = go.Figure()
        
        # Market IVs
        fig.add_trace(
            go.Scatter(
                x=strikes,
                y=market_ivs * 100,
                mode='markers',
                name='Market',
                marker=dict(size=10, color='cyan', symbol='circle'),
                hovertemplate='Strike: %{x:.2f}<br>Market IV: %{y:.2f}%<extra></extra>'
            )
        )
        
        # Model IVs
        fig.add_trace(
            go.Scatter(
                x=strikes,
                y=model_ivs * 100,
                mode='lines+markers',
                name='Model',
                line=dict(color='orange', width=2),
                marker=dict(size=6),
                hovertemplate='Strike: %{x:.2f}<br>Model IV: %{y:.2f}%<extra></extra>'
            )
        )
        
        # Error bars
        errors = (model_ivs - market_ivs) * 100
        fig.add_trace(
            go.Bar(
                x=strikes,
                y=errors,
                name='Error',
                marker=dict(color='red'),
                opacity=0.3,
                yaxis='y2',
                hovertemplate='Strike: %{x:.2f}<br>Error: %{y:.2f}%<extra></extra>'
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Strike Price',
            yaxis=dict(title='Implied Volatility (%)'),
            yaxis2=dict(
                title='Calibration Error (%)',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            template=self.theme,
            width=1000,
            height=600,
            hovermode='x unified'
        )
        
        if save_html:
            filename = f"{self.export_path}/calibration_fit_T{maturity:.2f}.html"
            fig.write_html(filename)
            print(f"Saved calibration fit to {filename}")
        
        return fig
    
    def plot_calibration_error_surface(self, strikes: np.ndarray, 
                                      maturities: np.ndarray,
                                      error_surface: np.ndarray,
                                      title: str = "Calibration Error Surface",
                                      save_html: bool = True) -> go.Figure:
        """
        3D surface of calibration errors across strikes and maturities
        
        Args:
            strikes: Strike prices
            maturities: Maturities
            error_surface: 2D array of errors [maturities × strikes]
            title: Plot title
            save_html: Save as HTML
        
        Returns:
            Plotly Figure
        """
        K_mesh, T_mesh = np.meshgrid(strikes, maturities)
        
        fig = go.Figure(data=[
            go.Surface(
                x=K_mesh,
                y=T_mesh,
                z=error_surface * 100,  # Percentage
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Error (%)"),
                hovertemplate='Strike: %{x:.2f}<br>Maturity: %{y:.2f}y<br>Error: %{z:.2f}%<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Strike',
                yaxis_title='Maturity (years)',
                zaxis_title='Error (%)'
            ),
            template=self.theme,
            width=1000,
            height=700
        )
        
        if save_html:
            filename = f"{self.export_path}/calibration_error_surface.html"
            fig.write_html(filename)
            print(f"Saved calibration error surface to {filename}")
        
        return fig
    
    # ========== Model Comparison ==========
    
    def compare_models(self, strikes: np.ndarray,
                      model_prices: Dict[str, np.ndarray],
                      market_prices: Optional[np.ndarray] = None,
                      maturity: float = 1.0,
                      title: str = "Model Comparison",
                      save_html: bool = True) -> go.Figure:
        """
        Compare option prices across different models
        
        Args:
            strikes: Strike prices
            model_prices: Dictionary mapping model name to price array
            market_prices: Optional market prices for reference
            maturity: Time to maturity
            title: Plot title
            save_html: Save as HTML
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # Market prices (if provided)
        if market_prices is not None:
            fig.add_trace(
                go.Scatter(
                    x=strikes,
                    y=market_prices,
                    mode='markers',
                    name='Market',
                    marker=dict(size=10, color='black', symbol='circle'),
                    hovertemplate='Strike: %{x:.2f}<br>Market: %{y:.2f}<extra></extra>'
                )
            )
        
        # Model prices
        colors = px.colors.qualitative.Set1
        for idx, (model_name, prices) in enumerate(model_prices.items()):
            fig.add_trace(
                go.Scatter(
                    x=strikes,
                    y=prices,
                    mode='lines+markers',
                    name=model_name,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    hovertemplate=f'{model_name}: ' + '%{y:.2f}<extra></extra>'
                )
            )
        
        fig.update_layout(
            title=f"{title} (T={maturity:.2f}y)",
            xaxis_title='Strike Price',
            yaxis_title='Option Price',
            template=self.theme,
            width=1000,
            height=600,
            hovermode='x unified'
        )
        
        if save_html:
            filename = f"{self.export_path}/model_comparison.html"
            fig.write_html(filename)
            print(f"Saved model comparison to {filename}")
        
        return fig
    
    # ========== Distribution Analysis ==========
    
    def plot_return_distribution(self, returns: np.ndarray,
                                title: str = "Return Distribution",
                                save_html: bool = True) -> go.Figure:
        """
        Plot histogram of returns with fitted normal distribution
        
        Useful for assessing model fit to empirical data
        
        Args:
            returns: Array of log-returns
            title: Plot title
            save_html: Save as HTML
        
        Returns:
            Plotly Figure
        """
        from scipy import stats
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns * 100,
                nbinsx=50,
                name='Empirical',
                histnorm='probability density',
                marker=dict(color='cyan', opacity=0.7),
                hovertemplate='Return: %{x:.2f}%<br>Density: %{y:.4f}<extra></extra>'
            )
        )
        
        # Fitted normal distribution
        mu = np.mean(returns)
        sigma = np.std(returns)
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_pdf = stats.norm.pdf(x_range, mu, sigma)
        
        fig.add_trace(
            go.Scatter(
                x=x_range * 100,
                y=normal_pdf / 100,  # Scale for percentage
                mode='lines',
                name='Normal Fit',
                line=dict(color='red', width=2),
                hovertemplate='Return: %{x:.2f}%<br>Normal PDF: %{y:.4f}<extra></extra>'
            )
        )
        
        # Add statistics annotation
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        fig.add_annotation(
            text=f"μ = {mu*100:.2f}%<br>σ = {sigma*100:.2f}%<br>" +
                 f"Skew = {skew:.2f}<br>Kurt = {kurt:.2f}",
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            xanchor='right', yanchor='top',
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Log-Return (%)',
            yaxis_title='Probability Density',
            template=self.theme,
            width=900,
            height=600,
            showlegend=True
        )
        
        if save_html:
            filename = f"{self.export_path}/return_distribution.html"
            fig.write_html(filename)
            print(f"Saved return distribution to {filename}")
        
        return fig
    
    # ========== Export All Plots ==========
    
    def create_dashboard_html(self, figures: Dict[str, go.Figure],
                             filename: str = "dashboard.html") -> None:
        """
        Combine multiple plots into single HTML dashboard
        
        Perfect for GitHub Pages or Vercel deployment
        
        Args:
            figures: Dictionary mapping plot name to Plotly Figure
            filename: Output HTML filename
        """
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Volatility Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #0e1117;
            color: white;
        }
        h1 {
            text-align: center;
            color: #00d4ff;
        }
        .plot-container {
            margin: 30px 0;
            padding: 20px;
            background-color: #1a1d24;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .plot-title {
            font-size: 24px;
            margin-bottom: 15px;
            color: #00d4ff;
        }
    </style>
</head>
<body>
    <h1>📊 Volatility Models Analysis Dashboard</h1>
"""
        
        for plot_name, fig in figures.items():
            plot_div = fig.to_html(full_html=False, include_plotlyjs=False)
            html_content += f"""
    <div class="plot-container">
        <div class="plot-title">{plot_name}</div>
        {plot_div}
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        full_path = f"{self.export_path}/{filename}"
        with open(full_path, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Created dashboard: {full_path}")
        print(f"   Open in browser or deploy to Vercel/GitHub Pages!")

# ========== Example Usage ==========

def example_visualization():
    """
    Demonstration of all visualization capabilities
    """
    viz = VolatilityVisualizer(theme='plotly_dark')
    
    # 1. IV Surface
    strikes = np.linspace(80, 120, 50)
    maturities = np.array([0.25, 0.5, 1.0, 2.0])
    iv_surface = np.random.rand(len(maturities), len(strikes)) * 0.2 + 0.2
    
    fig1 = viz.plot_iv_surface_3d(strikes, maturities, iv_surface, spot=100)
    
    # 2. Volatility Smile
    ivs = 0.2 + 0.05 * ((strikes - 100) / 10)**2
    fig2 = viz.plot_volatility_smile(strikes, ivs, maturity=1.0, spot=100)
    
    # 3. Price Paths
    times = np.linspace(0, 1, 252)
    paths = 100 * np.exp(np.cumsum(np.random.randn(100, 252) * 0.2 / np.sqrt(252), axis=1))
    fig3 = viz.plot_price_paths(times, paths, spot=100)
    
    # 4. Create Dashboard
    dashboard_figs = {
        "Implied Volatility Surface": fig1,
        "Volatility Smile": fig2,
        "Simulated Price Paths": fig3
    }
    
    viz.create_dashboard_html(dashboard_figs)
    
    print("\n✅ All visualizations created!")
    print(f"📁 Files saved to: {viz.export_path}")
    print("🌐 Ready for web deployment (Vercel/GitHub Pages)")

if __name__ == "__main__":
    example_visualization()
