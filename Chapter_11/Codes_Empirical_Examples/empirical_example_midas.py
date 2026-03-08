"""
MIDAS (Mixed Data Sampling) Estimation in Python
=================================================
Author: Alessia Paccagnini
Textbook: Macroeconometrics
Date: January 2026

This script demonstrates MIDAS regression for nowcasting/forecasting
quarterly GDP growth using monthly Federal Funds Rate data.

Key References:
- Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). "The MIDAS Touch"
- Ghysels, E., & Marcellino, M. (2018). "Applied Economic Forecasting Using Time Series Methods"
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gamma
import matplotlib.pyplot as plt
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================

def detect_environment():
    """Detect whether running in Google Colab, local Jupyter, or plain script."""
    try:
        import google.colab
        return 'colab'
    except ImportError:
        pass
    if 'ipykernel' in sys.modules:
        return 'jupyter'
    return 'script'

ENV = detect_environment()

def get_output_dir():
    """Return appropriate output directory for the current environment."""
    if ENV == 'colab':
        return '/content/'
    elif ENV == 'jupyter':
        return './'          # current working directory
    else:
        return '/home/claude/'

OUTPUT_DIR = get_output_dir()

def setup_colab_files():
    """
    If running in Colab, prompt the user to upload the three required Excel files:
      - FEDFUNDS.xlsx  (sheet: Monthly)
      - GDPC1.xlsx     (sheet: Quarterly)
      - GDPDEF.xlsx    (sheet: Quarterly)
    Returns a dict mapping filename -> upload path.
    """
    from google.colab import files
    print("Please upload the three required Excel files:")
    print("  1. FEDFUNDS.xlsx  (monthly Fed Funds Rate)")
    print("  2. GDPC1.xlsx     (quarterly real GDP)")
    print("  3. GDPDEF.xlsx    (quarterly GDP deflator)")
    uploaded = files.upload()   # opens Colab's file picker
    # Files land in /content/ by default
    paths = {name: f'/content/{name}' for name in uploaded.keys()}
    print(f"\nUploaded: {list(paths.keys())}")
    return paths

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data(file_paths=None):
    """
    Load FRED data and prepare for MIDAS estimation.
    Quarterly GDP (low frequency) regressed on monthly Fed Funds (high frequency).

    Parameters
    ----------
    file_paths : dict, optional
        Mapping like {'FEDFUNDS.xlsx': <path>, 'GDPC1.xlsx': <path>, 'GDPDEF.xlsx': <path>}.
        If None, paths are inferred from the detected environment:
          - Colab  : expects files uploaded via setup_colab_files() to /content/
          - Jupyter: looks in the current working directory
          - Script : looks in /mnt/user-data/uploads/ (Claude/JupyterHub layout)
    """
    # --- Resolve paths ---
    if file_paths is not None:
        path_fedfunds = file_paths.get('FEDFUNDS.xlsx', file_paths.get('fedfunds'))
        path_gdp      = file_paths.get('GDPC1.xlsx',   file_paths.get('gdp'))
        path_gdpdef   = file_paths.get('GDPDEF.xlsx',  file_paths.get('gdpdef'))
    elif ENV == 'colab':
        path_fedfunds = '/content/FEDFUNDS.xlsx'
        path_gdp      = '/content/GDPC1.xlsx'
        path_gdpdef   = '/content/GDPDEF.xlsx'
    elif ENV == 'jupyter':
        path_fedfunds = 'FEDFUNDS.xlsx'
        path_gdp      = 'GDPC1.xlsx'
        path_gdpdef   = 'GDPDEF.xlsx'
    else:
        path_fedfunds = '/mnt/user-data/uploads/FEDFUNDS.xlsx'
        path_gdp      = '/mnt/user-data/uploads/GDPC1.xlsx'
        path_gdpdef   = '/mnt/user-data/uploads/GDPDEF.xlsx'

    # Load data
    fedfunds = pd.read_excel(path_fedfunds, sheet_name='Monthly')
    gdp      = pd.read_excel(path_gdp,      sheet_name='Quarterly')
    gdpdef   = pd.read_excel(path_gdpdef,   sheet_name='Quarterly')
    
    # Rename columns
    fedfunds.columns = ['date', 'fedfunds']
    gdp.columns = ['date', 'gdp']
    gdpdef.columns = ['date', 'gdpdef']
    
    # Convert to datetime
    fedfunds['date'] = pd.to_datetime(fedfunds['date'])
    gdp['date'] = pd.to_datetime(gdp['date'])
    gdpdef['date'] = pd.to_datetime(gdpdef['date'])
    
    # Compute quarterly GDP growth (annualized)
    gdp = gdp.sort_values('date').reset_index(drop=True)
    gdp['gdp_growth'] = 400 * np.log(gdp['gdp'] / gdp['gdp'].shift(1))
    
    # Compute inflation from GDP deflator (annualized)
    gdpdef = gdpdef.sort_values('date').reset_index(drop=True)
    gdpdef['inflation'] = 400 * np.log(gdpdef['gdpdef'] / gdpdef['gdpdef'].shift(1))
    
    # Merge GDP and deflator
    quarterly = pd.merge(gdp, gdpdef[['date', 'inflation']], on='date', how='inner')
    
    return fedfunds, quarterly


def align_mixed_frequency(quarterly_df, monthly_df, m=3, K=4):
    """
    Align quarterly and monthly data for MIDAS regression.
    
    Parameters:
    -----------
    quarterly_df : DataFrame with quarterly data (low frequency target)
    monthly_df : DataFrame with monthly data (high frequency predictor)
    m : int, number of high-frequency observations per low-frequency period (3 for monthly-quarterly)
    K : int, number of quarterly lags to include (each with m monthly observations)
    
    Returns:
    --------
    y : array, quarterly GDP growth
    X_hf : array, shape (T, m*K) high-frequency lags matrix
    dates : quarterly dates
    """
    quarterly_df = quarterly_df.sort_values('date').reset_index(drop=True)
    monthly_df = monthly_df.sort_values('date').reset_index(drop=True)
    
    # Create quarter identifier for monthly data
    monthly_df['quarter'] = monthly_df['date'].dt.to_period('Q').dt.to_timestamp()
    
    y_list = []
    X_hf_list = []
    dates_list = []
    
    for idx in range(K, len(quarterly_df)):
        q_date = quarterly_df.loc[idx, 'date']
        y_val = quarterly_df.loc[idx, 'gdp_growth']
        
        if pd.isna(y_val):
            continue
        
        # Get monthly observations for current and lagged quarters
        # For nowcasting: use months within the quarter
        # For forecasting with lags: use months from previous quarters
        
        hf_values = []
        for lag in range(K):
            # Get the quarter date for this lag
            lag_q_date = quarterly_df.loc[idx - lag, 'date']
            
            # Get monthly values for this quarter (3 months)
            q_monthly = monthly_df[monthly_df['quarter'] == lag_q_date].sort_values('date')
            
            if len(q_monthly) < m:
                break
            
            # Take the m monthly observations (most recent first within quarter)
            monthly_vals = q_monthly['fedfunds'].values[-m:][::-1]  # Reverse for most recent first
            hf_values.extend(monthly_vals)
        
        if len(hf_values) == m * K:
            y_list.append(y_val)
            X_hf_list.append(hf_values)
            dates_list.append(q_date)
    
    y = np.array(y_list)
    X_hf = np.array(X_hf_list)
    dates = pd.to_datetime(dates_list)
    
    return y, X_hf, dates


# =============================================================================
# 2. MIDAS WEIGHTING SCHEMES
# =============================================================================

def exponential_almon_weights(K, theta1, theta2):
    """
    Exponential Almon lag polynomial weights.
    
    w(k; θ) = exp(θ₁k + θ₂k²) / Σⱼ exp(θ₁j + θ₂j²)
    
    Parameters:
    -----------
    K : int, number of lags
    theta1, theta2 : float, polynomial parameters
    
    Returns:
    --------
    weights : array of normalized weights
    """
    k = np.arange(1, K + 1)
    weights = np.exp(theta1 * k + theta2 * k**2)
    weights = weights / np.sum(weights)
    return weights


def beta_weights(K, theta1, theta2, eps=1e-6):
    """
    Beta polynomial weights (Ghysels et al., 2007).
    
    w(k; θ₁, θ₂) ∝ (k/K)^(θ₁-1) * (1 - k/K)^(θ₂-1)
    
    Parameters:
    -----------
    K : int, number of lags
    theta1, theta2 : float, shape parameters (must be > 0)
    eps : float, small constant for numerical stability
    
    Returns:
    --------
    weights : array of normalized weights
    """
    # Ensure positive parameters
    theta1 = max(theta1, eps)
    theta2 = max(theta2, eps)
    
    k = np.arange(1, K + 1)
    x = k / (K + 1)  # Normalize to (0, 1)
    
    # Beta density (unnormalized)
    weights = x**(theta1 - 1) * (1 - x)**(theta2 - 1)
    
    # Handle numerical issues
    weights = np.maximum(weights, eps)
    weights = weights / np.sum(weights)
    
    return weights


def pdl_weights(K, degree=2):
    """
    Polynomial Distributed Lag (PDL) / Almon weights.
    Returns the polynomial basis for unrestricted estimation.
    
    Parameters:
    -----------
    K : int, number of lags
    degree : int, polynomial degree
    
    Returns:
    --------
    P : array, shape (K, degree+1), polynomial basis matrix
    """
    k = np.arange(1, K + 1)
    P = np.column_stack([k**d for d in range(degree + 1)])
    return P


def step_function_weights(K, n_steps=3):
    """
    Step function weights (piecewise constant).
    
    Parameters:
    -----------
    K : int, number of lags
    n_steps : int, number of steps
    
    Returns:
    --------
    step_matrix : array, shape (K, n_steps), step function basis
    """
    step_size = K // n_steps
    step_matrix = np.zeros((K, n_steps))
    
    for s in range(n_steps):
        start = s * step_size
        end = (s + 1) * step_size if s < n_steps - 1 else K
        step_matrix[start:end, s] = 1
    
    return step_matrix


# =============================================================================
# 3. MIDAS REGRESSION MODELS
# =============================================================================

class MIDASRegression:
    """
    MIDAS Regression with various weighting schemes.
    
    Model: y_t = α + β * Σₖ w(k;θ) * x_{t-k/m} + ε_t
    
    where:
    - y_t is the low-frequency variable (quarterly GDP growth)
    - x_{t-k/m} are high-frequency observations (monthly Fed Funds)
    - w(k;θ) are the MIDAS weights
    """
    
    def __init__(self, weight_type='beta'):
        """
        Initialize MIDAS regression.
        
        Parameters:
        -----------
        weight_type : str, one of 'beta', 'exp_almon', 'unrestricted'
        """
        self.weight_type = weight_type
        self.params = None
        self.weights = None
        self.fitted_values = None
        self.residuals = None
        self.results = {}
        
    def _get_weights(self, K, theta):
        """Get weights based on weight type and parameters."""
        if self.weight_type == 'beta':
            return beta_weights(K, theta[0], theta[1])
        elif self.weight_type == 'exp_almon':
            return exponential_almon_weights(K, theta[0], theta[1])
        else:
            raise ValueError(f"Unknown weight type: {self.weight_type}")
    
    def _objective(self, params, y, X_hf):
        """
        Objective function (sum of squared residuals).
        
        params = [alpha, beta, theta1, theta2]
        """
        alpha, beta = params[0], params[1]
        theta = params[2:]
        
        K = X_hf.shape[1]
        weights = self._get_weights(K, theta)
        
        # Weighted sum of high-frequency variable
        X_weighted = X_hf @ weights
        
        # Fitted values
        y_hat = alpha + beta * X_weighted
        
        # Sum of squared residuals
        ssr = np.sum((y - y_hat)**2)
        
        return ssr
    
    def fit(self, y, X_hf, theta_init=None):
        """
        Fit MIDAS regression via nonlinear least squares.
        
        Parameters:
        -----------
        y : array, low-frequency dependent variable
        X_hf : array, high-frequency regressors matrix
        theta_init : array, initial values for theta parameters
        
        Returns:
        --------
        self
        """
        T, K = X_hf.shape
        
        # Initial parameter values
        if theta_init is None:
            if self.weight_type == 'beta':
                theta_init = [1.0, 1.0]  # Uniform weights
            else:
                theta_init = [0.0, 0.0]
        
        # Simple OLS for initial alpha, beta (using equal weights)
        X_equal = X_hf.mean(axis=1)
        X_ols = np.column_stack([np.ones(T), X_equal])
        beta_ols = np.linalg.lstsq(X_ols, y, rcond=None)[0]
        
        # Full initial parameters
        params_init = np.concatenate([beta_ols, theta_init])
        
        # Bounds for beta weights (must be positive)
        if self.weight_type == 'beta':
            bounds = [(None, None), (None, None), (0.01, 10), (0.01, 10)]
        else:
            bounds = [(None, None), (None, None), (-5, 5), (-5, 5)]
        
        # Optimize
        result = minimize(
            self._objective,
            params_init,
            args=(y, X_hf),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'disp': False}
        )
        
        self.params = result.x
        self.weights = self._get_weights(K, result.x[2:])
        
        # Compute fitted values and residuals
        X_weighted = X_hf @ self.weights
        self.fitted_values = self.params[0] + self.params[1] * X_weighted
        self.residuals = y - self.fitted_values
        
        # Compute standard errors (numerical Hessian approximation)
        self._compute_inference(y, X_hf)
        
        # Store results
        self.results = {
            'alpha': self.params[0],
            'beta': self.params[1],
            'theta': self.params[2:],
            'weights': self.weights,
            'ssr': result.fun,
            'converged': result.success,
            'T': T,
            'K': K
        }
        
        return self
    
    def _compute_inference(self, y, X_hf, eps=1e-5):
        """Compute standard errors via numerical Hessian."""
        T = len(y)
        k_params = len(self.params)
        
        # Numerical Hessian
        H = np.zeros((k_params, k_params))
        f0 = self._objective(self.params, y, X_hf)
        
        for i in range(k_params):
            for j in range(k_params):
                params_pp = self.params.copy()
                params_pm = self.params.copy()
                params_mp = self.params.copy()
                params_mm = self.params.copy()
                
                params_pp[i] += eps
                params_pp[j] += eps
                params_pm[i] += eps
                params_pm[j] -= eps
                params_mp[i] -= eps
                params_mp[j] += eps
                params_mm[i] -= eps
                params_mm[j] -= eps
                
                f_pp = self._objective(params_pp, y, X_hf)
                f_pm = self._objective(params_pm, y, X_hf)
                f_mp = self._objective(params_mp, y, X_hf)
                f_mm = self._objective(params_mm, y, X_hf)
                
                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
        
        # Variance-covariance matrix
        sigma2 = np.sum(self.residuals**2) / (T - k_params)
        try:
            H_inv = np.linalg.inv(H)
            self.vcov = 2 * sigma2 * H_inv
            self.std_errors = np.sqrt(np.diag(self.vcov))
        except:
            self.vcov = None
            self.std_errors = np.full(k_params, np.nan)
        
        # R-squared
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum(self.residuals**2)
        self.r_squared = 1 - ss_res / ss_tot
        self.adj_r_squared = 1 - (1 - self.r_squared) * (T - 1) / (T - k_params)
        self.sigma = np.sqrt(sigma2)
        self.aic = T * np.log(ss_res / T) + 2 * k_params
        self.bic = T * np.log(ss_res / T) + k_params * np.log(T)
    
    def predict(self, X_hf_new):
        """Generate predictions for new high-frequency data."""
        X_weighted = X_hf_new @ self.weights
        return self.params[0] + self.params[1] * X_weighted
    
    def summary(self):
        """Print estimation summary."""
        print("\n" + "="*70)
        print("MIDAS REGRESSION RESULTS")
        print("="*70)
        print(f"Weight function: {self.weight_type.upper()}")
        print(f"Number of observations: {self.results['T']}")
        print(f"Number of HF lags: {self.results['K']}")
        print("-"*70)
        print("\nParameter Estimates:")
        print("-"*70)
        print(f"{'Parameter':<15} {'Estimate':>12} {'Std.Err':>12} {'t-stat':>12}")
        print("-"*70)
        
        param_names = ['alpha', 'beta'] + [f'theta{i+1}' for i in range(len(self.params)-2)]
        for i, name in enumerate(param_names):
            est = self.params[i]
            se = self.std_errors[i] if not np.isnan(self.std_errors[i]) else np.nan
            t_stat = est / se if not np.isnan(se) and se > 0 else np.nan
            print(f"{name:<15} {est:>12.4f} {se:>12.4f} {t_stat:>12.2f}")
        
        print("-"*70)
        print("\nModel Fit:")
        print(f"  R-squared:          {self.r_squared:.4f}")
        print(f"  Adj. R-squared:     {self.adj_r_squared:.4f}")
        print(f"  Residual std. err:  {self.sigma:.4f}")
        print(f"  AIC:                {self.aic:.2f}")
        print(f"  BIC:                {self.bic:.2f}")
        print(f"  SSR:                {self.results['ssr']:.4f}")
        print("="*70)


class UnrestrictedMIDAS:
    """
    Unrestricted MIDAS (U-MIDAS) regression.
    
    Directly estimates coefficients for each high-frequency lag
    without parametric weight restrictions.
    
    Model: y_t = α + Σₖ βₖ * x_{t-k/m} + ε_t
    """
    
    def __init__(self):
        self.params = None
        self.fitted_values = None
        self.residuals = None
        
    def fit(self, y, X_hf):
        """Fit unrestricted MIDAS via OLS."""
        T, K = X_hf.shape
        
        # Add constant
        X = np.column_stack([np.ones(T), X_hf])
        
        # OLS estimation
        self.params = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Fitted values and residuals
        self.fitted_values = X @ self.params
        self.residuals = y - self.fitted_values
        
        # Inference
        sigma2 = np.sum(self.residuals**2) / (T - K - 1)
        XtX_inv = np.linalg.inv(X.T @ X)
        self.vcov = sigma2 * XtX_inv
        self.std_errors = np.sqrt(np.diag(self.vcov))
        
        # R-squared
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum(self.residuals**2)
        self.r_squared = 1 - ss_res / ss_tot
        self.adj_r_squared = 1 - (1 - self.r_squared) * (T - 1) / (T - K - 1)
        self.sigma = np.sqrt(sigma2)
        self.aic = T * np.log(ss_res / T) + 2 * (K + 1)
        self.bic = T * np.log(ss_res / T) + (K + 1) * np.log(T)
        
        self.T = T
        self.K = K
        
        return self
    
    def summary(self):
        """Print estimation summary."""
        print("\n" + "="*70)
        print("UNRESTRICTED MIDAS (U-MIDAS) REGRESSION RESULTS")
        print("="*70)
        print(f"Number of observations: {self.T}")
        print(f"Number of HF lags: {self.K}")
        print("-"*70)
        print("\nParameter Estimates (first 5 and last 5 lags):")
        print("-"*70)
        print(f"{'Parameter':<15} {'Estimate':>12} {'Std.Err':>12} {'t-stat':>12}")
        print("-"*70)
        
        # Constant
        print(f"{'alpha':<15} {self.params[0]:>12.4f} {self.std_errors[0]:>12.4f} {self.params[0]/self.std_errors[0]:>12.2f}")
        
        # First 5 lag coefficients
        for i in range(1, min(6, len(self.params))):
            name = f'beta_lag{i}'
            est = self.params[i]
            se = self.std_errors[i]
            t_stat = est / se
            print(f"{name:<15} {est:>12.4f} {se:>12.4f} {t_stat:>12.2f}")
        
        if len(self.params) > 10:
            print("  ...")
            # Last 5 lag coefficients
            for i in range(len(self.params)-5, len(self.params)):
                name = f'beta_lag{i}'
                est = self.params[i]
                se = self.std_errors[i]
                t_stat = est / se
                print(f"{name:<15} {est:>12.4f} {se:>12.4f} {t_stat:>12.2f}")
        
        print("-"*70)
        print("\nModel Fit:")
        print(f"  R-squared:          {self.r_squared:.4f}")
        print(f"  Adj. R-squared:     {self.adj_r_squared:.4f}")
        print(f"  Residual std. err:  {self.sigma:.4f}")
        print(f"  AIC:                {self.aic:.2f}")
        print(f"  BIC:                {self.bic:.2f}")
        print("="*70)


# =============================================================================
# 4. VISUALIZATION
# =============================================================================

def plot_midas_weights(model, title="MIDAS Weights"):
    """Plot the estimated MIDAS weights."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    K = len(model.weights)
    lags = np.arange(1, K + 1)
    
    ax.bar(lags, model.weights, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Lag (months)', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(lags[::2] if K > 6 else lags)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_fitted_vs_actual(y, fitted, dates, title="MIDAS Fitted vs Actual"):
    """Plot fitted values against actual values."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time series plot
    ax1 = axes[0]
    ax1.plot(dates, y, 'b-', label='Actual GDP Growth', linewidth=1.5)
    ax1.plot(dates, fitted, 'r--', label='MIDAS Fitted', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('GDP Growth (%)', fontsize=11)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    
    # Scatter plot
    ax2 = axes[1]
    ax2.scatter(fitted, y, alpha=0.5, edgecolors='black', linewidth=0.5)
    min_val = min(y.min(), fitted.min())
    max_val = max(y.max(), fitted.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='45° line')
    ax2.set_xlabel('Fitted Values', fontsize=11)
    ax2.set_ylabel('Actual Values', fontsize=11)
    ax2.set_title('Fitted vs Actual Scatter', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_weight_functions(K=12):
    """Compare different MIDAS weighting schemes."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    lags = np.arange(1, K + 1)
    
    # Beta weights with different parameters
    ax1 = axes[0, 0]
    for (t1, t2), label in [((1, 1), 'Uniform (1,1)'), 
                             ((1, 5), 'Declining (1,5)'),
                             ((2, 2), 'Hump (2,2)'),
                             ((5, 1), 'Increasing (5,1)')]:
        w = beta_weights(K, t1, t2)
        ax1.plot(lags, w, 'o-', label=label, markersize=4)
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Weight')
    ax1.set_title('Beta Polynomial Weights')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Exponential Almon weights
    ax2 = axes[0, 1]
    for (t1, t2), label in [((0, 0), 'Uniform (0,0)'),
                             ((-0.1, 0), 'Declining (-0.1,0)'),
                             ((0.1, -0.02), 'Hump (0.1,-0.02)'),
                             ((0.1, 0), 'Increasing (0.1,0)')]:
        w = exponential_almon_weights(K, t1, t2)
        ax2.plot(lags, w, 'o-', label=label, markersize=4)
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Weight')
    ax2.set_title('Exponential Almon Weights')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # Step function weights
    ax3 = axes[1, 0]
    for n_steps, color in [(2, 'blue'), (3, 'green'), (4, 'red')]:
        step_mat = step_function_weights(K, n_steps)
        # Assume equal step coefficients for visualization
        w = step_mat @ np.ones(n_steps) / n_steps
        ax3.step(lags, w, where='mid', label=f'{n_steps} steps', linewidth=2)
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('Weight')
    ax3.set_title('Step Function Weights')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # PDL weights
    ax4 = axes[1, 1]
    P2 = pdl_weights(K, degree=2)
    P3 = pdl_weights(K, degree=3)
    ax4.plot(lags, P2[:, 0], 'o-', label='Constant', markersize=4)
    ax4.plot(lags, P2[:, 1]/K, 'o-', label='Linear (scaled)', markersize=4)
    ax4.plot(lags, P2[:, 2]/K**2, 'o-', label='Quadratic (scaled)', markersize=4)
    ax4.set_xlabel('Lag')
    ax4.set_ylabel('Basis Value (scaled)')
    ax4.set_title('PDL Polynomial Basis')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.suptitle('Comparison of MIDAS Weighting Schemes', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

# Main execution (runs in both script and notebook contexts)

print("\n" + "="*70)
print("MIDAS ESTIMATION: Nowcasting GDP Growth with Fed Funds Rate")
print("="*70)

# Load data
print("\n>>> Loading and preparing data...")

def _ask_upload_or_reuse_midas():
    """Ask whether to upload files or reuse data already in the environment."""
    already_loaded = True
    for _v in ['fedfunds', 'quarterly']:
        if _v not in dir():
            already_loaded = False
            break
    # Check globals too (Jupyter keeps variables in global scope)
    import builtins
    _globals = vars(builtins)
    try:
        _globals = globals()
    except NameError:
        pass
    already_loaded = ('fedfunds' in _globals and 'quarterly' in _globals)

    if already_loaded:
        ans = input("Data already found in environment. Use existing data? [y/n] (default: y): ").strip().lower()
        if ans in ('', 'y', 'yes'):
            print("  Reusing existing fedfunds and quarterly variables.")
            return None   # signal: reuse
    return 'upload'       # signal: upload/load fresh

_load_signal = _ask_upload_or_reuse_midas()

if _load_signal is None:
    pass  # reuse existing fedfunds, quarterly
elif ENV == 'colab':
    print("Environment: Google Colab — uploading files now...")
    _uploaded = setup_colab_files()
    fedfunds, quarterly = load_and_prepare_data(file_paths=_uploaded)
else:
    print(f"Environment: {ENV} | Output dir: {OUTPUT_DIR}")
    fedfunds, quarterly = load_and_prepare_data()


print(f"Monthly Fed Funds: {len(fedfunds)} observations")
print(f"  Range: {fedfunds['date'].min().strftime('%Y-%m')} to {fedfunds['date'].max().strftime('%Y-%m')}")
print(f"Quarterly GDP: {len(quarterly)} observations")
print(f"  Range: {quarterly['date'].min().strftime('%Y-Q%q')} to {quarterly['date'].max().strftime('%Y-Q%q')}")

# Align data for MIDAS
# m=3 (3 months per quarter), K=12 (12 monthly lags = 4 quarters of history)
print("\n>>> Aligning mixed-frequency data...")
y, X_hf, dates = align_mixed_frequency(quarterly, fedfunds, m=3, K=12)
print(f"Aligned sample: {len(y)} quarterly observations")
print(f"High-frequency matrix: {X_hf.shape[0]} x {X_hf.shape[1]}")
print(f"  Sample period: {dates[0].strftime('%Y-Q%q')} to {dates[-1].strftime('%Y-Q%q')}")

# =========================================================================
# Estimate MIDAS with Beta weights
# =========================================================================
print("\n>>> Estimating MIDAS with Beta polynomial weights...")
midas_beta = MIDASRegression(weight_type='beta')
midas_beta.fit(y, X_hf, theta_init=[1.0, 3.0])
midas_beta.summary()

# =========================================================================
# Estimate MIDAS with Exponential Almon weights
# =========================================================================
print("\n>>> Estimating MIDAS with Exponential Almon weights...")
midas_almon = MIDASRegression(weight_type='exp_almon')
midas_almon.fit(y, X_hf, theta_init=[-0.05, -0.01])
midas_almon.summary()

# =========================================================================
# Estimate Unrestricted MIDAS (U-MIDAS)
# =========================================================================
print("\n>>> Estimating Unrestricted MIDAS (U-MIDAS)...")
umidas = UnrestrictedMIDAS()
umidas.fit(y, X_hf)
umidas.summary()

# =========================================================================
# Model Comparison
# =========================================================================
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(f"{'Model':<25} {'R²':>10} {'Adj.R²':>10} {'AIC':>12} {'BIC':>12}")
print("-"*70)
print(f"{'MIDAS (Beta)':<25} {midas_beta.r_squared:>10.4f} {midas_beta.adj_r_squared:>10.4f} {midas_beta.aic:>12.2f} {midas_beta.bic:>12.2f}")
print(f"{'MIDAS (Exp. Almon)':<25} {midas_almon.r_squared:>10.4f} {midas_almon.adj_r_squared:>10.4f} {midas_almon.aic:>12.2f} {midas_almon.bic:>12.2f}")
print(f"{'U-MIDAS (Unrestricted)':<25} {umidas.r_squared:>10.4f} {umidas.adj_r_squared:>10.4f} {umidas.aic:>12.2f} {umidas.bic:>12.2f}")
print("="*70)

# =========================================================================
# Generate Plots
# =========================================================================
print("\n>>> Generating visualizations...")

# Plot 1: Compare weight functions
fig1 = compare_weight_functions(K=12)
fig1.savefig(os.path.join(OUTPUT_DIR, 'midas_weight_comparison.png'), dpi=150, bbox_inches='tight')
print("  Saved: midas_weight_comparison.png")

# Plot 2: Estimated Beta weights
fig2 = plot_midas_weights(midas_beta, "Estimated Beta Polynomial Weights")
fig2.savefig(os.path.join(OUTPUT_DIR, 'midas_beta_weights.png'), dpi=150, bbox_inches='tight')
print("  Saved: midas_beta_weights.png")

# Plot 3: Fitted vs Actual
fig3 = plot_fitted_vs_actual(y, midas_beta.fitted_values, dates, 
                              "MIDAS (Beta): GDP Growth - Fitted vs Actual")
fig3.savefig(os.path.join(OUTPUT_DIR, 'midas_fitted_vs_actual.png'), dpi=150, bbox_inches='tight')
print("  Saved: midas_fitted_vs_actual.png")

# Plot 4: Compare U-MIDAS coefficients with parametric weights
fig4, ax = plt.subplots(figsize=(10, 5))
lags = np.arange(1, X_hf.shape[1] + 1)

# Normalize U-MIDAS coefficients for comparison
umidas_coefs = umidas.params[1:]  # Exclude constant
umidas_sum = np.sum(np.abs(umidas_coefs))
umidas_normalized = np.abs(umidas_coefs) / umidas_sum

ax.bar(lags - 0.2, umidas_normalized, width=0.4, label='U-MIDAS (normalized |β|)', alpha=0.6)
ax.plot(lags, midas_beta.weights, 'ro-', label='Beta weights', markersize=6)
ax.plot(lags, midas_almon.weights, 'gs--', label='Exp. Almon weights', markersize=6)
ax.set_xlabel('Lag (months)', fontsize=12)
ax.set_ylabel('Weight', fontsize=12)
ax.set_title('Comparison: Parametric vs Unrestricted MIDAS Weights', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, 'midas_weights_comparison.png'), dpi=150, bbox_inches='tight')
print("  Saved: midas_weights_comparison.png")

plt.close('all')

print("\n>>> MIDAS estimation complete!")
print("="*70)
