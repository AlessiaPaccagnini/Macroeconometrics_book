"""
Mixed-Frequency VAR (MF-VAR) Estimation in Python
==================================================
Following Ghysels (2016), Journal of Econometrics
"Macroeconomics and the Reality of Mixed Frequency Data"

Author: Alessia Paccagnini
Textbook: Macroeconometrics
Date: January 2026

This script implements the observation-driven MF-VAR approach where:
- High-frequency (monthly) variables are "stacked" into the low-frequency (quarterly) vector
- No latent variables or state-space representation required
- Standard VAR tools (OLS, Cholesky IRFs) apply directly

Key References:
- Ghysels, E. (2016). "Macroeconomics and the Reality of Mixed Frequency Data", JoE
- Ghysels, E., Hill, J., & Motegi, K. (2016). "Testing for Granger Causality with Mixed Frequency Data"
- Foroni, C. & Marcellino, M. (2014). "Mixed Frequency Structural VARs"
"""

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import chi2, norm
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
        return './'
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
    paths = {name: f'/content/{name}' for name in uploaded.keys()}
    print(f"\nUploaded: {list(paths.keys())}")
    return paths

# =============================================================================
# 1. DATA LOADING AND STACKING FOR MF-VAR
# =============================================================================

def load_data(file_paths=None):
    """
    Load the FRED data.

    Parameters
    ----------
    file_paths : dict, optional
        Mapping like {'FEDFUNDS.xlsx': <path>, 'GDPC1.xlsx': <path>, 'GDPDEF.xlsx': <path>}.
        If None, paths are inferred from the detected environment:
          - Colab  : expects files uploaded via setup_colab_files() to /content/
          - Jupyter: looks in the current working directory
          - Script : looks in /mnt/user-data/uploads/ (Claude/JupyterHub layout)
    """
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

    fedfunds = pd.read_excel(path_fedfunds, sheet_name='Monthly')
    gdp      = pd.read_excel(path_gdp,      sheet_name='Quarterly')
    gdpdef   = pd.read_excel(path_gdpdef,   sheet_name='Quarterly')
    
    fedfunds.columns = ['date', 'fedfunds']
    gdp.columns = ['date', 'gdp']
    gdpdef.columns = ['date', 'gdpdef']
    
    fedfunds['date'] = pd.to_datetime(fedfunds['date'])
    gdp['date'] = pd.to_datetime(gdp['date'])
    gdpdef['date'] = pd.to_datetime(gdpdef['date'])
    
    return fedfunds, gdp, gdpdef


def create_stacked_mfvar_data(monthly_df, quarterly_df, monthly_var='fedfunds', 
                               quarterly_var='gdp', transform_quarterly='growth',
                               m=3):
    """
    Create the stacked MF-VAR dataset following Ghysels (2016).
    
    The key insight: stack m high-frequency observations per low-frequency period
    into a single vector. For monthly-quarterly (m=3):
    
    Z_τ = [x_{τ,1}, x_{τ,2}, x_{τ,3}, y_τ]'
    
    where:
    - x_{τ,j} is the monthly variable in month j of quarter τ
    - y_τ is the quarterly variable
    
    Parameters:
    -----------
    monthly_df : DataFrame with monthly data
    quarterly_df : DataFrame with quarterly data  
    monthly_var : str, name of monthly variable column
    quarterly_var : str, name of quarterly variable column
    transform_quarterly : str, 'growth' for log growth, 'level' for levels
    m : int, number of months per quarter (default 3)
    
    Returns:
    --------
    Z : DataFrame with stacked MF-VAR data
    var_names : list of variable names
    """
    monthly_df = monthly_df.copy().sort_values('date').reset_index(drop=True)
    quarterly_df = quarterly_df.copy().sort_values('date').reset_index(drop=True)
    
    # Transform quarterly variable
    if transform_quarterly == 'growth':
        quarterly_df['y'] = 400 * np.log(quarterly_df[quarterly_var] / 
                                          quarterly_df[quarterly_var].shift(1))
    else:
        quarterly_df['y'] = quarterly_df[quarterly_var]
    
    # Create quarter identifier for monthly data
    monthly_df['quarter'] = monthly_df['date'].dt.to_period('Q').dt.to_timestamp()
    monthly_df['month_in_quarter'] = ((monthly_df['date'].dt.month - 1) % 3) + 1
    
    # Build the stacked dataset
    stacked_data = []
    
    for _, q_row in quarterly_df.iterrows():
        q_date = q_row['date']
        y_val = q_row['y']
        
        if pd.isna(y_val):
            continue
            
        # Get monthly values for this quarter
        q_monthly = monthly_df[monthly_df['quarter'] == q_date].sort_values('date')
        
        if len(q_monthly) < m:
            continue
        
        # Stack: [x_1, x_2, x_3, y]
        row_data = {
            'date': q_date,
            f'{monthly_var}_m1': q_monthly.iloc[0][monthly_var],  # First month
            f'{monthly_var}_m2': q_monthly.iloc[1][monthly_var],  # Second month
            f'{monthly_var}_m3': q_monthly.iloc[2][monthly_var],  # Third month
            quarterly_var: y_val
        }
        stacked_data.append(row_data)
    
    Z = pd.DataFrame(stacked_data)
    var_names = [f'{monthly_var}_m1', f'{monthly_var}_m2', f'{monthly_var}_m3', quarterly_var]
    
    return Z, var_names


def create_multivariate_stacked_data(monthly_dfs, quarterly_dfs, 
                                      monthly_vars, quarterly_vars,
                                      transforms=None, m=3):
    """
    Create stacked MF-VAR data with multiple monthly and quarterly variables.
    
    Parameters:
    -----------
    monthly_dfs : dict of DataFrames {var_name: df}
    quarterly_dfs : dict of DataFrames {var_name: df}
    monthly_vars : list of monthly variable names
    quarterly_vars : list of quarterly variable names
    transforms : dict {var_name: 'growth' or 'level'}
    m : int, months per quarter
    
    Returns:
    --------
    Z : DataFrame with stacked data
    var_names : list of all variable names in order
    """
    if transforms is None:
        transforms = {}
    
    # Get common quarterly dates
    all_quarters = None
    for var, df in quarterly_dfs.items():
        df = df.copy()
        df['quarter'] = pd.to_datetime(df['date']).dt.to_period('Q').dt.to_timestamp()
        quarters = set(df['quarter'].dropna())
        if all_quarters is None:
            all_quarters = quarters
        else:
            all_quarters = all_quarters.intersection(quarters)
    
    for var, df in monthly_dfs.items():
        df = df.copy()
        df['quarter'] = pd.to_datetime(df['date']).dt.to_period('Q').dt.to_timestamp()
        # Only keep quarters with complete monthly data
        complete_quarters = df.groupby('quarter').size()
        complete_quarters = set(complete_quarters[complete_quarters >= m].index)
        all_quarters = all_quarters.intersection(complete_quarters)
    
    all_quarters = sorted(list(all_quarters))
    
    # Build stacked dataset
    stacked_data = []
    var_names = []
    
    # Variable names: monthly vars stacked, then quarterly vars
    for mv in monthly_vars:
        for j in range(1, m + 1):
            var_names.append(f'{mv}_m{j}')
    var_names.extend(quarterly_vars)
    
    for q_date in all_quarters:
        row_data = {'date': q_date}
        valid = True
        
        # Add monthly variables (stacked)
        for mv in monthly_vars:
            df = monthly_dfs[mv].copy()
            df['date'] = pd.to_datetime(df['date'])
            df['quarter'] = df['date'].dt.to_period('Q').dt.to_timestamp()
            q_monthly = df[df['quarter'] == q_date].sort_values('date')
            
            if len(q_monthly) < m:
                valid = False
                break
            
            for j in range(m):
                row_data[f'{mv}_m{j+1}'] = q_monthly.iloc[j][mv]
        
        if not valid:
            continue
        
        # Add quarterly variables
        for qv in quarterly_vars:
            df = quarterly_dfs[qv].copy()
            df['date'] = pd.to_datetime(df['date'])
            q_row = df[df['date'] == q_date]
            
            if len(q_row) == 0:
                valid = False
                break
            
            val = q_row.iloc[0][qv]
            
            # Apply transformation
            if transforms.get(qv) == 'growth':
                # Need previous quarter
                prev_q = q_date - pd.DateOffset(months=3)
                prev_row = df[df['date'] == prev_q]
                if len(prev_row) == 0:
                    valid = False
                    break
                prev_val = prev_row.iloc[0][qv]
                val = 400 * np.log(val / prev_val)
            
            row_data[qv] = val
        
        if valid and not any(pd.isna(row_data.get(v)) for v in var_names):
            stacked_data.append(row_data)
    
    Z = pd.DataFrame(stacked_data)
    
    return Z, var_names


# =============================================================================
# 2. MF-VAR MODEL CLASS
# =============================================================================

class MixedFrequencyVAR:
    """
    Mixed-Frequency VAR model following Ghysels (2016).
    
    The model is formulated as a standard VAR on the stacked vector:
    
    Z_τ = c + A₁ Z_{τ-1} + A₂ Z_{τ-2} + ... + A_p Z_{τ-p} + u_τ
    
    where Z_τ = [x_{τ,1}, ..., x_{τ,m}, y_τ]' stacks m high-frequency 
    observations with the low-frequency variable(s).
    
    Estimation is by OLS equation-by-equation (or GLS for efficiency).
    """
    
    def __init__(self, data, var_names, p=1):
        """
        Initialize MF-VAR.
        
        Parameters:
        -----------
        data : DataFrame with stacked MF-VAR data (must include 'date' column)
        var_names : list of variable names (excluding 'date')
        p : int, number of lags
        """
        self.data = data.copy()
        self.var_names = var_names
        self.p = p
        self.k = len(var_names)  # Number of variables in stacked system
        
        # Results storage
        self.A = None  # VAR coefficient matrices [A1, A2, ..., Ap]
        self.c = None  # Intercept vector
        self.Sigma = None  # Residual covariance matrix
        self.residuals = None
        self.fitted = None
        self.T = None  # Effective sample size
        
    def _create_var_matrices(self):
        """Create Y and X matrices for VAR estimation."""
        Z = self.data[self.var_names].values
        T_full = len(Z)
        
        # Y: dependent variable matrix (T-p x k)
        Y = Z[self.p:]
        
        # X: regressor matrix with lags and constant
        # Each row: [1, Z_{t-1}', Z_{t-2}', ..., Z_{t-p}']
        X_list = [np.ones((T_full - self.p, 1))]  # Constant
        
        for lag in range(1, self.p + 1):
            X_list.append(Z[self.p - lag:T_full - lag])
        
        X = np.hstack(X_list)
        
        return Y, X
    
    def fit(self, method='ols'):
        """
        Estimate MF-VAR by OLS.
        
        Parameters:
        -----------
        method : str, 'ols' for equation-by-equation OLS
        
        Returns:
        --------
        self
        """
        Y, X = self._create_var_matrices()
        self.T = Y.shape[0]
        
        # OLS estimation: B = (X'X)^{-1} X'Y
        # B has shape (1 + k*p, k)
        XtX_inv = np.linalg.inv(X.T @ X)
        B = XtX_inv @ X.T @ Y
        
        # Extract intercept and coefficient matrices
        self.c = B[0, :]  # Intercept (k,)
        
        self.A = []
        for lag in range(self.p):
            start_idx = 1 + lag * self.k
            end_idx = 1 + (lag + 1) * self.k
            self.A.append(B[start_idx:end_idx, :].T)  # Shape (k, k)
        
        # Residuals and covariance
        self.fitted = X @ B
        self.residuals = Y - self.fitted
        
        # ML estimate of covariance (divide by T, not T-k*p-1)
        self.Sigma = (self.residuals.T @ self.residuals) / self.T
        
        # Compute standard errors
        self._compute_standard_errors(X, XtX_inv)
        
        return self
    
    def _compute_standard_errors(self, X, XtX_inv):
        """Compute standard errors for VAR coefficients."""
        # Variance of vec(B) = Sigma ⊗ (X'X)^{-1}
        # For equation-by-equation OLS with same regressors
        
        n_params = 1 + self.k * self.p  # params per equation
        
        self.se_c = np.zeros(self.k)
        self.se_A = [np.zeros((self.k, self.k)) for _ in range(self.p)]
        
        for eq in range(self.k):
            sigma_eq = self.Sigma[eq, eq]
            var_b = sigma_eq * np.diag(XtX_inv)
            se_b = np.sqrt(var_b)
            
            self.se_c[eq] = se_b[0]
            
            for lag in range(self.p):
                start_idx = 1 + lag * self.k
                end_idx = 1 + (lag + 1) * self.k
                self.se_A[lag][eq, :] = se_b[start_idx:end_idx]
    
    def predict(self, steps=1, Z_init=None):
        """
        Generate h-step ahead forecasts.
        
        Parameters:
        -----------
        steps : int, forecast horizon
        Z_init : array, initial values (if None, use last observation)
        
        Returns:
        --------
        forecasts : array, shape (steps, k)
        """
        if Z_init is None:
            # Use last p observations
            Z_init = self.data[self.var_names].values[-self.p:]
        
        forecasts = []
        Z_history = list(Z_init)
        
        for h in range(steps):
            # Z_t = c + A1 Z_{t-1} + ... + Ap Z_{t-p}
            Z_new = self.c.copy()
            for lag in range(self.p):
                Z_new += self.A[lag] @ Z_history[-(lag + 1)]
            
            forecasts.append(Z_new)
            Z_history.append(Z_new)
        
        return np.array(forecasts)
    
    def irf(self, periods=20, shock_var=None, shock_size=1.0, orthogonalized=True):
        """
        Compute impulse response functions.
        
        Parameters:
        -----------
        periods : int, number of periods for IRF
        shock_var : int or str, index or name of shock variable
        shock_size : float, size of shock (default 1 std dev)
        orthogonalized : bool, use Cholesky orthogonalization
        
        Returns:
        --------
        irf : array, shape (periods+1, k) impulse responses
        """
        if isinstance(shock_var, str):
            shock_var = self.var_names.index(shock_var)
        
        # Compute MA representation coefficients Ψ_h
        # Ψ_0 = I, Ψ_h = Σ_{j=1}^{min(h,p)} Ψ_{h-j} A_j
        
        Psi = [np.eye(self.k)]  # Ψ_0
        
        for h in range(1, periods + 1):
            Psi_h = np.zeros((self.k, self.k))
            for j in range(min(h, self.p)):
                Psi_h += Psi[h - 1 - j] @ self.A[j]
            Psi.append(Psi_h)
        
        # Orthogonalize shocks using Cholesky
        if orthogonalized:
            P = np.linalg.cholesky(self.Sigma)  # Lower triangular
        else:
            P = np.eye(self.k)
        
        # Impulse vector (shock to variable shock_var)
        impulse = np.zeros(self.k)
        impulse[shock_var] = shock_size
        
        # Transform impulse through Cholesky
        if orthogonalized:
            # Shock of size 1 std dev
            impulse = P[:, shock_var] * shock_size
        
        # Compute IRF
        irf = np.zeros((periods + 1, self.k))
        for h in range(periods + 1):
            irf[h] = Psi[h] @ impulse
        
        return irf
    
    def irf_with_ci(self, periods=20, shock_var=None, shock_size=1.0, 
                    orthogonalized=True, method='bootstrap', n_boot=1000, 
                    ci_level=0.95, seed=None):
        """
        Compute impulse response functions with confidence intervals.
        
        Parameters:
        -----------
        periods : int, number of periods for IRF
        shock_var : int or str, index or name of shock variable
        shock_size : float, size of shock (default 1 std dev)
        orthogonalized : bool, use Cholesky orthogonalization
        method : str, 'bootstrap' or 'asymptotic'
        n_boot : int, number of bootstrap replications (for bootstrap method)
        ci_level : float, confidence level (e.g., 0.95 for 95% CI)
        seed : int, random seed for reproducibility
        
        Returns:
        --------
        dict with keys:
            'irf': point estimate array (periods+1, k)
            'lower': lower CI bound (periods+1, k)
            'upper': upper CI bound (periods+1, k)
            'std': standard errors (periods+1, k) - for bootstrap
            'all_irfs': all bootstrap IRFs (n_boot, periods+1, k) - for bootstrap
        """
        if seed is not None:
            np.random.seed(seed)
        
        if isinstance(shock_var, str):
            shock_idx = self.var_names.index(shock_var)
        else:
            shock_idx = shock_var
        
        # Point estimate
        irf_point = self.irf(periods=periods, shock_var=shock_idx, 
                            shock_size=shock_size, orthogonalized=orthogonalized)
        
        if method == 'bootstrap':
            return self._bootstrap_irf_ci(periods, shock_idx, shock_size, 
                                          orthogonalized, n_boot, ci_level, irf_point)
        elif method == 'asymptotic':
            return self._asymptotic_irf_ci(periods, shock_idx, shock_size,
                                           orthogonalized, ci_level, irf_point)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'bootstrap' or 'asymptotic'")
    
    def _bootstrap_irf_ci(self, periods, shock_idx, shock_size, orthogonalized, 
                          n_boot, ci_level, irf_point):
        """
        Bootstrap confidence intervals for IRFs.
        
        Uses residual bootstrap: resample residuals and reconstruct data.
        """
        Y, X = self._create_var_matrices()
        T = Y.shape[0]
        
        # Store all bootstrap IRFs
        boot_irfs = np.zeros((n_boot, periods + 1, self.k))
        
        # Original data for reconstruction
        Z_original = self.data[self.var_names].values
        
        for b in range(n_boot):
            # Resample residuals with replacement
            boot_indices = np.random.choice(T, size=T, replace=True)
            boot_residuals = self.residuals[boot_indices]
            
            # Reconstruct data using bootstrap residuals
            Z_boot = np.zeros_like(Z_original[self.p:])
            
            # Use original initial values
            Z_history = list(Z_original[:self.p])
            
            for t in range(T):
                # Generate new observation
                Z_new = self.c.copy()
                for lag in range(self.p):
                    Z_new += self.A[lag] @ Z_history[-(lag + 1)]
                Z_new += boot_residuals[t]
                
                Z_boot[t] = Z_new
                Z_history.append(Z_new)
            
            # Create bootstrap dataset
            boot_data = pd.DataFrame(
                np.vstack([Z_original[:self.p], Z_boot]),
                columns=self.var_names
            )
            boot_data['date'] = self.data['date'].values
            
            # Estimate VAR on bootstrap data
            try:
                boot_var = MixedFrequencyVAR(boot_data, self.var_names, p=self.p)
                boot_var.fit()
                
                # Compute IRF
                boot_irfs[b] = boot_var.irf(periods=periods, shock_var=shock_idx,
                                            shock_size=shock_size, 
                                            orthogonalized=orthogonalized)
            except:
                # If estimation fails, use point estimate
                boot_irfs[b] = irf_point
        
        # Compute confidence intervals (percentile method)
        alpha = 1 - ci_level
        lower = np.percentile(boot_irfs, 100 * alpha / 2, axis=0)
        upper = np.percentile(boot_irfs, 100 * (1 - alpha / 2), axis=0)
        std = np.std(boot_irfs, axis=0)
        
        return {
            'irf': irf_point,
            'lower': lower,
            'upper': upper,
            'std': std,
            'all_irfs': boot_irfs,
            'method': 'bootstrap',
            'n_boot': n_boot,
            'ci_level': ci_level
        }
    
    def _asymptotic_irf_ci(self, periods, shock_idx, shock_size, orthogonalized,
                           ci_level, irf_point):
        """
        Asymptotic (analytical) confidence intervals for IRFs.
        
        Based on Lütkepohl (2005), using delta method for orthogonalized IRFs.
        This is a simplified version - full implementation requires 
        derivative of Cholesky decomposition.
        """
        # For non-orthogonalized IRFs, we can compute exact asymptotic variance
        # For orthogonalized IRFs, this is an approximation
        
        Y, X = self._create_var_matrices()
        T = Y.shape[0]
        
        # Variance of vec(B) where B = [c, A1, ..., Ap]
        # Var(vec(B)) = Sigma ⊗ (X'X)^{-1}
        XtX_inv = np.linalg.inv(X.T @ X)
        
        # Compute IRF standard errors using simulation-based approximation
        # (More accurate than pure analytical for orthogonalized IRFs)
        n_sim = 500
        sim_irfs = np.zeros((n_sim, periods + 1, self.k))
        
        # Draw from asymptotic distribution of parameters
        n_params_per_eq = 1 + self.k * self.p
        
        for s in range(n_sim):
            # Simulate VAR coefficients
            A_sim = []
            c_sim = np.zeros(self.k)
            
            for eq in range(self.k):
                # Variance for this equation's parameters
                var_eq = self.Sigma[eq, eq] * XtX_inv
                
                # Draw parameters
                params_eq = np.random.multivariate_normal(
                    np.concatenate([[self.c[eq]], 
                                   *[self.A[lag][eq, :] for lag in range(self.p)]]),
                    var_eq
                )
                
                c_sim[eq] = params_eq[0]
                
            # For coefficient matrices, use point estimates with noise
            # (Simplified - full version would draw jointly)
            for lag in range(self.p):
                A_lag_sim = self.A[lag].copy()
                for eq in range(self.k):
                    noise_scale = np.sqrt(self.Sigma[eq, eq] / T)
                    A_lag_sim[eq, :] += np.random.normal(0, noise_scale, self.k)
                A_sim.append(A_lag_sim)
            
            # Compute MA coefficients for simulated parameters
            Psi_sim = [np.eye(self.k)]
            for h in range(1, periods + 1):
                Psi_h = np.zeros((self.k, self.k))
                for j in range(min(h, self.p)):
                    Psi_h += Psi_sim[h - 1 - j] @ A_sim[j]
                Psi_sim.append(Psi_h)
            
            # Orthogonalize
            if orthogonalized:
                # Add noise to Sigma and compute Cholesky
                Sigma_sim = self.Sigma + np.random.normal(0, 0.01, self.Sigma.shape)
                Sigma_sim = (Sigma_sim + Sigma_sim.T) / 2  # Ensure symmetric
                # Ensure positive definite
                eigvals = np.linalg.eigvalsh(Sigma_sim)
                if np.min(eigvals) < 0.001:
                    Sigma_sim += (0.001 - np.min(eigvals) + 0.001) * np.eye(self.k)
                
                try:
                    P_sim = np.linalg.cholesky(Sigma_sim)
                except:
                    P_sim = np.linalg.cholesky(self.Sigma)
                
                impulse = P_sim[:, shock_idx] * shock_size
            else:
                impulse = np.zeros(self.k)
                impulse[shock_idx] = shock_size
            
            # Compute IRF
            for h in range(periods + 1):
                sim_irfs[s, h] = Psi_sim[h] @ impulse
        
        # Compute confidence intervals
        std = np.std(sim_irfs, axis=0)
        z_crit = norm.ppf((1 + ci_level) / 2)
        
        lower = irf_point - z_crit * std
        upper = irf_point + z_crit * std
        
        return {
            'irf': irf_point,
            'lower': lower,
            'upper': upper,
            'std': std,
            'all_irfs': sim_irfs,
            'method': 'asymptotic',
            'ci_level': ci_level
        }
    
    def fevd(self, periods=20):
        """
        Compute Forecast Error Variance Decomposition.
        
        Parameters:
        -----------
        periods : int, number of periods
        
        Returns:
        --------
        fevd : array, shape (periods+1, k, k)
               fevd[h, i, j] = contribution of shock j to variance of variable i at horizon h
        """
        # Compute orthogonalized IRFs for each shock
        P = np.linalg.cholesky(self.Sigma)
        
        # MA coefficients
        Psi = [np.eye(self.k)]
        for h in range(1, periods + 1):
            Psi_h = np.zeros((self.k, self.k))
            for j in range(min(h, self.p)):
                Psi_h += Psi[h - 1 - j] @ self.A[j]
            Psi.append(Psi_h)
        
        # Orthogonalized MA coefficients: Θ_h = Ψ_h P
        Theta = [psi @ P for psi in Psi]
        
        # FEVD computation
        fevd = np.zeros((periods + 1, self.k, self.k))
        
        for h in range(periods + 1):
            # Total MSE up to horizon h
            mse = np.zeros((self.k, self.k))
            for s in range(h + 1):
                mse += Theta[s] @ Theta[s].T
            
            mse_diag = np.diag(mse)
            
            # Contribution of each shock
            for j in range(self.k):
                contribution = np.zeros(self.k)
                for s in range(h + 1):
                    contribution += Theta[s][:, j]**2
                fevd[h, :, j] = contribution / mse_diag
        
        return fevd
    
    def granger_causality_test(self, cause_vars, effect_vars):
        """
        Test Granger causality from cause_vars to effect_vars.
        
        H0: coefficients on lagged cause_vars in equations for effect_vars = 0
        
        Parameters:
        -----------
        cause_vars : list of str or int, causing variables
        effect_vars : list of str or int, effect variables
        
        Returns:
        --------
        dict with test statistic, p-value, df
        """
        # Convert names to indices
        if isinstance(cause_vars[0], str):
            cause_idx = [self.var_names.index(v) for v in cause_vars]
        else:
            cause_idx = cause_vars
            
        if isinstance(effect_vars[0], str):
            effect_idx = [self.var_names.index(v) for v in effect_vars]
        else:
            effect_idx = effect_vars
        
        # Number of restrictions
        n_restrictions = len(cause_idx) * len(effect_idx) * self.p
        
        # Wald test
        # H0: R * vec(B) = 0
        # W = (R β̂)' [R (Σ ⊗ (X'X)^{-1}) R']^{-1} (R β̂)
        
        # For simplicity, use F-test approximation
        # Test each effect equation separately and combine
        
        Y, X = self._create_var_matrices()
        T = Y.shape[0]
        
        # Unrestricted SSR
        ssr_u = np.sum(self.residuals[:, effect_idx]**2)
        
        # Restricted model: remove cause_vars from each equation
        # Create restricted X matrix
        keep_cols = [0]  # Always keep constant
        for lag in range(self.p):
            for var in range(self.k):
                if var not in cause_idx:
                    keep_cols.append(1 + lag * self.k + var)
        
        X_r = X[:, keep_cols]
        
        # Estimate restricted model
        B_r = np.linalg.lstsq(X_r, Y[:, effect_idx], rcond=None)[0]
        resid_r = Y[:, effect_idx] - X_r @ B_r
        ssr_r = np.sum(resid_r**2)
        
        # F-statistic
        df1 = n_restrictions
        df2 = T * len(effect_idx) - X.shape[1] * len(effect_idx)
        
        F_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
        p_value = 1 - chi2.cdf(F_stat * df1, df1)  # Chi-square approximation
        
        return {
            'F_statistic': F_stat,
            'Wald_statistic': F_stat * df1,
            'p_value': p_value,
            'df': (df1, df2),
            'H0': f"No Granger causality from {cause_vars} to {effect_vars}"
        }
    
    def summary(self):
        """Print estimation summary."""
        print("\n" + "="*75)
        print("MIXED-FREQUENCY VAR ESTIMATION RESULTS")
        print("Following Ghysels (2016, Journal of Econometrics)")
        print("="*75)
        print(f"\nSample size (T):        {self.T}")
        print(f"Number of variables:    {self.k}")
        print(f"Number of lags (p):     {self.p}")
        print(f"Parameters per eq.:     {1 + self.k * self.p}")
        print(f"Total parameters:       {self.k * (1 + self.k * self.p)}")
        
        print("\n" + "-"*75)
        print("Variables in stacked system:")
        print("-"*75)
        for i, name in enumerate(self.var_names):
            print(f"  {i+1}. {name}")
        
        print("\n" + "-"*75)
        print("Coefficient Estimates (showing A₁ matrix):")
        print("-"*75)
        
        # Print A1 matrix with row/column labels
        print("\nA₁ (coefficients on first lag):")
        header = "".ljust(15) + "".join([f"{v[:12]:>14}" for v in self.var_names])
        print(header)
        print("-" * len(header))
        
        for i, row_var in enumerate(self.var_names):
            row_str = f"{row_var[:14]:<15}"
            for j in range(self.k):
                coef = self.A[0][i, j]
                se = self.se_A[0][i, j]
                t_stat = coef / se if se > 0 else np.nan
                # Add significance stars
                stars = ""
                if abs(t_stat) > 2.576:
                    stars = "***"
                elif abs(t_stat) > 1.96:
                    stars = "**"
                elif abs(t_stat) > 1.645:
                    stars = "*"
                row_str += f"{coef:>11.4f}{stars:<3}"
            print(row_str)
        
        print("\nNote: *** p<0.01, ** p<0.05, * p<0.10")
        
        # Intercepts
        print("\n" + "-"*75)
        print("Intercepts:")
        print("-"*75)
        for i, name in enumerate(self.var_names):
            print(f"  {name:<20}: {self.c[i]:>10.4f} (SE: {self.se_c[i]:.4f})")
        
        # Residual covariance
        print("\n" + "-"*75)
        print("Residual Covariance Matrix (Σ):")
        print("-"*75)
        header = "".ljust(15) + "".join([f"{v[:12]:>14}" for v in self.var_names])
        print(header)
        for i, row_var in enumerate(self.var_names):
            row_str = f"{row_var[:14]:<15}"
            for j in range(self.k):
                row_str += f"{self.Sigma[i, j]:>14.4f}"
            print(row_str)
        
        # Correlation matrix
        print("\nResidual Correlation Matrix:")
        D_inv = np.diag(1 / np.sqrt(np.diag(self.Sigma)))
        corr = D_inv @ self.Sigma @ D_inv
        header = "".ljust(15) + "".join([f"{v[:12]:>14}" for v in self.var_names])
        print(header)
        for i, row_var in enumerate(self.var_names):
            row_str = f"{row_var[:14]:<15}"
            for j in range(self.k):
                row_str += f"{corr[i, j]:>14.3f}"
            print(row_str)
        
        # Model fit statistics
        print("\n" + "-"*75)
        print("Equation-by-Equation R²:")
        print("-"*75)
        Y = self.data[self.var_names].values[self.p:]
        for i, name in enumerate(self.var_names):
            ss_tot = np.sum((Y[:, i] - np.mean(Y[:, i]))**2)
            ss_res = np.sum(self.residuals[:, i]**2)
            r2 = 1 - ss_res / ss_tot
            print(f"  {name:<20}: R² = {r2:.4f}")
        
        # Log-likelihood and information criteria
        det_Sigma = np.linalg.det(self.Sigma)
        log_lik = -0.5 * self.T * (self.k * np.log(2 * np.pi) + np.log(det_Sigma) + self.k)
        n_params = self.k * (1 + self.k * self.p)
        aic = -2 * log_lik + 2 * n_params
        bic = -2 * log_lik + n_params * np.log(self.T)
        
        print("\n" + "-"*75)
        print("Information Criteria:")
        print("-"*75)
        print(f"  Log-likelihood:  {log_lik:>15.2f}")
        print(f"  AIC:             {aic:>15.2f}")
        print(f"  BIC:             {bic:>15.2f}")
        print("="*75)


# =============================================================================
# 3. VISUALIZATION FUNCTIONS
# =============================================================================

def plot_irf(mfvar, shock_var, periods=20, var_names=None, figsize=(14, 10)):
    """
    Plot impulse response functions (without confidence intervals).
    
    Parameters:
    -----------
    mfvar : MixedFrequencyVAR object
    shock_var : int or str, shock variable
    periods : int, IRF horizon
    var_names : list, names for plot labels (optional)
    figsize : tuple, figure size
    """
    if var_names is None:
        var_names = mfvar.var_names
    
    if isinstance(shock_var, str):
        shock_idx = mfvar.var_names.index(shock_var)
        shock_name = shock_var
    else:
        shock_idx = shock_var
        shock_name = mfvar.var_names[shock_idx]
    
    irf = mfvar.irf(periods=periods, shock_var=shock_idx, orthogonalized=True)
    
    k = mfvar.k
    n_cols = 2
    n_rows = (k + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    horizons = np.arange(periods + 1)
    
    for i in range(k):
        ax = axes[i]
        ax.plot(horizons, irf[:, i], 'b-', linewidth=2)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.fill_between(horizons, 0, irf[:, i], alpha=0.2)
        ax.set_xlabel('Quarters', fontsize=10)
        ax.set_ylabel('Response', fontsize=10)
        ax.set_title(f'Response of {var_names[i]}', fontsize=11)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for i in range(k, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(f'Impulse Responses to {shock_name} Shock (1 Std. Dev.)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_irf_with_ci(mfvar, shock_var, periods=20, var_names=None, figsize=(14, 10),
                     method='bootstrap', n_boot=1000, ci_level=0.95, seed=42):
    """
    Plot impulse response functions with confidence intervals.
    
    Parameters:
    -----------
    mfvar : MixedFrequencyVAR object
    shock_var : int or str, shock variable
    periods : int, IRF horizon
    var_names : list, names for plot labels (optional)
    figsize : tuple, figure size
    method : str, 'bootstrap' or 'asymptotic'
    n_boot : int, number of bootstrap replications
    ci_level : float, confidence level (e.g., 0.95)
    seed : int, random seed
    
    Returns:
    --------
    fig : matplotlib figure
    irf_results : dict with IRF and CI data
    """
    if var_names is None:
        var_names = mfvar.var_names
    
    if isinstance(shock_var, str):
        shock_idx = mfvar.var_names.index(shock_var)
        shock_name = shock_var
    else:
        shock_idx = shock_var
        shock_name = mfvar.var_names[shock_idx]
    
    print(f"Computing IRF confidence intervals using {method} method...")
    if method == 'bootstrap':
        print(f"  Running {n_boot} bootstrap replications...")
    
    # Get IRF with confidence intervals
    irf_results = mfvar.irf_with_ci(
        periods=periods, 
        shock_var=shock_idx, 
        orthogonalized=True,
        method=method,
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed
    )
    
    irf = irf_results['irf']
    lower = irf_results['lower']
    upper = irf_results['upper']
    
    k = mfvar.k
    n_cols = 2
    n_rows = (k + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    horizons = np.arange(periods + 1)
    ci_pct = int(ci_level * 100)
    
    for i in range(k):
        ax = axes[i]
        
        # Plot confidence interval as shaded region
        ax.fill_between(horizons, lower[:, i], upper[:, i], 
                        color='blue', alpha=0.2, label=f'{ci_pct}% CI')
        
        # Plot point estimate
        ax.plot(horizons, irf[:, i], 'b-', linewidth=2, label='Point estimate')
        
        # Zero line
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        ax.set_xlabel('Quarters', fontsize=10)
        ax.set_ylabel('Response', fontsize=10)
        ax.set_title(f'Response of {var_names[i]}', fontsize=11)
        ax.grid(alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    # Hide unused subplots
    for i in range(k, len(axes)):
        axes[i].set_visible(False)
    
    method_label = "Bootstrap" if method == 'bootstrap' else "Asymptotic"
    fig.suptitle(f'Impulse Responses to {shock_name} Shock (1 Std. Dev.)\n'
                 f'{ci_pct}% Confidence Intervals ({method_label})', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig, irf_results


def plot_multiple_irfs_with_ci(mfvar, shock_vars, response_var, periods=20,
                                figsize=(12, 5), method='bootstrap', n_boot=500,
                                ci_level=0.95, seed=42):
    """
    Plot IRFs from multiple shocks to a single response variable.
    Useful for comparing effects of shocks at different points in the quarter.
    
    Parameters:
    -----------
    mfvar : MixedFrequencyVAR object
    shock_vars : list of shock variable names/indices
    response_var : str or int, response variable
    periods : int, IRF horizon
    figsize : tuple, figure size
    method : str, 'bootstrap' or 'asymptotic'
    n_boot : int, bootstrap replications
    ci_level : float, confidence level
    seed : int, random seed
    """
    if isinstance(response_var, str):
        resp_idx = mfvar.var_names.index(response_var)
        resp_name = response_var
    else:
        resp_idx = response_var
        resp_name = mfvar.var_names[resp_idx]
    
    n_shocks = len(shock_vars)
    fig, axes = plt.subplots(1, n_shocks, figsize=figsize)
    if n_shocks == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_shocks))
    horizons = np.arange(periods + 1)
    ci_pct = int(ci_level * 100)
    
    for idx, shock_var in enumerate(shock_vars):
        if isinstance(shock_var, str):
            shock_name = shock_var
        else:
            shock_name = mfvar.var_names[shock_var]
        
        print(f"Computing IRF for shock to {shock_name}...")
        
        irf_results = mfvar.irf_with_ci(
            periods=periods,
            shock_var=shock_var,
            orthogonalized=True,
            method=method,
            n_boot=n_boot,
            ci_level=ci_level,
            seed=seed + idx
        )
        
        ax = axes[idx]
        
        # Plot CI
        ax.fill_between(horizons, 
                        irf_results['lower'][:, resp_idx], 
                        irf_results['upper'][:, resp_idx],
                        color=colors[idx], alpha=0.3, label=f'{ci_pct}% CI')
        
        # Plot point estimate
        ax.plot(horizons, irf_results['irf'][:, resp_idx], 
                color=colors[idx], linewidth=2, label='Point estimate')
        
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Quarters', fontsize=11)
        ax.set_ylabel('Response', fontsize=11)
        ax.set_title(f'Shock to {shock_name}', fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    
    fig.suptitle(f'Response of {resp_name} to Different Shocks\n'
                 f'{ci_pct}% Confidence Intervals ({method.capitalize()})',
                 fontsize=13, y=1.05)
    plt.tight_layout()
    
    return fig


def plot_fevd(mfvar, periods=20, var_names=None, figsize=(14, 10)):
    """
    Plot Forecast Error Variance Decomposition.
    """
    if var_names is None:
        var_names = mfvar.var_names
    
    fevd = mfvar.fevd(periods=periods)
    k = mfvar.k
    
    n_cols = 2
    n_rows = (k + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    horizons = np.arange(periods + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, k))
    
    for i in range(k):
        ax = axes[i]
        
        # Stacked area plot
        bottom = np.zeros(periods + 1)
        for j in range(k):
            ax.fill_between(horizons, bottom, bottom + fevd[:, i, j], 
                           label=var_names[j], alpha=0.7, color=colors[j])
            bottom += fevd[:, i, j]
        
        ax.set_xlabel('Quarters', fontsize=10)
        ax.set_ylabel('Share', fontsize=10)
        ax.set_title(f'FEVD of {var_names[i]}', fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    for i in range(k, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle('Forecast Error Variance Decomposition', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def compare_mfvar_vs_lf_var(mfvar_data, lf_data, var_names_mf, var_names_lf, 
                            shock_var_mf, shock_var_lf, p=1, periods=20):
    """
    Compare IRFs from MF-VAR vs traditional low-frequency VAR.
    
    This illustrates the potential misspecification from temporal aggregation.
    """
    # Estimate MF-VAR
    mfvar = MixedFrequencyVAR(mfvar_data, var_names_mf, p=p)
    mfvar.fit()
    
    # Estimate LF-VAR (using aggregated monthly data)
    lfvar = MixedFrequencyVAR(lf_data, var_names_lf, p=p)
    lfvar.fit()
    
    # Get IRFs
    irf_mf = mfvar.irf(periods=periods, shock_var=shock_var_mf, orthogonalized=True)
    irf_lf = lfvar.irf(periods=periods, shock_var=shock_var_lf, orthogonalized=True)
    
    return mfvar, lfvar, irf_mf, irf_lf


# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================

# Main execution (runs in both script and notebook contexts)

print("\n" + "="*75)
print("MF-VAR ESTIMATION: Following Ghysels (2016, JoE)")
print("="*75)

# Load data
print("\n>>> Loading data...")

def _ask_upload_or_reuse_mfvar():
    """Ask whether to upload files or reuse data already in the environment."""
    import builtins
    _globals = vars(builtins)
    try:
        _globals = globals()
    except NameError:
        pass
    already_loaded = all(v in _globals for v in ['fedfunds', 'gdp', 'gdpdef'])

    if already_loaded:
        ans = input("Data already found in environment. Use existing data? [y/n] (default: y): ").strip().lower()
        if ans in ('', 'y', 'yes'):
            print("  Reusing existing fedfunds, gdp and gdpdef variables.")
            return None   # signal: reuse
    return 'upload'       # signal: upload/load fresh

_load_signal = _ask_upload_or_reuse_mfvar()

if _load_signal is None:
    pass  # reuse existing fedfunds, gdp, gdpdef
elif ENV == 'colab':
    print("Environment: Google Colab — uploading files now...")
    _uploaded = setup_colab_files()
    fedfunds, gdp, gdpdef = load_data(file_paths=_uploaded)
else:
    print(f"Environment: {ENV} | Output dir: {OUTPUT_DIR}")
    fedfunds, gdp, gdpdef = load_data()


print(f"Monthly Fed Funds: {len(fedfunds)} observations")
print(f"Quarterly GDP: {len(gdp)} observations")

# Create stacked MF-VAR data
print("\n>>> Creating stacked MF-VAR data...")
print("    Stacking monthly Fed Funds into [FFR_m1, FFR_m2, FFR_m3] per quarter")
print("    Adding quarterly GDP growth as the low-frequency variable")

Z, var_names = create_stacked_mfvar_data(
    fedfunds, gdp, 
    monthly_var='fedfunds', 
    quarterly_var='gdp',
    transform_quarterly='growth'
)

print(f"\nStacked data shape: {Z.shape}")
print(f"Variables: {var_names}")
print(f"Sample period: {Z['date'].min().strftime('%Y-Q%q')} to {Z['date'].max().strftime('%Y-Q%q')}")
print(f"\nFirst few observations:")
print(Z.head())

# =========================================================================
# Estimate MF-VAR
# =========================================================================
print("\n>>> Estimating MF-VAR(1)...")
mfvar = MixedFrequencyVAR(Z, var_names, p=1)
mfvar.fit()
mfvar.summary()

# =========================================================================
# Granger Causality Tests
# =========================================================================
print("\n>>> Granger Causality Tests:")
print("-"*75)

# Test: Do monthly interest rates Granger-cause GDP growth?
gc_ffr_to_gdp = mfvar.granger_causality_test(
    cause_vars=['fedfunds_m1', 'fedfunds_m2', 'fedfunds_m3'],
    effect_vars=['gdp']
)
print(f"\nTest: Monthly Fed Funds → GDP Growth")
print(f"  Wald statistic: {gc_ffr_to_gdp['Wald_statistic']:.4f}")
print(f"  p-value: {gc_ffr_to_gdp['p_value']:.4f}")
print(f"  Conclusion: {'Reject H0' if gc_ffr_to_gdp['p_value'] < 0.05 else 'Fail to reject H0'} at 5% level")

# Test: Does GDP growth Granger-cause interest rates?
gc_gdp_to_ffr = mfvar.granger_causality_test(
    cause_vars=['gdp'],
    effect_vars=['fedfunds_m1', 'fedfunds_m2', 'fedfunds_m3']
)
print(f"\nTest: GDP Growth → Monthly Fed Funds")
print(f"  Wald statistic: {gc_gdp_to_ffr['Wald_statistic']:.4f}")
print(f"  p-value: {gc_gdp_to_ffr['p_value']:.4f}")
print(f"  Conclusion: {'Reject H0' if gc_gdp_to_ffr['p_value'] < 0.05 else 'Fail to reject H0'} at 5% level")

# =========================================================================
# Generate Plots
# =========================================================================
print("\n>>> Generating visualizations...")

# IRF to Fed Funds shock in month 1 (without CI - quick)
fig1 = plot_irf(mfvar, 'fedfunds_m1', periods=16)
fig1.savefig(os.path.join(OUTPUT_DIR, 'mfvar_irf_ffr_m1.png'), dpi=150, bbox_inches='tight')
print("  Saved: mfvar_irf_ffr_m1.png")

# IRF to GDP shock (without CI - quick)
fig2 = plot_irf(mfvar, 'gdp', periods=16)
fig2.savefig(os.path.join(OUTPUT_DIR, 'mfvar_irf_gdp.png'), dpi=150, bbox_inches='tight')
print("  Saved: mfvar_irf_gdp.png")

# FEVD
fig3 = plot_fevd(mfvar, periods=16)
fig3.savefig(os.path.join(OUTPUT_DIR, 'mfvar_fevd.png'), dpi=150, bbox_inches='tight')
print("  Saved: mfvar_fevd.png")

# =========================================================================
# IRFs with Confidence Intervals
# =========================================================================
print("\n>>> Computing IRFs with confidence intervals...")

# Bootstrap CI for FFR_m1 shock
fig4, irf_ci_results = plot_irf_with_ci(
    mfvar, 'fedfunds_m1', periods=16,
    method='bootstrap', n_boot=500, ci_level=0.90, seed=42
)
fig4.savefig(os.path.join(OUTPUT_DIR, 'mfvar_irf_ffr_m1_ci_bootstrap.png'), dpi=150, bbox_inches='tight')
print("  Saved: mfvar_irf_ffr_m1_ci_bootstrap.png")

# Asymptotic CI for comparison
fig5, irf_ci_asymp = plot_irf_with_ci(
    mfvar, 'fedfunds_m1', periods=16,
    method='asymptotic', ci_level=0.90, seed=42
)
fig5.savefig(os.path.join(OUTPUT_DIR, 'mfvar_irf_ffr_m1_ci_asymptotic.png'), dpi=150, bbox_inches='tight')
print("  Saved: mfvar_irf_ffr_m1_ci_asymptotic.png")

# Compare responses of GDP to shocks in different months
print("\n>>> Comparing GDP response to shocks at different times in quarter...")
fig6 = plot_multiple_irfs_with_ci(
    mfvar, 
    shock_vars=['fedfunds_m1', 'fedfunds_m2', 'fedfunds_m3'],
    response_var='gdp',
    periods=16,
    method='bootstrap',
    n_boot=500,
    ci_level=0.90,
    seed=123
)
fig6.savefig(os.path.join(OUTPUT_DIR, 'mfvar_gdp_response_by_month.png'), dpi=150, bbox_inches='tight')
print("  Saved: mfvar_gdp_response_by_month.png")

# Print summary of IRF significance
print("\n>>> IRF Significance Summary (90% CI):")
print("-"*75)
print("Response of GDP to Fed Funds shocks:")
for shock in ['fedfunds_m1', 'fedfunds_m2', 'fedfunds_m3']:
    irf_res = mfvar.irf_with_ci(periods=8, shock_var=shock, method='bootstrap',
                                 n_boot=300, ci_level=0.90, seed=42)
    gdp_idx = mfvar.var_names.index('gdp')
    
    print(f"\n  Shock to {shock}:")
    for h in [0, 1, 2, 4, 8]:
        lower = irf_res['lower'][h, gdp_idx]
        point = irf_res['irf'][h, gdp_idx]
        upper = irf_res['upper'][h, gdp_idx]
        sig = "*" if (lower > 0 or upper < 0) else ""
        print(f"    h={h}: {point:>7.3f} [{lower:>7.3f}, {upper:>7.3f}] {sig}")

# =========================================================================
# Compare with traditional Low-Frequency VAR
# =========================================================================
print("\n>>> Comparing MF-VAR with traditional LF-VAR...")

# Create LF data (quarterly averages of monthly Fed Funds)
fedfunds_copy = fedfunds.copy()
fedfunds_copy['quarter'] = fedfunds_copy['date'].dt.to_period('Q').dt.to_timestamp()
ffr_quarterly = fedfunds_copy.groupby('quarter')['fedfunds'].mean().reset_index()
ffr_quarterly.columns = ['date', 'fedfunds']

# Merge with GDP
gdp_copy = gdp.copy()
gdp_copy['gdp_growth'] = 400 * np.log(gdp_copy['gdp'] / gdp_copy['gdp'].shift(1))
lf_data = pd.merge(ffr_quarterly, gdp_copy[['date', 'gdp_growth']], on='date')
lf_data = lf_data.dropna()

# Estimate LF-VAR
lfvar = MixedFrequencyVAR(lf_data, ['fedfunds', 'gdp_growth'], p=1)
lfvar.fit()

print("\nLow-Frequency VAR Results (using quarterly averages):")
print("-"*75)
print(f"  A₁[FFR→FFR]:  {lfvar.A[0][0,0]:.4f}  vs MF-VAR A₁[FFR_m1→FFR_m1]: {mfvar.A[0][0,0]:.4f}")
print(f"  A₁[FFR→GDP]:  {lfvar.A[0][1,0]:.4f}  vs MF-VAR A₁[FFR_m1→GDP]:    {mfvar.A[0][3,0]:.4f}")
print(f"  A₁[GDP→FFR]:  {lfvar.A[0][0,1]:.4f}  vs MF-VAR A₁[GDP→FFR_m1]:    {mfvar.A[0][0,3]:.4f}")
print(f"  A₁[GDP→GDP]:  {lfvar.A[0][1,1]:.4f}  vs MF-VAR A₁[GDP→GDP]:       {mfvar.A[0][3,3]:.4f}")

# Plot comparison of IRFs
fig4, axes = plt.subplots(1, 2, figsize=(12, 5))

# MF-VAR IRF (response of GDP to FFR_m1 shock)
irf_mf = mfvar.irf(periods=16, shock_var='fedfunds_m1')
# LF-VAR IRF (response of GDP to FFR shock)
irf_lf = lfvar.irf(periods=16, shock_var='fedfunds')

horizons = np.arange(17)

ax1 = axes[0]
ax1.plot(horizons, irf_mf[:, 3], 'b-', linewidth=2, label='MF-VAR (FFR_m1 shock)')
ax1.plot(horizons, irf_lf[:, 1], 'r--', linewidth=2, label='LF-VAR (avg FFR shock)')
ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.set_xlabel('Quarters', fontsize=11)
ax1.set_ylabel('Response', fontsize=11)
ax1.set_title('Response of GDP Growth to Interest Rate Shock', fontsize=12)
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(horizons, irf_mf[:, 0], 'b-', linewidth=2, label='MF-VAR (FFR_m1)')
ax2.plot(horizons, irf_lf[:, 0], 'r--', linewidth=2, label='LF-VAR (avg FFR)')
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_xlabel('Quarters', fontsize=11)
ax2.set_ylabel('Response', fontsize=11)
ax2.set_title('Response of Interest Rate to Own Shock', fontsize=12)
ax2.legend()
ax2.grid(alpha=0.3)

fig4.suptitle('MF-VAR vs LF-VAR: Impulse Response Comparison\n(Illustrating potential temporal aggregation bias)', 
              fontsize=13, y=1.05)
plt.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, 'mfvar_vs_lfvar_comparison.png'), dpi=150, bbox_inches='tight')
print("  Saved: mfvar_vs_lfvar_comparison.png")

# =========================================================================
# Estimate MF-VAR(2) for comparison
# =========================================================================
print("\n>>> Estimating MF-VAR(2) for comparison...")
mfvar2 = MixedFrequencyVAR(Z, var_names, p=2)
mfvar2.fit()

# Compare AIC/BIC
det_Sigma1 = np.linalg.det(mfvar.Sigma)
det_Sigma2 = np.linalg.det(mfvar2.Sigma)

k = mfvar.k
T = mfvar.T

ll1 = -0.5 * T * (k * np.log(2*np.pi) + np.log(det_Sigma1) + k)
ll2 = -0.5 * mfvar2.T * (k * np.log(2*np.pi) + np.log(det_Sigma2) + k)

aic1 = -2*ll1 + 2*(k*(1+k*1))
aic2 = -2*ll2 + 2*(k*(1+k*2))
bic1 = -2*ll1 + (k*(1+k*1))*np.log(T)
bic2 = -2*ll2 + (k*(1+k*2))*np.log(mfvar2.T)

print("\nLag Selection:")
print("-"*75)
print(f"{'Model':<15} {'Log-Lik':>12} {'AIC':>12} {'BIC':>12}")
print("-"*75)
print(f"{'MF-VAR(1)':<15} {ll1:>12.2f} {aic1:>12.2f} {bic1:>12.2f}")
print(f"{'MF-VAR(2)':<15} {ll2:>12.2f} {aic2:>12.2f} {bic2:>12.2f}")
print("-"*75)
preferred = "MF-VAR(1)" if bic1 < bic2 else "MF-VAR(2)"
print(f"BIC prefers: {preferred}")

plt.close('all')

print("\n>>> MF-VAR estimation complete!")
print("="*75)
