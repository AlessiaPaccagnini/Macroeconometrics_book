"""
============================================================================
Empirical Application: Forecasting U.S. GDP Growth and Inflation
Traditional Methods vs. Machine Learning
============================================================================

Author   : Alessia Paccagnini
Textbook : Macroeconometrics

DATA  (FRED):
    GDP.xlsx      — Real GDP (GDPC1), quarterly
    GDPDEFL.xlsx  — GDP Deflator (GDPDEF), quarterly
    FFR.xlsx      — Federal Funds Rate (FEDFUNDS), monthly → quarterly avg

SAMPLE: 1954:Q4 – 2025:Q3  (284 quarterly observations)
TRANSFORMATIONS:
    GDP Growth  = 400 × Δln(GDPC1)   [annualised %]
    Inflation   = 400 × Δln(GDPDEF)  [annualised %]
    FedFunds    = level               [%]

METHODS:
    Traditional : VAR(4), BVAR(4) with Minnesota prior (λ₁=0.2)
    ML          : LASSO, Ridge, Elastic Net (5-fold CV), Random Forest (500 trees)

EVALUATION FRAMEWORK:
    Window      : expanding
    OOS period  : final 60 observations ≈ 2010:Q1 – 2025:Q3
    Horizons    : h = 1, 4 quarters ahead
    Point       : RMSE, MAE
    Density     : 90% PI coverage, average interval width
    Significance: Diebold-Mariano test vs VAR benchmark

============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import inv, solve
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


# ============================================================================
# SECTION 1: DATA LOADING AND PREPARATION
# ============================================================================

def load_fred_data(gdp_path, deflator_path, fedfunds_path):
    """
    Load and process FRED data from Excel files.

    Parameters
    ----------
    gdp_path : str
        Path to GDP.xlsx (Real GDP, GDPC1)
    deflator_path : str
        Path to GDPDEFL.xlsx (GDP Deflator, GDPDEF)
    fedfunds_path : str
        Path to FFR.xlsx (Federal Funds Rate, monthly)

    Returns
    -------
    df : pd.DataFrame
        Quarterly DataFrame: GDP_growth, Inflation, FedFunds
        Sample: 1954:Q4 – 2025:Q3 (284 observations)
    """
    # --- Real GDP (quarterly) ---
    gdp = pd.read_excel(gdp_path, header=None, skiprows=1)
    gdp.columns = ['date', 'GDPC1']
    gdp['date']  = pd.to_datetime(gdp['date'])
    gdp['GDPC1'] = pd.to_numeric(gdp['GDPC1'], errors='coerce')
    gdp['quarter'] = gdp['date'].dt.to_period('Q')
    gdp = gdp.set_index('quarter')[['GDPC1']]

    # --- GDP Deflator (quarterly) ---
    deflator = pd.read_excel(deflator_path, header=None, skiprows=1)
    deflator.columns = ['date', 'GDPDEF']
    deflator['date']   = pd.to_datetime(deflator['date'])
    deflator['GDPDEF'] = pd.to_numeric(deflator['GDPDEF'], errors='coerce')
    deflator['quarter'] = deflator['date'].dt.to_period('Q')
    deflator = deflator.set_index('quarter')[['GDPDEF']]

    # --- Federal Funds Rate (monthly → quarterly average) ---
    fedfunds = pd.read_excel(fedfunds_path, header=None, skiprows=1)
    fedfunds.columns = ['date', 'FEDFUNDS']
    fedfunds['date']     = pd.to_datetime(fedfunds['date'])
    fedfunds['FEDFUNDS'] = pd.to_numeric(fedfunds['FEDFUNDS'], errors='coerce')
    fedfunds['quarter']  = fedfunds['date'].dt.to_period('Q')
    fedfunds_q = fedfunds.groupby('quarter')['FEDFUNDS'].mean().to_frame()

    # --- Merge ---
    df = gdp.join(deflator, how='inner').join(fedfunds_q, how='inner')

    # --- Transformations ---
    df['GDP_growth'] = np.log(df['GDPC1']).diff() * 400   # annualised %
    df['Inflation']  = np.log(df['GDPDEF']).diff() * 400  # annualised %
    df['FedFunds']   = df['FEDFUNDS']                      # level %

    df = df[['GDP_growth', 'Inflation', 'FedFunds']].dropna()

    # --- Restrict to book sample 1954:Q4 – 2025:Q3 ---
    df = df.loc['1954Q4':'2025Q3']

    # Convert PeriodIndex → DatetimeIndex for plotting
    df.index = df.index.to_timestamp()

    def _qstr(ts):
        return f"{ts.year}:Q{(ts.month - 1) // 3 + 1}"

    print(f"Sample: {_qstr(df.index[0])} – {_qstr(df.index[-1])}  "
          f"({len(df)} observations)")
    print("Variables: GDP_growth [ann. %], Inflation [ann. %], FedFunds [%]")

    return df


def prepare_forecast_data(df, target_var, h=1, p=4):
    """
    Prepare data for direct h-step ahead forecasting.
    
    Model: y_{t+h} = f(y_t, y_{t-1}, ..., y_{t-p+1}, x_t, x_{t-1}, ..., x_{t-p+1})
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all variables
    target_var : str
        Name of target variable to forecast
    h : int
        Forecast horizon
    p : int
        Number of lags to include
        
    Returns
    -------
    y : np.ndarray
        Target values y_{t+h}
    X : np.ndarray
        Feature matrix with lagged values
    dates : pd.DatetimeIndex
        Dates corresponding to forecast origins
    feature_names : list
        Names of features
    """
    T = len(df)
    variables = df.columns.tolist()
    
    # Determine valid sample range
    start_idx = p  # Need p lags
    end_idx = T - h  # Need h periods ahead for target
    
    # Build feature matrix
    X_list = []
    feature_names = []
    
    for lag in range(p):
        for var in variables:
            col_data = df[var].values[start_idx - lag - 1:end_idx - lag]
            X_list.append(col_data.reshape(-1, 1))
            feature_names.append(f'{var}_lag{lag + 1}')
    
    X = np.hstack(X_list)
    
    # Target variable
    y = df[target_var].values[start_idx + h - 1:end_idx + h]
    
    # Dates (forecast origins)
    dates = df.index[start_idx:end_idx + 1]
    
    return y, X, dates, feature_names


# ============================================================================
# SECTION 2: FORECASTING MODELS
# ============================================================================

class VARModel:
    """
    Vector Autoregression estimated by OLS.
    
    For a VAR(p) with n variables:
        Y_t = c + A_1 Y_{t-1} + ... + A_p Y_{t-p} + e_t
        
    We estimate equation by equation for direct forecasting.
    
    Density forecasts use residual variance assuming normality.
    """
    
    def __init__(self, p=4):
        self.p = p
        self.coef_ = None
        self.intercept_ = None
        self.sigma_ = None
        
    def fit(self, y, X):
        """Fit by OLS."""
        # Add intercept
        X_aug = np.column_stack([np.ones(len(y)), X])
        
        # OLS estimation
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        
        # Residual variance for density forecasts
        y_fitted = X_aug @ beta
        residuals = y - y_fitted
        self.sigma_ = np.std(residuals, ddof=X_aug.shape[1])
        
        return self
    
    def predict(self, X):
        """Point forecast."""
        return self.intercept_ + X @ self.coef_
    
    def predict_interval(self, X, alpha=0.1):
        """
        Prediction interval assuming normal errors.
        
        Returns lower and upper bounds for (1-alpha) interval.
        """
        y_pred = self.predict(X)
        z = stats.norm.ppf(1 - alpha / 2)
        lower = y_pred - z * self.sigma_
        upper = y_pred + z * self.sigma_
        return lower, upper
    
    def predict_distribution(self, X, n_samples=1000):
        """Sample from predictive distribution."""
        y_pred = self.predict(X)
        samples = np.random.normal(y_pred.reshape(-1, 1), self.sigma_, 
                                   size=(len(y_pred), n_samples))
        return samples


class BVARModel:
    """
    Bayesian VAR with Minnesota (Litterman) Prior.
    
    Prior specification:
        - Own lags: prior mean = 1 for first lag, 0 for others
        - Cross-variable lags: prior mean = 0
        - Prior variance decreases with lag length
        
    Hyperparameters:
        lambda_1: Overall tightness
        lambda_2: Cross-variable tightness (relative to own)
        lambda_3: Lag decay
        
    Density forecasts come from the posterior predictive distribution.
    """
    
    def __init__(self, p=4, lambda_1=0.2, lambda_2=0.5, lambda_3=1.0):
        self.p = p
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.coef_ = None
        self.intercept_ = None
        self.sigma_ = None
        self.V_post_ = None  # Posterior variance of coefficients
        
    def fit(self, y, X, var_names=None):
        """
        Fit BVAR using Minnesota prior.
        
        Parameters
        ----------
        y : np.ndarray
            Target variable
        X : np.ndarray
            Feature matrix (lags of all variables)
        var_names : list
            Names of features to identify own vs cross lags
        """
        T, k = X.shape
        
        # Compute prior variance for each coefficient
        # This is a simplified version - assumes features are ordered by lag
        n_vars = k // self.p  # Number of variables
        
        prior_var = np.ones(k)
        for j in range(k):
            lag = (j // n_vars) + 1  # Which lag (1, 2, ...)
            var_idx = j % n_vars  # Which variable
            
            # Lag decay
            prior_var[j] = (self.lambda_1 / (lag ** self.lambda_3)) ** 2
            
            # Cross-variable penalty (apply to non-own variables)
            # Assuming first variable in each lag block is own lag
            if var_idx > 0:
                prior_var[j] *= self.lambda_2 ** 2
        
        # Prior mean (0 for all coefficients in direct forecasting)
        prior_mean = np.zeros(k)
        
        # Add intercept (diffuse prior)
        X_aug = np.column_stack([np.ones(T), X])
        prior_var_full = np.concatenate([[100.0], prior_var])  # Large variance for intercept
        prior_mean_full = np.concatenate([[0.0], prior_mean])
        
        # Posterior (conjugate normal-normal)
        V_prior_inv = np.diag(1.0 / prior_var_full)
        V_post_inv = V_prior_inv + X_aug.T @ X_aug
        V_post = np.linalg.inv(V_post_inv)
        
        beta_post = V_post @ (V_prior_inv @ prior_mean_full + X_aug.T @ y)
        
        self.intercept_ = beta_post[0]
        self.coef_ = beta_post[1:]
        self.V_post_ = V_post
        
        # Residual variance
        y_fitted = X_aug @ beta_post
        residuals = y - y_fitted
        self.sigma_ = np.std(residuals, ddof=1)
        
        return self
    
    def predict(self, X):
        """Point forecast (posterior mean)."""
        return self.intercept_ + X @ self.coef_
    
    def predict_interval(self, X, alpha=0.1):
        """
        Prediction interval from posterior predictive.
        
        Accounts for both parameter uncertainty and forecast error.
        """
        n = X.shape[0]
        X_aug = np.column_stack([np.ones(n), X])
        
        y_pred = self.predict(X)
        
        # Predictive variance = parameter uncertainty + residual variance
        pred_var = np.zeros(n)
        for i in range(n):
            x_i = X_aug[i, :]
            pred_var[i] = x_i @ self.V_post_ @ x_i + self.sigma_ ** 2
        
        pred_std = np.sqrt(pred_var)
        z = stats.norm.ppf(1 - alpha / 2)
        
        lower = y_pred - z * pred_std
        upper = y_pred + z * pred_std
        
        return lower, upper
    
    def predict_distribution(self, X, n_samples=1000):
        """Sample from posterior predictive distribution."""
        n = X.shape[0]
        X_aug = np.column_stack([np.ones(n), X])
        
        samples = np.zeros((n, n_samples))
        
        for i in range(n):
            x_i = X_aug[i, :]
            pred_mean = x_i @ np.concatenate([[self.intercept_], self.coef_])
            pred_var = x_i @ self.V_post_ @ x_i + self.sigma_ ** 2
            samples[i, :] = np.random.normal(pred_mean, np.sqrt(pred_var), n_samples)
        
        return samples


class PenalizedRegressionModel:
    """
    Penalized regression models: LASSO, Ridge, Elastic Net.
    
    Uses cross-validation for hyperparameter selection.
    
    Density forecasts via bootstrap:
        1. Resample residuals
        2. Generate pseudo-samples
        3. Re-estimate model
        4. Collect predictions
    """
    
    def __init__(self, method='lasso', n_bootstrap=200):
        """
        Parameters
        ----------
        method : str
            'lasso', 'ridge', or 'elastic_net'
        n_bootstrap : int
            Number of bootstrap samples for density forecasts
        """
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.sigma_ = None
        self.residuals_ = None
        self.X_train_ = None
        self.y_train_ = None
        
    def fit(self, y, X):
        """Fit with cross-validation for hyperparameter selection."""
        # Standardize features
        X_scaled = self.scaler_.fit_transform(X)
        
        # Select model
        if self.method == 'lasso':
            self.model_ = LassoCV(cv=5, random_state=42, max_iter=10000)
        elif self.method == 'ridge':
            self.model_ = RidgeCV(cv=5)
        elif self.method == 'elastic_net':
            self.model_ = ElasticNetCV(cv=5, random_state=42, max_iter=10000,
                                        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99])
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.model_.fit(X_scaled, y)
        
        # Store for bootstrap
        self.X_train_ = X_scaled
        self.y_train_ = y
        
        # Compute residuals
        y_fitted = self.model_.predict(X_scaled)
        self.residuals_ = y - y_fitted
        self.sigma_ = np.std(self.residuals_, ddof=1)
        
        return self
    
    def predict(self, X):
        """Point forecast."""
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)
    
    def predict_interval(self, X, alpha=0.1, method='residual_bootstrap'):
        """
        Prediction interval via bootstrap.
        
        Parameters
        ----------
        X : np.ndarray
            Features for prediction
        alpha : float
            Significance level (default 0.1 for 90% interval)
        method : str
            'residual_bootstrap' or 'normal' (assumes normality)
        """
        if method == 'normal':
            # Simple normal approximation
            y_pred = self.predict(X)
            z = stats.norm.ppf(1 - alpha / 2)
            lower = y_pred - z * self.sigma_
            upper = y_pred + z * self.sigma_
            return lower, upper
        
        # Residual bootstrap
        samples = self.predict_distribution(X, n_samples=self.n_bootstrap)
        
        lower = np.percentile(samples, 100 * alpha / 2, axis=1)
        upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=1)
        
        return lower, upper
    
    def predict_distribution(self, X, n_samples=None):
        """
        Bootstrap predictive distribution.
        
        Uses residual bootstrap: resample residuals, create pseudo-data,
        re-estimate, predict.
        """
        if n_samples is None:
            n_samples = self.n_bootstrap
            
        X_scaled = self.scaler_.transform(X)
        n_pred = X_scaled.shape[0]
        T = len(self.y_train_)
        
        samples = np.zeros((n_pred, n_samples))
        
        for b in range(n_samples):
            # Resample residuals
            idx = np.random.choice(T, size=T, replace=True)
            resid_boot = self.residuals_[idx]
            
            # Create pseudo-sample
            y_boot = self.model_.predict(self.X_train_) + resid_boot
            
            # Re-estimate model
            if self.method == 'lasso':
                model_boot = LassoCV(cv=5, random_state=b, max_iter=10000)
            elif self.method == 'ridge':
                model_boot = RidgeCV(cv=5)
            else:
                model_boot = ElasticNetCV(cv=5, random_state=b, max_iter=10000,
                                          l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99])
            
            model_boot.fit(self.X_train_, y_boot)
            
            # Predict and add residual uncertainty
            y_pred_boot = model_boot.predict(X_scaled)
            samples[:, b] = y_pred_boot + np.random.choice(self.residuals_, size=n_pred)
        
        return samples


class RandomForestModel:
    """
    Random Forest for macroeconomic forecasting.
    
    Density forecasts using:
        1. Variance across trees (natural uncertainty measure)
        2. Prediction intervals from tree predictions distribution
    """
    
    def __init__(self, n_estimators=500, max_depth=None, min_samples_leaf=5,
                 max_features='sqrt', random_state=42):
        self.model_ = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler_ = StandardScaler()
        self.sigma_ = None
        self.residuals_ = None
        
    def fit(self, y, X):
        """Fit Random Forest."""
        X_scaled = self.scaler_.fit_transform(X)
        self.model_.fit(X_scaled, y)
        
        # Compute residuals for additional uncertainty
        y_fitted = self.model_.predict(X_scaled)
        self.residuals_ = y - y_fitted
        self.sigma_ = np.std(self.residuals_, ddof=1)
        
        return self
    
    def predict(self, X):
        """Point forecast (mean across trees)."""
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)
    
    def predict_interval(self, X, alpha=0.1):
        """
        Prediction interval using tree variance + residual variance.
        
        The total uncertainty combines:
        - Variance across trees (model uncertainty)
        - Residual variance (irreducible error)
        """
        X_scaled = self.scaler_.transform(X)
        
        # Get predictions from all trees
        tree_preds = np.array([tree.predict(X_scaled) 
                               for tree in self.model_.estimators_])
        
        # Mean prediction
        y_pred = tree_preds.mean(axis=0)
        
        # Variance across trees
        tree_var = tree_preds.var(axis=0)
        
        # Total variance = tree variance + residual variance
        total_var = tree_var + self.sigma_ ** 2
        total_std = np.sqrt(total_var)
        
        # Assuming approximate normality for intervals
        z = stats.norm.ppf(1 - alpha / 2)
        lower = y_pred - z * total_std
        upper = y_pred + z * total_std
        
        return lower, upper
    
    def predict_distribution(self, X, n_samples=1000):
        """
        Sample from predictive distribution.
        
        Combines tree predictions with residual resampling.
        """
        X_scaled = self.scaler_.transform(X)
        n_pred = X_scaled.shape[0]
        
        # Get predictions from all trees
        tree_preds = np.array([tree.predict(X_scaled) 
                               for tree in self.model_.estimators_])
        n_trees = tree_preds.shape[0]
        
        samples = np.zeros((n_pred, n_samples))
        
        for s in range(n_samples):
            # Randomly select a tree
            tree_idx = np.random.randint(0, n_trees)
            # Add resampled residual
            resid = np.random.choice(self.residuals_, size=n_pred)
            samples[:, s] = tree_preds[tree_idx, :] + resid
        
        return samples
    
    def feature_importance(self, feature_names):
        """Return feature importance as DataFrame."""
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model_.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance


# ============================================================================
# SECTION 3: EVALUATION METRICS
# ============================================================================

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def coverage(y_true, lower, upper):
    """Empirical coverage probability of prediction interval."""
    return np.mean((y_true >= lower) & (y_true <= upper))


def avg_interval_width(lower, upper):
    """Average width of prediction interval."""
    return np.mean(upper - lower)


def crps_gaussian(y_true, mu, sigma):
    """
    Continuous Ranked Probability Score for Gaussian predictive distribution.
    
    CRPS measures the quality of probabilistic forecasts.
    Lower is better.
    """
    z = (y_true - mu) / sigma
    crps = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 
                    2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return np.mean(crps)


def crps_empirical(y_true, samples):
    """
    CRPS computed from empirical samples of predictive distribution.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual values (n,)
    samples : np.ndarray
        Samples from predictive distribution (n, n_samples)
    """
    n = len(y_true)
    crps_values = np.zeros(n)
    
    for i in range(n):
        # Sort samples
        sorted_samples = np.sort(samples[i, :])
        n_samples = len(sorted_samples)
        
        # Compute CRPS using the formula with empirical CDF
        crps_i = 0
        for j, s in enumerate(sorted_samples):
            # Empirical CDF at s
            F_s = (j + 1) / n_samples
            # Indicator
            indicator = 1 if y_true[i] <= s else 0
            crps_i += (F_s - indicator) ** 2
        
        crps_values[i] = crps_i / n_samples
    
    return np.mean(crps_values)


# ============================================================================
# DIEBOLD-MARIANO TEST
# ============================================================================

def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano (1995) test for equal predictive ability.

    H0: E[d_t] = 0,  where d_t = L(e1_t) − L(e2_t),  L = squared error.
    Negative DM → model 1 has lower loss (outperforms model 2).

    HAC variance uses Newey-West with bandwidth = h − 1.

    Parameters
    ----------
    e1, e2 : np.ndarray  — forecast errors from model 1 and model 2
    h      : int         — forecast horizon (determines HAC bandwidth)

    Returns
    -------
    DM      : float — test statistic
    p_value : float — two-sided p-value
    """
    from scipy.stats import norm

    d  = e1 ** 2 - e2 ** 2
    d  = d[~np.isnan(d)]
    T  = len(d)
    d_bar = np.mean(d)

    # HAC variance: Newey-West, bandwidth = h − 1
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    bandwidth = h - 1          # 0 for h=1 → no autocovariance correction
    for k in range(1, bandwidth + 1):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        weight   = 1.0 - k / (bandwidth + 1)   # Bartlett kernel
        gamma_sum += 2.0 * weight * gamma_k

    var_d_bar = (gamma_0 + gamma_sum) / T

    if var_d_bar > 0:
        DM      = d_bar / np.sqrt(var_d_bar)
        p_value = 2.0 * (1.0 - norm.cdf(abs(DM)))
    else:
        DM, p_value = np.nan, np.nan

    return DM, p_value


def compute_dm_table(results, h):
    """
    Compute DM statistics for all methods vs VAR benchmark.

    Returns a list of dicts ready for pretty-printing or LaTeX export.
    """
    methods    = ['BVAR', 'LASSO', 'Ridge', 'ElasticNet', 'RandomForest']
    actuals    = results[h]['actuals']
    forecasts  = results[h]['forecasts']

    e_var = actuals - forecasts['VAR']

    rows = []
    for m in methods:
        e_m  = actuals - forecasts[m]
        dm, p = diebold_mariano_test(e_m, e_var, h=h)
        sig  = ('***' if p < 0.01 else
                '**'  if p < 0.05 else
                '*'   if p < 0.10 else '')
        rows.append({'Method': m, 'DM': dm, 'p_value': p, 'sig': sig})
    return rows


# ============================================================================
# SECTION 4: FORECASTING EXERCISE
# ============================================================================

def run_forecasting_comparison(df, target_var, horizons=[1, 4], 
                               test_size=60, p=4, verbose=True):
    """
    Run out-of-sample forecasting comparison across all methods.
    
    Uses expanding window: estimate on t=1,...,T0, forecast T0+h,
    then expand to t=1,...,T0+1, forecast T0+1+h, etc.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with all variables
    target_var : str
        Variable to forecast
    horizons : list
        Forecast horizons to evaluate
    test_size : int
        Number of out-of-sample periods
    p : int
        Number of lags
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        Dictionary with results for each horizon
    """
    results = {}
    
    for h in horizons:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Forecasting {target_var} at horizon h={h}")
            print('='*60)
        
        # Prepare full dataset
        y_full, X_full, dates_full, feature_names = prepare_forecast_data(
            df, target_var, h=h, p=p
        )
        
        T = len(y_full)
        T0 = T - test_size  # Initial training sample size
        
        if T0 < 50:
            print(f"Warning: Training sample too small ({T0} obs). Skipping h={h}.")
            continue
        
        # Initialize storage
        methods = ['VAR', 'BVAR', 'LASSO', 'Ridge', 'ElasticNet', 'RandomForest']
        forecasts = {m: np.zeros(test_size) for m in methods}
        lower_bounds = {m: np.zeros(test_size) for m in methods}
        upper_bounds = {m: np.zeros(test_size) for m in methods}
        actuals = np.zeros(test_size)
        forecast_dates = []
        
        # Expanding window forecasting
        for t in range(test_size):
            train_end = T0 + t
            
            y_train = y_full[:train_end]
            X_train = X_full[:train_end]
            
            y_test = y_full[train_end]
            X_test = X_full[train_end:train_end+1]
            
            actuals[t] = y_test
            forecast_dates.append(dates_full[train_end])
            
            if verbose and t % 20 == 0:
                print(f"  Forecast origin {t+1}/{test_size}: {dates_full[train_end-1].strftime('%Y-%m')}")
            
            # ----- VAR -----
            model_var = VARModel(p=p)
            model_var.fit(y_train, X_train)
            forecasts['VAR'][t] = model_var.predict(X_test)[0]
            lb, ub = model_var.predict_interval(X_test, alpha=0.1)
            lower_bounds['VAR'][t], upper_bounds['VAR'][t] = lb[0], ub[0]
            
            # ----- BVAR -----
            model_bvar = BVARModel(p=p, lambda_1=0.2, lambda_2=0.5, lambda_3=1.0)
            model_bvar.fit(y_train, X_train)
            forecasts['BVAR'][t] = model_bvar.predict(X_test)[0]
            lb, ub = model_bvar.predict_interval(X_test, alpha=0.1)
            lower_bounds['BVAR'][t], upper_bounds['BVAR'][t] = lb[0], ub[0]
            
            # ----- LASSO -----
            model_lasso = PenalizedRegressionModel(method='lasso', n_bootstrap=100)
            model_lasso.fit(y_train, X_train)
            forecasts['LASSO'][t] = model_lasso.predict(X_test)[0]
            lb, ub = model_lasso.predict_interval(X_test, alpha=0.1, method='normal')
            lower_bounds['LASSO'][t], upper_bounds['LASSO'][t] = lb[0], ub[0]
            
            # ----- Ridge -----
            model_ridge = PenalizedRegressionModel(method='ridge', n_bootstrap=100)
            model_ridge.fit(y_train, X_train)
            forecasts['Ridge'][t] = model_ridge.predict(X_test)[0]
            lb, ub = model_ridge.predict_interval(X_test, alpha=0.1, method='normal')
            lower_bounds['Ridge'][t], upper_bounds['Ridge'][t] = lb[0], ub[0]
            
            # ----- Elastic Net -----
            model_enet = PenalizedRegressionModel(method='elastic_net', n_bootstrap=100)
            model_enet.fit(y_train, X_train)
            forecasts['ElasticNet'][t] = model_enet.predict(X_test)[0]
            lb, ub = model_enet.predict_interval(X_test, alpha=0.1, method='normal')
            lower_bounds['ElasticNet'][t], upper_bounds['ElasticNet'][t] = lb[0], ub[0]
            
            # ----- Random Forest -----
            model_rf = RandomForestModel(n_estimators=200, min_samples_leaf=5)
            model_rf.fit(y_train, X_train)
            forecasts['RandomForest'][t] = model_rf.predict(X_test)[0]
            lb, ub = model_rf.predict_interval(X_test, alpha=0.1)
            lower_bounds['RandomForest'][t], upper_bounds['RandomForest'][t] = lb[0], ub[0]
        
        # Compute evaluation metrics
        metrics = {}
        for m in methods:
            metrics[m] = {
                'RMSE': rmse(actuals, forecasts[m]),
                'MAE': mae(actuals, forecasts[m]),
                'Coverage_90': coverage(actuals, lower_bounds[m], upper_bounds[m]),
                'Avg_Width': avg_interval_width(lower_bounds[m], upper_bounds[m])
            }
        
        results[h] = {
            'forecasts': forecasts,
            'actuals': actuals,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'metrics': metrics,
            'dates': forecast_dates,
            'feature_names': feature_names
        }
        
        # Print results
        if verbose:
            print(f"\n  Results for h={h}:")
            print(f"  {'Method':<15} {'RMSE':>8} {'MAE':>8} {'Coverage':>10} {'Avg Width':>10}")
            print(f"  {'-'*51}")
            for m in methods:
                print(f"  {m:<15} {metrics[m]['RMSE']:>8.3f} {metrics[m]['MAE']:>8.3f} "
                      f"{metrics[m]['Coverage_90']:>10.1%} {metrics[m]['Avg_Width']:>10.2f}")
    
    return results


# ============================================================================
# SECTION 5: VISUALIZATION
# ============================================================================

def plot_forecast_comparison(results, target_var, h=1, save_path=None):
    """
    Plot forecast comparison for a given horizon.
    """
    if h not in results:
        print(f"No results for horizon h={h}")
        return
    
    res = results[h]
    dates = pd.to_datetime(res['dates'])
    actuals = res['actuals']
    forecasts = res['forecasts']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    methods = ['VAR', 'BVAR', 'LASSO', 'Ridge', 'ElasticNet', 'RandomForest']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (method, color) in enumerate(zip(methods, colors)):
        ax = axes[i]
        
        # Plot actual values
        ax.plot(dates, actuals, 'k-', linewidth=1.5, label='Actual', alpha=0.8)
        
        # Plot forecasts
        ax.plot(dates, forecasts[method], color=color, linewidth=1.2, 
                label=f'{method} forecast', linestyle='--')
        
        # Plot prediction interval
        ax.fill_between(dates, res['lower_bounds'][method], res['upper_bounds'][method],
                        color=color, alpha=0.2, label='90% PI')
        
        # Metrics annotation
        rmse_val = res['metrics'][method]['RMSE']
        cov_val = res['metrics'][method]['Coverage_90']
        ax.text(0.02, 0.98, f'RMSE: {rmse_val:.3f}\nCov: {cov_val:.1%}',
                transform=ax.transAxes, verticalalignment='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'{method}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'{target_var} Forecasts (h={h})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_metrics_comparison(results, target_var, save_path=None):
    """
    Bar plot comparing metrics across methods and horizons.
    """
    horizons = list(results.keys())
    methods = ['VAR', 'BVAR', 'LASSO', 'Ridge', 'ElasticNet', 'RandomForest']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # RMSE comparison
    ax = axes[0]
    x = np.arange(len(methods))
    width = 0.35
    
    for i, h in enumerate(horizons):
        rmse_vals = [results[h]['metrics'][m]['RMSE'] for m in methods]
        offset = width * (i - len(horizons)/2 + 0.5)
        bars = ax.bar(x + offset, rmse_vals, width, label=f'h={h}')
    
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('Point Forecast Accuracy (RMSE)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Coverage comparison
    ax = axes[1]
    for i, h in enumerate(horizons):
        cov_vals = [results[h]['metrics'][m]['Coverage_90'] for m in methods]
        offset = width * (i - len(horizons)/2 + 0.5)
        bars = ax.bar(x + offset, cov_vals, width, label=f'h={h}')
    
    ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, label='Nominal (90%)')
    ax.set_ylabel('Coverage Probability', fontsize=11)
    ax.set_title('Density Forecast Accuracy (Coverage)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.suptitle(f'Forecasting Comparison: {target_var}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def create_results_table(results, target_var):
    """
    Create a publication-ready results table.
    """
    methods = ['VAR', 'BVAR', 'LASSO', 'Ridge', 'ElasticNet', 'RandomForest']
    horizons = list(results.keys())
    
    # Create multi-level columns
    data = []
    for m in methods:
        row = {'Method': m}
        for h in horizons:
            row[f'RMSE (h={h})'] = results[h]['metrics'][m]['RMSE']
            row[f'MAE (h={h})'] = results[h]['metrics'][m]['MAE']
            row[f'Coverage (h={h})'] = results[h]['metrics'][m]['Coverage_90']
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Highlight best RMSE for each horizon
    print(f"\nForecasting Results: {target_var}")
    print("=" * 80)
    
    for h in horizons:
        print(f"\nHorizon h={h}:")
        rmse_col = f'RMSE (h={h})'
        mae_col = f'MAE (h={h})'
        cov_col = f'Coverage (h={h})'
        
        best_rmse = df[rmse_col].min()
        best_mae = df[mae_col].min()
        
        for _, row in df.iterrows():
            rmse_star = '*' if row[rmse_col] == best_rmse else ' '
            mae_star = '*' if row[mae_col] == best_mae else ' '
            print(f"  {row['Method']:<15} RMSE: {row[rmse_col]:>7.3f}{rmse_star}  "
                  f"MAE: {row[mae_col]:>7.3f}{mae_star}  "
                  f"Coverage: {row[cov_col]:>6.1%}")
    
    print("\n* indicates best performance")
    
    return df


# ============================================================================
# SECTION 6: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Forecasting U.S. GDP Growth and Inflation")
    print("Traditional Methods vs. Machine Learning")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Cell 1 — Load data
    #
    # THREE WAYS TO PROVIDE THE DATA FILES:
    #
    # Option A — Google Colab interactive upload (recommended for students)
    #   Run this cell in Colab; a file-picker will appear.
    #   Upload GDP.xlsx, GDPDEFL.xlsx, FFR.xlsx in one go.
    #
    # Option B — Google Drive mount
    #   Uncomment the Drive block below, mount your Drive, and set MY_DRIVE_FOLDER
    #   to the subfolder where the three files live.
    #
    # Option C — Local / server path  (default when NOT running in Colab)
    #   Set LOCAL_GDP_PATH etc. to wherever the files are on your machine.
    # -------------------------------------------------------------------------

    import os, sys

    def _in_colab():
        """Return True when running inside Google Colaboratory."""
        return 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ

    if _in_colab():
        # ------------------------------------------------------------------
        # OPTION A  — interactive upload  (un-comment Option B block below
        #             if you prefer Google Drive instead)
        # ------------------------------------------------------------------
        from google.colab import files as _colab_files

        print("\nPlease upload the three FRED data files:")
        print("  → GDP.xlsx   (Real GDP, GDPC1 — quarterly)")
        print("  → GDPDEFL.xlsx (GDP Deflator — quarterly)")
        print("  → FFR.xlsx   (Federal Funds Rate — monthly)\n")

        _uploaded = _colab_files.upload()      # opens file-picker widget

        # Map uploaded filenames to paths (Colab saves them in the cwd)
        # Match by stem keyword (handles Colab's " (1)", " (2)" suffixes)
        _name_map = {
            'gdpdefl': 'DEFLATOR_PATH',   # check before 'gdp' to avoid false match
            'gdp':     'GDP_PATH',
            'ffr':     'FEDFUNDS_PATH',
        }
        _paths = {}
        for _fname in _uploaded:
            _flower = _fname.lower()
            for _key, _var in _name_map.items():
                if _key in _flower and _var not in _paths:
                    _paths[_var] = _fname
                    print(f"  ✓ {_var} → {_fname}")
                    break

        # Safety check
        for _var in ('GDP_PATH', 'DEFLATOR_PATH', 'FEDFUNDS_PATH'):
            if _var not in _paths:
                raise FileNotFoundError(
                    f"Could not find a file matching '{_var}' among the uploads. "
                    f"Uploaded files: {list(_uploaded.keys())}"
                )

        GDP_PATH      = _paths['GDP_PATH']
        DEFLATOR_PATH = _paths['DEFLATOR_PATH']
        FEDFUNDS_PATH = _paths['FEDFUNDS_PATH']

        # ------------------------------------------------------------------
        # OPTION B  — Google Drive (uncomment this block and comment out
        #             Option A above if you prefer Drive)
        # ------------------------------------------------------------------
        # from google.colab import drive as _drive
        # _drive.mount('/content/drive')
        # MY_DRIVE_FOLDER = 'MyDrive/Macroeconometrics/Chapter9'   # ← adjust
        # GDP_PATH      = f'/content/drive/{MY_DRIVE_FOLDER}/GDP.xlsx'
        # DEFLATOR_PATH = f'/content/drive/{MY_DRIVE_FOLDER}/GDPDEFL.xlsx'
        # FEDFUNDS_PATH = f'/content/drive/{MY_DRIVE_FOLDER}/FFR.xlsx'

    else:
        # ------------------------------------------------------------------
        # OPTION C  — Local / server paths
        # ------------------------------------------------------------------
        GDP_PATH      = 'GDP.xlsx'        # ← adjust if files are elsewhere
        DEFLATOR_PATH = 'GDPDEFL.xlsx'
        FEDFUNDS_PATH = 'FFR.xlsx'

    print("\n[1/4] Loading data...")
    df = load_fred_data(GDP_PATH, DEFLATOR_PATH, FEDFUNDS_PATH)

    print("\nSummary Statistics:")
    print(df.describe().round(2))

    # -------------------------------------------------------------------------
    # Cell 2 — Run expanding-window forecasting comparison
    # test_size=60  ≈ 2010:Q1 – 2025:Q3
    # -------------------------------------------------------------------------
    print("\n[2/4] Running forecasting comparison (expanding window, 60 OOS obs)...")

    results_gdp = run_forecasting_comparison(
        df,
        target_var='GDP_growth',
        horizons=[1, 4],
        test_size=60,
        p=4,
        verbose=True
    )

    results_inf = run_forecasting_comparison(
        df,
        target_var='Inflation',
        horizons=[1, 4],
        test_size=60,
        p=4,
        verbose=True
    )

    # -------------------------------------------------------------------------
    # Cell 3 — Results tables
    # -------------------------------------------------------------------------
    print("\n[3/4] Building results tables...")

    methods_order = ['VAR', 'BVAR', 'LASSO', 'Ridge', 'ElasticNet', 'RandomForest']

    # --- Point forecast accuracy (RMSE, MAE) ---
    print("\n" + "=" * 72)
    print("Point Forecast Accuracy: RMSE and MAE")
    print("=" * 72)
    header = f"{'Method':<15}" + "".join(
        [f"{'RMSE':>8}{'MAE':>8}" for _ in [1, 4]]
    )
    print(f"{'':15}{'h=1':^16}{'h=4':^16}")
    print(f"{'Method':<15}{'RMSE':>8}{'MAE':>8}  {'RMSE':>8}{'MAE':>8}")
    print("-" * 55)
    for target_label, results in [('GDP Growth', results_gdp),
                                   ('Inflation',  results_inf)]:
        print(f"\n  {target_label}")
        for m in methods_order:
            row = f"  {m:<13}"
            for h in [1, 4]:
                r = results[h]['metrics'][m]
                row += f"{r['RMSE']:>8.2f}{r['MAE']:>8.2f}  "
            print(row)

    # --- Density forecast (Coverage, Width) ---
    print("\n" + "=" * 72)
    print("Density Forecast Accuracy: Coverage and Interval Width")
    print("(Nominal coverage = 90%)")
    print("=" * 72)
    print(f"{'':15}{'h=1':^22}{'h=4':^22}")
    print(f"{'Method':<15}{'Cov.':>8}{'Width':>10}  {'Cov.':>8}{'Width':>10}")
    print("-" * 60)
    for target_label, results in [('GDP Growth', results_gdp),
                                   ('Inflation',  results_inf)]:
        print(f"\n  {target_label}")
        for m in methods_order:
            row = f"  {m:<13}"
            for h in [1, 4]:
                r = results[h]['metrics'][m]
                row += f"{r['Coverage_90']:>8.1%}{r['Avg_Width']:>10.2f}  "
            print(row)

    # --- Diebold-Mariano vs VAR ---
    print("\n" + "=" * 72)
    print("Diebold-Mariano Tests: Comparison to VAR Benchmark")
    print("(Negative DM → method outperforms VAR; HAC with h−1 lags)")
    print("=" * 72)
    print(f"{'':15}{'GDP Growth':^30}{'Inflation':^30}")
    print(f"{'Method':<15}{'h=1 DM':>9}{'p-val':>8}  {'h=4 DM':>9}{'p-val':>8}  "
          f"{'h=1 DM':>9}{'p-val':>8}  {'h=4 DM':>9}{'p-val':>8}")
    print("-" * 95)
    dm_store = {}
    for target_label, results in [('GDP', results_gdp), ('INF', results_inf)]:
        dm_store[target_label] = {}
        for h in [1, 4]:
            dm_store[target_label][h] = compute_dm_table(results, h)

    non_var = ['BVAR', 'LASSO', 'Ridge', 'ElasticNet', 'RandomForest']
    for m in non_var:
        row = f"{m:<15}"
        for target_label in ['GDP', 'INF']:
            for h in [1, 4]:
                entry = next(r for r in dm_store[target_label][h] if r['Method'] == m)
                dm_val = f"{entry['DM']:>+.2f}{entry['sig']}"
                p_val  = f"{entry['p_value']:.3f}"
                row   += f"{dm_val:>11}{p_val:>8}  "
        print(row)
    print("\n  * p<0.10  ** p<0.05  *** p<0.01")

    # -------------------------------------------------------------------------
    # Cell 4 — Figures
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating figures...")

    fig1 = plot_forecast_comparison(results_gdp, 'GDP Growth', h=1,
                                    save_path='figure9_gdp_h1.png')
    fig2 = plot_forecast_comparison(results_gdp, 'GDP Growth', h=4,
                                    save_path='figure9_gdp_h4.png')
    fig3 = plot_forecast_comparison(results_inf,  'Inflation',  h=1,
                                    save_path='figure9_inf_h1.png')
    fig4 = plot_forecast_comparison(results_inf,  'Inflation',  h=4,
                                    save_path='figure9_inf_h4.png')
    fig5 = plot_metrics_comparison(results_gdp, 'GDP Growth',
                                   save_path='figure9_metrics_gdp.png')
    fig6 = plot_metrics_comparison(results_inf,  'Inflation',
                                   save_path='figure9_metrics_inf.png')

    print("\n" + "=" * 70)
    print("Replication complete.")
    print("Outputs: figure9_*.png")
    print("=" * 70)
