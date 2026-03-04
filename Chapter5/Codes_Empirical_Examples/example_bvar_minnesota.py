"""
================================================================================
EXAMPLE: BAYESIAN VAR WITH MINNESOTA PRIOR
Comparison with Frequentist VAR (Chapter 4)
================================================================================
Data: U.S. FRED (GDPC1, GDPDEF, FEDFUNDS)
Sample: 1970:Q1 - 2007:Q4

This example demonstrates:
1. Minnesota prior implementation via dummy observations
2. Posterior simulation from Normal-Inverse Wishart
3. Bayesian IRFs with credible intervals
4. Comparison with frequentist (OLS) estimates
5. Prior sensitivity analysis

Author: Alessia Paccagnini
Textbook: Macroeconometrics
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, inv
from scipy.stats import invwishart
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# UPLOAD DATA FILES (Google Colab)
# =============================================================================
from google.colab import files
print("Please upload the three Excel files: GDPC1.xlsx, GDPDEF.xlsx, FEDFUNDS.xlsx")
uploaded = files.upload()

# =============================================================================
# CONFIGURATION
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (14, 4),
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False
})

COLORS = {
    'gdp': '#2E86AB',
    'inflation': '#A23B72', 
    'rate': '#F18F01',
    'bayes': '#E63946',
    'ols': '#457B9D',
}

print("=" * 70)
print("BAYESIAN VAR WITH MINNESOTA PRIOR")
print("Comparison with Frequentist Estimation")
print("=" * 70)

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================
print("\n[1/7] Loading FRED data...")

# Load GDP (Real GDP, Quarterly)
gdp_df = pd.read_excel('GDPC1.xlsx', sheet_name='Quarterly', header=None)
gdp_df.columns = ['date', 'gdp']
gdp_df = gdp_df.iloc[1:]
gdp_df['date'] = pd.to_datetime(gdp_df['date'])
gdp_df['gdp'] = pd.to_numeric(gdp_df['gdp'])
gdp_df = gdp_df.set_index('date')

# Load GDP Deflator
deflator_df = pd.read_excel('GDPDEF.xlsx', sheet_name='Quarterly', header=None)
deflator_df.columns = ['date', 'deflator']
deflator_df = deflator_df.iloc[1:]
deflator_df['date'] = pd.to_datetime(deflator_df['date'])
deflator_df['deflator'] = pd.to_numeric(deflator_df['deflator'])
deflator_df = deflator_df.set_index('date')

# Load Federal Funds Rate (Monthly -> Quarterly)
fedfunds_df = pd.read_excel('FEDFUNDS.xlsx', sheet_name='Monthly', header=None)
fedfunds_df.columns = ['date', 'fedfunds']
fedfunds_df = fedfunds_df.iloc[1:]
fedfunds_df['date'] = pd.to_datetime(fedfunds_df['date'])
fedfunds_df['fedfunds'] = pd.to_numeric(fedfunds_df['fedfunds'])
fedfunds_df = fedfunds_df.set_index('date')
fedfunds_q = fedfunds_df.resample('QE').mean()

# Compute growth rates (annualized)
gdp_growth = 400 * np.log(gdp_df['gdp'] / gdp_df['gdp'].shift(1))
inflation = 400 * np.log(deflator_df['deflator'] / deflator_df['deflator'].shift(1))

# Align dates
gdp_growth.index = gdp_growth.index + pd.offsets.QuarterEnd(0)
inflation.index = inflation.index + pd.offsets.QuarterEnd(0)

# Merge
data = pd.DataFrame({
    'gdp_growth': gdp_growth,
    'inflation': inflation,
    'fedfunds': fedfunds_q['fedfunds']
}).dropna()

# Sample: 1970:Q1 to 2007:Q4
data = data.loc['1970-01-01':'2007-12-31']

print(f"   ✓ Data loaded: {len(data)} observations")
print(f"   Sample: {data.index[0].strftime('%Y-Q1')} to {data.index[-1].strftime('%Y-Q4')}")

# =============================================================================
# SECTION 2: OLS VAR ESTIMATION (BENCHMARK)
# =============================================================================
print("\n[2/7] Estimating frequentist VAR(4) by OLS...")

def estimate_var_ols(Y, p):
    """Estimate VAR(p) by OLS"""
    T, K = Y.shape
    T_eff = T - p
    
    Y_dep = Y[p:]
    X = np.ones((T_eff, 1))
    for lag in range(1, p + 1):
        X = np.hstack([X, Y[p - lag:T - lag]])
    
    B_hat = np.linalg.lstsq(X, Y_dep, rcond=None)[0]
    u = Y_dep - X @ B_hat
    Sigma_u = (u.T @ u) / (T_eff - K * p - 1)
    
    return B_hat, Sigma_u, u, X, Y_dep

var_names = ['gdp_growth', 'inflation', 'fedfunds']
Y = data[var_names].values
dates = data.index
p = 4  # Lag order
K = len(var_names)

B_ols, Sigma_ols, u_ols, X_ols, Y_dep = estimate_var_ols(Y, p)
T_eff = len(u_ols)

print(f"   ✓ OLS estimation complete")
print(f"   Variables: {K}, Lags: {p}, Observations: {T_eff}")

# =============================================================================
# SECTION 3: MINNESOTA PRIOR SPECIFICATION
# =============================================================================
print("\n[3/7] Setting up Minnesota prior...")

def create_minnesota_prior_dummies(Y, p, lambda1=0.2, lambda2=0.5, lambda3=1.0, lambda4=1e5):
    """
    Create Minnesota prior via dummy observations
    
    Parameters:
    -----------
    Y : ndarray (T x K)
        Data matrix
    p : int
        Number of lags
    lambda1 : float
        Overall tightness (controls shrinkage toward prior mean)
    lambda2 : float
        Cross-variable shrinkage (0 < lambda2 <= 1)
    lambda3 : float
        Lag decay (higher = faster decay of prior variance with lag)
    lambda4 : float
        Constant term tightness (large = diffuse prior on constant)
    
    Returns:
    --------
    Y_d : ndarray
        Dummy observations for dependent variable
    X_d : ndarray
        Dummy observations for regressors
    sigma : ndarray
        AR(1) residual standard deviations (for scaling)
    """
    T, K = Y.shape
    
    # Estimate AR(1) for each variable to get scaling factors
    sigma = np.zeros(K)
    delta = np.zeros(K)  # AR(1) coefficients (for random walk prior, delta=1)
    
    for i in range(K):
        y_i = Y[1:, i]
        y_lag = Y[:-1, i]
        X_ar = np.column_stack([np.ones(len(y_i)), y_lag])
        beta_ar = np.linalg.lstsq(X_ar, y_i, rcond=None)[0]
        resid = y_i - X_ar @ beta_ar
        sigma[i] = np.std(resid, ddof=2)
        delta[i] = 1.0  # Random walk prior (can set to beta_ar[1] for AR(1) prior)
    
    # Number of dummy observations
    # Block 1: Shrinkage on own lags (K*p dummies)
    # Block 2: Shrinkage on cross-variable coefficients (K*p dummies, combined with Block 1)
    # Block 3: Sum-of-coefficients prior (K dummies)
    # Block 4: Initial observation prior (1 dummy with K variables)
    # Block 5: Constant term (1 dummy)
    
    n_dummy = K * p + K + 1 + 1  # Simplified version
    n_regressors = 1 + K * p  # constant + K*p lag coefficients
    
    Y_d = np.zeros((K * p + K + 1, K))
    X_d = np.zeros((K * p + K + 1, n_regressors))
    
    # ----- Block 1: Prior on VAR coefficients -----
    # These dummies implement: A_{l,ij} ~ N(delta_i * I(i=j, l=1), (lambda1 * sigma_i / (l^lambda3 * sigma_j))^2)
    row = 0
    for l in range(1, p + 1):
        for i in range(K):
            # Dummy for coefficient on lag l of variable i in equation i
            Y_d[row, i] = delta[i] * sigma[i] / (lambda1 * (l ** lambda3))
            col_idx = 1 + (l - 1) * K + i  # Position in regressor matrix
            X_d[row, col_idx] = sigma[i] / (lambda1 * (l ** lambda3))
            row += 1
    
    # ----- Block 2: Sum-of-coefficients prior -----
    # Implements belief that sum of lag coefficients on own variable = 1
    for i in range(K):
        Y_d[row, i] = delta[i] * sigma[i] / lambda1
        for l in range(1, p + 1):
            col_idx = 1 + (l - 1) * K + i
            X_d[row, col_idx] = sigma[i] / lambda1
        row += 1
    
    # ----- Block 3: Prior on constant -----
    Y_d[row, :] = 0
    X_d[row, 0] = lambda4
    
    return Y_d, X_d, sigma

# Set hyperparameters
lambda1 = 0.2   # Overall tightness (smaller = more shrinkage)
lambda2 = 0.5   # Cross-variable shrinkage
lambda3 = 1.0   # Lag decay

print(f"   Minnesota prior hyperparameters:")
print(f"   λ₁ (overall tightness) = {lambda1}")
print(f"   λ₂ (cross-variable)    = {lambda2}")
print(f"   λ₃ (lag decay)         = {lambda3}")

# Create dummy observations
Y_d, X_d, sigma_scale = create_minnesota_prior_dummies(Y, p, lambda1=lambda1)

print(f"   ✓ Created {len(Y_d)} dummy observations")

# =============================================================================
# SECTION 4: BAYESIAN ESTIMATION
# =============================================================================
print("\n[4/7] Bayesian VAR estimation...")

def estimate_bvar_minnesota(Y, p, lambda1=0.2, lambda2=0.5, lambda3=1.0):
    """
    Estimate BVAR with Minnesota prior using dummy observations
    Returns posterior parameters for Normal-Inverse Wishart distribution
    """
    T, K = Y.shape
    T_eff = T - p
    
    # Construct data matrices
    Y_dep = Y[p:]
    X = np.ones((T_eff, 1))
    for lag in range(1, p + 1):
        X = np.hstack([X, Y[p - lag:T - lag]])
    
    # Create dummy observations
    Y_d, X_d, sigma = create_minnesota_prior_dummies(Y, p, lambda1, lambda2, lambda3)
    
    # Stack actual data with dummy observations
    Y_star = np.vstack([Y_d, Y_dep])
    X_star = np.vstack([X_d, X])
    
    # Posterior parameters (conjugate Normal-Inverse Wishart)
    # Posterior mean of coefficients
    XtX_star = X_star.T @ X_star
    XtY_star = X_star.T @ Y_star
    
    B_post = np.linalg.solve(XtX_star, XtY_star)
    
    # Posterior scale matrix for Inverse Wishart
    resid_star = Y_star - X_star @ B_post
    S_post = resid_star.T @ resid_star
    
    # Posterior degrees of freedom
    nu_post = len(Y_star) - X_star.shape[1]
    
    # Posterior precision for coefficients
    V_post_inv = XtX_star
    
    return B_post, S_post, nu_post, V_post_inv, X

# Estimate BVAR
B_bvar, S_bvar, nu_bvar, V_bvar_inv, X_data = estimate_bvar_minnesota(Y, p, lambda1)

# Posterior mean of Sigma
Sigma_bvar = S_bvar / (nu_bvar - K - 1)

print(f"   ✓ BVAR estimation complete")
print(f"   Posterior degrees of freedom: {nu_bvar}")

# Compare coefficient estimates
print("\n   Comparison: First lag coefficients (own effects)")
print("   " + "-" * 50)
print(f"   {'Variable':<15} {'OLS':>12} {'BVAR':>12} {'Shrinkage':>12}")
print("   " + "-" * 50)
for i, var in enumerate(var_names):
    ols_coef = B_ols[1 + i, i]  # First own lag
    bvar_coef = B_bvar[1 + i, i]
    shrink = (1 - bvar_coef / ols_coef) * 100 if abs(ols_coef) > 0.01 else 0
    print(f"   {var:<15} {ols_coef:>12.4f} {bvar_coef:>12.4f} {shrink:>11.1f}%")

# =============================================================================
# SECTION 5: POSTERIOR SIMULATION
# =============================================================================
print("\n[5/7] Drawing from posterior distribution...")

def draw_posterior_niw(B_post, S_post, nu_post, V_post_inv, n_draws=2000):
    """
    Draw from Normal-Inverse Wishart posterior
    
    Sigma | Y ~ IW(nu_post, S_post)
    vec(B) | Sigma, Y ~ N(vec(B_post), Sigma ⊗ V_post_inv^{-1})
    """
    K = B_post.shape[1]
    n_coefs = B_post.shape[0]
    
    B_draws = np.zeros((n_draws, n_coefs, K))
    Sigma_draws = np.zeros((n_draws, K, K))
    
    V_post = np.linalg.inv(V_post_inv)
    
    for d in range(n_draws):
        # Draw Sigma from Inverse Wishart
        Sigma_draw = invwishart.rvs(df=nu_post, scale=S_post)
        Sigma_draws[d] = Sigma_draw
        
        # Draw B | Sigma from matrix normal
        # vec(B) ~ N(vec(B_post), Sigma ⊗ V_post)
        # Use Cholesky for efficient sampling
        L_sigma = cholesky(Sigma_draw, lower=True)
        L_V = cholesky(V_post, lower=True)
        
        # B = B_post + L_V @ Z @ L_sigma' where Z ~ N(0, I)
        Z = np.random.randn(n_coefs, K)
        B_draw = B_post + L_V @ Z @ L_sigma.T
        B_draws[d] = B_draw
    
    return B_draws, Sigma_draws

n_draws = 2000
n_burn = 500
total_draws = n_draws + n_burn

print(f"   Drawing {total_draws} samples (discarding {n_burn} burn-in)...")

B_draws, Sigma_draws = draw_posterior_niw(B_bvar, S_bvar, nu_bvar, V_bvar_inv, total_draws)

# Discard burn-in
B_draws = B_draws[n_burn:]
Sigma_draws = Sigma_draws[n_burn:]

print(f"   ✓ {n_draws} posterior draws retained")

# =============================================================================
# SECTION 6: COMPUTE IRFs FROM POSTERIOR
# =============================================================================
print("\n[6/7] Computing Bayesian IRFs with credible intervals...")

H = 40  # Horizon

def compute_irf_cholesky(B, Sigma, p, K, H):
    """Compute IRFs using Cholesky identification for a single draw"""
    # Cholesky decomposition
    P = cholesky(Sigma, lower=True)
    
    # Companion form
    A_comp = np.zeros((K * p, K * p))
    for l in range(p):
        A_comp[:K, l * K:(l + 1) * K] = B[1 + l * K:1 + (l + 1) * K, :].T
    if p > 1:
        A_comp[K:, :K * (p - 1)] = np.eye(K * (p - 1))
    
    # IRFs
    IRF = np.zeros((H + 1, K, K))
    IRF[0] = P
    
    A_power = np.eye(K * p)
    for h in range(1, H + 1):
        A_power = A_power @ A_comp
        Phi_h = A_power[:K, :K]
        IRF[h] = Phi_h @ P
    
    return IRF

# Compute IRFs for each posterior draw
print("   Computing IRFs for each posterior draw...")
IRF_draws = np.zeros((n_draws, H + 1, K, K))

for d in range(n_draws):
    IRF_draws[d] = compute_irf_cholesky(B_draws[d], Sigma_draws[d], p, K, H)

# Compute posterior median and credible intervals
IRF_median = np.median(IRF_draws, axis=0)
IRF_lower_68 = np.percentile(IRF_draws, 16, axis=0)
IRF_upper_68 = np.percentile(IRF_draws, 84, axis=0)
IRF_lower_90 = np.percentile(IRF_draws, 5, axis=0)
IRF_upper_90 = np.percentile(IRF_draws, 95, axis=0)

print("   ✓ IRF computation complete")

# Compute OLS IRFs for comparison
P_ols = cholesky(Sigma_ols, lower=True)
IRF_ols = compute_irf_cholesky(B_ols, Sigma_ols, p, K, H)

# =============================================================================
# SECTION 7: GENERATE FIGURES
# =============================================================================
print("\n[7/7] Generating figures...")

var_labels = ['GDP Growth', 'Inflation', 'Federal Funds Rate']
shock_idx = 2  # Monetary policy shock (third variable)

# --- FIGURE 1: Bayesian IRFs to MP Shock with Credible Intervals ---
print("   Figure 1: Bayesian IRFs to MP shock...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

colors_var = [COLORS['gdp'], COLORS['inflation'], COLORS['rate']]

for i, (ax, label, color) in enumerate(zip(axes, var_labels, colors_var)):
    horizons = range(H + 1)
    
    # 90% credible interval
    ax.fill_between(horizons, IRF_lower_90[:, i, shock_idx], IRF_upper_90[:, i, shock_idx],
                    alpha=0.15, color=color, label='90% CI')
    
    # 68% credible interval
    ax.fill_between(horizons, IRF_lower_68[:, i, shock_idx], IRF_upper_68[:, i, shock_idx],
                    alpha=0.3, color=color, label='68% CI')
    
    # Posterior median
    ax.plot(horizons, IRF_median[:, i, shock_idx], color=color, linewidth=2.5,
            label='Posterior median')
    
    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    ax.set_xlabel('Quarters after shock')
    ax.set_ylabel('Percentage points')
    ax.set_title(f'Response of {label}')
    ax.set_xlim(0, H)
    ax.legend(loc='upper right', fontsize=8)

fig.suptitle('Bayesian VAR: Impulse Responses to Monetary Policy Shock\n'
             f'(Minnesota Prior, λ₁={lambda1}, U.S. Data 1970-2007)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figure1_bvar_irf.png', dpi=150, bbox_inches='tight')
plt.show()

# --- FIGURE 2: Comparison BVAR vs OLS ---
print("   Figure 2: BVAR vs OLS comparison...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for i, (ax, label) in enumerate(zip(axes, var_labels)):
    horizons = range(H + 1)
    
    # BVAR credible interval
    ax.fill_between(horizons, IRF_lower_68[:, i, shock_idx], IRF_upper_68[:, i, shock_idx],
                    alpha=0.3, color=COLORS['bayes'], label='BVAR 68% CI')
    
    # BVAR median
    ax.plot(horizons, IRF_median[:, i, shock_idx], color=COLORS['bayes'], linewidth=2.5,
            label='BVAR median')
    
    # OLS point estimate
    ax.plot(horizons, IRF_ols[:, i, shock_idx], color=COLORS['ols'], linewidth=2.5,
            linestyle='--', label='OLS')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Quarters after shock')
    ax.set_ylabel('Percentage points')
    ax.set_title(f'Response of {label}')
    ax.set_xlim(0, H)
    ax.legend(loc='upper right', fontsize=8)

fig.suptitle('Comparison: Bayesian VAR (Minnesota Prior) vs OLS\n'
             '(U.S. Data 1970-2007, Cholesky Identification)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figure2_bvar_vs_ols.png', dpi=150, bbox_inches='tight')
plt.show()

# --- FIGURE 3: Prior Sensitivity Analysis ---
print("   Figure 3: Prior sensitivity analysis...")

# Estimate BVAR with different lambda1 values
lambda_values = [0.05, 0.1, 0.2, 0.5, 1.0]
IRF_sensitivity = {}

for lam in lambda_values:
    B_temp, S_temp, nu_temp, V_temp, _ = estimate_bvar_minnesota(Y, p, lambda1=lam)
    Sigma_temp = S_temp / (nu_temp - K - 1)
    IRF_sensitivity[lam] = compute_irf_cholesky(B_temp, Sigma_temp, p, K, H)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
cmap = plt.cm.viridis
colors_sens = [cmap(i / len(lambda_values)) for i in range(len(lambda_values))]

for i, (ax, label) in enumerate(zip(axes, var_labels)):
    for j, (lam, irf) in enumerate(IRF_sensitivity.items()):
        ax.plot(range(H + 1), irf[:, i, shock_idx], color=colors_sens[j],
                linewidth=2, label=f'λ₁={lam}')
    
    # OLS for reference
    ax.plot(range(H + 1), IRF_ols[:, i, shock_idx], 'k--', linewidth=1.5,
            label='OLS', alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Quarters after shock')
    ax.set_ylabel('Percentage points')
    ax.set_title(f'Response of {label}')
    ax.set_xlim(0, H)
    ax.legend(loc='upper right', fontsize=7)

fig.suptitle('Prior Sensitivity: Effect of Overall Tightness (λ₁) on IRFs\n'
             '(Smaller λ₁ = More Shrinkage Toward Random Walk)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figure3_bvar_sensitivity.png', dpi=150, bbox_inches='tight')
plt.show()

# --- FIGURE 4: Posterior Distribution of Impact Effects ---
print("   Figure 4: Posterior distributions of impact effects...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for i, (ax, label, color) in enumerate(zip(axes, var_labels, colors_var)):
    impact_draws = IRF_draws[:, 0, i, shock_idx]
    
    ax.hist(impact_draws, bins=50, density=True, alpha=0.7, color=color,
            edgecolor='white', linewidth=0.5)
    
    # Posterior mean and median
    post_mean = np.mean(impact_draws)
    post_median = np.median(impact_draws)
    
    ax.axvline(post_mean, color='darkred', linestyle='-', linewidth=2,
               label=f'Mean: {post_mean:.4f}')
    ax.axvline(post_median, color='darkblue', linestyle='--', linewidth=2,
               label=f'Median: {post_median:.4f}')
    
    # OLS estimate
    ax.axvline(IRF_ols[0, i, shock_idx], color='black', linestyle=':',
               linewidth=2, label=f'OLS: {IRF_ols[0, i, shock_idx]:.4f}')
    
    ax.set_xlabel('Impact effect (h=0)')
    ax.set_ylabel('Posterior density')
    ax.set_title(f'{label}')
    ax.legend(loc='upper right', fontsize=8)

fig.suptitle('Posterior Distribution of Impact Effects (h=0)\n'
             '(Response to Monetary Policy Shock)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figure4_bvar_posterior.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n   ✓ All figures saved!")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY RESULTS")
print("=" * 70)

print("\n--- Coefficient Shrinkage (BVAR vs OLS) ---")
print(f"{'Coefficient':<25} {'OLS':>12} {'BVAR':>12} {'Shrinkage':>12}")
print("-" * 61)

# Compare key coefficients
for i, var in enumerate(var_names):
    for lag in [1, 2]:
        ols_c = B_ols[1 + (lag-1)*K + i, i]
        bvar_c = B_bvar[1 + (lag-1)*K + i, i]
        shrink = abs(bvar_c - ols_c) / abs(ols_c) * 100 if abs(ols_c) > 0.01 else 0
        print(f"{var} lag {lag:<17} {ols_c:>12.4f} {bvar_c:>12.4f} {shrink:>11.1f}%")

print("\n--- Impact Effects of MP Shock (h=0) ---")
print(f"{'Variable':<20} {'OLS':>12} {'BVAR Median':>12} {'68% CI':>20}")
print("-" * 64)
for i, label in enumerate(var_labels):
    ols_impact = IRF_ols[0, i, shock_idx]
    bvar_impact = IRF_median[0, i, shock_idx]
    ci_low = IRF_lower_68[0, i, shock_idx]
    ci_high = IRF_upper_68[0, i, shock_idx]
    print(f"{label:<20} {ols_impact:>12.4f} {bvar_impact:>12.4f} [{ci_low:>7.4f}, {ci_high:>7.4f}]")

print("\n--- Peak Effects of MP Shock ---")
for i, label in enumerate(var_labels):
    # Find peak (trough for GDP/inflation, peak for Fed Funds)
    if i < 2:
        peak_idx = np.argmin(IRF_median[:, i, shock_idx])
    else:
        peak_idx = np.argmax(IRF_median[:20, i, shock_idx])
    
    peak_val = IRF_median[peak_idx, i, shock_idx]
    ci_low = IRF_lower_68[peak_idx, i, shock_idx]
    ci_high = IRF_upper_68[peak_idx, i, shock_idx]
    
    print(f"{label}: Peak at h={peak_idx}, value = {peak_val:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

print("\n--- Price Puzzle Check ---")
inflation_irf_median = IRF_median[:20, 1, shock_idx]
puzzle_quarters = np.where(inflation_irf_median > 0)[0]
if len(puzzle_quarters) > 0:
    print(f"Price puzzle in posterior median: quarters {puzzle_quarters}")
else:
    print("No price puzzle in posterior median")

# Check probability of positive inflation response
prob_positive = np.mean(IRF_draws[:, :20, 1, shock_idx] > 0, axis=0)
max_prob = np.max(prob_positive)
max_prob_h = np.argmax(prob_positive)
print(f"Max probability of positive inflation response: {max_prob:.2%} at h={max_prob_h}")

print("\n" + "=" * 70)
print("FILES GENERATED:")
print("=" * 70)
print("   figure1_bvar_irf.png         - Bayesian IRFs with credible intervals")
print("   figure2_bvar_vs_ols.png      - Comparison BVAR vs OLS")
print("   figure3_bvar_sensitivity.png - Prior sensitivity analysis")
print("   figure4_bvar_posterior.png   - Posterior distributions")
