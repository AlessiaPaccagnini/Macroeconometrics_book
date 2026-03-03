"""
================================================================================
EXAMPLE 2: PROXY SVAR (EXTERNAL INSTRUMENTS)
Complete Analysis with IRFs and Comparison to Cholesky
================================================================================
Data: U.S. FRED (GDPC1, GDPDEF, FEDFUNDS)
Instrument: Romer-Romer (2004) shocks, updated by Wieland-Yang (2020)
Sample: 1970:Q1 - 2007:Q4

Author: Alessia Paccagnini
Textbook: Macroeconometrics

NOTE: This script is designed to run in Google Colab.
      It will prompt you to upload the required Excel files.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# GOOGLE COLAB: UPLOAD DATA FILES
# =============================================================================
# Upload the following 4 files when prompted:
#   - GDPC1.xlsx
#   - GDPDEF.xlsx
#   - FEDFUNDS.xlsx
#   - RR_monetary_shock_quarterly.xlsx

from google.colab import files
print("Please upload the 4 data files: GDPC1.xlsx, GDPDEF.xlsx, FEDFUNDS.xlsx, RR_monetary_shock_quarterly.xlsx")
uploaded = files.upload()
print(f"\nUploaded {len(uploaded)} file(s): {list(uploaded.keys())}")

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
    'cholesky': '#E94F37',
    'proxy': '#2E86AB',
}

print("=" * 70)
print("EXAMPLE 2: PROXY SVAR (EXTERNAL INSTRUMENTS)")
print("Using Romer-Romer (2004) Monetary Policy Shocks")
print("=" * 70)

# =============================================================================
# SECTION 1: LOAD FRED DATA
# =============================================================================
print("\n[1/7] Loading FRED data...")

# Load GDP
gdp_df = pd.read_excel('GDPC1.xlsx', sheet_name='Quarterly')
gdp_df['observation_date'] = pd.to_datetime(gdp_df['observation_date'])
gdp_df['GDPC1'] = pd.to_numeric(gdp_df['GDPC1'], errors='coerce')
gdp_df = gdp_df.set_index('observation_date')

# Load Deflator
deflator_df = pd.read_excel('GDPDEF.xlsx', sheet_name='Quarterly')
deflator_df['observation_date'] = pd.to_datetime(deflator_df['observation_date'])
deflator_df['GDPDEF'] = pd.to_numeric(deflator_df['GDPDEF'], errors='coerce')
deflator_df = deflator_df.set_index('observation_date')

# Load Fed Funds
fedfunds_df = pd.read_excel('FEDFUNDS.xlsx', sheet_name='Monthly')
fedfunds_df['observation_date'] = pd.to_datetime(fedfunds_df['observation_date'])
fedfunds_df['FEDFUNDS'] = pd.to_numeric(fedfunds_df['FEDFUNDS'], errors='coerce')
fedfunds_df = fedfunds_df.set_index('observation_date')

# Convert to quarterly
fedfunds_q = fedfunds_df.resample('QE').mean()

# Compute growth rates
gdp_growth = 400 * np.log(gdp_df['GDPC1'] / gdp_df['GDPC1'].shift(1))
inflation = 400 * np.log(deflator_df['GDPDEF'] / deflator_df['GDPDEF'].shift(1))

# Align dates
gdp_growth.index = gdp_growth.index + pd.offsets.QuarterEnd(0)
inflation.index = inflation.index + pd.offsets.QuarterEnd(0)

print("   ✓ FRED data loaded")

# =============================================================================
# SECTION 2: LOAD ROMER-ROMER SHOCKS
# =============================================================================
print("\n[2/7] Loading Romer-Romer monetary policy shocks...")

# Load Romer-Romer shocks (Wieland-Yang 2020 update)
rr = pd.read_excel('RR_monetary_shock_quarterly.xlsx')
rr['date'] = pd.to_datetime(rr['date']) + pd.offsets.QuarterEnd(0)
rr = rr.set_index('date')

print(f"   Source: Romer & Romer (2004), updated by Wieland & Yang (2020)")
print(f"   Available: {rr.index[0]} to {rr.index[-1]}")
print(f"   Observations: {len(rr)}")

# =============================================================================
# SECTION 3: MERGE DATA
# =============================================================================
print("\n[3/7] Merging datasets...")

data = pd.DataFrame({
    'gdp_growth': gdp_growth,
    'inflation': inflation,
    'fedfunds': fedfunds_q['FEDFUNDS'],
    'rr_shock': rr['resid_romer']
}).dropna()

# Sample: 1970-2007 (R-R shock availability and pre-ZLB)
data = data.loc['1970-01-01':'2007-12-31']

print(f"   ✓ Final sample: {data.index[0]} to {data.index[-1]}")
print(f"   Observations: {len(data)}")

# =============================================================================
# SECTION 4: VAR ESTIMATION
# =============================================================================
print("\n[4/7] Estimating VAR(4)...")

def estimate_var(Y, p):
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
    return T_eff, B_hat, u, Sigma_u, X

var_names = ['gdp_growth', 'inflation', 'fedfunds']
Y = data[var_names].values
dates = data.index

p = 4
T_eff, B_hat, u, Sigma_u, X_var = estimate_var(Y, p)
K = len(var_names)

print(f"   VAR({p}) estimated")
print(f"   Effective sample: {T_eff} observations")

# =============================================================================
# SECTION 5: IDENTIFICATION
# =============================================================================
print("\n[5/7] Structural Identification...")

# --- Cholesky (for comparison) ---
P_chol = cholesky(Sigma_u, lower=True)

print("\n   A) CHOLESKY IDENTIFICATION")
print("   " + "-" * 50)
chol_df = pd.DataFrame(P_chol,
                        index=['GDP', 'Inflation', 'FedFunds'],
                        columns=['eps_GDP', 'eps_pi', 'eps_MP'])
print(chol_df.round(4).to_string())

# --- Proxy SVAR ---
print("\n   B) PROXY SVAR IDENTIFICATION")
print("   " + "-" * 50)
print("   Instrument: Romer-Romer (2004) narrative shocks")

# Align instrument with VAR residuals
z = data['rr_shock'].values[p:]

def proxy_svar_identification(u, z):
    """
    Proxy SVAR identification using external instrument

    The key insight: if z is correlated with the MP shock but not other shocks,
    then we can identify the structural impact coefficients from:

    b_{-mp} / b_mp = Cov(u_{-mp}, z) / Cov(u_mp, z)

    Normalizing b_mp = 1, we get: b_{-mp} = Cov(u_{-mp}, z) / Cov(u_mp, z)
    """
    K = u.shape[1]
    min_len = min(len(u), len(z))
    u, z = u[:min_len], z[:min_len]

    # Remove NaN
    valid = ~np.isnan(z)
    u, z = u[valid], z[valid]
    n = len(z)

    # Compute covariances
    cov_uz = np.array([np.cov(u[:, i], z)[0, 1] for i in range(K)])
    cov_mp_z = cov_uz[-1]  # Cov(u_mp, z)

    # Impact coefficients (normalized so b_mp = 1)
    b = cov_uz / cov_mp_z

    # First-stage regression: u_mp = alpha + beta*z + error
    X_fs = np.column_stack([np.ones(n), z])
    beta_fs = np.linalg.lstsq(X_fs, u[:, -1], rcond=None)[0]
    resid_fs = u[:, -1] - X_fs @ beta_fs

    # F-statistic
    TSS = np.sum((u[:, -1] - np.mean(u[:, -1]))**2)
    RSS = np.sum(resid_fs**2)
    R2 = 1 - RSS / TSS
    F_stat = (R2 / 1) / ((1 - R2) / (n - 2))

    # Bootstrap standard errors
    n_boot = 500
    b_boot = np.zeros((n_boot, K))
    for i in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        cov_uz_b = np.array([np.cov(u[idx, j], z[idx])[0, 1] for j in range(K)])
        if np.abs(cov_uz_b[-1]) > 1e-10:
            b_boot[i] = cov_uz_b / cov_uz_b[-1]
        else:
            b_boot[i] = np.nan
    se = np.nanstd(b_boot, axis=0)

    return b, se, n, F_stat, R2

np.random.seed(42)
b_proxy, se_proxy, n_valid, F_stat, R2 = proxy_svar_identification(u, z)

print(f"\n   First-stage diagnostics:")
print(f"   F-statistic: {F_stat:.1f}")
print(f"   R-squared: {R2:.4f}")
print(f"   Observations: {n_valid}")

if F_stat > 10:
    print("   ✓ Strong instrument (F > 10)")
else:
    print("   ⚠ Weak instrument (F < 10)")

print(f"\n   Structural impact multipliers:")
print(f"   {'Variable':<20} {'Estimate':>12} {'Std. Error':>12} {'t-stat':>10}")
print("   " + "-" * 54)
for i, var in enumerate(var_names):
    t_stat = b_proxy[i] / se_proxy[i] if se_proxy[i] > 0 else np.nan
    print(f"   {var:<20} {b_proxy[i]:>12.4f} {se_proxy[i]:>12.4f} {t_stat:>10.2f}")

# =============================================================================
# SECTION 6: COMPUTE IRFs
# =============================================================================
print("\n\n[6/7] Computing IRFs...")

H = 40

def compute_ma_coefficients(B_hat, p, K, H):
    B = B_hat[1:].reshape(p, K, K).transpose(0, 2, 1)
    Phi = np.zeros((H + 1, K, K))
    Phi[0] = np.eye(K)
    for h in range(1, H + 1):
        for j in range(min(h, p)):
            Phi[h] += B[j] @ Phi[h - j - 1]
    return Phi

def compute_irfs_cholesky(B_hat, P, p, K, H):
    Phi = compute_ma_coefficients(B_hat, p, K, H)
    IRF = np.zeros((H + 1, K, K))
    for h in range(H + 1):
        IRF[h] = Phi[h] @ P
    return IRF

def compute_irfs_proxy(B_hat, b, p, K, H):
    """
    Compute IRFs for Proxy SVAR

    Unlike Cholesky which identifies all K shocks,
    Proxy SVAR only identifies ONE shock (the MP shock).

    IRF_h = Phi_h @ b

    where b is the structural impact vector
    """
    Phi = compute_ma_coefficients(B_hat, p, K, H)
    IRF = np.zeros((H + 1, K))
    for h in range(H + 1):
        IRF[h] = Phi[h] @ b
    return IRF

# Cholesky IRFs
IRF_chol = compute_irfs_cholesky(B_hat, P_chol, p, K, H)
irf_chol_mp = IRF_chol[:, :, 2]  # MP shock

# Proxy SVAR IRFs
irf_proxy = compute_irfs_proxy(B_hat, b_proxy, p, K, H)

# Bootstrap CIs
print("   Computing bootstrap confidence intervals...")

def bootstrap_irfs_proxy(Y, z_full, p, H, n_boot=2000, ci=0.90):
    """
    Wild bootstrap for Proxy SVAR IRFs.
    Following Mertens and Ravn (2013) and Gertler and Karadi (2015),
    as described in Section 4.18.11 of the textbook.

    Key: the SAME Mammen weights multiply both residuals and instrument,
    preserving their covariance structure.

    Returns: (median, lower, upper)
      - median: bootstrap median IRF (plotted as the point estimate)
      - lower/upper: percentile confidence bands
      The median is guaranteed to lie inside the bands.
    """
    T, K = Y.shape

    # Estimate baseline VAR
    T_eff_base, B_base, u_base, _, X_base = estimate_var(Y, p)

    # Align instrument with residuals
    z_aligned = z_full[p:]
    min_len = min(len(u_base), len(z_aligned))
    z_trim = z_aligned[:min_len]

    irfs_boot = np.zeros((n_boot, H + 1, K))
    n_failures = 0

    for b in range(n_boot):
        try:
            # Mammen (1993) two-point distribution
            p_mammen = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
            w = np.where(
                np.random.uniform(size=T_eff_base) < p_mammen,
                -(np.sqrt(5) - 1) / 2,
                 (np.sqrt(5) + 1) / 2
            )

            # Wild bootstrap residuals (same weights for u and z!)
            u_star = u_base * w[:, np.newaxis]

            # Reconstruct data recursively
            Y_star = np.zeros_like(Y)
            Y_star[:p] = Y[:p]
            for t in range(T_eff_base):
                Y_star[p + t] = X_base[t] @ B_base + u_star[t]

            # Re-estimate VAR
            _, B_b, u_b, _, _ = estimate_var(Y_star, p)

            # Wild bootstrap instrument (same weights)
            w_z = w[:min_len]
            z_star = z_trim * w_z

            # Re-identify
            b_est, _, _, _, _ = proxy_svar_identification(u_b, z_star)

            # Bootstrap IRFs
            irfs_boot[b] = compute_irfs_proxy(B_b, b_est, p, K, H)
        except:
            irfs_boot[b] = np.nan
            n_failures += 1

    print(f"   Wild bootstrap: {n_boot} replications, {n_failures} failures")

    alpha = (1 - ci) / 2
    median = np.nanmedian(irfs_boot, axis=0)
    lower  = np.nanpercentile(irfs_boot, alpha * 100, axis=0)
    upper  = np.nanpercentile(irfs_boot, (1 - alpha) * 100, axis=0)

    return median, lower, upper

def bootstrap_irfs_cholesky(Y, p, H, n_boot=500, ci=0.90):
    """Residual bootstrap for Cholesky IRFs. Returns (median, lower, upper)."""
    T, K = Y.shape
    irfs_boot = np.zeros((n_boot, H + 1, K))
    _, B_orig, u_orig, _, X_orig = estimate_var(Y, p)
    T_eff = len(u_orig)

    for b in range(n_boot):
        idx = np.random.choice(T_eff, T_eff, replace=True)
        u_boot = u_orig[idx]
        Y_boot = np.zeros_like(Y)
        Y_boot[:p] = Y[:p]
        Y_boot[p:] = X_orig @ B_orig + u_boot

        try:
            _, B_b, _, Sigma_b, _ = estimate_var(Y_boot, p)
            P_b = cholesky(Sigma_b, lower=True)
            IRF_b = compute_irfs_cholesky(B_b, P_b, p, K, H)
            irfs_boot[b] = IRF_b[:, :, 2]
        except:
            irfs_boot[b] = np.nan

    alpha = (1 - ci) / 2
    median = np.nanmedian(irfs_boot, axis=0)
    lower  = np.nanpercentile(irfs_boot, alpha * 100, axis=0)
    upper  = np.nanpercentile(irfs_boot, (1 - alpha) * 100, axis=0)
    return median, lower, upper

np.random.seed(42)
print("   Proxy SVAR: wild bootstrap (Mammen weights, 2000 reps, 90% CI)...")
irf_median_proxy, irf_lower_proxy, irf_upper_proxy = bootstrap_irfs_proxy(
    Y, data['rr_shock'].values, p, H, n_boot=2000, ci=0.90)
print("   Cholesky: residual bootstrap (500 reps, 90% CI)...")
irf_median_chol, irf_lower_chol, irf_upper_chol = bootstrap_irfs_cholesky(
    Y, p, H, n_boot=500, ci=0.90)

print("   ✓ IRFs computed")

# =============================================================================
# SECTION 7: GENERATE FIGURES
# =============================================================================
print("\n\n[7/7] Generating figures...")

var_labels = ['GDP Growth', 'Inflation', 'Federal Funds Rate']

# --- FIGURE 1: Proxy SVAR IRFs ---
print("   Figure 1: Proxy SVAR IRFs...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for i, (ax, label) in enumerate(zip(axes, var_labels)):
    ax.plot(range(H + 1), irf_proxy[:, i], color=COLORS['proxy'], linewidth=2.5)
    ax.fill_between(range(H + 1), irf_lower_proxy[:, i], irf_upper_proxy[:, i],
                    alpha=0.25, color=COLORS['proxy'])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Quarters after shock')
    ax.set_ylabel('Percentage points')
    ax.set_title(f'Response of {label}')
    ax.set_xlim(0, H)

fig.suptitle('Impulse Responses to Monetary Policy Shock\n(Proxy SVAR, Romer-Romer Instrument, 90% Wild Bootstrap CI, U.S. 1970-2007)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figure1_proxy_irf.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: figure1_proxy_irf.png")

# --- FIGURE 2: Proxy vs Cholesky Comparison ---
print("   Figure 2: Proxy SVAR vs Cholesky comparison...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for i, (ax, label) in enumerate(zip(axes, var_labels)):
    # Proxy SVAR
    ax.plot(range(H + 1), irf_proxy[:, i], color=COLORS['proxy'],
            linewidth=2.5, label='Proxy SVAR (R-R)')
    ax.fill_between(range(H + 1), irf_lower_proxy[:, i], irf_upper_proxy[:, i],
                    alpha=0.2, color=COLORS['proxy'])

    # Cholesky
    ax.plot(range(H + 1), irf_chol_mp[:, i], color=COLORS['cholesky'],
            linewidth=2.5, linestyle='--', label='Cholesky')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Quarters after shock')
    ax.set_ylabel('Percentage points')
    ax.set_title(f'Response of {label}')
    ax.set_xlim(0, H)
    ax.legend(loc='upper right')

fig.suptitle('Monetary Policy Shock: Proxy SVAR vs Cholesky\n(90% Bootstrap CI, U.S. Data 1970-2007)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figure2_proxy_vs_cholesky.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: figure2_proxy_vs_cholesky.png")

print("\n   ✓ All figures displayed!")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY RESULTS")
print("=" * 70)

print("\n--- First-Stage Diagnostics ---")
print(f"F-statistic: {F_stat:.1f}")
print(f"R-squared: {R2:.4f}")
print(f"Instrument: Romer-Romer (2004) narrative shocks")

print("\n--- Impact Multipliers Comparison ---")
print(f"{'Variable':<25} {'Cholesky':>12} {'Proxy SVAR':>12}")
print("-" * 50)
for i, label in enumerate(var_labels):
    print(f"{label:<25} {irf_chol_mp[0, i]:>12.4f} {irf_proxy[0, i]:>12.4f}")

print("\n--- Price Puzzle Check ---")
print("Cholesky:")
puzzle_chol = np.where(irf_chol_mp[:20, 1] > 0)[0]
print(f"  Quarters with inflation > 0: {list(puzzle_chol) if len(puzzle_chol) > 0 else 'None'}")

print("Proxy SVAR:")
puzzle_proxy = np.where(irf_proxy[:20, 1] > 0)[0]
print(f"  Quarters with inflation > 0: {list(puzzle_proxy) if len(puzzle_proxy) > 0 else 'None'}")

print("\n--- Comparison with Textbook Table 4.5 ---")
print(f"{'Variable':<25} {'Book Chol':>10} {'Book Proxy':>11} {'Code Chol':>10} {'Code Proxy':>11}")
print("-" * 68)
book_chol = [0.00, 0.00, 0.90]
book_proxy = [0.78, 0.19, 1.00]
for i, label in enumerate(var_labels):
    print(f"{label:<25} {book_chol[i]:>10.2f} {book_proxy[i]:>11.2f} {irf_chol_mp[0, i]:>10.4f} {irf_proxy[0, i]:>11.4f}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

# Download figures to your local machine
from google.colab import files
files.download('figure1_proxy_irf.png')
files.download('figure2_proxy_vs_cholesky.png')
