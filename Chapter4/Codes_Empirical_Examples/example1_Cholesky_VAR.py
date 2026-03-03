"""
COLAB VERSION - FIXED for VISIBLE PLOTS
Section 4.18.1 - All figures with white background
Author: Alessia Paccagnini
Textbook: Macroeconometrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, inv
import warnings
warnings.filterwarnings('ignore')

try:
    import google.colab
    from google.colab import files
    uploaded = files.upload()
    IN_COLAB = True
except:
    IN_COLAB = False

# CRITICAL: Force white background BEFORE any plotting
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'figure.dpi': 100,
    'savefig.dpi': 150,
})

plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'gdp': '#2E86AB',
    'oil': '#06A77D',  # GREEN!
    'inflation': '#A23B72',
    'rate': '#F18F01',
}

print("="*70)
print("4-VARIABLE CHOLESKY - VISIBLE PLOTS VERSION")
print("="*70)

# Load data
print("\n[1/7] Loading data...")
df_raw = pd.read_excel('2026-01-QD.xlsx')
df_raw = df_raw.iloc[2:].copy()
df_raw['sasdate'] = pd.to_datetime(df_raw['sasdate'])
df_raw = df_raw.set_index('sasdate')

for var in ['GDPC1', 'GDPCTPI', 'FEDFUNDS', 'OILPRICEx']:
    df_raw[var] = pd.to_numeric(df_raw[var], errors='coerce')

df = pd.DataFrame({
    'gdp_growth': 100 * np.log(df_raw['GDPC1'] / df_raw['GDPC1'].shift(1)),
    'oil_growth': 100 * np.log(df_raw['OILPRICEx'] / df_raw['OILPRICEx'].shift(1)),
    'inflation': 100 * np.log(df_raw['GDPCTPI'] / df_raw['GDPCTPI'].shift(1)),
    'fedfunds': df_raw['FEDFUNDS']
})

df = df['1971-01-01':'2007-12-31'].dropna()
print(f"   Sample: {len(df)} observations")

# VAR
print("\n[2/7] Estimating VAR(4)...")
p, K, H = 4, 4, 20
Y = df.values
T = len(Y)

Y_lagged = [Y[p-lag:T-lag] for lag in range(1, p+1)]
X = np.column_stack([np.ones(T-p)] + Y_lagged)
Y_est = Y[p:]

B_hat = inv(X.T @ X) @ (X.T @ Y_est)
U = Y_est - X @ B_hat
Sigma_u = (U.T @ U) / (T - p - K*p - 1)

# Cholesky
print("\n[3/7] Cholesky...")
P = cholesky(Sigma_u, lower=True)
mp_shock = P[3,3]
print(f"   MP shock: {mp_shock:.2f} pp")

# IRF
print("\n[4/7] Computing IRFs...")
def compute_irf(B, P, H, K, p):
    F = np.zeros((K*p, K*p))
    F[:K, :] = B[1:, :].T
    if p > 1:
        F[K:, :K*(p-1)] = np.eye(K*(p-1))
    J = np.hstack([np.eye(K), np.zeros((K, K*(p-1)))])
    IRF = np.zeros((H+1, K, K))
    IRF[0] = P
    Fh = np.eye(K*p)
    for h in range(1, H+1):
        Fh = Fh @ F
        IRF[h] = J @ Fh @ J.T @ P
    return IRF

IRF = compute_irf(B_hat, P, H, K, p)
irf_mp = IRF[:, :, 3]

# Bootstrap
print("\n[5/7] Bootstrap (300 reps)...")
B_sim = 300
IRF_boot = np.zeros((B_sim, H+1, K, K))
U_ctr = U - U.mean(axis=0)
np.random.seed(42)

for b in range(B_sim):
    if (b+1) % 100 == 0:
        print(f"   {b+1}/{B_sim}")
    idx = np.random.randint(0, len(U_ctr), len(U_ctr))
    U_star = U_ctr[idx]
    Y_star = np.zeros((T, K))
    Y_star[:p] = Y[:p]
    for t in range(p, T):
        lags = Y_star[t-p:t][::-1].flatten()
        X_t = np.concatenate([[1], lags])
        Y_star[t] = X_t @ B_hat + U_star[t-p]
    Y_boot = Y_star[p:]
    Y_lag_boot = [Y_star[p-lag:T-lag] for lag in range(1, p+1)]
    X_boot = np.column_stack([np.ones(T-p)] + Y_lag_boot)
    B_boot = inv(X_boot.T @ X_boot) @ (X_boot.T @ Y_boot)
    U_boot = Y_boot - X_boot @ B_boot
    Sig_boot = (U_boot.T @ U_boot) / (T - p - K*p - 1)
    try:
        P_boot = cholesky(Sig_boot, lower=True)
        IRF_boot[b] = compute_irf(B_boot, P_boot, H, K, p)
    except:
        IRF_boot[b] = IRF

bias = IRF_boot.mean(axis=0) - IRF
IRF_ctr = IRF_boot - bias
CI_68 = np.percentile(IRF_ctr, [16, 84], axis=0)
CI_90 = np.percentile(IRF_ctr, [5, 95], axis=0)

# FEVD
print("\n[6/7] Computing FEVD...")
def compute_fevd(IRF, H, K):
    FEVD = np.zeros((H+1, K, K))
    for h in range(H+1):
        mse = np.zeros((K, K))
        for j in range(h+1):
            mse += IRF[j] @ IRF[j].T
        for i in range(K):
            for j in range(K):
                shock_contrib = sum(IRF[s][i, j]**2 for s in range(h+1))
                FEVD[h, i, j] = shock_contrib / mse[i, i] if mse[i, i] > 0 else 0
    return FEVD

FEVD = compute_fevd(IRF, H, K)

# Historical Decomposition
print("\n[7/7] Historical decomposition...")
def historical_decomposition(Y, B_hat, P, p):
    T, K = Y.shape
    Y_lagged_all = [Y[p-lag:T-lag] for lag in range(1, p+1)]
    X_all = np.column_stack([np.ones(T-p)] + Y_lagged_all)
    Y_fitted = X_all @ B_hat
    U_all = Y[p:] - Y_fitted
    eps = U_all @ inv(P).T
    HD = np.zeros((T-p, K, K))
    F = np.zeros((K*p, K*p))
    F[:K, :] = B_hat[1:, :].T
    if p > 1:
        F[K:, :K*(p-1)] = np.eye(K*(p-1))
    J = np.hstack([np.eye(K), np.zeros((K, K*(p-1)))])
    for t in range(T-p):
        for shock in range(K):
            contrib = np.zeros(K)
            for s in range(t+1):
                Fs = np.linalg.matrix_power(F, t-s)
                impact = J @ Fs @ J.T @ P[:, shock]
                contrib += impact * eps[s, shock]
            HD[t, :, shock] = contrib
    return HD, eps

HD, eps = historical_decomposition(Y, B_hat, P, p)

# PLOT 1: IRFs
print("\nCreating figures...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor='white')
names = ['GDP Growth', 'Oil Price Growth', 'Inflation', 'Federal Funds']
colors = [COLORS['gdp'], COLORS['oil'], COLORS['inflation'], COLORS['rate']]
h = np.arange(H+1)

for i, (ax, nm, col) in enumerate(zip(axes, names, colors)):
    ax.set_facecolor('white')
    ax.plot(h, irf_mp[:,i], color=col, linewidth=2.5, label='IRF', zorder=3)
    ax.fill_between(h, CI_68[0,:,i,3], CI_68[1,:,i,3], alpha=0.3, color=col, label='68%', zorder=2)
    ax.fill_between(h, CI_90[0,:,i,3], CI_90[1,:,i,3], alpha=0.15, color=col, label='90%', zorder=1)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Quarters')
    ax.set_title(nm, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.set_ylabel('Percentage Points')
    if i == 3:
        ax.legend(fontsize=8)

plt.suptitle('Impulse Responses to Monetary Policy Shock', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('IRFs_MP_shock.png', dpi=150, facecolor='white', edgecolor='none', bbox_inches='tight')
plt.show()
plt.close()
print("✓ IRFs_MP_shock.png")

# PLOT 2: Cumulative
irf_cum = np.cumsum(irf_mp, axis=0)
CI_68_cum = np.cumsum(CI_68, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='white')
for ax in axes:
    ax.set_facecolor('white')

axes[0].plot(h, irf_cum[:,0], color=COLORS['gdp'], linewidth=2.5)
axes[0].fill_between(h, CI_68_cum[0,:,0,3], CI_68_cum[1,:,0,3], alpha=0.3, color=COLORS['gdp'])
axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[0].set_title('GDP Level Response', fontweight='bold')
axes[0].set_xlabel('Quarters')
axes[0].set_ylabel('Percentage Points')
axes[0].grid(True, alpha=0.3)

axes[1].plot(h, irf_cum[:,2], color=COLORS['inflation'], linewidth=2.5)
axes[1].fill_between(h, CI_68_cum[0,:,2,3], CI_68_cum[1,:,2,3], alpha=0.3, color=COLORS['inflation'])
axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[1].set_title('Price Level Response', fontweight='bold')
axes[1].set_xlabel('Quarters')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Cumulative Responses: GDP and Price Level Effects', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('Cumulative_responses.png', dpi=150, facecolor='white', edgecolor='none', bbox_inches='tight')
plt.show()
plt.close()
print("✓ Cumulative_responses.png")

# PLOT 3: FEVD
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
axes = axes.flatten()
var_names = ['GDP Growth', 'Oil Price Growth', 'Inflation', 'Federal Funds']
shock_names = ['GDP Shock', 'Oil Shock', 'Inflation Shock', 'MP Shock']
colors_fevd = [COLORS['gdp'], COLORS['oil'], COLORS['inflation'], COLORS['rate']]

for i, (ax, var_name) in enumerate(zip(axes, var_names)):
    ax.set_facecolor('white')
    bottom = np.zeros(H+1)
    for j in range(K):
        ax.fill_between(h, bottom, bottom + FEVD[:, i, j]*100, label=shock_names[j], color=colors_fevd[j], alpha=0.8)
        bottom += FEVD[:, i, j]*100
    ax.set_xlabel('Quarters')
    ax.set_ylabel('Percent')
    ax.set_title(var_name, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=9)

plt.suptitle('Forecast Error Variance Decomposition', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('FEVD.png', dpi=150, facecolor='white', edgecolor='none', bbox_inches='tight')
plt.show()
plt.close()
print("✓ FEVD.png")

# PLOT 4: Historical Decomposition
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
axes = axes.flatten()
dates_hd = df.index[p:]

for i, (ax, var_name) in enumerate(zip(axes, var_names)):
    ax.set_facecolor('white')
    bottom = np.zeros(len(dates_hd))
    bottom_neg = np.zeros(len(dates_hd))
    for j in range(K):
        values = HD[:, i, j]
        pos_vals = np.maximum(values, 0)
        neg_vals = np.minimum(values, 0)
        ax.fill_between(dates_hd, bottom, bottom + pos_vals, label=shock_names[j], color=colors_fevd[j], alpha=0.7)
        bottom += pos_vals
        ax.fill_between(dates_hd, bottom_neg, bottom_neg + neg_vals, color=colors_fevd[j], alpha=0.7)
        bottom_neg += neg_vals
    actual_demeaned = Y[p:, i] - Y[p:, i].mean()
    ax.plot(dates_hd, actual_demeaned, 'k-', linewidth=1.5, label='Actual', alpha=0.8, zorder=10)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.set_ylabel('Percentage Points')
    ax.set_title(var_name, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=8)
    ax.tick_params(axis='x', rotation=45, labelsize=9)

plt.suptitle('Historical Decomposition', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Historical_decomposition.png', dpi=150, facecolor='white', edgecolor='none', bbox_inches='tight')
plt.show()
plt.close()
print("✓ Historical_decomposition.png")

# Summary
print("\n" + "="*70)
print("✅ ALL 4 FIGURES SAVED - WITH WHITE BACKGROUNDS!")
print("="*70)
print("\nFiles created with visible plots:")
print("  1. IRFs_MP_shock.png")
print("  2. Cumulative_responses.png")
print("  3. FEVD.png")
print("  4. Historical_decomposition.png")
if IN_COLAB:
    print("\n📥 Download from file browser!")
print("="*70)
