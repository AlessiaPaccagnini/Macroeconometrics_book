"""
=============================================================================
VAR EMPIRICAL EXAMPLES FOR MACROECONOMETRICS TEXTBOOK
Chapter: VAR Models
=============================================================================

Textbook: Macroeconometrics
Author: Alessia Paccagnini


Data Source: FRED St. Louis McCracken & Ng Dataset (2026-01-QD.xlsx)

This script provides three comprehensive examples using real quarterly US data:
1. VAR Estimation with Information Criteria, Residual Diagnostics, and Granger Causality
2. VECM Example with Cointegration Testing (Consumption-Income)
3. Reduced-Form Impulse Response Functions (motivating the need for identification)

Period: 1960:Q2 to 2026:Q1 (264 observations for VAR, 265 for cointegration)
Updated: March 3, 2026

IMPORTANT — Data loading:
  The Excel file has: row 0 = column names, rows 1-2 = metadata, rows 3-4 = 1959 Q1-Q2.
  We start from iloc[5] (= 1959:Q3). After computing quarterly growth rates and dropping
  the first NaN, the VAR sample begins at 1960:Q2.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Remove this line if running interactively
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR, VECM
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.figsize': (12, 8), 'font.size': 10, 'figure.dpi': 150})

# ============================================================================
# FILE PATH — adjust for your environment
# ============================================================================

# --- Option A: Google Colab (uncomment these 3 lines) ---
# from google.colab import files
# uploaded = files.upload()  # This will prompt you to select the file
# file_path = list(uploaded.keys())[0]

# --- Option B: Local / other environment ---
file_path = "2026-01-QD.xlsx"

# ============================================================================
# LOAD DATA
# ============================================================================
df = pd.read_excel(file_path, sheet_name='in', header=None)
colnames = df.iloc[0].values

gdp_idx    = np.where(colnames == 'GDPC1')[0][0]
cpi_idx    = np.where(colnames == 'CPIAUCSL')[0][0]
ff_idx     = np.where(colnames == 'FEDFUNDS')[0][0]
cons_idx   = np.where(colnames == 'PCECC96')[0][0]
income_idx = np.where(colnames == 'DPIC96')[0][0]

# iloc[5:] starts at 1959:Q3 in the raw file
gdp    = pd.to_numeric(df.iloc[5:, gdp_idx].values,    errors='coerce')
cpi    = pd.to_numeric(df.iloc[5:, cpi_idx].values,    errors='coerce')
ff     = pd.to_numeric(df.iloc[5:, ff_idx].values,     errors='coerce')
cons   = pd.to_numeric(df.iloc[5:, cons_idx].values,   errors='coerce')
income = pd.to_numeric(df.iloc[5:, income_idx].values, errors='coerce')

dates = pd.date_range(start='1960-01-01', periods=len(gdp), freq='QE')

# Annualised quarter-on-quarter growth rates
temp_df    = pd.DataFrame({'GDP': gdp, 'CPI': cpi, 'FF': ff}, index=dates)
gdp_growth = 100 * 4 * (np.log(temp_df['GDP']) - np.log(temp_df['GDP'].shift(1)))
inflation  = 100 * 4 * (np.log(temp_df['CPI']) - np.log(temp_df['CPI'].shift(1)))

macro_data = pd.DataFrame(
    {'GDP_Growth': gdp_growth, 'Inflation': inflation, 'FedFunds': ff},
    index=dates).dropna()

# Cointegration data — full series, no differencing needed
coint_data = pd.DataFrame(
    {'LogConsumption': np.log(cons), 'LogIncome': np.log(income)},
    index=dates).dropna()

var_data = macro_data[['GDP_Growth', 'Inflation', 'FedFunds']].dropna()

first_q = f"{var_data.index[0].year}:Q{var_data.index[0].quarter}"
last_q  = f"{var_data.index[-1].year}:Q{var_data.index[-1].quarter}"

print("="*80)
print(f"Sample: {first_q} to {last_q}  |  VAR obs: {len(var_data)}  |  Coint obs: {len(coint_data)}")
print("="*80)

# ============================================================================
# EXAMPLE 1 — VAR
# ============================================================================
model      = VAR(var_data)
lag_results = model.select_order(maxlags=12)

# NOTE: BIC(1)=3.835 and BIC(2)=3.834 differ by only 0.0006.
# Due to floating-point differences between MATLAB and Python in det()/log(),
# MATLAB may select p=2 while Python selects p=1. To ensure consistency
# across platforms, we fix p=1 (the textbook specification).
optimal_lag = 1  # Textbook specification; BIC near-tied between p=1 and p=2
var_result  = model.fit(optimal_lag)
residuals   = var_result.resid

print(var_result.summary())

# Granger causality
# grangercausalitytests(data[[X, Y]]) tests: does Y Granger-cause X?
print("\nGranger Causality (lag = 1):")
for x_var in var_data.columns:
    for y_var in var_data.columns:
        if x_var == y_var: continue
        gc = grangercausalitytests(var_data[[x_var, y_var]],
                                   maxlag=optimal_lag, verbose=False)
        f_stat, p_val, _, _ = gc[optimal_lag][0]['ssr_ftest']
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else "")
        print(f"  {y_var:>12s} -> {x_var:<12s}  F = {float(f_stat):7.2f}  p = {float(p_val):.4f} {sig}")

# ---- Plot 1: Residual diagnostics ----
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle('VAR Residual Diagnostics\n(GDP Growth, Inflation, Federal Funds Rate)',
             fontsize=13, fontweight='bold')
for i, col in enumerate(var_data.columns):
    r = residuals.iloc[:, i]
    axes[i,0].plot(r.index, r.values, 'b-', lw=.8)
    axes[i,0].axhline(0, color='red', lw=1, ls='--')
    axes[i,0].set_title(f'{col}: Time Series', fontsize=10); axes[i,0].set_ylabel('Residuals')
    axes[i,1].hist(r.values, bins=30, density=True, alpha=.7, color='steelblue', edgecolor='white')
    mu, sig = r.mean(), r.std()
    xg = np.linspace(mu-4*sig, mu+4*sig, 100)
    axes[i,1].plot(xg, stats.norm.pdf(xg, mu, sig), 'r-', lw=2, label='Normal')
    axes[i,1].set_title(f'{col}: Histogram', fontsize=10); axes[i,1].legend()
    plot_acf(r, lags=20, ax=axes[i,2], color='steelblue',
             vlines_kwargs={'colors':'steelblue'})
    axes[i,2].set_title(f'{col}: ACF', fontsize=10)
plt.tight_layout()
plt.savefig('ex1_var_diagnostics.png', dpi=150, bbox_inches='tight'); plt.close()

# ---- Plot 2: Lag selection ----
fig, ax = plt.subplots(figsize=(10, 6))
lags_r = range(1, 13)
aic_v = [model.fit(p).aic  for p in lags_r]
bic_v = [model.fit(p).bic  for p in lags_r]
hq_v  = [model.fit(p).hqic for p in lags_r]
ax.plot(lags_r, aic_v, 'g-o', lw=2, ms=6, label='AIC')
ax.plot(lags_r, bic_v, 'r-s', lw=2, ms=6, label='BIC')
ax.plot(lags_r, hq_v,  'b-^', lw=2, ms=6, label='HQIC')
for x, c in [(lag_results.aic,'green'),(lag_results.bic,'red'),(lag_results.hqic,'blue')]:
    ax.axvline(x, color=c, lw=1.5, ls='--', alpha=.7)
ax.set_xlabel('Number of Lags'); ax.set_ylabel('Information Criterion Value')
ax.set_title(f'VAR Lag Order Selection\n(Real Data: {first_q} – {last_q})',
             fontsize=12, fontweight='bold')
ax.legend(); ax.grid(True, alpha=.3)
plt.tight_layout()
plt.savefig('ex1_lag_selection.png', dpi=150, bbox_inches='tight'); plt.close()

# ============================================================================
# EXAMPLE 2 — COINTEGRATION
# ============================================================================
vecm_data = coint_data[['LogConsumption','LogIncome']].dropna()

# Johansen test with det_order=1 (Case 4: linear trend in data, constant in CE)
# This is the correct specification when variables have deterministic trends
# (productivity growth). Using det_order=0 spuriously finds r=2 because the
# test confuses trend-stationarity with I(0).
johansen = coint_johansen(vecm_data, det_order=1, k_ar_diff=1)

print("\nJohansen Trace Test (Case 4: linear trend in data):")
for i in range(2):
    print(f"  r<={i}: stat={johansen.lr1[i]:.2f}  "
          f"90%={johansen.cvt[i,0]:.2f}  95%={johansen.cvt[i,1]:.2f}  99%={johansen.cvt[i,2]:.2f}")
print("\nJohansen Max Eigenvalue Test:")
for i in range(2):
    print(f"  r={i}: stat={johansen.lr2[i]:.2f}  "
          f"90%={johansen.cvm[i,0]:.2f}  95%={johansen.cvm[i,1]:.2f}  99%={johansen.cvm[i,2]:.2f}")

# OLS long-run relationship (avoids VECM beta sign ambiguity)
X_ols = np.column_stack([np.ones(len(vecm_data)), vecm_data['LogIncome'].values])
b_ols = np.linalg.lstsq(X_ols, vecm_data['LogConsumption'].values, rcond=None)[0]
ols_int, ols_slope = b_ols
print(f"OLS long-run: LogC = {ols_int:.4f} + {ols_slope:.4f} * LogY")

# ---- Plot 3: Cointegration ----
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle(f'Cointegration Analysis: Consumption and Income\n(Real Data: {first_q} – {last_q})',
             fontsize=13, fontweight='bold')
axes[0].plot(vecm_data.index, vecm_data['LogConsumption'], lw=1.5, label='Log Consumption')
axes[0].plot(vecm_data.index, vecm_data['LogIncome'], lw=1.5, color='orange', label='Log Income')
axes[0].set_title('Time Series of Log Consumption and Log Income'); axes[0].set_ylabel('Log Scale')
axes[0].legend(); axes[0].grid(True, alpha=.3)

xd, yd = vecm_data['LogIncome'].values, vecm_data['LogConsumption'].values
axes[1].scatter(xd, yd, alpha=.4, s=10, color='steelblue')
axes[1].plot(xd, ols_int + ols_slope*xd, 'r-', lw=2,
             label=f'Long-run: C = {ols_int:.2f} + {ols_slope:.3f} × Y')
axes[1].set_xlabel('Log Income'); axes[1].set_ylabel('Log Consumption')
axes[1].set_title('Long-run Relationship'); axes[1].legend(); axes[1].grid(True, alpha=.3)

cr = yd - (ols_int + ols_slope*xd)
cr_n = (cr - cr.mean()) / cr.std()
axes[2].plot(vecm_data.index, cr_n, 'b-', lw=1)
axes[2].axhline(0, color='red', lw=1, ls='--')
axes[2].fill_between(vecm_data.index, 0, cr_n, alpha=.2)
axes[2].set_title('Cointegrating Residual (Standardized)')
axes[2].set_xlabel('Quarter'); axes[2].set_ylabel('Deviation from Equilibrium')
axes[2].grid(True, alpha=.3)
plt.tight_layout()
plt.savefig('ex2_vecm_cointegration.png', dpi=150, bbox_inches='tight'); plt.close()

# ============================================================================
# EXAMPLE 3 — IDENTIFICATION PROBLEM
# ============================================================================
irf_obj = var_result.irf(periods=20)
corr_resid = residuals.corr()
vn = ['GDP_Growth','Inflation','FedFunds']

# ---- Plot 4: Correlation heatmap ----
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_resid.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(vn, fontsize=11); ax.set_yticklabels(vn, fontsize=11)
for i in range(3):
    for j in range(3):
        clr = 'white' if abs(corr_resid.iloc[i,j]) > .5 else 'black'
        ax.text(j, i, f'{corr_resid.iloc[i,j]:.3f}', ha='center', va='center',
                color=clr, fontsize=14, fontweight='bold')
ax.set_title('Correlation of Reduced-Form Innovations\n(The Root of the Identification Problem)',
             fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Correlation Coefficient')
plt.tight_layout()
plt.savefig('ex3_innovation_correlation.png', dpi=150, bbox_inches='tight'); plt.close()

# ---- Plot 5: Reduced-form IRFs ----
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle('Reduced-Form Impulse Response Functions\n'
             '\u26A0\uFE0F NOT Economically Identified \u2014 These Are Statistical Responses Only',
             fontsize=13, fontweight='bold')
hor = np.arange(21)
for i, imp in enumerate(vn):
    for j, resp in enumerate(vn):
        ax = axes[j, i]; v = irf_obj.irfs[:, j, i]
        ax.plot(hor, v, 'b-', lw=2); ax.axhline(0, color='black', lw=.8)
        ax.fill_between(hor, 0, v, alpha=.2, color='blue')
        ax.set_title(f'Response of {resp}\nto {imp} innovation', fontsize=9)
        if j == 2: ax.set_xlabel('Quarters')
        ax.set_ylabel('Response'); ax.grid(True, alpha=.3)
fig.text(.5, .01,
         'WARNING: These are REDUCED-FORM IRFs to correlated innovations. '
         'They conflate multiple structural effects and cannot be given causal interpretation.',
         ha='center', fontsize=10, style='italic', color='darkred',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=.9))
plt.tight_layout(rect=[0, .04, 1, .96])
plt.savefig('ex3_reduced_form_irf.png', dpi=150, bbox_inches='tight'); plt.close()

print("\nAll 5 plots saved. Done.")
