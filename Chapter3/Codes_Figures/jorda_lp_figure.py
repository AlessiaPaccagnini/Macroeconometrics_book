"""
Jordà (2005) Local Projections Example with Real U.S. Data
==========================================================
Comparing LP vs VAR impulse responses for monetary policy analysis
Using FRED data: GDP, GDP Deflator, Federal Funds Rate

Google Colab version — upload Excel files when prompted.
"""

# ============================================================
# 0. Install dependencies (Colab)
# ============================================================
# !pip install statsmodels openpyxl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.edgecolor'] = 'black'

# ============================================================
# 1. Upload and Load FRED Data (Colab file upload)
# ============================================================
print("=" * 60)
print("Please upload the three FRED Excel files when prompted:")
print("  1) GDPC1.xlsx   (Real GDP)")
print("  2) GDPDEF.xlsx  (GDP Deflator)")
print("  3) FEDFUNDS.xlsx (Federal Funds Rate)")
print("=" * 60)

from google.colab import files

print("\n>>> Upload GDPC1.xlsx (Real GDP):")
uploaded = files.upload()
gdp_file = list(uploaded.keys())[0]

print("\n>>> Upload GDPDEF.xlsx (GDP Deflator):")
uploaded = files.upload()
deflator_file = list(uploaded.keys())[0]

print("\n>>> Upload FEDFUNDS.xlsx (Federal Funds Rate):")
uploaded = files.upload()
fedfunds_file = list(uploaded.keys())[0]

# ============================================================
# 2. Read and Process Data
# ============================================================
print("\nLoading data...")

gdp = pd.read_excel(gdp_file, sheet_name='Quarterly')
deflator = pd.read_excel(deflator_file, sheet_name='Quarterly')

# Fed funds is monthly
try:
    fedfunds = pd.read_excel(fedfunds_file, sheet_name='Monthly')
except Exception:
    fedfunds = pd.read_excel(fedfunds_file, sheet_name='Quarterly')

print(f"  GDP shape:       {gdp.shape}")
print(f"  Deflator shape:  {deflator.shape}")
print(f"  Fed Funds shape: {fedfunds.shape}")

# Rename columns (FRED format: observation_date, SERIES_NAME)
gdp.columns = ['date', 'gdp']
deflator.columns = ['date', 'deflator']
fedfunds.columns = ['date', 'fedfunds']

# Convert dates
gdp['date'] = pd.to_datetime(gdp['date'])
deflator['date'] = pd.to_datetime(deflator['date'])
fedfunds['date'] = pd.to_datetime(fedfunds['date'])

# Fed funds: monthly → quarterly (average)
fedfunds['quarter'] = fedfunds['date'].dt.to_period('Q')
fedfunds_q = fedfunds.groupby('quarter')['fedfunds'].mean().reset_index()
fedfunds_q['date'] = fedfunds_q['quarter'].dt.to_timestamp()
fedfunds_q = fedfunds_q[['date', 'fedfunds']]

# Align GDP and deflator to quarter start
gdp['date'] = gdp['date'].dt.to_period('Q').dt.to_timestamp()
deflator['date'] = deflator['date'].dt.to_period('Q').dt.to_timestamp()

# Merge
df = gdp.merge(deflator, on='date', how='inner')
df = df.merge(fedfunds_q, on='date', how='inner')
df = df.set_index('date').sort_index()

print(f"\nMerged data: {df.shape[0]} observations")
print(f"Date range:  {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")

# ============================================================
# 3. Construct Variables
# ============================================================

# Output gap via HP filter (λ = 1600 for quarterly data)
df['log_gdp'] = 100 * np.log(df['gdp'])
cycle, trend = hpfilter(df['log_gdp'].dropna(), lamb=1600)
df['output_gap'] = cycle

# Inflation: annualised 4-quarter change in log GDP deflator
df['log_deflator'] = 100 * np.log(df['deflator'])
df['inflation'] = df['log_deflator'].diff(4)

# Federal funds rate (already in percent)
df['fed_funds'] = df['fedfunds']

# Sample: 1960Q1 – 2007Q4 (pre-crisis, avoids ZLB)
df_sample = df.loc['1960-01-01':'2007-12-31',
                    ['output_gap', 'inflation', 'fed_funds']].dropna()

start_q = f"{df_sample.index[0].year}Q{(df_sample.index[0].month - 1) // 3 + 1}"
end_q   = f"{df_sample.index[-1].year}Q{(df_sample.index[-1].month - 1) // 3 + 1}"

print(f"\nEstimation sample: {start_q} – {end_q}  (T = {len(df_sample)})")
print("\nDescriptive statistics:")
print(df_sample.describe().round(2))

# ============================================================
# 4. Estimate VAR Models
# ============================================================
print("\n" + "=" * 60)
print("Estimating VAR models...")
print("=" * 60)

model_var = VAR(df_sample)

# VAR(4) — standard for quarterly data
results_var4 = model_var.fit(4)
print(f"\n  VAR(4)  AIC = {results_var4.aic:.2f}   BIC = {results_var4.bic:.2f}")

# VAR(1) — deliberately misspecified for comparison
results_var1 = model_var.fit(1)
print(f"  VAR(1)  AIC = {results_var1.aic:.2f}   BIC = {results_var1.bic:.2f}")

# Cholesky IRFs (ordering: output_gap → inflation → fed_funds)
H = 20  # 20-quarter horizon
irf_var4 = results_var4.irf(H)
irf_var1 = results_var1.irf(H)

# ============================================================
# 5. Estimate Local Projections
# ============================================================
print("\n" + "=" * 60)
print("Estimating Local Projections...")
print("=" * 60)


def estimate_lp(data, shock_var, response_var, H, n_lags):
    """
    Jordà (2005) Local Projections with Newey–West standard errors.

    Parameters
    ----------
    data : DataFrame — panel of variables
    shock_var : str — name of the shock variable
    response_var : str — name of the response variable
    H : int — maximum horizon
    n_lags : int — number of control lags

    Returns
    -------
    irfs, ses, ci_lower, ci_upper : arrays of length H+1
    """
    irfs = np.zeros(H + 1)
    ses  = np.zeros(H + 1)

    for h in range(H + 1):
        # Dependent variable: y_{t+h}
        y = data[response_var].shift(-h)

        # Controls: lags 1..n_lags of every variable
        X_list = []
        for var in data.columns:
            for lag in range(1, n_lags + 1):
                X_list.append(data[var].shift(lag))

        # Contemporaneous shock + controls + constant
        X_df = pd.concat([data[shock_var]] + X_list, axis=1)
        X_df = add_constant(X_df)

        # Drop NaN rows
        combined = pd.concat([y, X_df], axis=1).dropna()
        y_clean = combined.iloc[:, 0].values
        X_clean = combined.iloc[:, 1:].values

        # OLS with HAC (Newey–West) standard errors
        nw_lags = max(h + 1, 4)
        res = OLS(y_clean, X_clean).fit(
            cov_type='HAC', cov_kwds={'maxlags': nw_lags}
        )

        # Coefficient on the contemporaneous shock (index 1 after constant)
        irfs[h] = res.params[1]
        ses[h]  = res.bse[1]

    ci_lower = irfs - 1.96 * ses
    ci_upper = irfs + 1.96 * ses
    return irfs, ses, ci_lower, ci_upper


n_lags = 4

print("  Output gap response...")
irf_lp_y, se_lp_y, ci_lo_y, ci_hi_y = estimate_lp(
    df_sample, 'fed_funds', 'output_gap', H, n_lags)

print("  Inflation response...")
irf_lp_pi, se_lp_pi, ci_lo_pi, ci_hi_pi = estimate_lp(
    df_sample, 'fed_funds', 'inflation', H, n_lags)

print("  Fed funds own response...")
irf_lp_ff, se_lp_ff, ci_lo_ff, ci_hi_ff = estimate_lp(
    df_sample, 'fed_funds', 'fed_funds', H, n_lags)

print("  Done!")

# ============================================================
# 6. Publication-Quality Figure
# ============================================================
print("\n" + "=" * 60)
print("Creating figure...")
print("=" * 60)

# Cholesky indices (ordering: output_gap=0, inflation=1, fed_funds=2)
shock_idx   = 2
resp_y_idx  = 0
resp_pi_idx = 1

# VAR(4) IRFs and 95 % bands
var4_y   = irf_var4.irfs[:, resp_y_idx,  shock_idx]
var4_pi  = irf_var4.irfs[:, resp_pi_idx, shock_idx]
var4_se_y  = irf_var4.stderr()[:, resp_y_idx,  shock_idx]
var4_se_pi = irf_var4.stderr()[:, resp_pi_idx, shock_idx]
var4_lo_y, var4_hi_y   = var4_y - 1.96 * var4_se_y, var4_y + 1.96 * var4_se_y
var4_lo_pi, var4_hi_pi = var4_pi - 1.96 * var4_se_pi, var4_pi + 1.96 * var4_se_pi

# VAR(1) IRFs (no bands — shown as misspecified benchmark)
var1_y  = irf_var1.irfs[:, resp_y_idx,  shock_idx]
var1_pi = irf_var1.irfs[:, resp_pi_idx, shock_idx]

horizons = np.arange(H + 1)

# Colours
c_var4 = '#2166AC'   # blue
c_lp   = '#B2182B'   # red
c_var1 = '#4DAF4A'   # green
alpha  = 0.15

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Panel (a): Output Gap ---
ax = axes[0]
ax.fill_between(horizons, var4_lo_y, var4_hi_y, color=c_var4, alpha=alpha)
ax.plot(horizons, var4_y, color=c_var4, lw=2.5, label='VAR(4)')
ax.fill_between(horizons, ci_lo_y, ci_hi_y, color=c_lp, alpha=alpha)
ax.plot(horizons, irf_lp_y, color=c_lp, lw=2.5, ls='--', label='Local Projections')
ax.plot(horizons, var1_y, color=c_var1, lw=2, ls=':', label='VAR(1) — misspecified')
ax.axhline(0, color='black', lw=1)
ax.set_xlabel('Quarters after shock', fontsize=13)
ax.set_ylabel('Percent', fontsize=13)
ax.set_title('(a) Response of Output Gap to Fed Funds Shock',
             fontsize=14, fontweight='bold', pad=10)
ax.legend(loc='lower right', fontsize=11, framealpha=0.95,
          edgecolor='gray', fancybox=False)
ax.set_xlim(0, 20)
ax.set_xticks([0, 4, 8, 12, 16, 20])
ax.tick_params(labelsize=11)
ym = max(abs(min(var4_lo_y.min(), ci_lo_y.min(), var1_y.min())),
         abs(max(var4_hi_y.max(), ci_hi_y.max(), var1_y.max()))) * 1.15
ax.set_ylim(-ym, ym)

# --- Panel (b): Inflation ---
ax = axes[1]
ax.fill_between(horizons, var4_lo_pi, var4_hi_pi, color=c_var4, alpha=alpha)
ax.plot(horizons, var4_pi, color=c_var4, lw=2.5, label='VAR(4)')
ax.fill_between(horizons, ci_lo_pi, ci_hi_pi, color=c_lp, alpha=alpha)
ax.plot(horizons, irf_lp_pi, color=c_lp, lw=2.5, ls='--', label='Local Projections')
ax.plot(horizons, var1_pi, color=c_var1, lw=2, ls=':', label='VAR(1) — misspecified')
ax.axhline(0, color='black', lw=1)
ax.set_xlabel('Quarters after shock', fontsize=13)
ax.set_ylabel('Percent', fontsize=13)
ax.set_title('(b) Response of Inflation to Fed Funds Shock',
             fontsize=14, fontweight='bold', pad=10)
ax.legend(loc='upper right', fontsize=11, framealpha=0.95,
          edgecolor='gray', fancybox=False)
ax.set_xlim(0, 20)
ax.set_xticks([0, 4, 8, 12, 16, 20])
ax.tick_params(labelsize=11)
ym_pi = max(abs(min(var4_lo_pi.min(), ci_lo_pi.min(), var1_pi.min())),
            abs(max(var4_hi_pi.max(), ci_hi_pi.max(), var1_pi.max()))) * 1.15
ax.set_ylim(-ym_pi, ym_pi)

fig.suptitle(
    'Impulse Responses to a Monetary Policy Shock: LP vs VAR\n'
    f'U.S. Quarterly Data, {start_q}–{end_q}',
    fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('jorda_lp_example_real_data.png', dpi=200,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()
print("Figure saved as 'jorda_lp_example_real_data.png'")

# ============================================================
# 7. Summary Statistics
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY: LP vs VAR Comparison (Real U.S. Data)")
print("=" * 60)
print(f"\nSample: {start_q} – {end_q}  (T = {len(df_sample)})")
print("Variables: Output Gap, Inflation, Federal Funds Rate")
print("Identification: Cholesky (output → inflation → fed funds)")

print(f"\n--- Output Gap Response (h = 8) ---")
print(f"  VAR(4): {var4_y[8]: .4f}  [{var4_lo_y[8]: .4f}, {var4_hi_y[8]: .4f}]")
print(f"  LP:     {irf_lp_y[8]: .4f}  [{ci_lo_y[8]: .4f}, {ci_hi_y[8]: .4f}]")
print(f"  VAR(1): {var1_y[8]: .4f}  (misspecified)")

print(f"\n--- Inflation Response (h = 8) ---")
print(f"  VAR(4): {var4_pi[8]: .4f}  [{var4_lo_pi[8]: .4f}, {var4_hi_pi[8]: .4f}]")
print(f"  LP:     {irf_lp_pi[8]: .4f}  [{ci_lo_pi[8]: .4f}, {ci_hi_pi[8]: .4f}]")
print(f"  VAR(1): {var1_pi[8]: .4f}  (misspecified)")

print("\n" + "=" * 60)
