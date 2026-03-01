"""
Example 2: Spurious Regression and Error Correction Models
==========================================================
This example demonstrates:
1. The spurious regression problem with independent I(1) series
2. Cointegration testing using Engle-Granger method
3. Error Correction Model estimation and interpretation

Author: Alessia Paccagnini
Textbook: Macroeconometrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

# =============================================================================
# Find a seed that produces high R² for spurious regression
# =============================================================================
def find_high_r2_seed(n=200, target_r2=0.85, max_seeds=2000):
    """Search for a seed that produces a dramatic spurious regression."""
    best_seed = 0
    best_r2 = 0
    
    for seed in range(max_seeds):
        np.random.seed(seed)
        eps1 = np.random.normal(0, 1, n)
        eps2 = np.random.normal(0, 1, n)
        y = np.cumsum(eps1)
        x = np.cumsum(eps2)
        
        X = sm.add_constant(x)
        model = OLS(y, X).fit()
        
        if model.rsquared > best_r2:
            best_r2 = model.rsquared
            best_seed = seed
            if best_r2 >= target_r2:
                break
    
    return best_seed, best_r2

print("=" * 70)
print("EXAMPLE 2: SPURIOUS REGRESSION AND ERROR CORRECTION MODELS")
print("=" * 70)

# Find a seed with high R²
n = 200
best_seed, best_r2 = find_high_r2_seed(n, target_r2=0.85)
print(f"\nUsing seed {best_seed} which produces R² = {best_r2:.3f} for spurious regression")

# =============================================================================
# Part A: SPURIOUS REGRESSION
# =============================================================================
print("\n" + "=" * 70)
print("PART A: SPURIOUS REGRESSION PROBLEM")
print("=" * 70)

# Generate two INDEPENDENT random walks with the high-R² seed
np.random.seed(best_seed)
eps1 = np.random.normal(0, 1, n)
eps2 = np.random.normal(0, 1, n)

y_spurious = np.cumsum(eps1)  # Random walk 1
x_spurious = np.cumsum(eps2)  # Random walk 2 (independent!)

print("\nData Generation:")
print("    y_t = y_{t-1} + ε₁_t  (Random Walk)")
print("    x_t = x_{t-1} + ε₂_t  (Random Walk, INDEPENDENT of y)")
print("    By construction, there is NO relationship between y and x!")

# Run the spurious regression
X_spur = sm.add_constant(x_spurious)
spurious_model = OLS(y_spurious, X_spur).fit()

dw = sm.stats.durbin_watson(spurious_model.resid)
r2 = spurious_model.rsquared

print("\n" + "-" * 50)
print("SPURIOUS REGRESSION: y_t = α + β·x_t + u_t")
print("-" * 50)
print(f"    α (constant):  {spurious_model.params[0]:.4f}")
print(f"    β coefficient: {spurious_model.params[1]:.4f}")
print(f"    t-statistic:   {spurious_model.tvalues[1]:.2f}")
print(f"    p-value:       {spurious_model.pvalues[1]:.2e}")
print(f"    R-squared:     {r2:.4f}")
print(f"    Durbin-Watson: {dw:.4f}")

print("\n    ⚠️  WARNING: The regression appears highly significant!")
print(f"    ⚠️  R² = {r2:.1%} and t-statistic = {spurious_model.tvalues[1]:.1f}")
print("    ⚠️  But we KNOW y and x are independent by construction!")
print("    ⚠️  This is the SPURIOUS REGRESSION problem.")

print(f"\n    Granger-Newbold rule: R² > DW suggests spurious regression")
print(f"    R² = {r2:.3f}, DW = {dw:.3f} → {'SPURIOUS!' if r2 > dw else 'OK'}")

# Test residuals for unit root
adf_resid = adfuller(spurious_model.resid, autolag='AIC')
print(f"\n    ADF test on residuals: stat = {adf_resid[0]:.3f}, p-value = {adf_resid[1]:.3f}")
print(f"    Residuals are {'stationary' if adf_resid[1] < 0.05 else 'NON-STATIONARY'}")
print(f"    → {'Valid regression' if adf_resid[1] < 0.05 else 'SPURIOUS REGRESSION CONFIRMED!'}")

# =============================================================================
# Figure 1: Spurious Regression Illustration
# =============================================================================
fig = plt.figure(figsize=(14, 10))

# Panel 1: Time series of both random walks
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(y_spurious, color='steelblue', linewidth=1.2, label=r'$y_t$ (Random Walk 1)')
ax1.plot(x_spurious, color='darkred', linewidth=1.2, label=r'$x_t$ (Random Walk 2)')
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.set_title('(a) Two Independent Random Walks', fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Panel 2: Scatter plot with regression line
ax2 = fig.add_subplot(2, 2, 2)
ax2.scatter(x_spurious, y_spurious, alpha=0.5, s=25, color='purple', edgecolor='none')

# Add regression line
x_line = np.linspace(x_spurious.min(), x_spurious.max(), 100)
y_line = spurious_model.params[0] + spurious_model.params[1] * x_line
ax2.plot(x_line, y_line, color='red', linewidth=2.5, 
         label=f'OLS: $\\hat{{\\beta}}$ = {spurious_model.params[1]:.3f}')

ax2.set_xlabel(r'$x_t$')
ax2.set_ylabel(r'$y_t$')
ax2.set_title(f'(b) Spurious Correlation: $R^2$ = {r2:.3f}, $t$ = {spurious_model.tvalues[1]:.1f}', 
              fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# Add text box with warning
textstr = f'$R^2 = {r2:.3f}$\n$t = {spurious_model.tvalues[1]:.1f}$\n$p < 0.001$\n\nYet NO true\nrelationship!'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='orange')
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# Panel 3: Residuals from spurious regression
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(spurious_model.resid, color='darkgreen', linewidth=0.9)
ax3.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax3.fill_between(range(n), spurious_model.resid, alpha=0.3, color='darkgreen')
ax3.set_xlabel('Time')
ax3.set_ylabel('Residuals')
ax3.set_title(f'(c) Residuals: DW = {dw:.3f} (Should be ≈ 2.0)', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Panel 4: ACF of residuals
ax4 = fig.add_subplot(2, 2, 4)
plot_acf(spurious_model.resid, ax=ax4, lags=30, alpha=0.05)
ax4.set_title('(d) ACF of Residuals (Highly Persistent!)', fontweight='bold')
ax4.set_xlabel('Lag')

plt.tight_layout()
plt.savefig('example2_spurious_regression.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# Part B: COINTEGRATION - Genuine Long-Run Relationship
# =============================================================================
print("\n" + "=" * 70)
print("PART B: COINTEGRATION - A GENUINE LONG-RUN RELATIONSHIP")
print("=" * 70)

# Generate COINTEGRATED series (consumption and income)
np.random.seed(456)  # Different seed for cointegration example

# Income follows a random walk with drift
income_shocks = np.random.normal(0.5, 1.0, n)
income = 100 + np.cumsum(income_shocks)

# Consumption is cointegrated with income: c_t = α + β*y_t + stationary error
# The error follows a stationary AR(1) process
stationary_error = np.zeros(n)
for t in range(1, n):
    stationary_error[t] = 0.6 * stationary_error[t-1] + np.random.normal(0, 1.5)
    
consumption = 15 + 0.85 * income + stationary_error

print("\nData Generation (Permanent Income Hypothesis):")
print("    income_t = income_{t-1} + 0.5 + ε_t  (Random Walk with drift)")
print("    consumption_t = 15 + 0.85·income_t + u_t  (Cointegrated)")
print("    where u_t = 0.6·u_{t-1} + ν_t is a stationary AR(1) process")
print("    TRUE cointegrating vector: (1, -0.85)")
print("    TRUE intercept: 15")

# Unit root tests on levels
print("\n" + "-" * 50)
print("STEP 1: Test for Unit Roots in Levels")
print("-" * 50)

adf_income = adfuller(income, autolag='AIC')
adf_cons = adfuller(consumption, autolag='AIC')

print(f"    Income:      ADF stat = {adf_income[0]:.3f}, p-value = {adf_income[1]:.3f} → I(1)")
print(f"    Consumption: ADF stat = {adf_cons[0]:.3f}, p-value = {adf_cons[1]:.3f} → I(1)")

# Unit root tests on first differences
adf_dincome = adfuller(np.diff(income), autolag='AIC')
adf_dcons = adfuller(np.diff(consumption), autolag='AIC')

print(f"\n    Δ Income:      ADF stat = {adf_dincome[0]:.3f}, p-value = {adf_dincome[1]:.4f} → I(0)")
print(f"    Δ Consumption: ADF stat = {adf_dcons[0]:.3f}, p-value = {adf_dcons[1]:.4f} → I(0)")
print("\n    Both series are I(1): nonstationary in levels, stationary in first differences.")

# Engle-Granger cointegration test
print("\n" + "-" * 50)
print("STEP 2: Engle-Granger Cointegration Test")
print("-" * 50)

# Step 2a: Estimate cointegrating regression
X_coint = sm.add_constant(income)
coint_reg = OLS(consumption, X_coint).fit()

print("\nCointegrating Regression: consumption_t = α + β·income_t + z_t")
print(f"    α̂ (constant): {coint_reg.params[0]:.4f}  (true = 15)")
print(f"    β̂ (income):   {coint_reg.params[1]:.4f}  (true = 0.85)")
print(f"    R-squared:    {coint_reg.rsquared:.4f}")
print(f"    Durbin-Watson: {sm.stats.durbin_watson(coint_reg.resid):.4f}")

# Step 2b: Test residuals for stationarity
residuals_coint = coint_reg.resid
adf_coint_resid = adfuller(residuals_coint, autolag='AIC')

print("\nADF Test on Cointegrating Residuals:")
print(f"    Test statistic: {adf_coint_resid[0]:.4f}")
print(f"    p-value:        {adf_coint_resid[1]:.4f}")

print("\n    Engle-Granger Critical Values (2 variables):")
print("    1%: -3.90,  5%: -3.34,  10%: -3.04")
print(f"\n    Test statistic ({adf_coint_resid[0]:.2f}) < 5% CV (-3.34)?", end=" ")

if adf_coint_resid[0] < -3.34:
    print("YES")
    print("\n    ✓ CONCLUSION: Reject H₀ of no cointegration")
    print("    ✓ Consumption and income are COINTEGRATED")
else:
    print("NO")
    print("\n    ✗ CONCLUSION: Cannot reject H₀ → No cointegration")

# =============================================================================
# Figure 2: Cointegration Illustration
# =============================================================================
fig = plt.figure(figsize=(14, 10))

# Panel 1: Time series of cointegrated series
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(income, color='steelblue', linewidth=1.2, label='Income')
ax1.plot(consumption, color='darkred', linewidth=1.2, label='Consumption')
ax1.set_xlabel('Time')
ax1.set_ylabel('Level')
ax1.set_title('(a) Cointegrated Series: Consumption and Income', fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Panel 2: Scatter plot with cointegrating regression
ax2 = fig.add_subplot(2, 2, 2)
ax2.scatter(income, consumption, alpha=0.5, s=25, color='purple', edgecolor='none')

# Add regression line
inc_line = np.linspace(income.min(), income.max(), 100)
cons_line = coint_reg.params[0] + coint_reg.params[1] * inc_line
ax2.plot(inc_line, cons_line, color='red', linewidth=2.5, 
         label=f'$\\hat{{c}}_t = {coint_reg.params[0]:.1f} + {coint_reg.params[1]:.3f} \\cdot y_t$')

ax2.set_xlabel('Income')
ax2.set_ylabel('Consumption')
ax2.set_title(f'(b) Cointegrating Regression: $R^2$ = {coint_reg.rsquared:.3f}', fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Panel 3: Equilibrium error (cointegrating residuals)
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(residuals_coint, color='darkgreen', linewidth=0.9)
ax3.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax3.fill_between(range(n), residuals_coint, alpha=0.3, color='darkgreen')
ax3.set_xlabel('Time')
ax3.set_ylabel('Equilibrium Error')
ax3.set_title(f'(c) Equilibrium Error $z_t$ = $c_t - \\hat{{α}} - \\hat{{β}}y_t$ (Stationary)', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Panel 4: ACF of cointegrating residuals
ax4 = fig.add_subplot(2, 2, 4)
plot_acf(residuals_coint, ax=ax4, lags=30, alpha=0.05)
ax4.set_title('(d) ACF of Equilibrium Error (Decays Quickly)', fontweight='bold')
ax4.set_xlabel('Lag')

plt.tight_layout()
plt.savefig('example2_cointegration.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# Part C: ERROR CORRECTION MODEL
# =============================================================================
print("\n" + "=" * 70)
print("PART C: ERROR CORRECTION MODEL (ECM)")
print("=" * 70)

# Create ECM variables
dc = np.diff(consumption)  # Δc_t
dy = np.diff(income)       # Δy_t
ec_term = residuals_coint[:-1]  # z_{t-1} = c_{t-1} - α̂ - β̂·y_{t-1}

# Estimate ECM: Δc_t = γ + α·z_{t-1} + δ·Δy_t + ε_t
ecm_data = pd.DataFrame({
    'dc': dc,
    'dy': dy,
    'ec_lag': ec_term
})

X_ecm = sm.add_constant(ecm_data[['ec_lag', 'dy']])
ecm_model = OLS(ecm_data['dc'], X_ecm).fit()

print("\nECM Specification:")
print("    Δc_t = γ + α·(c_{t-1} - β̂·y_{t-1}) + δ·Δy_t + ε_t")
print("\nEstimation Results:")
print("-" * 60)
print(f"    {'Parameter':<25} {'Estimate':>10} {'Std.Err':>10} {'t-stat':>10}")
print("-" * 60)
print(f"    {'Constant (γ)':<25} {ecm_model.params['const']:>10.4f} {ecm_model.bse['const']:>10.4f} {ecm_model.tvalues['const']:>10.2f}")
print(f"    {'Error Correction (α)':<25} {ecm_model.params['ec_lag']:>10.4f} {ecm_model.bse['ec_lag']:>10.4f} {ecm_model.tvalues['ec_lag']:>10.2f}")
print(f"    {'Δ Income (δ)':<25} {ecm_model.params['dy']:>10.4f} {ecm_model.bse['dy']:>10.4f} {ecm_model.tvalues['dy']:>10.2f}")
print("-" * 60)
print(f"    R-squared: {ecm_model.rsquared:.4f}")
print(f"    Durbin-Watson: {sm.stats.durbin_watson(ecm_model.resid):.4f}")

alpha_ec = ecm_model.params['ec_lag']
delta = ecm_model.params['dy']
beta_lr = coint_reg.params[1]

print("\n" + "-" * 50)
print("INTERPRETATION")
print("-" * 50)

if alpha_ec < 0:
    print(f"\n    ✓ Error Correction Coefficient (α) = {alpha_ec:.4f} < 0")
    print("      → System corrects toward equilibrium")
    
    # Speed of adjustment
    print(f"\n    Speed of Adjustment:")
    print(f"      {abs(alpha_ec)*100:.1f}% of disequilibrium corrected per period")
    
    # Half-life
    half_life = np.log(0.5) / np.log(1 + alpha_ec)
    print(f"      Half-life: {half_life:.2f} periods")
    print(f"      (Time for half of any deviation to be eliminated)")
else:
    print(f"\n    ✗ α = {alpha_ec:.4f} > 0: System is explosive!")

print(f"\n    Short-run vs Long-run Effects:")
print(f"      Short-run effect of Δy on Δc (δ): {delta:.4f}")
print(f"      Long-run effect of y on c (β):    {beta_lr:.4f}")
print(f"\n      Interpretation: A $1 increase in income raises consumption by")
print(f"      ${delta:.2f} immediately, with further adjustment to ${beta_lr:.2f}")
print(f"      occurring gradually through error correction.")

# =============================================================================
# Figure 3: Error Correction Model Diagnostics
# =============================================================================
fig = plt.figure(figsize=(14, 10))

# Panel 1: Equilibrium error over time
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(residuals_coint, color='steelblue', linewidth=0.9)
ax1.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax1.fill_between(range(n), residuals_coint, alpha=0.3, color='steelblue')
ax1.set_xlabel('Time')
ax1.set_ylabel('Equilibrium Error ($z_t$)')
ax1.set_title('(a) Equilibrium Error: Deviations from Long-Run Relationship', fontweight='bold')
ax1.grid(True, alpha=0.3)

# Panel 2: ECM residuals
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(ecm_model.resid, color='darkgreen', linewidth=0.9)
ax2.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax2.set_xlabel('Time')
ax2.set_ylabel('ECM Residuals')
ax2.set_title(f'(b) ECM Residuals (DW = {sm.stats.durbin_watson(ecm_model.resid):.2f})', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Panel 3: Actual vs Fitted Δc
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(dc, color='steelblue', linewidth=0.8, alpha=0.7, label='Actual $\\Delta c_t$')
ax3.plot(ecm_model.fittedvalues, color='red', linewidth=1.2, label='Fitted')
ax3.set_xlabel('Time')
ax3.set_ylabel('$\\Delta c_t$')
ax3.set_title(f'(c) Actual vs Fitted: $R^2$ = {ecm_model.rsquared:.3f}', fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Panel 4: Error Correction Mechanism
ax4 = fig.add_subplot(2, 2, 4)
ax4.scatter(ec_term, dc, alpha=0.5, s=30, color='purple', edgecolor='none')
ax4.axhline(0, color='gray', linestyle='--', alpha=0.7)
ax4.axvline(0, color='gray', linestyle='--', alpha=0.7)

# Add regression line showing error correction
z_range = np.linspace(ec_term.min(), ec_term.max(), 100)
ec_line = alpha_ec * z_range + ecm_model.params['const']
ax4.plot(z_range, ec_line, color='red', linewidth=2.5, 
         label=f'$\\alpha$ = {alpha_ec:.3f}')

ax4.set_xlabel('Lagged Equilibrium Error ($z_{t-1}$)')
ax4.set_ylabel('Change in Consumption ($\\Delta c_t$)')
ax4.set_title('(d) Error Correction Mechanism', fontweight='bold')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

# Add annotations
ax4.annotate('Above equilibrium\n→ consumption falls', 
             xy=(ec_term.max()*0.6, ec_line[80]), fontsize=9,
             ha='center', color='darkred')
ax4.annotate('Below equilibrium\n→ consumption rises', 
             xy=(ec_term.min()*0.6, ec_line[20]), fontsize=9,
             ha='center', color='darkgreen')

plt.tight_layout()
plt.savefig('example2_ecm.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 4: Comparison - Spurious vs Cointegrated
# =============================================================================
fig = plt.figure(figsize=(14, 6))

# Panel 1: Spurious regression residuals
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(spurious_model.resid, color='crimson', linewidth=0.9)
ax1.axhline(0, color='black', linestyle='--', linewidth=1)
ax1.fill_between(range(n), spurious_model.resid, alpha=0.3, color='crimson')
ax1.set_xlabel('Time')
ax1.set_ylabel('Residuals')
ax1.set_title(f'Spurious Regression Residuals\n$R^2$ = {r2:.3f}, DW = {dw:.3f}', fontweight='bold')
ax1.grid(True, alpha=0.3)

# Add text
ax1.text(0.05, 0.95, 'Non-stationary!\nWander without\nreturning to zero', 
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel 2: Cointegration residuals
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(residuals_coint, color='steelblue', linewidth=0.9)
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.fill_between(range(n), residuals_coint, alpha=0.3, color='steelblue')
ax2.set_xlabel('Time')
ax2.set_ylabel('Residuals')
dw_coint = sm.stats.durbin_watson(coint_reg.resid)
ax2.set_title(f'Cointegrating Regression Residuals\n$R^2$ = {coint_reg.rsquared:.3f}, DW = {dw_coint:.3f}', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add text
ax2.text(0.05, 0.95, 'Stationary!\nMean-reverting\naround zero', 
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('example2_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    SPURIOUS vs COINTEGRATED REGRESSION              │
├─────────────────────────────────────────────────────────────────────┤
│                        SPURIOUS            COINTEGRATED             │
├─────────────────────────────────────────────────────────────────────┤
│  True relationship:    NONE                 c = 15 + 0.85·y         │
│  R²:                   {r2:.3f}                {coint_reg.rsquared:.3f}                 │
│  t-statistic:          {spurious_model.tvalues[1]:.1f}                 {coint_reg.tvalues[1]:.1f}                 │
│  Durbin-Watson:        {dw:.3f}                {dw_coint:.3f}                 │
│  Residuals:            Non-stationary       Stationary              │
│  Interpretation:       MEANINGLESS          Long-run equilibrium    │
└─────────────────────────────────────────────────────────────────────┘

KEY LESSONS:

1. SPURIOUS REGRESSION: Independent I(1) series can produce highly 
   significant but meaningless regression results (R² = {r2:.0%}!). 
   The Granger-Newbold rule (R² > DW) helps detect this problem.

2. COINTEGRATION: When I(1) series share a common stochastic trend,
   their linear combination is stationary. This implies a genuine
   long-run equilibrium relationship.

3. ERROR CORRECTION MODEL: Cointegrated variables adjust toward their
   long-run equilibrium:
   - Short-run effect (δ): {delta:.3f} 
   - Long-run effect (β):  {beta_lr:.3f}
   - Speed of adjustment:  {abs(alpha_ec)*100:.1f}% per period
   - Half-life:            {half_life:.1f} periods
""")
print("=" * 70)
print("\nFigures saved:")
print("  - example2_spurious_regression.png")
print("  - example2_cointegration.png")
print("  - example2_ecm.png")
print("  - example2_comparison.png")

# =============================================================================
# Export data for R and MATLAB compatibility
# =============================================================================
# Save the cointegrated series so R and MATLAB can load identical values.
# Required when USE_PYTHON_DATA = TRUE in the R/MATLAB scripts.
import pandas as pd
pd.DataFrame({'income': income, 'consumption': consumption}).to_csv('ex2_data.csv', index=False)
print("\nData exported: ex2_data.csv (for R and MATLAB)")
