"""
Example 3: GARCH Models for Volatility Modeling
================================================
This example demonstrates:
1. Testing for ARCH effects (ARCH-LM test)
2. Estimating GARCH(1,1) model
3. Interpreting results and volatility persistence
4. Comparing symmetric vs asymmetric GARCH models

Author: Alessia Paccagnini
Textbook: Macroeconometrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
import warnings
warnings.filterwarnings('ignore')

np.random.seed(456)
n = 1000

print("=" * 70)
print("EXAMPLE 3: GARCH MODELS FOR VOLATILITY MODELING")
print("=" * 70)

# =============================================================================
# Generate data with GARCH(1,1) volatility
# =============================================================================
print("\n" + "-" * 50)
print("Data Generation: Simulated Stock Returns with GARCH(1,1)")
print("-" * 50)

# True parameters
omega_true = 0.05
alpha_true = 0.10
beta_true = 0.85

# Simulate GARCH(1,1) process
returns = np.zeros(n)
sigma2 = np.zeros(n)
sigma2[0] = omega_true / (1 - alpha_true - beta_true)  # Unconditional variance

for t in range(1, n):
    sigma2[t] = omega_true + alpha_true * returns[t-1]**2 + beta_true * sigma2[t-1]
    returns[t] = np.sqrt(sigma2[t]) * np.random.standard_normal()

# Create DataFrame
dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
data = pd.DataFrame({'returns': returns, 'sigma2': sigma2}, index=dates)

print(f"\nTrue GARCH(1,1) parameters:")
print(f"    ω (omega): {omega_true}")
print(f"    α (alpha): {alpha_true}")
print(f"    β (beta):  {beta_true}")
print(f"    α + β:     {alpha_true + beta_true} (persistence)")
print(f"    Unconditional variance: {omega_true / (1 - alpha_true - beta_true):.4f}")

# =============================================================================
# Visualize the data
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(11, 9))

# Returns
axes[0].plot(data['returns'], 'b-', linewidth=0.5)
axes[0].axhline(0, color='black', linewidth=0.5)
axes[0].set_title('Simulated Stock Returns', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Returns')
axes[0].grid(True, alpha=0.3)

# Squared returns (proxy for volatility)
axes[1].plot(data['returns']**2, 'r-', linewidth=0.5, alpha=0.7, label='Squared Returns')
axes[1].plot(data['sigma2'], 'b-', linewidth=1, label='True Conditional Variance')
axes[1].set_title('Volatility Clustering', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Squared Returns / Variance')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Distribution of returns
axes[2].hist(data['returns'], bins=50, density=True, alpha=0.7, edgecolor='black', label='Returns')
x = np.linspace(data['returns'].min(), data['returns'].max(), 100)
axes[2].plot(x, stats.norm.pdf(x, data['returns'].mean(), data['returns'].std()), 
             'r-', linewidth=2, label='Normal Distribution')
axes[2].set_title('Distribution of Returns (Fat Tails)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Returns')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('example3_garch_data.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSee 'example3_garch_data.png' for visualization")

# Summary statistics
print("\nSummary Statistics:")
print(f"    Mean:     {data['returns'].mean():.4f}")
print(f"    Std Dev:  {data['returns'].std():.4f}")
print(f"    Skewness: {stats.skew(data['returns']):.4f}")
print(f"    Kurtosis: {stats.kurtosis(data['returns']):.4f} (excess, normal = 0)")
print("    → Excess kurtosis indicates fat tails (leptokurtosis)")

# =============================================================================
# Step 1: Test for ARCH Effects
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: TESTING FOR ARCH EFFECTS (ARCH-LM Test)")
print("=" * 70)

print("\nARCH-LM Test: H₀: No ARCH effects (homoskedasticity)")
print("             H₁: ARCH effects present (heteroskedasticity)")
print("-" * 50)

for lags in [1, 5, 10]:
    arch_test = het_arch(data['returns'], nlags=lags)
    print(f"    Lags = {lags:2d}: LM stat = {arch_test[0]:8.3f}, p-value = {arch_test[1]:.6f} {'***' if arch_test[1] < 0.01 else ''}")

print("-" * 50)
print("    *** indicates significance at 1% level")
print("\n    ✓ Strong evidence of ARCH effects → GARCH modeling is appropriate")

# =============================================================================
# Step 2: Estimate GARCH(1,1) Model
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: GARCH(1,1) ESTIMATION")
print("=" * 70)

# Fit GARCH(1,1)
model_garch = arch_model(data['returns'], vol='Garch', p=1, q=1, mean='Constant')
results_garch = model_garch.fit(disp='off')

print("\n" + str(results_garch.summary().tables[0]))
print("\nParameter Estimates:")
print("-" * 60)
print(results_garch.summary().tables[1])
print("-" * 60)

# Extract parameters
omega_hat = results_garch.params['omega']
alpha_hat = results_garch.params['alpha[1]']
beta_hat = results_garch.params['beta[1]']
persistence = alpha_hat + beta_hat

print(f"\nParameter Interpretation:")
print(f"    ω (omega):    {omega_hat:.6f} (intercept)")
print(f"    α (alpha):    {alpha_hat:.4f} (reaction to shocks)")
print(f"    β (beta):     {beta_hat:.4f} (volatility persistence)")
print(f"    α + β:        {persistence:.4f} (total persistence)")

# Compare to true values
print(f"\nComparison with True Values:")
print(f"    {'Parameter':<12} {'True':>10} {'Estimated':>10}")
print(f"    {'-'*32}")
print(f"    {'omega':<12} {omega_true:>10.4f} {omega_hat:>10.4f}")
print(f"    {'alpha':<12} {alpha_true:>10.4f} {alpha_hat:>10.4f}")
print(f"    {'beta':<12} {beta_true:>10.4f} {beta_hat:>10.4f}")

# Unconditional variance and half-life
uncond_var = omega_hat / (1 - alpha_hat - beta_hat)
half_life = np.log(0.5) / np.log(persistence)

print(f"\nImplied Quantities:")
print(f"    Unconditional variance: {uncond_var:.4f}")
print(f"    Unconditional std dev:  {np.sqrt(uncond_var):.4f}")
print(f"    Half-life of shocks:    {half_life:.1f} periods")
print(f"    (Time for volatility shock to decay by 50%)")

# =============================================================================
# Step 3: Model Comparison (GARCH vs GJR-GARCH)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: MODEL COMPARISON (Symmetric vs Asymmetric)")
print("=" * 70)

# Fit GJR-GARCH (asymmetric)
model_gjr = arch_model(data['returns'], vol='Garch', p=1, o=1, q=1, mean='Constant')
results_gjr = model_gjr.fit(disp='off')

# Fit EGARCH
model_egarch = arch_model(data['returns'], vol='EGARCH', p=1, q=1, mean='Constant')
results_egarch = model_egarch.fit(disp='off')

print("\nModel Comparison:")
print("-" * 60)
print(f"{'Model':<20} {'Log-Lik':>12} {'AIC':>12} {'BIC':>12}")
print("-" * 60)
print(f"{'GARCH(1,1)':<20} {results_garch.loglikelihood:>12.2f} {results_garch.aic:>12.2f} {results_garch.bic:>12.2f}")
print(f"{'GJR-GARCH(1,1,1)':<20} {results_gjr.loglikelihood:>12.2f} {results_gjr.aic:>12.2f} {results_gjr.bic:>12.2f}")
print(f"{'EGARCH(1,1)':<20} {results_egarch.loglikelihood:>12.2f} {results_egarch.aic:>12.2f} {results_egarch.bic:>12.2f}")
print("-" * 60)

# Find best model
models = {'GARCH(1,1)': results_garch, 'GJR-GARCH': results_gjr, 'EGARCH': results_egarch}
best_model = min(models.items(), key=lambda x: x[1].bic)
print(f"\nBest model by BIC: {best_model[0]}")

# Check asymmetry in GJR-GARCH
gamma_gjr = results_gjr.params.get('gamma[1]', 0)
print(f"\nGJR-GARCH asymmetry parameter (γ): {gamma_gjr:.4f}")
if abs(gamma_gjr) > 0.01:
    print("    → Evidence of asymmetric volatility (leverage effect)")
else:
    print("    → No significant asymmetry (symmetric GARCH sufficient)")

# =============================================================================
# Step 4: Diagnostic Checking
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: DIAGNOSTIC CHECKING")
print("=" * 70)

# Standardized residuals
std_resid = results_garch.std_resid

print("\nStandardized Residuals Analysis:")
print(f"    Mean:     {std_resid.mean():.4f} (should be ≈ 0)")
print(f"    Std Dev:  {std_resid.std():.4f} (should be ≈ 1)")
print(f"    Skewness: {stats.skew(std_resid):.4f}")
print(f"    Kurtosis: {stats.kurtosis(std_resid):.4f}")

# ARCH-LM test on standardized residuals
print("\nARCH-LM Test on Standardized Residuals:")
print("    (Should show NO remaining ARCH effects)")
for lags in [5, 10]:
    arch_test_resid = het_arch(std_resid, nlags=lags)
    result = "No ARCH effects ✓" if arch_test_resid[1] > 0.05 else "ARCH effects remain ✗"
    print(f"    Lags = {lags:2d}: p-value = {arch_test_resid[1]:.4f} → {result}")

# Ljung-Box test on squared standardized residuals
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(std_resid**2, lags=[10, 20], return_df=True)
print("\nLjung-Box Test on Squared Standardized Residuals:")
print(lb_test.to_string())

# =============================================================================
# Visualization of results
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# Conditional volatility
cond_vol = results_garch.conditional_volatility
axes[0, 0].plot(data.index, cond_vol, 'b-', linewidth=0.8, label='Estimated')
axes[0, 0].plot(data.index, np.sqrt(data['sigma2']), 'r--', linewidth=0.8, alpha=0.7, label='True')
axes[0, 0].set_title('Conditional Volatility (σ_t)', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Returns with volatility bands
axes[0, 1].plot(data.index, data['returns'], 'gray', linewidth=0.3, alpha=0.7)
axes[0, 1].plot(data.index, 2*cond_vol, 'r-', linewidth=1, label='±2σ bands')
axes[0, 1].plot(data.index, -2*cond_vol, 'r-', linewidth=1)
axes[0, 1].set_title('Returns with 95% Volatility Bands', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Standardized residuals
axes[1, 0].plot(std_resid, 'b-', linewidth=0.5)
axes[1, 0].axhline(0, color='black', linewidth=0.5)
axes[1, 0].axhline(2, color='red', linestyle='--', linewidth=0.5)
axes[1, 0].axhline(-2, color='red', linestyle='--', linewidth=0.5)
axes[1, 0].set_title('Standardized Residuals', fontweight='bold')
axes[1, 0].set_ylabel('z_t = ε_t / σ_t')
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(std_resid, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot of Standardized Residuals', fontweight='bold')

plt.tight_layout()
plt.savefig('example3_garch_results.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSee 'example3_garch_results.png' for diagnostic plots")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
GARCH(1,1) Model Results:
-------------------------
    σ²_t = {omega_hat:.4f} + {alpha_hat:.4f}·ε²_{'{t-1}'} + {beta_hat:.4f}·σ²_{'{t-1}'}

Key Findings:
    • Strong ARCH effects detected (ARCH-LM test highly significant)
    • High persistence (α + β = {persistence:.3f}): volatility shocks are long-lasting
    • Half-life of {half_life:.1f} periods for volatility to return halfway to normal
    • Model diagnostics: standardized residuals show no remaining ARCH effects
    
Interpretation:
    • α = {alpha_hat:.3f}: {alpha_hat*100:.1f}% of yesterday's squared shock affects today's variance
    • β = {beta_hat:.3f}: {beta_hat*100:.1f}% of yesterday's variance persists to today
    • When α + β is close to 1, the process is nearly integrated (IGARCH)
""")
print("=" * 70)


# =============================================================================
# Export data for R and MATLAB compatibility
# =============================================================================
# Save the simulated returns so R and MATLAB can load identical values.
# Required when USE_PYTHON_DATA = TRUE in the R/MATLAB scripts.
pd.DataFrame({'returns': returns, 'sigma2': sigma2}).to_csv('ex3_data.csv', index=False)
print("\nData exported: ex3_data.csv (for R and MATLAB)")
