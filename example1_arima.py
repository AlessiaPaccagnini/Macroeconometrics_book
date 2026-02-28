"""
Example 1: ARIMA Model Selection using Box-Jenkins Methodology
==============================================================
This example demonstrates the three-stage Box-Jenkins approach:
1. Identification: Plot data, test for stationarity, examine ACF/PACF
2. Estimation: Fit ARIMA model
3. Diagnostic Checking: Analyze residuals

Author: Alessia Paccagnini
Textbook: Macroeconometrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Step 0: Load or simulate data (US GDP growth)
# =============================================================================
np.random.seed(42)

# Simulate GDP growth as an AR(2) process with some MA component
n = 200
eps = np.random.normal(0, 1, n)
y = np.zeros(n)
for t in range(2, n):
    y[t] = 0.5 + 0.4*y[t-1] - 0.2*y[t-2] + eps[t] + 0.3*eps[t-1]

# Create a pandas series with quarterly dates
dates = pd.date_range(start='1970-01-01', periods=n, freq='QE')
gdp_growth = pd.Series(y, index=dates, name='GDP_Growth')

print("=" * 70)
print("EXAMPLE 1: ARIMA MODEL SELECTION - BOX-JENKINS METHODOLOGY")
print("=" * 70)

# =============================================================================
# Stage 1: IDENTIFICATION
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 1: IDENTIFICATION")
print("=" * 70)

# 1.1 Visual inspection
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Time series plot
axes[0].plot(gdp_growth, 'b-', linewidth=0.8)
axes[0].axhline(y=gdp_growth.mean(), color='r', linestyle='--', label=f'Mean = {gdp_growth.mean():.2f}')
axes[0].set_title('GDP Growth Rate (Simulated)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Percent')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ACF
plot_acf(gdp_growth, lags=20, ax=axes[1], alpha=0.05)
axes[1].set_title('Autocorrelation Function (ACF)')

# PACF
plot_pacf(gdp_growth, lags=20, ax=axes[2], alpha=0.05, method='ywm')
axes[2].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.savefig('example1_identification.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n1.1 Visual Inspection:")
print("    - Time series plot shows fluctuations around a constant mean")
print("    - No obvious trend detected")
print("    - See 'example1_identification.png' for ACF/PACF plots")

# 1.2 Unit Root Tests
print("\n1.2 Unit Root Tests:")

# ADF Test
adf_result = adfuller(gdp_growth, autolag='AIC')
print(f"\n    ADF Test (H0: unit root):")
print(f"    Test Statistic: {adf_result[0]:.4f}")
print(f"    p-value: {adf_result[1]:.4f}")
print(f"    Critical Values: 1%: {adf_result[4]['1%']:.2f}, 5%: {adf_result[4]['5%']:.2f}, 10%: {adf_result[4]['10%']:.2f}")
print(f"    Conclusion: {'Reject H0 - Series is stationary' if adf_result[1] < 0.05 else 'Fail to reject H0 - Series has unit root'}")

# KPSS Test
kpss_result = kpss(gdp_growth, regression='c', nlags='auto')
print(f"\n    KPSS Test (H0: stationarity):")
print(f"    Test Statistic: {kpss_result[0]:.4f}")
print(f"    p-value: {kpss_result[1]:.4f}")
print(f"    Critical Values: 1%: {kpss_result[3]['1%']:.2f}, 5%: {kpss_result[3]['5%']:.2f}, 10%: {kpss_result[3]['10%']:.2f}")
print(f"    Conclusion: {'Reject H0 - Series is non-stationary' if kpss_result[1] < 0.05 else 'Fail to reject H0 - Series is stationary'}")

print("\n    Combined inference: Both tests suggest the series is STATIONARY (d=0)")

# 1.3 ACF/PACF interpretation
print("\n1.3 ACF/PACF Interpretation:")
print("    - ACF: Gradual decay suggests AR component")
print("    - PACF: Significant spikes at lags 1 and 2, then cuts off")
print("    - Tentative model: AR(2) or ARMA(2,1)")

# =============================================================================
# Stage 2: ESTIMATION
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 2: ESTIMATION")
print("=" * 70)

# Estimate multiple models and compare using AIC/BIC
models_to_try = [(1,0,0), (2,0,0), (1,0,1), (2,0,1), (2,0,2)]
results = []

print("\n2.1 Model Comparison (AIC/BIC):")
print("-" * 50)
print(f"{'Model':<15} {'AIC':<12} {'BIC':<12}")
print("-" * 50)

for order in models_to_try:
    try:
        model = ARIMA(gdp_growth, order=order)
        fitted = model.fit()
        results.append({
            'order': order,
            'aic': fitted.aic,
            'bic': fitted.bic,
            'model': fitted
        })
        print(f"ARIMA{order}      {fitted.aic:<12.2f} {fitted.bic:<12.2f}")
    except:
        print(f"ARIMA{order}      Failed to converge")

print("-" * 50)

# Select best model by BIC (more parsimonious)
best_model = min(results, key=lambda x: x['bic'])
print(f"\nBest model by BIC: ARIMA{best_model['order']}")

# Detailed estimation results
print("\n2.2 Estimation Results for Best Model:")
print(best_model['model'].summary().tables[1])

# =============================================================================
# Stage 3: DIAGNOSTIC CHECKING
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 3: DIAGNOSTIC CHECKING")
print("=" * 70)

fitted_model = best_model['model']
residuals = fitted_model.resid

# 3.1 Residual plots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Standardized residuals
axes[0, 0].plot(residuals, 'b-', linewidth=0.8)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_title('Standardized Residuals')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].grid(True, alpha=0.3)

# Histogram
axes[0, 1].hist(residuals, bins=25, density=True, alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Histogram of Residuals')
axes[0, 1].set_xlabel('Residuals')

# ACF of residuals
plot_acf(residuals, lags=20, ax=axes[1, 0], alpha=0.05)
axes[1, 0].set_title('ACF of Residuals')

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot')

plt.tight_layout()
plt.savefig('example1_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n3.1 Residual Analysis:")
print("    See 'example1_diagnostics.png' for diagnostic plots")

# 3.2 Ljung-Box test
print("\n3.2 Ljung-Box Test for Residual Autocorrelation:")
lb_test = acorr_ljungbox(residuals, lags=[5, 10, 15, 20],
                         model_df=best_model['order'][0] + best_model['order'][2],
                         return_df=True)
print(lb_test.to_string())

if all(lb_test['lb_pvalue'] > 0.05):
    print("\n    Conclusion: No significant autocorrelation in residuals (p > 0.05)")
    print("    The model passes the diagnostic check!")
else:
    print("\n    Conclusion: Some autocorrelation remains - consider revising the model")

# 3.3 Summary statistics
print("\n3.3 Residual Summary Statistics:")
print(f"    Mean: {residuals.mean():.4f} (should be ≈ 0)")
print(f"    Std Dev: {residuals.std():.4f}")
print(f"    Skewness: {stats.skew(residuals):.4f} (should be ≈ 0)")
print(f"    Kurtosis: {stats.kurtosis(residuals):.4f} (should be ≈ 0 for normal)")

# Jarque-Bera test for normality
jb_stat, jb_pvalue = stats.jarque_bera(residuals)
print(f"\n    Jarque-Bera test: statistic = {jb_stat:.2f}, p-value = {jb_pvalue:.4f}")
print(f"    {'Residuals appear normally distributed' if jb_pvalue > 0.05 else 'Residuals deviate from normality'}")

print("\n" + "=" * 70)
print("FINAL MODEL: ARIMA" + str(best_model['order']))
print("=" * 70)


# =============================================================================
# Export data for R and MATLAB compatibility
# =============================================================================
# Save the simulated data so R and MATLAB can load identical values.
# Required when USE_PYTHON_DATA = TRUE in the R/MATLAB scripts.
pd.DataFrame({'y': y}).to_csv('ex1_data.csv', index=False)
print("\nData exported: ex1_data.csv (for R and MATLAB)")
