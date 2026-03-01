"""
ARMA Process Simulation and Visualization
==========================================
This script generates artificial time series for:
- White Noise
- AR(1): y_t = phi * y_{t-1} + epsilon_t
- MA(1): y_t = epsilon_t + theta * epsilon_{t-1}
- ARMA(1,1): y_t = phi * y_{t-1} + epsilon_t + theta * epsilon_{t-1}

Author:   Alessia Paccagnini
Textbook: Macroeconometrics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Output directory: same folder as this script (portable across machines)
out_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'

# Set seed for reproducibility
np.random.seed(42)

# =============================================================================
# Parameters
# =============================================================================
T = 500          # Number of observations
sigma = 1.0      # Standard deviation of white noise
phi = 0.7        # AR(1) coefficient
theta = 0.5      # MA(1) coefficient

# =============================================================================
# Method 1: Manual Generation (for pedagogical clarity)
# =============================================================================

def generate_white_noise(T, sigma=1.0):
    """Generate white noise process: epsilon_t ~ N(0, sigma^2)"""
    return np.random.normal(0, sigma, T)

def generate_ar1(T, phi, sigma=1.0):
    """
    Generate AR(1) process: y_t = phi * y_{t-1} + epsilon_t
    Requires |phi| < 1 for stationarity
    """
    if abs(phi) >= 1:
        raise ValueError(f"AR(1) requires |phi| < 1 for stationarity. Got phi = {phi}")
    
    epsilon = np.random.normal(0, sigma, T)
    y = np.zeros(T)
    
    # Initialize with unconditional variance draw
    y[0] = np.random.normal(0, sigma / np.sqrt(1 - phi**2))
    
    for t in range(1, T):
        y[t] = phi * y[t-1] + epsilon[t]
    
    return y, epsilon

def generate_ma1(T, theta, sigma=1.0):
    """
    Generate MA(1) process: y_t = epsilon_t + theta * epsilon_{t-1}
    Always stationary for any theta
    """
    epsilon = np.random.normal(0, sigma, T + 1)  # Need one extra for t=0
    y = np.zeros(T)
    
    for t in range(T):
        y[t] = epsilon[t + 1] + theta * epsilon[t]
    
    return y, epsilon[1:]

def generate_arma11(T, phi, theta, sigma=1.0):
    """
    Generate ARMA(1,1) process: y_t = phi * y_{t-1} + epsilon_t + theta * epsilon_{t-1}
    Requires |phi| < 1 for stationarity
    """
    if abs(phi) >= 1:
        raise ValueError(f"ARMA(1,1) requires |phi| < 1 for stationarity. Got phi = {phi}")
    
    epsilon = np.random.normal(0, sigma, T + 1)
    y = np.zeros(T)
    
    # Initialize
    y[0] = epsilon[1] + theta * epsilon[0]
    
    for t in range(1, T):
        y[t] = phi * y[t-1] + epsilon[t + 1] + theta * epsilon[t]
    
    return y, epsilon[1:]

# =============================================================================
# Method 2: Using statsmodels ArmaProcess (for verification)
# =============================================================================

def generate_using_statsmodels(T, phi=0, theta=0):
    """
    Generate ARMA process using statsmodels
    Note: statsmodels uses the convention where AR coefficients have opposite sign
    AR polynomial: [1, -phi] means y_t - phi*y_{t-1} = epsilon_t
    MA polynomial: [1, theta] means epsilon_t + theta*epsilon_{t-1}
    """
    ar_params = np.array([1, -phi]) if phi != 0 else np.array([1])
    ma_params = np.array([1, theta]) if theta != 0 else np.array([1])
    
    arma_process = ArmaProcess(ar_params, ma_params)
    return arma_process.generate_sample(nsample=T)

# =============================================================================
# Generate all processes
# =============================================================================

# White noise
wn = generate_white_noise(T, sigma)

# AR(1)
ar1, eps_ar1 = generate_ar1(T, phi, sigma)

# MA(1)
ma1, eps_ma1 = generate_ma1(T, theta, sigma)

# ARMA(1,1)
arma11, eps_arma11 = generate_arma11(T, phi, theta, sigma)

# =============================================================================
# Figure 1: Time Series Plots
# =============================================================================

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(wn, color='steelblue', linewidth=0.8)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
axes[0].set_title(r'White Noise: $\varepsilon_t \sim N(0, 1)$', fontsize=12)
axes[0].set_ylabel(r'$\varepsilon_t$')

axes[1].plot(ar1, color='darkgreen', linewidth=0.8)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
axes[1].set_title(r'AR(1): $y_t = 0.7 y_{t-1} + \varepsilon_t$', fontsize=12)
axes[1].set_ylabel(r'$y_t$')

axes[2].plot(ma1, color='darkorange', linewidth=0.8)
axes[2].axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
axes[2].set_title(r'MA(1): $y_t = \varepsilon_t + 0.5 \varepsilon_{t-1}$', fontsize=12)
axes[2].set_ylabel(r'$y_t$')

axes[3].plot(arma11, color='purple', linewidth=0.8)
axes[3].axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
axes[3].set_title(r'ARMA(1,1): $y_t = 0.7 y_{t-1} + \varepsilon_t + 0.5 \varepsilon_{t-1}$', fontsize=12)
axes[3].set_ylabel(r'$y_t$')
axes[3].set_xlabel('Time')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'arma_time_series.png'), dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Figure 2: ACF Comparison
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# White Noise ACF
plot_acf(wn, ax=axes[0, 0], lags=20, title='ACF: White Noise')
axes[0, 0].set_xlabel('Lag')

# AR(1) ACF
plot_acf(ar1, ax=axes[0, 1], lags=20, title=r'ACF: AR(1) with $\phi = 0.7$')
axes[0, 1].set_xlabel('Lag')

# MA(1) ACF
plot_acf(ma1, ax=axes[1, 0], lags=20, title=r'ACF: MA(1) with $\theta = 0.5$')
axes[1, 0].set_xlabel('Lag')

# ARMA(1,1) ACF
plot_acf(arma11, ax=axes[1, 1], lags=20, title=r'ACF: ARMA(1,1) with $\phi = 0.7$, $\theta = 0.5$')
axes[1, 1].set_xlabel('Lag')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'arma_acf_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Figure 3: PACF Comparison
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# White Noise PACF
plot_pacf(wn, ax=axes[0, 0], lags=20, title='PACF: White Noise', method='ywm')
axes[0, 0].set_xlabel('Lag')

# AR(1) PACF
plot_pacf(ar1, ax=axes[0, 1], lags=20, title=r'PACF: AR(1) with $\phi = 0.7$', method='ywm')
axes[0, 1].set_xlabel('Lag')

# MA(1) PACF
plot_pacf(ma1, ax=axes[1, 0], lags=20, title=r'PACF: MA(1) with $\theta = 0.5$', method='ywm')
axes[1, 0].set_xlabel('Lag')

# ARMA(1,1) PACF
plot_pacf(arma11, ax=axes[1, 1], lags=20, title=r'PACF: ARMA(1,1) with $\phi = 0.7$, $\theta = 0.5$', method='ywm')
axes[1, 1].set_xlabel('Lag')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'arma_pacf_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Print Theoretical vs Sample Statistics
# =============================================================================

print("=" * 70)
print("THEORETICAL vs SAMPLE STATISTICS")
print("=" * 70)

# White Noise
print("\n--- White Noise ---")
print(f"Theoretical Mean: 0.000 | Sample Mean: {np.mean(wn):.3f}")
print(f"Theoretical Var:  {sigma**2:.3f} | Sample Var:  {np.var(wn):.3f}")

# AR(1)
print(f"\n--- AR(1): phi = {phi} ---")
ar1_theoretical_var = sigma**2 / (1 - phi**2)
ar1_theoretical_acf1 = phi
print(f"Theoretical Mean: 0.000 | Sample Mean: {np.mean(ar1):.3f}")
print(f"Theoretical Var:  {ar1_theoretical_var:.3f} | Sample Var:  {np.var(ar1):.3f}")
print(f"Theoretical ACF(1): {ar1_theoretical_acf1:.3f} | Sample ACF(1): {np.corrcoef(ar1[:-1], ar1[1:])[0,1]:.3f}")

# MA(1)
print(f"\n--- MA(1): theta = {theta} ---")
ma1_theoretical_var = sigma**2 * (1 + theta**2)
ma1_theoretical_acf1 = theta / (1 + theta**2)
print(f"Theoretical Mean: 0.000 | Sample Mean: {np.mean(ma1):.3f}")
print(f"Theoretical Var:  {ma1_theoretical_var:.3f} | Sample Var:  {np.var(ma1):.3f}")
print(f"Theoretical ACF(1): {ma1_theoretical_acf1:.3f} | Sample ACF(1): {np.corrcoef(ma1[:-1], ma1[1:])[0,1]:.3f}")

# ARMA(1,1)
print(f"\n--- ARMA(1,1): phi = {phi}, theta = {theta} ---")
arma11_theoretical_var = sigma**2 * (1 + theta**2 + 2*phi*theta) / (1 - phi**2)
arma11_theoretical_acf1 = (phi + theta) * (1 + phi*theta) / (1 + theta**2 + 2*phi*theta)
print(f"Theoretical Mean: 0.000 | Sample Mean: {np.mean(arma11):.3f}")
print(f"Theoretical Var:  {arma11_theoretical_var:.3f} | Sample Var:  {np.var(arma11):.3f}")
print(f"Theoretical ACF(1): {arma11_theoretical_acf1:.3f} | Sample ACF(1): {np.corrcoef(arma11[:-1], arma11[1:])[0,1]:.3f}")

print("\n" + "=" * 70)
print("ACF/PACF IDENTIFICATION PATTERNS")
print("=" * 70)
print("""
Process     | ACF Pattern                    | PACF Pattern
------------|--------------------------------|----------------------------------
White Noise | No significant autocorrelations| No significant partial autocorr.
AR(p)       | Geometric decay (or damped     | Cuts off after lag p
            | oscillations if complex roots) |
MA(q)       | Cuts off after lag q           | Geometric decay (or damped
            |                                | oscillations)
ARMA(p,q)   | Decays after lag q             | Decays after lag p
""")

print("\nFigures saved:")
print("  - arma_time_series.png")
print("  - arma_acf_comparison.png")
print("  - arma_pacf_comparison.png")
