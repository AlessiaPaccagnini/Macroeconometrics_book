"""
Nonstationary Processes: Simulation and Visualization
======================================================
This script generates artificial time series for:
- Deterministic Trend: y_t = alpha + beta*t + u_t (where u_t is AR(1))
- Random Walk: y_t = y_{t-1} + epsilon_t
- Random Walk with Drift: y_t = delta + y_{t-1} + epsilon_t

Author:   Alessia Paccagnini
Textbook: Macroeconometrics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf

# Output directory: same folder as this script (portable across machines)
out_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'

# Set seed for reproducibility
np.random.seed(42)

# Set matplotlib style for publication quality
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

# =============================================================================
# Parameters
# =============================================================================
T = 250          # Number of observations
sigma = 1.0      # Standard deviation of white noise

# Deterministic trend parameters
alpha = 2.0      # Intercept
beta = 0.1       # Trend slope
phi_u = 0.7      # AR(1) coefficient for stationary component

# Random walk with drift parameter
delta = 0.15     # Drift term

# =============================================================================
# Generate Processes
# =============================================================================

def generate_deterministic_trend(T, alpha, beta, phi, sigma=1.0):
    """
    Generate trend-stationary process: y_t = alpha + beta*t + u_t
    where u_t = phi*u_{t-1} + epsilon_t is AR(1)
    """
    epsilon = np.random.normal(0, sigma, T)
    u = np.zeros(T)
    
    # Generate AR(1) stationary component
    u[0] = np.random.normal(0, sigma / np.sqrt(1 - phi**2))
    for t in range(1, T):
        u[t] = phi * u[t-1] + epsilon[t]
    
    # Add deterministic trend
    t_index = np.arange(T)
    y = alpha + beta * t_index + u
    
    return y, u, t_index

def generate_random_walk(T, sigma=1.0, y0=0):
    """
    Generate random walk: y_t = y_{t-1} + epsilon_t
    """
    epsilon = np.random.normal(0, sigma, T)
    y = np.zeros(T)
    y[0] = y0 + epsilon[0]
    
    for t in range(1, T):
        y[t] = y[t-1] + epsilon[t]
    
    return y, epsilon

def generate_random_walk_drift(T, delta, sigma=1.0, y0=0):
    """
    Generate random walk with drift: y_t = delta + y_{t-1} + epsilon_t
    """
    epsilon = np.random.normal(0, sigma, T)
    y = np.zeros(T)
    y[0] = y0 + delta + epsilon[0]
    
    for t in range(1, T):
        y[t] = delta + y[t-1] + epsilon[t]
    
    return y, epsilon

# Generate all processes
y_det, u_det, t_index = generate_deterministic_trend(T, alpha, beta, phi_u, sigma)
y_rw, eps_rw = generate_random_walk(T, sigma)
y_rwd, eps_rwd = generate_random_walk_drift(T, delta, sigma)

# Also generate white noise for comparison
wn = np.random.normal(0, sigma, T)

# =============================================================================
# Figure 1: Individual Time Series Plots with Enhanced Detail
# =============================================================================

# --- Deterministic Trend ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_det, color='steelblue', linewidth=0.9, label=r'$y_t = 2 + 0.1t + u_t$')
ax.plot(t_index, alpha + beta * t_index, color='red', linestyle='--', 
        linewidth=2, label=r'Deterministic trend: $2 + 0.1t$')
ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
ax.set_xlabel('Time')
ax.set_ylabel(r'$y_t$')
ax.set_title(r'Deterministic Trend: $y_t = \alpha + \beta t + u_t$ where $u_t \sim AR(1)$')
ax.legend(loc='upper left')
ax.set_xlim(0, T-1)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'deterministic_trend.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Random Walk ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_rw, color='darkgreen', linewidth=0.9)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Initial level $y_0 = 0$')
ax.fill_between(range(T), y_rw, alpha=0.3, color='darkgreen')
ax.set_xlabel('Time')
ax.set_ylabel(r'$y_t$')
ax.set_title(r'Random Walk: $y_t = y_{t-1} + \varepsilon_t$')
ax.legend(loc='best')
ax.set_xlim(0, T-1)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'random_walk.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Random Walk with Drift ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_rwd, color='purple', linewidth=0.9, label=r'$y_t = 0.15 + y_{t-1} + \varepsilon_t$')
ax.plot(t_index, delta * t_index, color='red', linestyle='--', 
        linewidth=2, label=r'Drift component: $0.15t$')
ax.set_xlabel('Time')
ax.set_ylabel(r'$y_t$')
ax.set_title(r'Random Walk with Drift: $y_t = \delta + y_{t-1} + \varepsilon_t$')
ax.legend(loc='upper left')
ax.set_xlim(0, T-1)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'random_walk_drift.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 2: Combined Time Series Comparison
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Deterministic Trend
axes[0].plot(y_det, color='steelblue', linewidth=0.8, label='Series')
axes[0].plot(t_index, alpha + beta * t_index, color='red', linestyle='--', 
             linewidth=2, label='Trend')
axes[0].set_ylabel(r'$y_t$')
axes[0].set_title(r'Deterministic Trend: $y_t = 2 + 0.1t + u_t$ where $u_t = 0.7u_{t-1} + \varepsilon_t$')
axes[0].legend(loc='upper left')

# Random Walk
axes[1].plot(y_rw, color='darkgreen', linewidth=0.8)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].set_ylabel(r'$y_t$')
axes[1].set_title(r'Random Walk: $y_t = y_{t-1} + \varepsilon_t$ (variance grows with $t$)')

# Random Walk with Drift
axes[2].plot(y_rwd, color='purple', linewidth=0.8, label='Series')
axes[2].plot(t_index, delta * t_index, color='red', linestyle='--', 
             linewidth=2, label='Drift')
axes[2].set_ylabel(r'$y_t$')
axes[2].set_xlabel('Time')
axes[2].set_title(r'Random Walk with Drift: $y_t = 0.15 + y_{t-1} + \varepsilon_t$')
axes[2].legend(loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'nonstationary_time_series.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 3: ACF Comparison - Why Correlogram Fails for Unit Roots
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

nlags = 30

# White Noise ACF (for reference)
plot_acf(wn, ax=axes[0, 0], lags=nlags, title='ACF: White Noise (Stationary)')
axes[0, 0].set_xlabel('Lag')
axes[0, 0].set_ylim(-0.3, 1.1)

# Deterministic Trend ACF (detrended residuals would be stationary)
plot_acf(y_det, ax=axes[0, 1], lags=nlags, title='ACF: Deterministic Trend (before detrending)')
axes[0, 1].set_xlabel('Lag')
axes[0, 1].set_ylim(-0.3, 1.1)

# Random Walk ACF
plot_acf(y_rw, ax=axes[1, 0], lags=nlags, title='ACF: Random Walk')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylim(-0.3, 1.1)

# Random Walk with Drift ACF
plot_acf(y_rwd, ax=axes[1, 1], lags=nlags, title='ACF: Random Walk with Drift')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylim(-0.3, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'nonstationary_acf_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 4: PACF Comparison
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# White Noise PACF
plot_pacf(wn, ax=axes[0, 0], lags=nlags, title='PACF: White Noise (Stationary)', method='ywm')
axes[0, 0].set_xlabel('Lag')

# Deterministic Trend PACF
plot_pacf(y_det, ax=axes[0, 1], lags=nlags, title='PACF: Deterministic Trend', method='ywm')
axes[0, 1].set_xlabel('Lag')

# Random Walk PACF
plot_pacf(y_rw, ax=axes[1, 0], lags=nlags, title='PACF: Random Walk', method='ywm')
axes[1, 0].set_xlabel('Lag')

# Random Walk with Drift PACF
plot_pacf(y_rwd, ax=axes[1, 1], lags=nlags, title='PACF: Random Walk with Drift', method='ywm')
axes[1, 1].set_xlabel('Lag')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'nonstationary_pacf_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 5: ACF Comparison - Levels vs First Differences
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# First row: ACF in levels
plot_acf(y_det, ax=axes[0, 0], lags=nlags, title='Deterministic Trend (levels)')
axes[0, 0].set_xlabel('Lag')
axes[0, 0].set_ylim(-0.3, 1.1)

plot_acf(y_rw, ax=axes[0, 1], lags=nlags, title='Random Walk (levels)')
axes[0, 1].set_xlabel('Lag')
axes[0, 1].set_ylim(-0.3, 1.1)

plot_acf(y_rwd, ax=axes[0, 2], lags=nlags, title='Random Walk with Drift (levels)')
axes[0, 2].set_xlabel('Lag')
axes[0, 2].set_ylim(-0.3, 1.1)

# Second row: ACF in first differences
dy_det = np.diff(y_det)
dy_rw = np.diff(y_rw)
dy_rwd = np.diff(y_rwd)

plot_acf(dy_det, ax=axes[1, 0], lags=nlags, title='Deterministic Trend (first diff.)')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylim(-0.5, 1.1)

plot_acf(dy_rw, ax=axes[1, 1], lags=nlags, title='Random Walk (first diff.) = White Noise')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylim(-0.5, 1.1)

plot_acf(dy_rwd, ax=axes[1, 2], lags=nlags, title='RW with Drift (first diff.) = White Noise')
axes[1, 2].set_xlabel('Lag')
axes[1, 2].set_ylim(-0.5, 1.1)

fig.suptitle('ACF in Levels vs. First Differences: Diagnosing Nonstationarity', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'acf_levels_vs_differences.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 6: Variance Growth Comparison
# =============================================================================

# Generate multiple realizations to show variance growth
n_simulations = 100
T_var = 200

rw_simulations = np.zeros((n_simulations, T_var))
det_simulations = np.zeros((n_simulations, T_var))

for i in range(n_simulations):
    rw_simulations[i, :], _ = generate_random_walk(T_var, sigma)
    det_simulations[i, :], _, _ = generate_deterministic_trend(T_var, 0, 0, phi_u, sigma)

# Compute sample variance at each time point
rw_var = np.var(rw_simulations, axis=0)
det_var = np.var(det_simulations, axis=0)
theoretical_rw_var = sigma**2 * np.arange(1, T_var + 1)
theoretical_det_var = sigma**2 / (1 - phi_u**2) * np.ones(T_var)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Random Walk variance
axes[0].plot(rw_var, color='darkgreen', linewidth=1.5, label='Sample variance')
axes[0].plot(theoretical_rw_var, color='red', linestyle='--', linewidth=2, 
             label=r'Theoretical: $t\sigma^2$')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Variance')
axes[0].set_title('Random Walk: Variance Grows Linearly with Time')
axes[0].legend()

# Stationary AR(1) variance
axes[1].plot(det_var, color='steelblue', linewidth=1.5, label='Sample variance')
axes[1].axhline(y=theoretical_det_var[0], color='red', linestyle='--', linewidth=2, 
                label=r'Theoretical: $\sigma^2/(1-\phi^2)$')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Variance')
axes[1].set_title('Stationary AR(1): Variance Remains Constant')
axes[1].legend()
axes[1].set_ylim(0, max(det_var) * 1.5)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'variance_growth_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 7: Multiple Random Walk Realizations (Fan Chart)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot multiple random walk paths
for i in range(30):
    y_temp, _ = generate_random_walk(T, sigma)
    axes[0].plot(y_temp, alpha=0.4, linewidth=0.7)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Time')
axes[0].set_ylabel(r'$y_t$')
axes[0].set_title('Random Walk: Multiple Realizations (Variance Expansion)')

# Plot multiple trend-stationary paths
for i in range(30):
    y_temp, _, _ = generate_deterministic_trend(T, alpha, beta, phi_u, sigma)
    axes[1].plot(y_temp, alpha=0.4, linewidth=0.7)
axes[1].plot(t_index, alpha + beta * t_index, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Time')
axes[1].set_ylabel(r'$y_t$')
axes[1].set_title('Deterministic Trend: Multiple Realizations (Constant Variance)')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'multiple_realizations.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# Print Summary Statistics
# =============================================================================

print("=" * 70)
print("NONSTATIONARY PROCESSES: SUMMARY STATISTICS")
print("=" * 70)

print("\n--- Deterministic Trend: y_t = 2 + 0.1t + u_t ---")
print(f"Sample Mean: {np.mean(y_det):.3f}")
print(f"Sample Variance: {np.var(y_det):.3f}")
print(f"ACF(1): {acf(y_det, nlags=1)[1]:.3f}")
print(f"ACF(10): {acf(y_det, nlags=10)[10]:.3f}")

print("\n--- Random Walk: y_t = y_{t-1} + epsilon_t ---")
print(f"Sample Mean: {np.mean(y_rw):.3f}")
print(f"Sample Variance: {np.var(y_rw):.3f}")
print(f"Theoretical Variance at T={T}: {T * sigma**2:.3f}")
print(f"ACF(1): {acf(y_rw, nlags=1)[1]:.3f}")
print(f"ACF(10): {acf(y_rw, nlags=10)[10]:.3f}")

print("\n--- Random Walk with Drift: y_t = 0.15 + y_{t-1} + epsilon_t ---")
print(f"Sample Mean: {np.mean(y_rwd):.3f}")
print(f"Sample Variance: {np.var(y_rwd):.3f}")
print(f"ACF(1): {acf(y_rwd, nlags=1)[1]:.3f}")
print(f"ACF(10): {acf(y_rwd, nlags=10)[10]:.3f}")

print("\n--- First Differences ---")
print(f"Δy (Random Walk) - Sample Mean: {np.mean(dy_rw):.3f}, Var: {np.var(dy_rw):.3f}")
print(f"Δy (RW with Drift) - Sample Mean: {np.mean(dy_rwd):.3f}, Var: {np.var(dy_rwd):.3f}")

print("\n" + "=" * 70)
print("KEY INSIGHT: WHY CORRELOGRAM FAILS FOR UNIT ROOTS")
print("=" * 70)
print("""
For a random walk, the sample ACF at lag k is approximately:

    ρ_k ≈ (T - k) / T → 1  as T → ∞

This means:
1. ALL autocorrelations appear close to 1, regardless of lag
2. The slow linear decay is an artifact, not a true decay pattern
3. Standard confidence bands (based on 1/√T) are INVALID
4. The ACF cannot distinguish between:
   - A near-unit-root AR(1) with φ = 0.99
   - A true unit root with φ = 1.00
   
This is why we need FORMAL UNIT ROOT TESTS (ADF, PP, KPSS) rather
than visual inspection of the correlogram!
""")

print("\nFigures saved:")
print("  - deterministic_trend.png")
print("  - random_walk.png")
print("  - random_walk_drift.png")
print("  - nonstationary_time_series.png")
print("  - nonstationary_acf_comparison.png")
print("  - nonstationary_pacf_comparison.png")
print("  - acf_levels_vs_differences.png")
print("  - variance_growth_comparison.png")
print("  - multiple_realizations.png")
