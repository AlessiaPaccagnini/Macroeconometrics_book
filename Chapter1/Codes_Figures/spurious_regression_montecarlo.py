"""
Spurious Regression: Monte Carlo Simulations
=============================================
Replicates the three figures from the macroeconometrics textbook:
1. Monte Carlo evidence (rejection rates, mean R2, mean DW)
2. Granger-Newbold rule of thumb (R2 vs DW scatter)
3. t-statistic distributions (spurious vs valid)

Author:   Alessia Paccagnini
Textbook: Macroeconometrics

References:
- Granger, C.W.J. and Newbold, P. (1974). "Spurious Regressions in Econometrics."
  Journal of Econometrics, 2(2), 111-120.
- Phillips, P.C.B. (1986). "Understanding Spurious Regressions in Econometrics."
  Journal of Econometrics, 33(3), 311-340.
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import os

np.random.seed(42)

# Output directory: same folder as this script (portable across machines)
out_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'

# ============================================================
# Helper function: run one spurious or valid regression
# ============================================================
def run_regression(T, spurious=True):
    """
    Run a single regression of y on x.
    If spurious=True: both x and y are independent random walks (I(1)).
    If spurious=False: both x and y are independent white noise (I(0)).
    
    Returns: dict with r2, t_stat, dw, beta_hat
    """
    if spurious:
        x = np.cumsum(np.random.randn(T))
        y = np.cumsum(np.random.randn(T))
    else:
        x = np.random.randn(T)
        y = np.random.randn(T)
    
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    return {
        'r2': model.rsquared,
        't_stat': model.tvalues[1],
        'dw': durbin_watson(model.resid),
        'beta_hat': model.params[1]
    }


# ============================================================
# Figure 1: Monte Carlo evidence (rejection rates, mean R2, mean DW)
# ============================================================
print("=" * 60)
print("Figure 1: Monte Carlo Evidence on Spurious Regression")
print("=" * 60)

sample_sizes = [50, 100, 200, 500]
n_simulations = 1000
alpha = 0.05
critical_value = 1.96

mc_results = {T: [] for T in sample_sizes}

for T in sample_sizes:
    print(f"  Simulating T = {T}...")
    for _ in range(n_simulations):
        mc_results[T].append(run_regression(T, spurious=True))

# Compute summary statistics
rejection_rates = []
mean_r2 = []
mean_dw = []

for T in sample_sizes:
    t_stats = [r['t_stat'] for r in mc_results[T]]
    r2s = [r['r2'] for r in mc_results[T]]
    dws = [r['dw'] for r in mc_results[T]]
    
    rej_rate = np.mean([abs(t) > critical_value for t in t_stats]) * 100
    rejection_rates.append(rej_rate)
    mean_r2.append(np.mean(r2s))
    mean_dw.append(np.mean(dws))
    
    print(f"  T={T}: Rejection rate={rej_rate:.1f}%, "
          f"Mean R2={np.mean(r2s):.3f}, Mean DW={np.mean(dws):.3f}")

# Plot Figure 1
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Rejection rates
bars_a = axes[0].bar([str(T) for T in sample_sizes], rejection_rates, 
                      color='#E74C6F', edgecolor='white', width=0.6)
axes[0].axhline(y=5, color='black', linestyle='--', linewidth=1.5, label='Nominal 5% level')
axes[0].set_xlabel('Sample Size (T)', fontsize=11)
axes[0].set_ylabel('Rejection Rate (%)', fontsize=11)
axes[0].set_title('(a) False Rejection Rate at 5% Level', fontsize=12, fontweight='bold')
axes[0].set_ylim(0, 100)
axes[0].legend(fontsize=10)
for bar, rate in zip(bars_a, rejection_rates):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Panel (b): Mean R2
bars_b = axes[1].bar([str(T) for T in sample_sizes], mean_r2, 
                      color='#6C9BD1', edgecolor='white', width=0.6)
axes[1].set_xlabel('Sample Size (T)', fontsize=11)
axes[1].set_ylabel('Mean $R^2$', fontsize=11)
axes[1].set_title('(b) Mean $R^2$ from Spurious Regressions', fontsize=12, fontweight='bold')
axes[1].set_ylim(0, 0.5)

# Panel (c): Mean DW
bars_c = axes[2].bar([str(T) for T in sample_sizes], mean_dw, 
                      color='#2E7D32', edgecolor='white', width=0.6)
axes[2].axhline(y=2.0, color='red', linestyle='--', linewidth=1.5, label='DW = 2 (no autocorrelation)')
axes[2].set_xlabel('Sample Size (T)', fontsize=11)
axes[2].set_ylabel('Mean Durbin-Watson', fontsize=11)
axes[2].set_title('(c) Mean DW Statistic (Lower = More Autocorrelation)', fontsize=12, fontweight='bold')
axes[2].set_ylim(0, 2.5)
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'spurious_regression_montecarlo.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'spurious_regression_montecarlo.pdf'), bbox_inches='tight')
plt.close()
print("  -> Saved: spurious_regression_montecarlo.png/pdf\n")


# ============================================================
# Figure 2: Granger-Newbold Rule of Thumb (R2 vs DW scatter)
# ============================================================
print("=" * 60)
print("Figure 2: Granger-Newbold Rule of Thumb")
print("=" * 60)

n_sims_scatter = 1000
T_scatter = 200

# Spurious regressions (independent random walks)
print("  Simulating spurious regressions...")
spurious_results = [run_regression(T_scatter, spurious=True) for _ in range(n_sims_scatter)]
spurious_r2 = [r['r2'] for r in spurious_results]
spurious_dw = [r['dw'] for r in spurious_results]

# Valid regressions (stationary with true relationship)
print("  Simulating valid regressions...")
valid_r2 = []
valid_dw = []
for _ in range(n_sims_scatter):
    x = np.random.randn(T_scatter)
    y = 0.5 * x + np.random.randn(T_scatter)  # True relationship
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    valid_r2.append(model.rsquared)
    valid_dw.append(durbin_watson(model.resid))

# Plot Figure 2
fig, ax = plt.subplots(figsize=(10, 7))

# Shaded spurious region (R2 > DW)
dw_fill = np.linspace(0, 2.5, 300)
ax.fill_between(dw_fill, dw_fill, 1.0, alpha=0.15, color='red')
ax.text(0.12, 0.82, 'Spurious\nRegion\n($R^2 > DW$)', fontsize=14, 
        color='red', fontweight='bold', fontstyle='italic', transform=ax.transAxes)

# 45-degree line: R2 = DW
dw_line = np.linspace(0, 1.0, 100)
ax.plot(dw_line, dw_line, 'k--', linewidth=2.5, label='$R^2$ = DW line')

# Scatter: spurious
ax.scatter(spurious_dw, spurious_r2, color='#E74C6F', alpha=0.5, s=30, 
           edgecolors='#C0392B', linewidth=0.3, label='Spurious (independent I(1))')

# Scatter: valid
ax.scatter(valid_dw, valid_r2, color='#5DADE2', alpha=0.4, s=30, marker='^',
           edgecolors='#2874A6', linewidth=0.3, label='Valid (stationary with true relationship)')

ax.set_xlabel('Durbin-Watson Statistic', fontsize=12)
ax.set_ylabel('$R^2$', fontsize=12)
ax.set_title("Granger-Newbold Rule of Thumb: If $R^2 > DW$, Suspect Spurious Regression",
             fontsize=13, fontweight='bold')
ax.set_xlim(0, 2.5)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'spurious_regression_rule_of_thumb.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'spurious_regression_rule_of_thumb.pdf'), bbox_inches='tight')
plt.close()
print("  -> Saved: spurious_regression_rule_of_thumb.png/pdf\n")


# ============================================================
# Figure 3: t-statistic distributions (spurious vs valid)
# ============================================================
print("=" * 60)
print("Figure 3: Distribution of t-statistics")
print("=" * 60)

n_sims_tstat = 5000
T_tstat = 200

# Spurious t-statistics
print("  Simulating spurious regressions...")
spurious_t = [run_regression(T_tstat, spurious=True)['t_stat'] for _ in range(n_sims_tstat)]

# Valid t-statistics
print("  Simulating valid regressions...")
valid_t = [run_regression(T_tstat, spurious=False)['t_stat'] for _ in range(n_sims_tstat)]

# Theoretical t-distribution
t_grid = np.linspace(-15, 15, 500)
t_pdf = stats.t.pdf(t_grid, df=T_tstat - 2)

# Plot Figure 3
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel (a): Spurious
axes[0].hist(spurious_t, bins=50, density=True, color='#E74C6F', alpha=0.6, 
             edgecolor='white', label='Spurious (I(1) variables)')
axes[0].plot(t_grid, t_pdf, 'k--', linewidth=2, label='Standard $t$-distribution')
axes[0].axvline(x=-1.96, color='blue', linestyle=':', linewidth=1.5)
axes[0].axvline(x=1.96, color='blue', linestyle=':', linewidth=1.5, label='$\\pm 1.96$ critical values')
axes[0].set_xlabel('$t$-statistic', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title('(a) $t$-statistics from Spurious Regressions\n(Two Independent Random Walks)',
                   fontsize=12, fontweight='bold')
axes[0].set_xlim(-15, 15)
axes[0].set_ylim(0, 0.40)
axes[0].legend(fontsize=10)

# Panel (b): Valid
axes[1].hist(valid_t, bins=50, density=True, color='#5DADE2', alpha=0.6,
             edgecolor='white', label='Valid (stationary variables)')
axes[1].plot(t_grid, t_pdf, 'k--', linewidth=2, label='Standard $t$-distribution')
axes[1].axvline(x=-1.96, color='blue', linestyle=':', linewidth=1.5)
axes[1].axvline(x=1.96, color='blue', linestyle=':', linewidth=1.5, label='$\\pm 1.96$ critical values')
axes[1].set_xlabel('$t$-statistic', fontsize=11)
axes[1].set_ylabel('Density', fontsize=11)
axes[1].set_title('(b) $t$-statistics from Valid Regressions\n(Two Independent White Noise Series)',
                   fontsize=12, fontweight='bold')
axes[1].set_xlim(-15, 15)
axes[1].set_ylim(0, 0.40)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'spurious_regression_tstat_distribution.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'spurious_regression_tstat_distribution.pdf'), bbox_inches='tight')
plt.close()
print("  -> Saved: spurious_regression_tstat_distribution.png/pdf\n")

print("=" * 60)
print("All figures generated successfully!")
print("=" * 60)
