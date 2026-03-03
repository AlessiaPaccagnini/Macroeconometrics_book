"""
================================================================================
Figure 5.1: Common Prior Distributions for Bayesian Estimation
================================================================================
Author: Alessia Paccagnini
Textbook: Macroeconometrics
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set up the figure with three panels
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# Color palette - professional blues/greens
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# =============================================================================
# Panel (a): Normal Prior
# =============================================================================
ax = axes[0]
x_norm = np.linspace(-6, 6, 500)

# Different variance choices
normal_params = [
    (0, 0.5, r'$N(0, 0.5^2)$ — Tight'),
    (0, 1.0, r'$N(0, 1^2)$ — Moderate'),
    (0, 2.0, r'$N(0, 2^2)$ — Diffuse'),
    (1, 1.0, r'$N(1, 1^2)$ — Shifted'),
]

for i, (mu, sigma, label) in enumerate(normal_params):
    y = stats.norm.pdf(x_norm, mu, sigma)
    ax.plot(x_norm, y, color=colors[i], linewidth=2.5, label=label)

ax.set_xlabel(r'$\theta$', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('(a) Normal Prior\nLocation parameters', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.set_xlim(-6, 6)
ax.set_ylim(0, 0.85)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3)

# =============================================================================
# Panel (b): Inverse Gamma Prior
# =============================================================================
ax = axes[1]
x_ig = np.linspace(0.01, 5, 500)

# Different shape/scale choices
# scipy uses a, scale parameterization where a=alpha, scale=1/beta
ig_params = [
    (3, 2, r'$IG(3, 2)$ — Moderate'),
    (2, 1, r'$IG(2, 1)$ — Diffuse'),
    (1, 1, r'$IG(1, 1)$ — Heavy tail'),
    (8, 3, r'$IG(8, 3)$ — Informative'),
]

for i, (alpha, beta, label) in enumerate(ig_params):
    # scipy invgamma: pdf at x is beta^alpha / Gamma(alpha) * x^(-alpha-1) * exp(-beta/x)
    # We use scale = beta (the rate parameter in the "beta/x" term)
    y = stats.invgamma.pdf(x_ig, a=alpha, scale=beta)
    ax.plot(x_ig, y, color=colors[i], linewidth=2.5, label=label)

ax.set_xlabel(r'$\sigma^2$', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('(b) Inverse Gamma Prior\nVariance parameters', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.set_xlim(0, 5)
ax.set_ylim(0, 4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3)

# =============================================================================
# Panel (c): Beta Prior
# =============================================================================
ax = axes[2]
x_beta = np.linspace(0.001, 0.999, 500)

# Classic Beta shapes
beta_params = [
    (1, 1, r'$Beta(1,1)$ — Uniform'),
    (0.5, 0.5, r'$Beta(0.5,0.5)$ — U-shaped'),
    (2, 5, r'$Beta(2,5)$ — Skewed right'),
    (5, 2, r'$Beta(5,2)$ — Skewed left'),
    (5, 5, r'$Beta(5,5)$ — Symmetric'),
]

for i, (a, b, label) in enumerate(beta_params):
    y = stats.beta.pdf(x_beta, a, b)
    ax.plot(x_beta, y, color=colors[i], linewidth=2.5, label=label)

ax.set_xlabel(r'$p$', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('(c) Beta Prior\nProbabilities & proportions', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3)

# =============================================================================
# Final adjustments
# =============================================================================
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/prior_distributions.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/prior_distributions.png', dpi=300, bbox_inches='tight')
print("Figure saved successfully!")
