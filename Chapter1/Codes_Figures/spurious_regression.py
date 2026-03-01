"""
Spurious Regression: Nine Regressions of Independent Random Walks
==================================================================
Author:   Alessia Paccagnini
Textbook: Macroeconometrics
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import os

np.random.seed(None)  # We'll try seeds until we get good results

# Find 9 pairs with R2 > 0.80 and DW < 0.20
T = 300
results = []
max_attempts = 100000
attempt = 0

while len(results) < 9 and attempt < max_attempts:
    attempt += 1
    # Generate two independent random walks
    x = np.cumsum(np.random.randn(T))
    y = np.cumsum(np.random.randn(T))
    
    # Regress y on x
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    r2 = model.rsquared
    t_stat = model.tvalues[1]
    dw = durbin_watson(model.resid)
    
    if r2 > 0.80 and dw < 0.20:
        results.append({
            'x': x, 'y': y,
            'r2': r2, 't': t_stat, 'dw': dw,
            'fitted': model.fittedvalues
        })
        print(f"Found pair {len(results)}: R2={r2:.2f}, t={t_stat:.1f}, DW={dw:.2f}")

print(f"\nTotal attempts: {attempt}")
print(f"Pairs found: {len(results)}")

# Plot
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle('Nine Regressions of Independent Random Walks\n(All relationships are SPURIOUS)',
             fontsize=14, fontweight='bold', y=0.98)

for i, (ax, res) in enumerate(zip(axes.flatten(), results)):
    # Scatter plot
    ax.scatter(res['x'], res['y'], alpha=0.45, s=15, color='#9B59B6', edgecolors='#7D3C98', linewidth=0.3)
    
    # Regression line
    sort_idx = np.argsort(res['x'])
    ax.plot(res['x'][sort_idx], res['fitted'][sort_idx], color='red', linewidth=2)
    
    # Format t-stat with significance stars
    abs_t = abs(res['t'])
    if abs_t > 3.29:
        stars = '***'
    elif abs_t > 2.58:
        stars = '**'
    elif abs_t > 1.96:
        stars = '*'
    else:
        stars = ''
    
    t_sign = '-' if res['t'] < 0 else ''
    title = f"$R^2$={res['r2']:.2f}, t={t_sign}{abs_t:.1f}{stars}, DW={res['dw']:.2f}"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('$x_t$', fontsize=9)
    ax.set_ylabel('$y_t$', fontsize=9)
    ax.tick_params(labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.94])

# Save to the same directory as this script for easy replication
out_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
plt.savefig(os.path.join(out_dir, 'spurious_regression_multiple.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'spurious_regression_multiple.pdf'), bbox_inches='tight')
print("\nPlots saved!")
