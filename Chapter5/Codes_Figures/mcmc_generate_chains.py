"""
================================================================================
MCMC Chain Generator for Figure 5.2
================================================================================
Run this ONCE to generate mcmc_chain_good.csv and mcmc_chain_bad.csv.
These CSV files are then loaded by the Python, R, and MATLAB plotting
scripts to ensure identical figures across all three languages.

Author: Alessia Paccagnini
Textbook: Macroeconometrics
================================================================================
"""

import numpy as np

np.random.seed(42)

# Target: N(mu, Sigma) with rho = 0.7
mu = np.array([0.0, 0.0])
Sigma = np.array([[1.0, 0.7],
                   [0.7, 1.0]])
Sigma_inv = np.linalg.inv(Sigma)

def log_target(theta):
    diff = theta - mu
    return -0.5 * (diff @ Sigma_inv @ diff)

def rwmh(n_iter, c, theta_init):
    d = len(theta_init)
    chain = np.zeros((n_iter, d))
    chain[0] = theta_init
    n_accept = 0
    for t in range(1, n_iter):
        theta_star = chain[t-1] + c * np.random.randn(d)
        log_alpha = log_target(theta_star) - log_target(chain[t-1])
        if np.log(np.random.rand()) < log_alpha:
            chain[t] = theta_star
            n_accept += 1
        else:
            chain[t] = chain[t-1]
    return chain, n_accept / (n_iter - 1)

n_iter = 10000
chain_good, ar_good = rwmh(n_iter, c=1.70, theta_init=np.array([-2.0, 2.0]))
chain_bad,  ar_bad  = rwmh(n_iter, c=0.05, theta_init=np.array([-2.0, 2.0]))

# Save to CSV (header row: theta1,theta2)
np.savetxt('mcmc_chain_good.csv', chain_good, delimiter=',',
           header='theta1,theta2', comments='')
np.savetxt('mcmc_chain_bad.csv', chain_bad, delimiter=',',
           header='theta1,theta2', comments='')

print(f"Well-tuned:   acceptance rate = {ar_good:.1%}")
print(f"Poorly tuned: acceptance rate = {ar_bad:.1%}")
print("\nSaved: mcmc_chain_good.csv")
print("Saved: mcmc_chain_bad.csv")
print("\nNow run the plotting script in Python, R, or MATLAB.")
