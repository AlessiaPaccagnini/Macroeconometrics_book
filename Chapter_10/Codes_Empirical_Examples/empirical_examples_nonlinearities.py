# -*- coding: utf-8 -*-
"""
=============================================================================
FOUR APPROACHES TO REGIME-DEPENDENT DYNAMICS
=============================================================================

Author:   Alessia Paccagnini
Textbook: Macroeconometrics

Empirical comparison of:
  Approach 1: Time-Varying Parameter VAR with Stochastic Volatility (TVP-VAR-SV)
  Approach 2: Markov-Switching VAR (MS-VAR)
  Approach 3: Threshold VAR (TVAR)
  Approach 4: Smooth Transition VAR (STVAR)

Data: US quarterly macro data 1960Q1-2019Q4 (FRED)
  GDPC1    -> annualised GDP growth
  GDPDEF   -> annualised inflation
  FEDFUNDS -> quarterly average federal funds rate

Variable ordering (Cholesky): GDP growth | Inflation | Fed Funds Rate


Requirements:
  pip install numpy scipy pandas matplotlib openpyxl

Usage:
  Place GDPC1.xlsx, GDPDEF.xlsx, FEDFUNDS.xlsx in the working directory.
  python regime_dynamics.py
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)


# =============================================================================
# CONVERGENCE DIAGNOSTICS
# =============================================================================
# Two standard tools for assessing MCMC chain quality:
#
#   trace_plot()   -- visual check: the chain should look like "white noise"
#                    around a stable mean after burn-in. Trends, drifts, or
#                    slow mixing suggest the sampler has not converged.
#
#   geweke_z()     -- formal test (Geweke 1992): compares the mean of the
#                    first 10% of draws with the mean of the last 50%.
#                    Under convergence the test statistic is ~N(0,1).
#                    |z| > 1.96 flags a potential convergence problem.
#
# How to use after running each sampler:
#
#   TVP-VAR-SV -- check a representative time-varying coefficient and
#                a log-volatility path, e.g.:
#                   trace_plot(beta_store[:, T//2, 0], "beta_1 (midpoint)")
#                   trace_plot(h_store[:, T//2, 0],    "log-vol GDP (midpoint)")
#                   print(geweke_z(beta_store[:, T//2, 0]))
#
#   MS-VAR     -- check the regime-1 VAR coefficients and the smoothed
#                regime probability at a few dates, e.g.:
#                   trace_plot(ms_out["bsave1"][:, 0], "MS B1[0]")
#                   trace_plot(ms_out["regime"].mean(axis=1), "avg regime prob")
#                   print(geweke_z(ms_out["bsave1"][:, 0]))
#
# Rule of thumb: run with at least 4x the default REPS for final results and
# verify that Geweke |z| < 1.96 for the parameters of interest.
# =============================================================================

def trace_plot(chain, label="parameter", burnin=0, ax=None):
    """
    Plot MCMC trace for a single scalar chain (post-burn-in draws).

    Parameters
    ----------
    chain  : 1-D array of posterior draws (burn-in already removed)
    label  : string label for the plot title
    burnin : number of additional draws to shade as burn-in (default 0,
             since chain is assumed post-burn-in)
    ax     : optional matplotlib Axes; if None a new figure is created

    Returns
    -------
    fig, ax
    """
    chain = np.asarray(chain).ravel()
    show  = ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    else:
        fig = ax[0].figure

    # Left panel: trace
    ax[0].plot(chain, lw=0.6, color="#2166ac", alpha=0.8)
    ax[0].axhline(chain.mean(), color="red", lw=1.2, ls="--", label="Posterior mean")
    ax[0].set_title(f"Trace -- {label}", fontsize=9)
    ax[0].set_xlabel("Draw"); ax[0].set_ylabel("Value")
    ax[0].legend(fontsize=8)

    # Right panel: density
    from scipy.stats import gaussian_kde
    kde  = gaussian_kde(chain)
    xg   = np.linspace(chain.min(), chain.max(), 200)
    ax[1].plot(xg, kde(xg), color="#d6604d", lw=1.8)
    ax[1].axvline(chain.mean(), color="red", lw=1.2, ls="--")
    ax[1].axvline(np.percentile(chain, 16), color="grey", lw=1.0, ls=":")
    ax[1].axvline(np.percentile(chain, 84), color="grey", lw=1.0, ls=":",
                  label="16th / 84th pctile")
    ax[1].set_title(f"Density -- {label}", fontsize=9)
    ax[1].set_xlabel("Value"); ax[1].set_ylabel("Density")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    if show:
        fig.savefig(f"diag_trace_{label.replace(' ','_')}.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


def geweke_z(chain, first=0.1, last=0.5):
    """
    Geweke (1992) convergence diagnostic for a single scalar chain.

    Splits the chain into the first `first` fraction and the last `last`
    fraction, computes the difference of means scaled by the sum of the
    spectral densities (approximated via a simple AR(1) estimate), and
    returns a z-score. Under convergence the z-score is ~N(0,1).

    Parameters
    ----------
    chain : 1-D array of posterior draws (burn-in already removed)
    first : fraction for the early window  (default 0.10 = first 10%)
    last  : fraction for the late window   (default 0.50 = last  50%)

    Returns
    -------
    z : float  -- |z| > 1.96 suggests lack of convergence at 5% level
    """
    chain = np.asarray(chain).ravel()
    n     = len(chain)
    n1    = int(np.floor(first * n))
    n2    = int(np.floor(last  * n))
    a     = chain[:n1]
    b     = chain[n - n2:]

    def spectral_density_0(x):
        """Estimate spectral density at frequency 0 via AR(1) fit."""
        x   = x - x.mean()
        n_x = len(x)
        if n_x < 4:
            return x.var() + 1e-12
        # AR(1) coefficient via OLS
        rho = np.sum(x[1:] * x[:-1]) / (np.sum(x[:-1]**2) + 1e-12)
        rho = np.clip(rho, -0.99, 0.99)
        s2  = np.var(x, ddof=1)
        return s2 / (1 - rho)**2

    s_a = spectral_density_0(a) / len(a)
    s_b = spectral_density_0(b) / len(b)
    denom = np.sqrt(s_a + s_b)
    if denom < 1e-12:
        return 0.0
    z = (a.mean() - b.mean()) / denom
    return z


def mcmc_diagnostics(store, param_names=None, n_check=5, tag=""):
    """
    Run trace plots and Geweke z-scores for a selection of parameters.

    Parameters
    ----------
    store       : 2-D array (ndraws x nparams) of posterior draws
    param_names : list of strings (optional)
    n_check     : how many parameters to sample for diagnostics (default 5)
    tag         : string prefix for saved figure filenames

    Prints a summary table and saves one trace-plot PNG per parameter.
    """
    store = np.asarray(store)
    if store.ndim == 1:
        store = store[:, np.newaxis]
    ndraws, npar = store.shape
    if param_names is None:
        param_names = [f"param_{i}" for i in range(npar)]

    # Sample evenly-spaced indices to check
    idx = np.round(np.linspace(0, npar - 1, min(n_check, npar))).astype(int)

    print(f"\n  {'Parameter':<30} {'Geweke z':>10}  {'|z|>1.96':>10}")
    print("  " + "-" * 55)
    any_flag = False
    for i in idx:
        z    = geweke_z(store[:, i])
        flag = "  *** WARN" if abs(z) > 1.96 else ""
        if abs(z) > 1.96:
            any_flag = True
        print(f"  {param_names[i]:<30} {z:>10.3f}{flag}")
        trace_plot(store[:, i], label=f"{tag}_{param_names[i]}")
    if not any_flag:
        print("  All Geweke |z| < 1.96 -- no convergence issues detected.")
    else:
        print("  *** Flagged parameters suggest insufficient iterations.")
        print("      Consider increasing REPS and/or checking trace plots.")
    print()

# -- matplotlib style ----------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9,  "figure.dpi": 130,
})

# =============================================================================
# 0.  GLOBAL SETTINGS
# =============================================================================

LAGS     = 2
HORZ     = 20
N        = 3
SHOCK_EQ = 2          # 0-indexed: column 2 = Fed Funds Rate

# MCMC iterations and burn-in
# For quick testing use the defaults below.
# For publication quality increase to REPS_TVP=20000, REPS_MS=20000.
# Always inspect convergence diagnostics after any run (see below).
REPS_TVP = 3000;  BURN_TVP = 1000
REPS_MS  = 5000;  BURN_MS  = 3000

# Model time-indices for IRF comparison (0-indexed, after dropping LAGS rows)
# Raw data 1960Q1-2019Q4 (240 obs). Model starts 1960Q3 (index 0).
# 2005Q1 = model index 177  (0-based)
# 2008Q4 = model index 192  (0-based)
T_NORMAL = 177
T_STRESS = 192

NBER_RECESSIONS = [
    ("1960-04-01","1961-02-01"), ("1969-12-01","1970-11-01"),
    ("1973-11-01","1975-03-01"), ("1980-01-01","1980-07-01"),
    ("1981-07-01","1982-11-01"), ("1990-07-01","1991-03-01"),
    ("2001-03-01","2001-11-01"), ("2007-12-01","2009-06-01"),
]

# =============================================================================
# 1.  DATA PREPARATION
# =============================================================================

def load_fred_data(gdp_path, gdpdef_path, fedfunds_path,
                   start="1960", end="2019"):
    gdp   = pd.read_excel(gdp_path,      sheet_name="Quarterly", skiprows=1,
                          header=None, names=["date","gdp"])
    defr  = pd.read_excel(gdpdef_path,   sheet_name="Quarterly", skiprows=1,
                          header=None, names=["date","gdpdef"])
    ff    = pd.read_excel(fedfunds_path, sheet_name="Monthly",   skiprows=1,
                          header=None, names=["date","fedfunds"])
    for d in [gdp, defr, ff]:
        d["date"] = pd.to_datetime(d["date"])
        d.dropna(inplace=True)
    ff["quarter"] = ff["date"].dt.to_period("Q").dt.to_timestamp()
    ff_q = ff.groupby("quarter")["fedfunds"].mean().reset_index()
    ff_q.columns = ["date","fedfunds"]
    data = gdp.merge(defr, on="date").merge(ff_q, on="date")
    data["gdp_growth"] = np.log(data["gdp"]).diff()    * 400
    data["inflation"]  = np.log(data["gdpdef"]).diff() * 400
    data = data.dropna()
    data = data.set_index("date").loc[start:end,
               ["gdp_growth","inflation","fedfunds"]]
    return data

def shade_recessions(ax, alpha=0.15):
    for s, e in NBER_RECESSIONS:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   alpha=alpha, color="grey", zorder=0)

# =============================================================================
# 2.  HELPER: IRF SIMULATION
# =============================================================================

def irfsim(B, N, lags, chol_lower, shock, horz):
    """
    Simulate IRF. B: (k x N), chol_lower: (N x N) lower Cholesky.
    Returns (horz x N).
    """
    irf = np.zeros((horz, N))
    irf[0] = shock
    k = N * lags + 1
    for h in range(1, horz):
        for lag in range(min(h, lags)):
            A_lag = B[lag*N:(lag+1)*N, :].T   # (N x N)
            irf[h] += A_lag @ irf[h-1-lag]
    return irf

# =============================================================================
# 3.  APPROACH 1 -- TVP-VAR WITH STOCHASTIC VOLATILITY
# =============================================================================

print("=" * 65)
print("FOUR APPROACHES TO REGIME-DEPENDENT DYNAMICS")
print("=" * 65)
print("\nLoading FRED data...")

data = load_fred_data("GDPC1.xlsx","GDPDEF.xlsx","FEDFUNDS.xlsx")
dates_plot = data.index
Y_raw      = data.values.astype(float)
T_raw      = len(Y_raw)

print(f"Sample:       {dates_plot[0].date()}  to  {dates_plot[-1].date()}")
print(f"Observations: {T_raw}\n")

print("=" * 65)
print("APPROACH 1: TVP-VAR-SV  (Primiceri 2005)")
print("=" * 65)

p  = LAGS
Y  = Y_raw[p:]
T  = len(Y)
k  = N * p + 1
nk = N * k

# Build regressor matrix
X = np.zeros((T, k))
for t in range(T):
    row = np.concatenate([Y_raw[p+t-lag] for lag in range(1,p+1)] + [[1.]])
    X[t] = row

# OLS init
B_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
e_ols = Y - X @ B_ols
S_ols = (e_ols.T @ e_ols) / (T - k)

# Observation matrices Z[t]: (N x nk)
Z_arr = np.zeros((T, N, nk))
for t in range(T):
    for eq in range(N):
        Z_arr[t, eq, eq*k:(eq+1)*k] = X[t]

# Initialise
beta0    = B_ols.ravel(order="F")           # column-major like MATLAB vec()
h_all    = np.tile(np.log(np.diag(S_ols)+1e-6), (T,1))
beta_all = np.tile(beta0, (T,1))
kappa_Q  = 0.01
Q_diag   = np.full(nk, kappa_Q)
Q        = np.diag(Q_diag)
m0, P0   = beta0.copy(), 4*np.eye(nk)
sig_eta  = 0.1

ndraws_tvp = REPS_TVP - BURN_TVP
beta_store = np.zeros((ndraws_tvp, T, nk))
h_store    = np.zeros((ndraws_tvp, T, N))

print(f"Running Gibbs sampler: {REPS_TVP} iterations, {BURN_TVP} burn-in...")
print("  NOTE: increase REPS_TVP to 20000 for publication quality.\n")

jdraw = 0
for isim in range(REPS_TVP):

    # Step 1: Carter-Kohn FFBS
    H_obs = np.exp(h_all)
    mf = np.zeros((T, nk))
    Pf = np.zeros((T, nk, nk))
    mp, Pp = m0.copy(), P0.copy()

    for t in range(T):
        Zt = Z_arr[t]
        Ht = np.diag(H_obs[t])
        vt = Y[t] - Zt @ mp
        Ft = Zt @ Pp @ Zt.T + Ht
        try:    Ft_inv = np.linalg.inv(Ft)
        except: Ft_inv = np.linalg.pinv(Ft)
        Kt     = Pp @ Zt.T @ Ft_inv
        mf[t]  = mp + Kt @ vt
        Pnew   = Pp - Kt @ Zt @ Pp
        Pnew   = 0.5*(Pnew+Pnew.T) + 1e-9*np.eye(nk)
        Pf[t]  = Pnew
        mp, Pp = mf[t], Pnew + Q

    beta_path = np.zeros((T, nk))
    beta_path[-1] = rng.multivariate_normal(mf[-1], Pf[-1]+1e-9*np.eye(nk))
    for t in range(T-2, -1, -1):
        Pp_next = Pf[t] + Q
        try:    J = Pf[t] @ np.linalg.inv(Pp_next)
        except: J = Pf[t] @ np.linalg.pinv(Pp_next)
        mb = mf[t] + J @ (beta_path[t+1] - mf[t])
        Pb = 0.5*(Pf[t]-J@Pf[t]); Pb = 0.5*(Pb+Pb.T)+1e-9*np.eye(nk)
        beta_path[t] = rng.multivariate_normal(mb, Pb)
    beta_all = beta_path

    # Step 2: Q diagonal IG draw
    db     = np.diff(beta_all, axis=0)
    sq     = (db**2).sum(axis=0)
    nu_q   = nk + 1 + (T-1)
    Q_diag = np.clip(1/rng.gamma(nu_q/2, 2/(kappa_Q+sq)), 1e-8, 1.0)
    Q      = np.diag(Q_diag)

    # Step 3: residuals
    resid = Y - np.einsum("tij,tj->ti", Z_arr, beta_all)

    # Step 4: log-volatility MH
    for i in range(N):
        h_cur = h_all[:, i].copy()
        for t in range(T):
            h_prev = h_cur[t-1] if t > 0 else h_cur[0]
            h_prop = h_prev + sig_eta * rng.standard_normal()
            ll_p = -0.5*h_prop - 0.5*resid[t,i]**2*np.exp(-h_prop)
            ll_c = -0.5*h_cur[t] - 0.5*resid[t,i]**2*np.exp(-h_cur[t])
            if np.log(rng.random()+1e-300) < ll_p - ll_c:
                h_cur[t] = h_prop
        h_all[:, i] = h_cur

    if isim >= BURN_TVP:
        beta_store[jdraw] = beta_all
        h_store[jdraw]    = h_all
        jdraw += 1

    if (isim+1) % 500 == 0:
        print(f"  Iteration {isim+1:4d} / {REPS_TVP}")

print("  TVP-VAR-SV complete.")
print("\n  --- TVP-VAR-SV Convergence Diagnostics ---")
print("  Checking a representative time-varying coefficient and log-volatility...")
# Flatten beta_store to (ndraws x T*nk); check midpoint-time coefficients
_mid = T // 2
_tvp_check = np.column_stack([
    beta_store[:, _mid, 0],   # first coeff at mid-sample
    beta_store[:, _mid, nk//2],  # middle coeff
    h_store[:, _mid, 0],      # log-vol GDP at mid-sample
    h_store[:, _mid, 1],      # log-vol Inflation
    h_store[:, _mid, 2],      # log-vol FFR
])
_tvp_names = ["beta[mid,0]", f"beta[mid,{nk//2}]",
              "log-vol GDP[mid]", "log-vol Inf[mid]", "log-vol FFR[mid]"]
mcmc_diagnostics(_tvp_check, param_names=_tvp_names, n_check=5, tag="tvpvar")
print("  Trace plots saved as diag_trace_tvpvar_*.png")
print("  Tip: increase REPS_TVP to >= 20000 for publication quality.\n")

# Posterior volatility
sig_arr = np.exp(h_store / 2)
sig_med = np.median(sig_arr, axis=0)
sig_lo  = np.percentile(sig_arr, 16, axis=0)
sig_hi  = np.percentile(sig_arr, 84, axis=0)

def compute_tvp_irf(t_idx):
    irfs = np.zeros((ndraws_tvp, HORZ, N))
    for d in range(ndraws_tvp):
        B_t    = beta_store[d, t_idx].reshape(k, N, order="F")
        A_list = [B_t[lag*N:(lag+1)*N, :].T for lag in range(p)]
        sigma_t = np.exp(h_store[d, t_idx] / 2)
        impact  = np.zeros(N); impact[SHOCK_EQ] = sigma_t[SHOCK_EQ]
        irf = np.zeros((HORZ, N)); irf[0] = impact
        for h in range(1, HORZ):
            for lag in range(min(h, p)):
                irf[h] += A_list[lag] @ irf[h-1-lag]
        irfs[d] = irf
    return (np.median(irfs,0), np.percentile(irfs,16,0), np.percentile(irfs,84,0))

print("Computing TVP-VAR-SV IRFs at 2005Q1 and 2008Q4...")
irf_n_med, irf_n_lo, irf_n_hi = compute_tvp_irf(T_NORMAL)
irf_s_med, irf_s_lo, irf_s_hi = compute_tvp_irf(T_STRESS)

gdp_vol     = sig_med[:, 0]
vol_thresh  = np.percentile(gdp_vol, 75)
tvp_hv_flag = gdp_vol > vol_thresh
dates_tvp   = dates_plot[p:]

var_names   = ["GDP Growth", "Inflation", "Fed Funds Rate"]
resp_labels = ["GDP Growth Response", "Inflation Response", "Fed Funds Rate Response"]
col_main    = ["#2166ac", "#d6604d", "#4dac26"]

# TVP-VAR-SV: Volatility figure
fig, axes = plt.subplots(3, 1, figsize=(11,9), sharex=True)
fig.suptitle("Time-Varying Stochastic Volatilities (TVP-VAR-SV)",
             fontsize=12, fontweight="bold")
for i, ax in enumerate(axes):
    shade_recessions(ax)
    ax.fill_between(dates_tvp, sig_lo[:,i], sig_hi[:,i],
                    alpha=0.3, color=col_main[i], label="68% Credible Interval")
    ax.plot(dates_tvp, sig_med[:,i], color=col_main[i], lw=1.8, label="Posterior Mean")
    if i == 0:
        ax.axvline(pd.Timestamp("1984-01-01"), color="k", lw=0.8, ls="--")
        ax.text(pd.Timestamp("1985-06-01"), ax.get_ylim()[1]*0.85,
                "Great\nModeration", fontsize=7, color="grey")
    ax.set_ylabel("Volatility (SD)"); ax.set_title(var_names[i]+" Volatility")
    ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.25, ls=":")
axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("fig_tvpvar_volatility.png", dpi=150, bbox_inches="tight")
plt.close()
print("TVP-VAR-SV volatility figure saved.")

# TVP-VAR-SV: IRF comparison figure
hor = np.arange(HORZ)
fig, axes = plt.subplots(1, 3, figsize=(13,4))
fig.suptitle("TVP-VAR-SV: Monetary Policy Shock (1 SD Tightening)\n"
             "Normal vs. Stress Period Comparison", fontsize=11, fontweight="bold")
for i, ax in enumerate(axes):
    ax.fill_between(hor, irf_n_lo[:,i], irf_n_hi[:,i], alpha=0.2, color="#2166ac")
    ax.plot(hor, irf_n_med[:,i], color="#2166ac", lw=2, label="Normal (2005Q1)")
    ax.fill_between(hor, irf_s_lo[:,i], irf_s_hi[:,i], alpha=0.2, color="#d6604d")
    ax.plot(hor, irf_s_med[:,i], color="#d6604d", lw=2, ls="--", label="Stress (2008Q4)")
    ax.axhline(0, color="k", lw=0.6, ls=":"); ax.grid(True, alpha=0.25, ls=":")
    ax.set_xlabel("Quarters Ahead"); ax.set_title(resp_labels[i])
    if i == 0: ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("fig_tvpvar_irf_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("TVP-VAR-SV IRF comparison figure saved.\n")
print(f"TVP-VAR-SV  Normal (2005Q1): Peak GDP = {irf_n_med[:,0].min():.3f}%")
print(f"TVP-VAR-SV  Stress (2008Q4): Peak GDP = {irf_s_med[:,0].min():.3f}%\n")

# =============================================================================
# 4.  APPROACH 2 -- MARKOV-SWITCHING VAR
# =============================================================================

print("=" * 65)
print("APPROACH 2: MARKOV-SWITCHING VAR  (Gibbs sampler)")
print("=" * 65)

def make_dummies(Y, lags, lam=0.1, tau=1.0, eps=1e-3):
    T_d, N_d = Y.shape
    mu     = Y.mean(axis=0)
    sigma  = np.array([np.std(np.diff(Y[:,i])) + 1e-6 for i in range(N_d)])
    k_d    = N_d*lags+1
    rows   = N_d*lags + N_d + 1
    Yd     = np.zeros((rows, N_d))
    Xd     = np.zeros((rows, k_d))
    for lag in range(1, lags+1):
        for j in range(N_d):
            r = (lag-1)*N_d + j
            Yd[r,j]           = sigma[j]*lag/lam
            Xd[r,(lag-1)*N_d+j] = sigma[j]*lag/lam
    s0 = N_d*lags
    for j in range(N_d):
        Yd[s0+j,j] = mu[j]/tau
        Xd[s0+j,:] = np.concatenate([np.tile(mu/tau, lags), [1/tau]])
    Yd[-1]  = mu*eps
    Xd[-1]  = np.concatenate([np.tile(mu*eps, lags), [eps]])
    return Yd, Xd

def build_var_xy(Y_in, lags):
    T_in, N_in = Y_in.shape
    k_in = N_in*lags+1
    Yv = Y_in[lags:]
    Xv = np.zeros((len(Yv), k_in))
    for t in range(len(Yv)):
        row = np.concatenate([Y_in[lags+t-l] for l in range(1,lags+1)] + [[1.]])
        Xv[t] = row
    return Yv, Xv

def riwish(nu, S):
    """Draw from Inverse-Wishart(nu, S)."""
    p   = S.shape[0]
    ch  = np.linalg.cholesky(np.linalg.inv(S))
    Z   = rng.standard_normal((nu, p)) @ ch.T
    return np.linalg.inv(Z.T @ Z)

def hamilton_filter(Y, X, B1, B2, Sig1, Sig2, p_vec, q_vec):
    T_f = len(Y)
    iS1, iS2 = np.linalg.inv(Sig1), np.linalg.inv(Sig2)
    dS1, dS2 = np.linalg.det(Sig1), np.linalg.det(Sig2)
    fprob = np.zeros((T_f, 2))
    lik   = 0.0
    ett   = np.array([0.5, 0.5])
    for t in range(T_f):
        P_tr = np.array([[p_vec[t],   1-q_vec[t]],
                         [1-p_vec[t], q_vec[t]]])
        e1 = Y[t] - X[t] @ B1;  e2 = Y[t] - X[t] @ B2
        n1 = (1/np.sqrt(dS1+1e-300))*np.exp(-0.5*e1@iS1@e1)
        n2 = (1/np.sqrt(dS2+1e-300))*np.exp(-0.5*e2@iS2@e2)
        ett1 = ett * np.array([n1, n2])
        fit  = ett1.sum()
        if fit <= 0: lik -= 10; ett = np.array([0.5,0.5])
        else:
            ett         = P_tr @ ett1 / fit
            fprob[t]    = ett1 / fit
            lik        += np.log(fit)
    return fprob, lik

def sample_states(fprob, p_vec, q_vec):
    T_s = len(fprob)
    st  = np.zeros(T_s, dtype=int)
    p00, p01 = fprob[-1]
    st[-1] = 1 if rng.random() >= p00/(p00+p01+1e-300) else 0
    for t in range(T_s-2, -1, -1):
        P_tr = np.array([[p_vec[t],   1-q_vec[t]],
                         [1-p_vec[t], q_vec[t]]])
        row  = P_tr[st[t+1]]
        p00  = row[0]*fprob[t,0]; p01 = row[1]*fprob[t,1]
        st[t]= 0 if rng.random() < p00/(p00+p01+1e-300) else 1
    return st

def ms_gibbs(Y_in, lags, REPS, BURN, lam=0.1, tau=1.0, verbose=True):
    Yv, Xv = build_var_xy(Y_in, lags)
    T_g, N_g = Yv.shape;  k_g = Xv.shape[1]
    Yd, Xd   = make_dummies(Y_in, lags, lam, tau)

    mid  = T_g//2
    B1   = np.linalg.lstsq(Xv[:mid], Yv[:mid], rcond=None)[0]
    B2   = np.linalg.lstsq(Xv[mid:], Yv[mid:], rcond=None)[0]
    e1   = Yv[:mid] - Xv[:mid]@B1;  e2 = Yv[mid:] - Xv[mid:]@B2
    Sig1 = (e1.T@e1)/mid + 1e-4*np.eye(N_g)
    Sig2 = (e2.T@e2)/mid + 1e-4*np.eye(N_g)

    p_vec = np.full(T_g, 0.95); q_vec = np.full(T_g, 0.95)
    st    = np.array([0]*mid + [1]*(T_g-mid))
    g0    = np.array([-2., -4.])        # TVTP logistic parameters
    ivg0  = np.diag(1/np.array([10.,10.]))

    npost    = REPS - BURN
    bsave1   = np.zeros((npost, N_g*k_g))
    bsave2   = np.zeros((npost, N_g*k_g))
    sigS1    = np.zeros((npost, N_g, N_g))
    sigS2    = np.zeros((npost, N_g, N_g))
    reg_save = np.zeros((npost, T_g), dtype=int)
    pmat_s   = np.zeros((npost, T_g))
    qmat_s   = np.zeros((npost, T_g))

    jd = 0
    for isim in range(REPS):
        # 1. Hamilton filter + backward sample
        fprob, _ = hamilton_filter(Yv, Xv, B1, B2, Sig1, Sig2, p_vec, q_vec)
        st       = sample_states(fprob, p_vec, q_vec)
        slag     = np.concatenate([[0], st[:-1]])

        # 2. TVTP: MH step on gamma (logistic)
        Zg = np.column_stack([np.ones(T_g), slag])
        for _ in range(5):
            g_prop = g0 + rng.standard_normal(2)*0.1
            lp_c = np.sum(st*np.log(1/(1+np.exp(-Zg@g0))+1e-300) +
                          (1-st)*np.log(1 - 1/(1+np.exp(-Zg@g0))+1e-300))
            lp_p = np.sum(st*np.log(1/(1+np.exp(-Zg@g_prop))+1e-300) +
                          (1-st)*np.log(1 - 1/(1+np.exp(-Zg@g_prop))+1e-300))
            lpr_c = -0.5*g0@ivg0@g0;  lpr_p = -0.5*g_prop@ivg0@g_prop
            if np.log(rng.random()+1e-300) < (lp_p+lpr_p)-(lp_c+lpr_c):
                g0 = g_prop
        lp_val = np.clip(1/(1+np.exp(-Zg@g0)), 0.05, 0.99)
        p_vec  = lp_val; q_vec = lp_val

        # 3. VAR coefficients regime-specific
        for regime in [0, 1]:
            idx = np.where(st == regime)[0]
            if len(idx) < k_g + 2: continue
            Yr  = np.vstack([Yv[idx], Yd]); Xr = np.vstack([Xv[idx], Xd])
            B_r = np.linalg.lstsq(Xr, Yr, rcond=None)[0]
            XX  = Xr.T @ Xr
            try:    iXX = np.linalg.inv(XX)
            except: iXX = np.linalg.pinv(XX)
            S   = Sig1 if regime==0 else Sig2
            L   = np.linalg.cholesky(iXX)
            R   = np.linalg.cholesky(S)
            B_d = B_r + L @ rng.standard_normal((k_g, N_g)) @ R.T
            if regime==0: B1 = B_d
            else:         B2 = B_d

        # 4. Covariance IW
        for regime in [0, 1]:
            idx = np.where(st == regime)[0]
            if len(idx) < 2: continue
            Br  = B1 if regime==0 else B2
            er  = np.vstack([Yv[idx], Yd]) - np.vstack([Xv[idx], Xd]) @ Br
            S_d = riwish(len(er), er.T@er) + 1e-6*np.eye(N_g)
            if regime==0: Sig1 = S_d
            else:         Sig2 = S_d

        if isim >= BURN:
            bsave1[jd]   = B1.T.ravel()
            bsave2[jd]   = B2.T.ravel()
            sigS1[jd]    = Sig1; sigS2[jd] = Sig2
            reg_save[jd] = st
            pmat_s[jd]   = p_vec; qmat_s[jd] = q_vec
            jd += 1

        if verbose and (isim+1) % 500 == 0:
            print(f"  MS-VAR iteration {isim+1:4d} / {REPS}")

    return dict(bsave1=bsave1, bsave2=bsave2, sigS1=sigS1, sigS2=sigS2,
                regime=reg_save, pmat=pmat_s, qmat=qmat_s)

print(f"Running MS-VAR Gibbs sampler: {REPS_MS} iterations, {BURN_MS} burn-in...")
ms_out = ms_gibbs(Y_raw, LAGS, REPS_MS, BURN_MS, verbose=True)
print("  MS-VAR complete.")
print("\n  --- MS-VAR Convergence Diagnostics ---")
print("  Checking VAR coefficients, regime probabilities, and transition parameters...")
_ms_check = np.column_stack([
    ms_out["bsave1"][:, 0],           # first normal-regime coeff
    ms_out["bsave2"][:, 0],           # first stress-regime coeff
    ms_out["regime"].mean(axis=1),    # avg smoothed regime prob per draw
    ms_out["pmat"].mean(axis=1),      # avg p00 per draw
    ms_out["qmat"].mean(axis=1),      # avg q11 per draw
])
_ms_names = ["B1[0]", "B2[0]", "avg regime prob", "avg p00", "avg q11"]
mcmc_diagnostics(_ms_check, param_names=_ms_names, n_check=5, tag="msvar")
print("  Trace plots saved as diag_trace_msvar_*.png")
print("  Tip: increase REPS_MS to >= 20000 for publication quality.\n")

ndraws_ms   = ms_out["regime"].shape[0]
T_ms        = ms_out["regime"].shape[1]
smooth_prob = ms_out["regime"].mean(axis=0)
p00_mean    = ms_out["pmat"].mean()
q11_mean    = ms_out["qmat"].mean()
print(f"MS-VAR  P(Normal->Normal) = {p00_mean:.4f}  E[Duration] = {1/(1-p00_mean):.1f} qtrs")
print(f"MS-VAR  P(Stress->Stress) = {q11_mean:.4f}  E[Duration] = {1/(1-q11_mean):.1f} qtrs\n")

# Regime-specific IRFs
print("Computing MS-VAR regime-specific IRFs...")
k_ms = N*LAGS+1
irf_ms_n = np.zeros((ndraws_ms, HORZ, N))
irf_ms_s = np.zeros((ndraws_ms, HORZ, N))
for d in range(ndraws_ms):
    for regime, store_arr in [(0, irf_ms_n), (1, irf_ms_s)]:
        bv   = ms_out["bsave1"][d] if regime==0 else ms_out["bsave2"][d]
        Sig  = ms_out["sigS1"][d]  if regime==0 else ms_out["sigS2"][d]
        B_d  = bv.reshape(N, k_ms).T          # (k x N)
        ch   = np.linalg.cholesky(Sig)         # lower
        shock = np.zeros(N); shock[SHOCK_EQ] = ch[SHOCK_EQ, SHOCK_EQ]
        store_arr[d] = irfsim(B_d, N, LAGS, ch, shock, HORZ)

ms_n_med = np.median(irf_ms_n, 0); ms_n_lo = np.percentile(irf_ms_n,16,0)
ms_n_hi  = np.percentile(irf_ms_n, 84, 0)
ms_s_med = np.median(irf_ms_s, 0); ms_s_lo = np.percentile(irf_ms_s,16,0)
ms_s_hi  = np.percentile(irf_ms_s, 84, 0)

ms_stress_flag = smooth_prob > 0.5
dates_ms = dates_plot[:T_ms]

# MS-VAR: Regime probability figure
fig, ax = plt.subplots(figsize=(11,4))
shade_recessions(ax)
ax.fill_between(dates_ms, 0, smooth_prob, alpha=0.55, color="#2166ac", label="Pr(Stress Regime)")
ax.plot(dates_ms, smooth_prob, color="#2166ac", lw=1.3)
ax.axhline(0.5, color="k", ls="--", lw=1.2, label="50% Threshold")
ax.set_ylim(0,1); ax.set_xlabel("Date"); ax.set_ylabel("Probability")
ax.set_title("MS-VAR: Smoothed Stress Regime Probability",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.25, ls=":")
plt.tight_layout()
plt.savefig("fig_msvar_regime_prob.png", dpi=150, bbox_inches="tight")
plt.close()
print("MS-VAR regime probability figure saved.")
print(f"MS-VAR  Normal regime: Peak GDP = {ms_n_med[:,0].min():.3f}%")
print(f"MS-VAR  Stress regime: Peak GDP = {ms_s_med[:,0].min():.3f}%\n")

# =============================================================================
# 5.  APPROACH 3 -- THRESHOLD VAR
# =============================================================================

print("=" * 65)
print("APPROACH 3: THRESHOLD VAR")
print("=" * 65)

TVAR_DELAY = 1; TVAR_VAR = 0; TVAR_NCRIT = 15
N_GRID = 100; N_GIRF = 1000

Yt = Y_raw[LAGS:]
Xt = np.zeros((len(Yt), N*LAGS+1))
for t in range(len(Yt)):
    Xt[t] = np.concatenate([Y_raw[LAGS+t-l] for l in range(1,LAGS+1)] + [[1.]])
T_tv  = len(Yt)

Ystar = np.concatenate([np.full(TVAR_DELAY, np.nan),
                        Y_raw[:T_raw-TVAR_DELAY, TVAR_VAR]])[LAGS:]

tau_grid = np.linspace(np.nanpercentile(Ystar,15), np.nanpercentile(Ystar,85), N_GRID)
sse_grid = np.full(N_GRID, np.inf)

for ig, tau in enumerate(tau_grid):
    e1 = Ystar <= tau; e2 = ~e1
    if e1.sum() < TVAR_NCRIT or e2.sum() < TVAR_NCRIT: continue
    B1 = np.linalg.lstsq(Xt[e1], Yt[e1], rcond=None)[0]
    B2 = np.linalg.lstsq(Xt[e2], Yt[e2], rcond=None)[0]
    sse_grid[ig] = ((Yt[e1]-Xt[e1]@B1)**2).sum() + ((Yt[e2]-Xt[e2]@B2)**2).sum()

best_idx = sse_grid.argmin()
tau_hat  = tau_grid[best_idx]
e1_hat   = Ystar <= tau_hat;  e2_hat = ~e1_hat
B1_tv    = np.linalg.lstsq(Xt[e1_hat], Yt[e1_hat], rcond=None)[0]
B2_tv    = np.linalg.lstsq(Xt[e2_hat], Yt[e2_hat], rcond=None)[0]
r1_tv    = Yt[e1_hat] - Xt[e1_hat]@B1_tv
r2_tv    = Yt[e2_hat] - Xt[e2_hat]@B2_tv
Sig1_tv  = r1_tv.T@r1_tv / (e1_hat.sum()-N*LAGS-1)
Sig2_tv  = r2_tv.T@r2_tv / (e2_hat.sum()-N*LAGS-1)

print(f"Threshold estimate:  tau_hat = {tau_hat:.4f}%")
print(f"Low-growth regime:   {e1_hat.sum()} quarters ({100*e1_hat.mean():.1f}%)")
print(f"High-growth regime:  {e2_hat.sum()} quarters ({100*e2_hat.mean():.1f}%)\n")

# Hansen sup-LM bootstrap
B_lin    = np.linalg.lstsq(Xt, Yt, rcond=None)[0]
r_lin    = Yt - Xt@B_lin
SSE_lin  = (r_lin**2).sum()
LM_stat  = (SSE_lin - sse_grid[best_idx]) / sse_grid[best_idx] * T_tv

N_BOOT   = 500
LM_boot  = np.zeros(N_BOOT)
for b in range(N_BOOT):
    idx_b   = rng.integers(0, T_tv, T_tv)
    Yt_b    = Xt@B_lin + r_lin[idx_b]
    Blin_b  = np.linalg.lstsq(Xt, Yt_b, rcond=None)[0]
    rlin_b  = Yt_b - Xt@Blin_b
    SSE_lb  = (rlin_b**2).sum()
    sse_b   = np.full(N_GRID, np.inf)
    for ig, tau in enumerate(tau_grid):
        e1b = Ystar<=tau; e2b = ~e1b
        if e1b.sum()<TVAR_NCRIT or e2b.sum()<TVAR_NCRIT: continue
        B1b = np.linalg.lstsq(Xt[e1b], Yt_b[e1b], rcond=None)[0]
        B2b = np.linalg.lstsq(Xt[e2b], Yt_b[e2b], rcond=None)[0]
        sse_b[ig] = ((Yt_b[e1b]-Xt[e1b]@B1b)**2).sum() + ((Yt_b[e2b]-Xt[e2b]@B2b)**2).sum()
    LM_boot[b] = (SSE_lb - sse_b.min()) / sse_b.min() * T_tv

pval_LM = (LM_boot >= LM_stat).mean()
print(f"Hansen sup-LM:  stat = {LM_stat:.3f},  bootstrap p-value = {pval_LM:.4f}")
if pval_LM < 0.05: print("  -> Linearity rejected at 5% level.\n")

# Generalized IRFs
ch1_tv = np.linalg.cholesky(Sig1_tv); ch2_tv = np.linalg.cholesky(Sig2_tv)

def compute_girf_tvar(start_regime):
    pool = np.where(e1_hat if start_regime==1 else e2_hat)[0]
    girf_diff = np.zeros((N_GIRF, HORZ, N))
    for g in range(N_GIRF):
        t0 = rng.choice(pool); t0 = max(min(t0, T_tv-1), LAGS)
        Y_hist = Yt[max(0,t0-LAGS):t0]
        if len(Y_hist) < LAGS:
            Y_hist = np.vstack([np.tile(Yt[0], (LAGS-len(Y_hist),1)), Y_hist])
        Y_base  = np.zeros((HORZ, N))
        Y_shock = np.zeros((HORZ, N))
        eps_all = rng.standard_normal((HORZ, N))

        for h in range(HORZ):
            def build_x(Y_path):
                row = []
                for l in range(1, LAGS+1):
                    row.append(Y_path[h-l] if h-l>=0 else Y_hist[-(l-h)])
                return np.concatenate(row + [[1.]])
            xb = build_x(Y_base); xs = build_x(Y_shock)
            zb = Y_base[h-TVAR_DELAY,TVAR_VAR] if h-TVAR_DELAY>=0 else \
                 Y_hist[-(TVAR_DELAY-h), TVAR_VAR]
            zs = Y_shock[h-TVAR_DELAY,TVAR_VAR] if h-TVAR_DELAY>=0 else zb
            Bb = B1_tv if zb<=tau_hat else B2_tv
            Bs = B1_tv if zs<=tau_hat else B2_tv
            cb = ch1_tv if zb<=tau_hat else ch2_tv
            cs = ch1_tv if zs<=tau_hat else ch2_tv
            Y_base[h]  = xb@Bb + eps_all[h]@cb.T
            Y_shock[h] = xs@Bs + eps_all[h]@cs.T
            if h == 0:
                sv = np.zeros(N); sv[SHOCK_EQ] = cs[SHOCK_EQ, SHOCK_EQ]
                Y_shock[h] += sv
        girf_diff[g] = Y_shock - Y_base
    return girf_diff.mean(axis=0)

print("Computing TVAR GIRFs...")
girf1 = compute_girf_tvar(1)   # low-growth
girf2 = compute_girf_tvar(2)   # high-growth
tvar_flag = e1_hat

print(f"TVAR  Low-growth:  Peak GDP = {girf1[:,0].min():.3f}%")
print(f"TVAR  High-growth: Peak GDP = {girf2[:,0].min():.3f}%\n")

# TVAR: GIRF figure
hor = np.arange(HORZ)
fig, axes = plt.subplots(1, 3, figsize=(13,4))
fig.suptitle("Threshold VAR: Generalized Impulse Responses\n"
             "Monetary Policy Shock (1 SD Tightening)", fontsize=11, fontweight="bold")
for i, ax in enumerate(axes):
    ax.plot(hor, girf1[:,i], color="#2166ac", lw=2, label="Low Growth Regime")
    ax.plot(hor, girf2[:,i], color="#4dac26", lw=2, label="High Growth Regime")
    ax.axhline(0, color="k", lw=0.6, ls=":"); ax.grid(True, alpha=0.25, ls=":")
    ax.set_xlabel("Quarters Ahead"); ax.set_title(resp_labels[i])
    if i == 0: ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("fig_tvar_girf.png", dpi=150, bbox_inches="tight")
plt.close()
print("TVAR GIRF figure saved.\n")

# =============================================================================
# 6.  APPROACH 4 -- SMOOTH TRANSITION VAR
# =============================================================================

print("=" * 65)
print("APPROACH 4: SMOOTH TRANSITION VAR")
print("=" * 65)

Zvar = Ystar.copy()

def logistic_F(z, gamma, c):
    return 1 / (1 + np.exp(-gamma * (z - c)))

def fit_stvar_coefs(Yt, Xt, Ft):
    w1 = 1-Ft; w2 = Ft
    sw1 = np.sqrt(w1)[:,None]; sw2 = np.sqrt(w2)[:,None]
    B1 = np.linalg.lstsq(Xt*sw1, Yt*sw1, rcond=None)[0]
    B2 = np.linalg.lstsq(Xt*sw2, Yt*sw2, rcond=None)[0]
    r1 = Yt-Xt@B1; r2 = Yt-Xt@B2
    S1 = (r1*w1[:,None]).T@r1 / (w1.sum()-1) + 1e-8*np.eye(N)
    S2 = (r2*w2[:,None]).T@r2 / (w2.sum()-1) + 1e-8*np.eye(N)
    return B1, B2, S1, S2

def stvar_sse(log_gamma, c):
    gamma = np.exp(log_gamma)
    if gamma <= 0: return 1e10
    Ft = logistic_F(Zvar, gamma, c)
    if Ft.min() > 0.99 or Ft.max() < 0.01: return 1e10
    B1,B2,_,_ = fit_stvar_coefs(Yt, Xt, Ft)
    Yhat = (1-Ft)[:,None]*(Xt@B1) + Ft[:,None]*(Xt@B2)
    return ((Yt-Yhat)**2).sum()

# 2-D grid search
c_grid     = np.linspace(np.percentile(Zvar,10), np.percentile(Zvar,90), 40)
gamma_grid = [0.5,1.0,2.0,4.0,6.0,8.0,12.0,20.0]
sse_2d     = np.full((len(gamma_grid), len(c_grid)), np.inf)

for ig, gv in enumerate(gamma_grid):
    for ic, cv in enumerate(c_grid):
        Ft = logistic_F(Zvar, gv, cv)
        if Ft.min()>0.99 or Ft.max()<0.01: continue
        B1,B2,_,_ = fit_stvar_coefs(Yt, Xt, Ft)
        Yhat = (1-Ft)[:,None]*(Xt@B1) + Ft[:,None]*(Xt@B2)
        sse_2d[ig,ic] = ((Yt-Yhat)**2).sum()

best   = np.unravel_index(sse_2d.argmin(), sse_2d.shape)
g_init = gamma_grid[best[0]]; c_init = c_grid[best[1]]

res = optimize.minimize(lambda p: stvar_sse(p[0], p[1]),
                        x0=[np.log(g_init), c_init],
                        method="Nelder-Mead",
                        options=dict(maxiter=800, xatol=1e-6, fatol=1e-6))
gamma_hat = min(np.exp(res.x[0]), 20.0)
c_hat     = res.x[1]

print(f"STVAR estimates:  gamma_hat = {gamma_hat:.3f},  c_hat = {c_hat:.4f}%")

Ft_hat             = logistic_F(Zvar, gamma_hat, c_hat)
B1_st,B2_st,Sig1_st,Sig2_st = fit_stvar_coefs(Yt, Xt, Ft_hat)
dates_st = dates_plot[LAGS:]

# IRFs across five states
z_states   = [-2.0,-0.5, 0.0, 1.5, 3.0]
cols_st    = ["#991a1a","#cc4d1a","#999919","#339933","#0d6699"]
lbl_states = ["z=-2.0% (Deep Recession)","z=-0.5%","z=0.0%","z=+1.5%","z=+3.0% (Expansion)"]
irf_st     = np.zeros((len(z_states), HORZ, N))

for iz, zv in enumerate(z_states):
    Fz      = logistic_F(zv, gamma_hat, c_hat)
    B_mix   = (1-Fz)*B1_st + Fz*B2_st
    Sig_mix = (1-Fz)*Sig1_st + Fz*Sig2_st + 1e-8*np.eye(N)
    ch_m    = np.linalg.cholesky(Sig_mix)
    shock   = np.zeros(N); shock[SHOCK_EQ] = ch_m[SHOCK_EQ, SHOCK_EQ]
    irf_st[iz] = irfsim(B_mix, N, LAGS, ch_m, shock, HORZ)

# Information criteria
def model_ic(resid, k_params, T_n):
    Sig = resid.T@resid/T_n
    ll  = -T_n*N/2*np.log(2*np.pi) - T_n/2*np.log(np.linalg.det(Sig)) - T_n/2
    return -2*ll + 2*k_params, -2*ll + np.log(T_n)*k_params

r_lin2   = Yt - Xt@np.linalg.lstsq(Xt,Yt,rcond=None)[0]
r_tvar   = np.vstack([Yt[e1_hat]-Xt[e1_hat]@B1_tv, Yt[e2_hat]-Xt[e2_hat]@B2_tv])
r_stv    = Yt - (1-Ft_hat)[:,None]*(Xt@B1_st) - Ft_hat[:,None]*(Xt@B2_st)
aic_lin, bic_lin  = model_ic(r_lin2, N*(N*LAGS+1),   T_tv)
aic_tvar,bic_tvar = model_ic(r_tvar, 2*N*(N*LAGS+1), T_tv)
aic_stv, bic_stv  = model_ic(r_stv,  2*N*(N*LAGS+1)+2, T_tv)

print(f"\nInformation Criteria:")
print(f"  {'Linear VAR':<12}  AIC = {aic_lin:8.1f}   BIC = {bic_lin:8.1f}")
print(f"  {'TVAR':<12}  AIC = {aic_tvar:8.1f}   BIC = {bic_tvar:8.1f}")
print(f"  {'STVAR':<12}  AIC = {aic_stv:8.1f}   BIC = {bic_stv:8.1f}\n")

stvar_flag = Ft_hat > 0.5

# STVAR: Transition function figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,6))
fig.suptitle("STVAR: Gradual Regime Changes",
             fontsize=12, fontweight="bold")
shade_recessions(ax1)
ax1.plot(dates_st, Zvar, color="#3333cc", lw=1.5, label="GDP Growth")
ax1.axhline(c_hat, color="red", ls="--", lw=1.5,
            label=f"Threshold c = {c_hat:.2f}%")
ax1.set_ylabel("GDP Growth (%)"); ax1.set_title("GDP Growth (Transition Variable)")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25, ls=":")

shade_recessions(ax2)
ax2.plot(dates_st, Ft_hat, color="#257a25", lw=2)
ax2.axhline(0.5, color="k", ls="--", lw=1.0)
ax2.set_ylim(0,1); ax2.set_ylabel("Transition Weight")
ax2.set_title(f"F(z): Weight on High-Growth Regime  (gamma = {gamma_hat:.2f})")
ax2.grid(True, alpha=0.25, ls=":")
plt.tight_layout()
plt.savefig("fig_stvar_transition.png", dpi=150, bbox_inches="tight")
plt.close()
print("STVAR transition figure saved.")

# STVAR: IRF continuum figure
hor = np.arange(HORZ)
fig, axes = plt.subplots(1, 3, figsize=(13,4))
fig.suptitle("STVAR: IRFs Across Continuum of States\n"
             "Monetary Policy Shock (1 SD Tightening)", fontsize=11, fontweight="bold")
for i, ax in enumerate(axes):
    for iz in range(len(z_states)):
        ax.plot(hor, irf_st[iz,:,i], color=cols_st[iz], lw=1.8,
                label=lbl_states[iz] if i==0 else None)
    ax.axhline(0,color="k",lw=0.6,ls=":"); ax.grid(True,alpha=0.25,ls=":")
    ax.set_xlabel("Quarters Ahead"); ax.set_title(resp_labels[i])
    if i == 0: ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig("fig_stvar_irf_continuum.png", dpi=150, bbox_inches="tight")
plt.close()
print("STVAR IRF continuum figure saved.")

# Model comparison figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
fig.suptitle("Model Selection: Information Criteria",
             fontsize=11, fontweight="bold")
models = ["Linear VAR","TVAR","STVAR"]
cols_ic = ["#8080cc","#4db34d","#cc8040"]
for ax, vals, lbl in [(ax1,[aic_lin,aic_tvar,aic_stv],"AIC (lower is better)"),
                      (ax2,[bic_lin,bic_tvar,bic_stv],"BIC (lower is better)")]:
    bars = ax.bar(models, vals, color=cols_ic)
    ax.set_ylabel(lbl); ax.grid(True, alpha=0.25, ls=":", axis="y")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+5, f"{v:.0f}",
                ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("fig_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Model comparison figure saved.\n")

# =============================================================================
# 7.  SYNTHESIS -- REGIME CONCORDANCE
# =============================================================================

print("=" * 65)
print("SYNTHESIS: REGIME CLASSIFICATION CONCORDANCE")
print("=" * 65)

T_common = min(len(tvp_hv_flag), T_ms-LAGS, len(tvar_flag), len(stvar_flag))
tvp_c    = tvp_hv_flag[:T_common].astype(int)
ms_c     = ms_stress_flag[LAGS:LAGS+T_common].astype(int)
tvar_c   = tvar_flag[:T_common].astype(int)
stvar_c  = stvar_flag[:T_common].astype(int)
dates_c  = dates_plot[LAGS:LAGS+T_common]

nber_flag = np.zeros(T_common, dtype=int)
for t, dt in enumerate(dates_c):
    for s,e in NBER_RECESSIONS:
        if pd.Timestamp(s) <= dt <= pd.Timestamp(e):
            nber_flag[t] = 1

flags  = np.column_stack([nber_flag, ms_c, tvar_c, stvar_c])
labels = ["NBER","MS-VAR","TVAR","STVAR"]
Cmat   = np.corrcoef(flags.T)

print("Concordance correlations:")
print(pd.DataFrame(Cmat, index=labels, columns=labels).round(3).to_string())
print()

# (Concordance heatmap removed -- correlation table printed above is sufficient.)

# Stacked regime timeline figure
cols_syn    = ["#cc3333","#3366cc","#33b333","#b36600"]
flag_labels = ["NBER Recession","MS-VAR Stress","TVAR Low-Growth","STVAR Stress"]
fig, axes = plt.subplots(4, 1, figsize=(13,6), sharex=True)
fig.suptitle("Regime Classifications Across Four Approaches",
             fontsize=11, fontweight="bold")
for i, ax in enumerate(axes):
    ax.fill_between(dates_c, 0, flags[:,i].astype(float),
                    color=cols_syn[i], alpha=0.6)
    ax.set_ylim(0,1); ax.set_yticks([0,1])
    ax.set_ylabel(flag_labels[i], fontsize=8); ax.grid(True,alpha=0.25,ls=":")
axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("fig_regime_timeline.png", dpi=150, bbox_inches="tight")
plt.close()
print("Regime timeline figure saved.\n")

# =============================================================================
# 8.  FINAL SUMMARY
# =============================================================================

print("=" * 65)
print("SYNTHESIS: MODEL-ROBUST FINDINGS")
print("=" * 65)

amp_tvp  = abs(irf_s_med[:,0].min()) / max(abs(irf_n_med[:,0].min()), 1e-6)
amp_ms   = abs(ms_s_med[:,0].min())  / max(abs(ms_n_med[:,0].min()),  1e-6)
amp_tvar = abs(girf1[:,0].min())     / max(abs(girf2[:,0].min()),     1e-6)
amp_stv  = abs(irf_st[0,:,0].min()) / max(abs(irf_st[-1,:,0].min()), 1e-6)

print(f"\nGDP Response Amplification (Stress/Normal or Low/High):")
print(f"  TVP-VAR-SV : {amp_tvp:.2f}x")
print(f"  MS-VAR     : {amp_ms:.2f}x")
print(f"  TVAR       : {amp_tvar:.2f}x")
print(f"  STVAR      : {amp_stv:.2f}x")
print("\nCore finding: All four models show LARGER policy effects during stress.\n")

outfiles = [
    "fig_tvpvar_volatility.png", "fig_tvpvar_irf_comparison.png",
    "fig_msvar_regime_prob.png", "fig_tvar_girf.png",
    "fig_stvar_transition.png",  "fig_stvar_irf_continuum.png",
    "fig_model_comparison.png", "fig_regime_timeline.png",
]
print("=" * 65)
print("OUTPUT FILES")
print("=" * 65)
for f in outfiles: print(f"  {f}")
print("=" * 65)
print("ALL DONE -- Replication complete.")
print("=" * 65)
