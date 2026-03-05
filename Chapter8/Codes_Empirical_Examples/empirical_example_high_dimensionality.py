"""
Replicating Bernanke, Boivin & Eliasz (2005) FAVAR Analysis
============================================================

Textbook: Macroeconometrics
Author:   Alessia Paccagnini

This script:
  1. Loads FRED-MD dataset (McCracken & Ng, 2016)
  2. Applies appropriate transformations for stationarity
  3. Extracts factors via PCA (excluding FFR)
  4. Estimates standard 3-variable VAR (IP, CPI, FFR)
  5. Estimates FAVAR (5 factors + FFR) — two-step approach
  6. Compares IRFs across subsamples (Pre-Volcker / Great Moderation / Full)
  7. Rolling out-of-sample forecasting: FAVAR vs VAR vs Random Walk
  8. Clark-West (2007) test for equal predictive ability (nested models)
  9. Giacomini-Rossi (2010) fluctuation test for forecast stability

References:
  Bernanke, Boivin & Eliasz (2005), QJE
  McCracken & Ng (2016), JBES
  Clark & West (2007), JoE
  Giacomini & Rossi (2010), ReStud

Data: 2025-12-MD.csv  (FRED-MD monthly)
Sample: 1962-01-01 to 2007-12-01 (pre-crisis, consistent with textbook)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import linalg, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# Figure style
# --------------------------------------------------------------------------
plt.rcParams.update({
    'figure.figsize': (12, 8), 'font.size': 11,
    'axes.labelsize': 12,      'axes.titlesize': 13,
    'legend.fontsize': 10,     'font.family': 'serif',
    'axes.spines.top': False,  'axes.spines.right': False,
})

# ==========================================================================
# SECTION 1 — DATA LOADING AND TRANSFORMATION
# ==========================================================================

def load_fred_md(filepath):
    """Load FRED-MD dataset; return raw DataFrame, transform codes, varnames."""
    df = pd.read_csv(filepath)
    transform_codes = df.iloc[0, 1:].values.astype(float)
    var_names = df.columns[1:].tolist()
    data = df.iloc[1:].copy()
    data['sasdate'] = pd.to_datetime(data['sasdate'], format='%m/%d/%Y')
    data = data.set_index('sasdate')
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data, transform_codes, var_names


def transform_series(x, tcode):
    """
    Apply FRED-MD transformation codes.
      1 = levels          2 = first difference     3 = second difference
      4 = log             5 = log first difference  6 = log second difference
      7 = Δ(x/x(-1) - 1)
    """
    small = 1e-6
    if   tcode == 1: return x
    elif tcode == 2: return x.diff()
    elif tcode == 3: return x.diff().diff()
    elif tcode == 4: return np.log(x.clip(lower=small))
    elif tcode == 5: return np.log(x.clip(lower=small)).diff()
    elif tcode == 6: return np.log(x.clip(lower=small)).diff().diff()
    elif tcode == 7: return (x / x.shift(1) - 1).diff()
    else:            return x


def prepare_data(data, transform_codes, var_names, start_date, end_date):
    """Transform all series, subset to sample, drop columns >10% missing."""
    transformed = pd.DataFrame(index=data.index)
    for i, var in enumerate(var_names):
        if var in data.columns:
            transformed[var] = transform_series(data[var], transform_codes[i])
    transformed = transformed.loc[start_date:end_date]
    # Keep columns with at most 10% missing
    missing_pct = transformed.isnull().mean()
    transformed = transformed.loc[:, missing_pct < 0.10]
    return transformed


# ==========================================================================
# SECTION 2 — FACTOR EXTRACTION
# ==========================================================================

def extract_factors(X, n_factors=5, var_to_exclude=None):
    """
    Extract principal component factors from panel X.
    Excludes var_to_exclude (typically ['FEDFUNDS']) before PCA.
    Returns factors, fitted PCA, fitted scaler, and the cleaned DataFrame.

    Note on implementation: this script uses sklearn.decomposition.PCA,
    whereas the companion MATLAB and R scripts use direct SVD decomposition
    (svds / svd). Both approaches are mathematically equivalent, but
    individual factor signs may differ across languages — this is expected
    (PCA sign is arbitrary) and does not affect IRF results since factors
    enter symmetrically. If comparing factor time series across languages,
    a sign flip on one or more factors is normal.
    """
    X_work = X.copy()
    if var_to_exclude:
        X_work = X_work.drop(columns=[v for v in var_to_exclude
                                       if v in X_work.columns], errors='ignore')
    # Drop rows with ANY missing value (explicit, avoids silent sample shrinkage)
    X_clean = X_work.dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(X_scaled)

    print(f"    Variables in panel : {X_clean.shape[1]}")
    print(f"    Observations       : {X_clean.shape[0]}")
    print(f"    Variance explained : {pca.explained_variance_ratio_.sum():.1%}")
    return factors, pca, scaler, X_clean


# ==========================================================================
# SECTION 3 — VAR / FAVAR ESTIMATION
# ==========================================================================

def estimate_var(Y, p):
    """
    Estimate VAR(p) by OLS.
    Returns B (K × Kp+1, intercept first), Sigma (K × K), residuals.
    """
    T, K = Y.shape
    # Build regressors: intercept + lags 1..p
    X = np.column_stack([np.ones(T - p)] +
                        [Y[p - i: T - i, :] for i in range(1, p + 1)])
    Y_dep = Y[p:, :]
    B = linalg.lstsq(X, Y_dep)[0].T          # K × (Kp+1)
    residuals = Y_dep - X @ B.T
    Sigma = (residuals.T @ residuals) / (T - p - K * p - 1)
    return B, Sigma, residuals


def var_companion_form(B, K, p):
    """Companion matrix from B (K × Kp+1, intercept in column 0)."""
    C = np.zeros((K * p, K * p))
    C[:K, :] = B[:, 1:]                       # drop intercept column
    if p > 1:
        C[K:, :K * (p - 1)] = np.eye(K * (p - 1))
    return C


# ==========================================================================
# SECTION 4 — IMPULSE RESPONSES
# ==========================================================================

def compute_irf(B, Sigma, K, p, horizon=48, shock_var=None, normalize_to=None):
    """
    Cholesky IRF.  normalize_to rescales so the shock variable's impact
    equals normalize_to (e.g. 0.25 for a 25bp FFR shock).
    """
    P = linalg.cholesky(Sigma, lower=True)
    C = var_companion_form(B, K, p)

    if shock_var is None:
        shock_var = K - 1
    e = np.zeros(K); e[shock_var] = 1.0
    impact = P @ e
    if normalize_to is not None:
        impact = impact * (normalize_to / impact[shock_var])

    irf = np.zeros((horizon, K))
    state = np.zeros(K * p)
    state[:K] = impact
    for h in range(horizon):
        irf[h, :] = state[:K]
        state = C @ state
    return irf


def bootstrap_var_irf(Y, p, horizon, n_boot=500, shock_var=None,
                      alpha=0.10, normalize_to=0.25):
    """
    Residual-bootstrap confidence intervals for Cholesky IRFs.
    Bug fix: lag reconstruction uses explicit indexing (avoids flatten/reverse
    ordering error).
    """
    T, K = Y.shape
    B, Sigma, residuals = estimate_var(Y, p)
    irf_point = compute_irf(B, Sigma, K, p, horizon, shock_var, normalize_to)

    irf_boot = np.zeros((n_boot, horizon, K))
    for b in range(n_boot):
        idx = np.random.choice(len(residuals), size=len(residuals), replace=True)
        resid_b = residuals[idx, :]
        Y_b = np.zeros_like(Y)
        Y_b[:p, :] = Y[:p, :]
        for t in range(p, T):
            # FIX: explicit lag stacking — no flatten/reverse ambiguity
            Y_lag = np.concatenate([Y_b[t - i, :] for i in range(1, p + 1)])
            Y_b[t, :] = B[:, 0] + B[:, 1:] @ Y_lag + resid_b[t - p, :]
        try:
            Bb, Sb, _ = estimate_var(Y_b, p)
            irf_boot[b] = compute_irf(Bb, Sb, K, p, horizon, shock_var, normalize_to)
        except Exception:
            irf_boot[b] = irf_point

    irf_lo = np.percentile(irf_boot, alpha / 2 * 100, axis=0)
    irf_hi = np.percentile(irf_boot, (1 - alpha / 2) * 100, axis=0)
    return irf_point, irf_lo, irf_hi


def recover_favar_irf(irf_factors, pca, scaler, X_clean, n_factors,
                      response_vars):
    """
    Recover IRFs in original units from FAVAR factor-space responses.
    Formula (eq. 8.29): ∂X_{i,t+h}/∂ε_t = σ_i · λ_i' · ∂F_{t+h}/∂ε_t
    """
    var_stds  = scaler.scale_
    var_names = X_clean.columns.tolist()
    irf_dict  = {}
    for var in response_vars:
        if var in var_names:
            idx = var_names.index(var)
            loadings = pca.components_[:, idx]         # (n_factors,)
            irf_dict[var] = (irf_factors[:, :n_factors] @ loadings
                             * var_stds[idx])
    return irf_dict


# ==========================================================================
# SECTION 5 — IRF ANALYSIS PER SUBSAMPLE
# ==========================================================================

def analyze_subsample(data, transform_codes, var_names,
                      start_date, end_date, sample_name,
                      n_factors=5, p=12, horizon=48, n_boot=500):
    print(f"\n{'='*60}")
    print(f"  {sample_name}")
    print(f"  {start_date}  →  {end_date}")
    print('='*60)

    transformed = prepare_data(data, transform_codes, var_names,
                                start_date, end_date)

    factors, pca, scaler, X_clean = extract_factors(
        transformed, n_factors=n_factors, var_to_exclude=['FEDFUNDS'])

    VAR_VARS = ['INDPRO', 'CPIAUCSL', 'FEDFUNDS']
    factor_dates = X_clean.index
    var_data = transformed.loc[factor_dates, VAR_VARS].dropna()
    common_dates = var_data.index

    F = pd.DataFrame(factors, index=factor_dates,
                     columns=[f'F{i+1}' for i in range(n_factors)]
                     ).loc[common_dates].values
    Y_var  = var_data.values
    Y_ffr  = var_data[['FEDFUNDS']].values
    Y_favar = np.column_stack([F, Y_ffr])

    print(f"    Final T = {len(common_dates)}")

    # VAR IRFs
    irf_var, irf_lo, irf_hi = bootstrap_var_irf(
        Y_var, p, horizon, n_boot=n_boot, shock_var=2, normalize_to=0.25)

    # FAVAR IRFs
    B_fav, Sig_fav, _ = estimate_var(Y_favar, p)
    K_fav = n_factors + 1
    irf_fav_raw = compute_irf(B_fav, Sig_fav, K_fav, p, horizon,
                               shock_var=n_factors, normalize_to=0.25)
    irf_favar = recover_favar_irf(irf_fav_raw, pca, scaler, X_clean,
                                   n_factors, ['INDPRO', 'CPIAUCSL'])
    irf_favar['FEDFUNDS'] = irf_fav_raw[:, -1:]

    var_cpi_pos   = int(np.sum(irf_var[:12, 1] > 0))
    favar_cpi_pos = int(np.sum(irf_favar['CPIAUCSL'][:12] > 0))
    print(f"    Price puzzle — VAR: {var_cpi_pos}/12 pos  "
          f"FAVAR: {favar_cpi_pos}/12 pos")

    return dict(sample_name=sample_name, T=len(common_dates),
                irf_var=irf_var, irf_lo=irf_lo, irf_hi=irf_hi,
                irf_favar=irf_favar,
                var_cpi_pos=var_cpi_pos, favar_cpi_pos=favar_cpi_pos,
                Y_var=Y_var, F=F, Y_favar=Y_favar,
                pca=pca, scaler=scaler, X_clean=X_clean,
                n_factors=n_factors, p=p)


# ==========================================================================
# SECTION 6 — ROLLING FORECASTING: VAR vs FAVAR vs RANDOM WALK
# ==========================================================================

def rolling_forecasts(Y_var, F, p=12, h=1,
                      initial_window=120, n_factors=5):
    """
    Rolling-window one-step-ahead (or h-step-ahead) forecasts for:
      - Random Walk (RW)
      - VAR(p)
      - FAVAR(p): [factors, FFR]

    Returns arrays of shape (n_forecasts,) for each model (FFR column),
    plus actuals.
    """
    T = Y_var.shape[0]
    Y_favar = np.column_stack([F, Y_var[:, -1:]])   # [factors, FFR]
    K_var   = Y_var.shape[1]
    K_favar = Y_favar.shape[1]

    fc_rw    = []
    fc_var   = []
    fc_favar = []
    actuals  = []

    for t in range(initial_window, T - h):
        # ---- Estimation windows ----
        Y_v_est  = Y_var[:t,   :]
        Y_f_est  = Y_favar[:t, :]

        actual = Y_var[t + h - 1, 2]   # FFR h-steps ahead
        actuals.append(actual)

        # Random Walk: last observed FFR
        fc_rw.append(Y_var[t - 1, 2])

        # VAR
        try:
            B_v, Sig_v, _ = estimate_var(Y_v_est, p)
            state_v = np.zeros(K_var * p)
            state_v[:K_var] = Y_var[t - 1, :]
            C_v = var_companion_form(B_v, K_var, p)
            fc_h = Y_var[t - 1, :]
            for _ in range(h):
                fc_h = B_v[:, 0] + B_v[:, 1:] @ state_v
                state_v = np.concatenate([fc_h,
                                          state_v[:K_var * (p - 1)]])
            fc_var.append(fc_h[2])
        except Exception:
            fc_var.append(Y_var[t - 1, 2])

        # FAVAR
        try:
            B_f, Sig_f, _ = estimate_var(Y_f_est, p)
            state_f = np.zeros(K_favar * p)
            state_f[:K_favar] = Y_favar[t - 1, :]
            fc_h_f = Y_favar[t - 1, :]
            for _ in range(h):
                fc_h_f = B_f[:, 0] + B_f[:, 1:] @ state_f
                state_f = np.concatenate([fc_h_f,
                                          state_f[:K_favar * (p - 1)]])
            fc_favar.append(fc_h_f[-1])
        except Exception:
            fc_favar.append(Y_var[t - 1, 2])

    return (np.array(fc_rw),   np.array(fc_var),
            np.array(fc_favar), np.array(actuals))


# ==========================================================================
# SECTION 7 — FORECAST EVALUATION TESTS
# ==========================================================================

def clark_west_test(actual, fc_benchmark, fc_model):
    """
    Clark & West (2007) MSPE-adjusted test for nested models.

    H₀: benchmark (RW or VAR) forecasts as well as the larger model.
    Under H₀ the larger model's MSPE is inflated by an estimation-error term;
    CW corrects for this.

    Statistic: t-test on ĉ_t = (e₁²_t) - (e₂²_t - (ŷ₁_t - ŷ₂_t)²)
    where e₁ = benchmark error, e₂ = model error.

    Returns: CW statistic, one-sided p-value, MSPE₁, MSPE₂
    Reference: Clark & West (2007), Journal of Econometrics 138, 291-311.
    """
    e1 = actual - fc_benchmark
    e2 = actual - fc_model
    c  = e1**2 - (e2**2 - (fc_benchmark - fc_model)**2)
    c_bar = c.mean()
    se    = c.std(ddof=1) / np.sqrt(len(c))
    cw_stat = c_bar / se if se > 0 else np.nan
    # One-sided: H₁ model beats benchmark
    p_val   = 1 - stats.norm.cdf(cw_stat) if not np.isnan(cw_stat) else np.nan
    mspe1   = (e1**2).mean()
    mspe2   = (e2**2).mean()
    return cw_stat, p_val, mspe1, mspe2


def giacomini_rossi_test(actual, fc1, fc2, window=None, alpha=0.10):
    """
    Giacomini & Rossi (2010) fluctuation test for forecast instability.

    Tests whether the relative forecasting performance of two models
    is stable over time — i.e. whether one model beats the other
    consistently or only in specific subperiods.

    Procedure:
      1. Compute loss differential d_t = L(e1_t) - L(e2_t) with L = squared error.
      2. Compute rolling mean d̄_{t,m} over window m.
      3. GR statistic: sup_t |√m · d̄_{t,m} / σ̂_d|
      4. Compare to asymptotic critical values (Giacomini & Rossi 2010, Table 1).

    Returns: GR statistic, critical value at alpha, rolling means, time index.
    Reference: Giacomini & Rossi (2010), Review of Economic Studies 77, 530-561.
    """
    e1 = actual - fc1
    e2 = actual - fc2
    d  = e1**2 - e2**2

    n = len(d)
    if window is None:
        window = max(int(np.floor(n * 0.2)), 10)   # 20% of sample, ≥10

    # Long-run variance of d (Newey-West, bandwidth = window-1)
    d_dm  = d - d.mean()
    bw    = window - 1
    lrv   = np.var(d, ddof=1)
    for k in range(1, bw + 1):
        w    = 1 - k / (bw + 1)
        lrv += 2 * w * np.mean(d_dm[k:] * d_dm[:-k])
    lrv = max(lrv, 1e-12)

    # Rolling means
    rolling = np.array([d[t - window: t].mean()
                        for t in range(window, n + 1)])
    gr_series = np.sqrt(window) * rolling / np.sqrt(lrv)
    gr_stat   = np.max(np.abs(gr_series))

    # Critical values from Giacomini & Rossi (2010), Table 1 (two-sided)
    # These are for the sup-norm of a Brownian bridge; λ = m/P ≈ window/n
    lam = window / n
    # Approximate critical values (alpha = 0.10, 0.05, 0.01)
    cv_table = {0.10: 2.49, 0.05: 2.80, 0.01: 3.40}
    cv = cv_table.get(alpha, 2.80)

    time_idx = np.arange(window, n + 1)
    return gr_stat, cv, gr_series, time_idx


def compute_forecast_metrics(actual, fc_rw, fc_var, fc_favar):
    """Compute RMSE, MAE, and test statistics for all three models."""
    models = {'RW': fc_rw, 'VAR': fc_var, 'FAVAR': fc_favar}
    metrics = {}
    for name, fc in models.items():
        e = actual - fc
        metrics[name] = {'RMSE': np.sqrt((e**2).mean()),
                          'MAE' : np.abs(e).mean()}

    # Clark-West tests (nested: RW ⊂ VAR ⊂ FAVAR)
    cw_var_rw,   p_var_rw,   _, _ = clark_west_test(actual, fc_rw,  fc_var)
    cw_favar_rw, p_favar_rw, _, _ = clark_west_test(actual, fc_rw,  fc_favar)
    cw_favar_var,p_favar_var,_, _ = clark_west_test(actual, fc_var, fc_favar)

    # Giacomini-Rossi fluctuation tests
    gr_var_rw,   cv, gr_s_vr,  ti_vr  = giacomini_rossi_test(actual, fc_rw,  fc_var)
    gr_favar_rw, cv, gr_s_fr,  ti_fr  = giacomini_rossi_test(actual, fc_rw,  fc_favar)
    gr_favar_var,cv, gr_s_fv,  ti_fv  = giacomini_rossi_test(actual, fc_var, fc_favar)

    tests = {
        'CW_VAR_vs_RW'   : (cw_var_rw,    p_var_rw),
        'CW_FAVAR_vs_RW' : (cw_favar_rw,  p_favar_rw),
        'CW_FAVAR_vs_VAR': (cw_favar_var, p_favar_var),
        'GR_VAR_vs_RW'   : (gr_var_rw,    cv),
        'GR_FAVAR_vs_RW' : (gr_favar_rw,  cv),
        'GR_FAVAR_vs_VAR': (gr_favar_var, cv),
        'GR_series'      : {'VAR_vs_RW'   : (gr_s_vr,  ti_vr),
                             'FAVAR_vs_RW' : (gr_s_fr,  ti_fr),
                             'FAVAR_vs_VAR': (gr_s_fv,  ti_fv)},
    }
    return metrics, tests


# ==========================================================================
# SECTION 8 — PLOTTING
# ==========================================================================

BLUE, RED, GREEN = '#1f77b4', '#d62728', '#2ca02c'


def plot_irf_comparison(results_list, savepath_prefix):
    """Figure 8.1: 2×3 IRF comparison across subsamples."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    titles = [r['sample_name'] for r in results_list]
    months = np.arange(results_list[0]['irf_var'].shape[0])

    for col, res in enumerate(results_list):
        # --- CPI row ---
        ax = axes[0, col]
        ax.fill_between(months,
                        res['irf_lo'][:, 1] * 100,
                        res['irf_hi'][:, 1] * 100,
                        color=BLUE, alpha=0.20)
        ax.plot(months, res['irf_var'][:, 1] * 100,
                color=BLUE, lw=2, label='VAR')
        ax.plot(months, res['irf_favar']['CPIAUCSL'] * 100,
                color=RED, lw=2, ls='--', label='FAVAR')
        ax.axhline(0, color='k', lw=0.7)
        ax.axvspan(0, 12, alpha=0.08, color='red')
        ax.set_title(titles[col], fontweight='bold')
        if col == 0:
            ax.set_ylabel('CPI Response (%)')
            ax.legend(loc='upper right')
        ax.text(0.05, 0.95,
                f"VAR: {res['var_cpi_pos']}/12 pos.",
                transform=ax.transAxes, fontsize=9,
                color=BLUE, va='top')
        ax.set_xlim([0, len(months) - 1])
        ax.grid(True, alpha=0.25)

        # --- IP row ---
        ax = axes[1, col]
        ax.fill_between(months,
                        res['irf_lo'][:, 0] * 100,
                        res['irf_hi'][:, 0] * 100,
                        color=BLUE, alpha=0.20)
        ax.plot(months, res['irf_var'][:, 0] * 100,
                color=BLUE, lw=2, label='VAR')
        ax.plot(months, res['irf_favar']['INDPRO'] * 100,
                color=RED, lw=2, ls='--', label='FAVAR')
        ax.axhline(0, color='k', lw=0.7)
        if col == 0:
            ax.set_ylabel('IP Response (%)')
        ax.set_xlabel('Months')
        ax.set_xlim([0, len(months) - 1])
        ax.grid(True, alpha=0.25)

    fig.suptitle('VAR vs FAVAR Across Monetary Policy Regimes\n'
                 'Response to 25bp Federal Funds Rate Increase',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{savepath_prefix}_irf.pdf', bbox_inches='tight')
    plt.savefig(f'{savepath_prefix}_irf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {savepath_prefix}_irf.pdf/.png")


def plot_forecast_evaluation(metrics, tests, fc_rw, fc_var, fc_favar,
                              actual, savepath_prefix):
    """
    Two-panel figure:
      Left  — RMSE bar chart (RW / VAR / FAVAR)
      Right — Giacomini-Rossi fluctuation paths vs critical value
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- RMSE bars ---
    names  = ['RW', 'VAR', 'FAVAR']
    rmses  = [metrics[n]['RMSE'] for n in names]
    colors = [BLUE, GREEN, RED]
    bars = ax1.bar(names, rmses, color=colors, alpha=0.80, edgecolor='k', lw=0.8)
    for bar, val in zip(bars, rmses):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 val + max(rmses) * 0.01,
                 f'{val:.4f}', ha='center', fontsize=10)
    ax1.set_ylabel('RMSE (FFR, percentage points)')
    ax1.set_title('Forecast Accuracy', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # CW annotations
    cw_pairs = [
        ('CW_VAR_vs_RW',    'VAR vs RW',    1.0),
        ('CW_FAVAR_vs_RW',  'FAVAR vs RW',  2.0),
        ('CW_FAVAR_vs_VAR', 'FAVAR vs VAR', 1.0),
    ]
    y_top = max(rmses) * 1.12
    for key, label, x_mid in cw_pairs:
        stat, pval = tests[key]
        stars = ('***' if pval < 0.01 else
                 '**'  if pval < 0.05 else
                 '*'   if pval < 0.10 else '')
        ax1.text(x_mid, y_top,
                 f'CW: {stat:.2f}{stars}',
                 ha='center', fontsize=8, color='dimgray')
    ax1.set_ylim([0, max(rmses) * 1.22])

    # --- GR fluctuation paths ---
    gr_data = tests['GR_series']
    cv      = tests['GR_FAVAR_vs_RW'][1]    # same CV for all three
    line_styles = [('-', BLUE,  'VAR vs RW'),
                   ('--', RED,  'FAVAR vs RW'),
                   (':', GREEN, 'FAVAR vs VAR')]
    for (ls, col, label), (key, (gr_s, ti)) in zip(
            line_styles, gr_data.items()):
        ax2.plot(ti, gr_s, lw=1.8, ls=ls, color=col, label=label)
    ax2.axhline( cv, color='k', lw=1.5, ls='--', label=f'CV (10%): ±{cv}')
    ax2.axhline(-cv, color='k', lw=1.5, ls='--')
    ax2.axhline(0, color='gray', lw=0.7)
    ax2.set_xlabel('Rolling window end (observation)')
    ax2.set_ylabel('GR fluctuation statistic')
    ax2.set_title('Giacomini–Rossi (2010) Fluctuation Test\n'
                  'Forecast Stability over Time', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    fig.suptitle('Out-of-Sample Forecast Comparison: FFR\n'
                 'RW vs VAR vs FAVAR  |  Clark-West & Giacomini-Rossi Tests',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{savepath_prefix}_forecast_eval.pdf', bbox_inches='tight')
    plt.savefig(f'{savepath_prefix}_forecast_eval.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: {savepath_prefix}_forecast_eval.pdf/.png")


# ==========================================================================
# SECTION 9 — PRINT RESULTS TABLES
# ==========================================================================

def print_forecast_table(metrics, tests):
    """Print forecast evaluation table to console."""
    print("\n" + "="*65)
    print("OUT-OF-SAMPLE FORECAST EVALUATION  (target: FFR, h=1)")
    print("="*65)
    print(f"{'Model':<10} {'RMSE':>10} {'MAE':>10}")
    print("-"*32)
    for name in ['RW', 'VAR', 'FAVAR']:
        print(f"{name:<10} {metrics[name]['RMSE']:>10.4f} "
              f"{metrics[name]['MAE']:>10.4f}")

    print("\n--- Clark-West (2007) Test  [one-sided, H₁: model > benchmark] ---")
    print(f"  {'Comparison':<20} {'CW stat':>10} {'p-value':>10} {'sig':>5}")
    print("  " + "-"*50)
    cw_rows = [('VAR vs RW',    'CW_VAR_vs_RW'),
               ('FAVAR vs RW',  'CW_FAVAR_vs_RW'),
               ('FAVAR vs VAR', 'CW_FAVAR_vs_VAR')]
    for label, key in cw_rows:
        stat, pval = tests[key]
        stars = ('***' if pval < 0.01 else '**' if pval < 0.05
                 else '*' if pval < 0.10 else '')
        print(f"  {label:<20} {stat:>10.3f} {pval:>10.3f} {stars:>5}")

    print("\n--- Giacomini-Rossi (2010) Fluctuation Test ---")
    print(f"  {'Comparison':<20} {'GR stat':>10} {'CV (10%)':>10} {'stable?':>8}")
    print("  " + "-"*52)
    gr_rows = [('VAR vs RW',    'GR_VAR_vs_RW'),
               ('FAVAR vs RW',  'GR_FAVAR_vs_RW'),
               ('FAVAR vs VAR', 'GR_FAVAR_vs_VAR')]
    for label, key in gr_rows:
        stat, cv = tests[key]
        stable = 'YES' if stat < cv else 'NO'
        print(f"  {label:<20} {stat:>10.3f} {cv:>10.3f} {stable:>8}")

    print("\n  Note: Clark-West corrects for the upward MSPE bias of nested")
    print("  larger models under H₀ of equal predictive ability.")
    print("  Giacomini-Rossi tests whether relative performance is stable")
    print("  over rolling subperiods (instability → structural change).")
    print("="*65)


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    print("="*70)
    print("BBE (2005) FAVAR Replication with FRED-MD Data")
    print("Section 8.13.5  |  Macroeconometrics  |  Alessia Paccagnini")
    print("="*70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    DATA_FILE = '2025-12-MD.csv'
    print(f"\n1. Loading {DATA_FILE} ...")
    data, transform_codes, var_names = load_fred_md(DATA_FILE)
    print(f"   Raw data: {data.shape[0]} obs, {data.shape[1]} variables")
    print(f"   Range  : {data.index[0].date()} — {data.index[-1].date()}")

    # ------------------------------------------------------------------
    # Subsample IRF analysis (Section 8.13.5 / Figure 8.1)
    # ------------------------------------------------------------------
    SUBSAMPLES = [
        ('1962-01-01', '1984-12-01', 'Pre-Volcker Era (1962-1984)'),
        ('1985-01-01', '2007-12-01', 'Great Moderation (1985-2007)'),
        ('1962-01-01', '2007-12-01', 'Full Sample (1962-2007)'),
    ]

    print("\n2. Subsample IRF analysis ...")
    results_irf = []
    for start, end, name in SUBSAMPLES:
        res = analyze_subsample(data, transform_codes, var_names,
                                start, end, name,
                                n_factors=5, p=12, horizon=48, n_boot=500)
        results_irf.append(res)

    plot_irf_comparison(results_irf, 'bbe_favar')

    # ------------------------------------------------------------------
    # Price puzzle summary table
    # ------------------------------------------------------------------
    print("\n" + "="*65)
    print("PRICE PUZZLE SUMMARY")
    print("="*65)
    print(f"{'Sample':<35} {'T':>5} {'VAR pos':>9} {'FAVAR pos':>10}")
    print("-"*65)
    for res in results_irf:
        print(f"{res['sample_name']:<35} {res['T']:>5} "
              f"{res['var_cpi_pos']:>6}/12 {res['favar_cpi_pos']:>7}/12")

    # ------------------------------------------------------------------
    # Rolling forecast comparison on Full Sample
    # ------------------------------------------------------------------
    print("\n3. Rolling forecasting on Full Sample (h=1, initial window=120) ...")
    full = results_irf[2]    # Full Sample result
    fc_rw, fc_var, fc_favar, actuals = rolling_forecasts(
        full['Y_var'], full['F'],
        p=full['p'], h=1, initial_window=120, n_factors=full['n_factors'])

    print(f"   Forecast origins: {len(actuals)}")

    # ------------------------------------------------------------------
    # Forecast evaluation + tests
    # ------------------------------------------------------------------
    print("\n4. Computing forecast metrics and test statistics ...")
    metrics, tests = compute_forecast_metrics(actuals, fc_rw, fc_var, fc_favar)
    print_forecast_table(metrics, tests)

    # ------------------------------------------------------------------
    # Save forecast evaluation figure
    # ------------------------------------------------------------------
    print("\n5. Saving figures ...")
    plot_forecast_evaluation(metrics, tests,
                              fc_rw, fc_var, fc_favar, actuals,
                              'bbe_favar')

    # ------------------------------------------------------------------
    # Save numerical results
    # ------------------------------------------------------------------
    months = np.arange(48)
    for res in results_irf:
        tag = (res['sample_name']
               .replace(' ', '_').replace('(', '').replace(')', '')
               .replace('-', '_'))
        pd.DataFrame({
            'Month'    : months,
            'VAR_IP'   : res['irf_var'][:, 0] * 100,
            'VAR_CPI'  : res['irf_var'][:, 1] * 100,
            'VAR_FFR'  : res['irf_var'][:, 2],
            'FAVAR_IP' : res['irf_favar']['INDPRO'] * 100,
            'FAVAR_CPI': res['irf_favar']['CPIAUCSL'] * 100,
        }).to_csv(f'bbe_irf_{tag}.csv', index=False)

    pd.DataFrame({
        'Actual' : actuals, 'RW': fc_rw,
        'VAR'    : fc_var,  'FAVAR': fc_favar,
    }).to_csv('bbe_forecast_results.csv', index=False)

    print("\n" + "="*70)
    print("Done. Files generated:")
    print("  bbe_favar_irf.pdf/.png        — Figure 8.1 (IRF comparison)")
    print("  bbe_favar_forecast_eval.pdf/.png — Forecast evaluation figure")
    print("  bbe_irf_*.csv                 — Numerical IRF results")
    print("  bbe_forecast_results.csv      — Rolling forecast series")
    print("="*70)


if __name__ == "__main__":
    main()
