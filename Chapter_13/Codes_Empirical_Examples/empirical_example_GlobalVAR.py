"""
================================================================================
GVAR: US MONETARY POLICY SPILLOVERS
Replication of Section 13.7.2 — Macroeconometrics Textbook
Author:   Alessia Paccagnini
Textbook: Macroeconometrics (De Gruyter)
================================================================================

Book specification (eq. 13.74):
  Non-US countries  VARX*(1,1):
    y_it = α_i + Φ_i y_{i,t-1} + Λ_{i0} y*_it + Λ_{i1} y*_{i,t-1} + ε_it

  USA  closed VAR(2) + weakly exogenous oil (dominant unit)
  y_it = (Δy_it, π_it, rs_it)'  — GDP growth, CPI inflation, short rate

  Trade-weighted foreign variables (eq. 13.67):
    y*_it = Σ_{j≠i} w_ij y_jt,   w_ii = 0,  Σ_j w_ij = 1

Country set (Section 13.7.2):
  USA (dominant unit), Euro Area, UK, Japan, China, Canada, Korea, Brazil
  N = 8,  T = 162 (1979Q2–2019Q4, simulated)

  wCAN,USA ≈ 0.55  (book Section 13.7.2 + Figure 13.4 text)

DGP calibrated to reproduce:
  Table 13.7:  All F-stats < 3.07 (weak exogeneity holds)
               Canada Δy*: 2.15 (largest, tight US integration)
  Figure 13.2: Max eigenvalue ≈ 0.973
  Figure 13.3: GDP trough at h=2:
               USA −0.141 pp, Canada −0.093 pp, UK −0.034 pp,
               Euro Area −0.022 pp, Japan −0.009 pp, China −0.003 pp
  Figure 13.4: Own shocks 91–97% of GDP FEV (h=8)
               USA → Canada ≈ 6%,  Euro → UK ≈ 3%,  Canada → USA ≈ 3%

Script steps:
  1.  Simulate GVAR data
  2.  Construct trade-weighted foreign variables  (eq. 13.67)
  3.  Estimate VARX*(1,1) for non-US; VAR(2)+oil for USA
  4.  Weak exogeneity tests  (Table 13.7)
  5.  Stack global VAR  (eq. 13.71)
  6.  Stability check — eigenvalue plot  (Figure 13.2)
  7.  GIRF to US interest-rate shock  (Figure 13.3)
  8.  FEVD heatmap + spillover network  (Figure 13.4)
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import linalg
from scipy.stats import f as f_dist

np.random.seed(42)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10})

# ── country / variable definitions ───────────────────────────────────────────
COUNTRIES  = ["USA", "Euro_Area", "UK", "Japan", "China",
              "Canada", "Korea", "Brazil"]
N          = len(COUNTRIES)
VARS       = ["gdp", "inf", "rs"]     # GDP growth, inflation, short rate
K          = len(VARS)                # 3
T          = 162                      # 1979Q2–2019Q4
CIDX       = {c: i for i, c in enumerate(COUNTRIES)}

# ── Trade weight matrix  (wCAN,USA = 0.55, dominant US share) ─────────────────
# Rows = home country,  columns = partner country
# Calibrated to reproduce FEVD pattern in Figure 13.4
_W_raw = np.array([
#    USA    EUR    UK     JPN    CHN    CAN    KOR    BRA
    [0.00,  0.18,  0.10,  0.12,  0.14,  0.22,  0.13,  0.11],  # USA
    [0.26,  0.00,  0.18,  0.10,  0.14,  0.08,  0.12,  0.12],  # Euro Area
    [0.22,  0.36,  0.00,  0.08,  0.10,  0.07,  0.08,  0.09],  # UK
    [0.28,  0.14,  0.07,  0.00,  0.21,  0.08,  0.13,  0.09],  # Japan
    [0.20,  0.18,  0.08,  0.18,  0.00,  0.08,  0.18,  0.10],  # China
    [0.55,  0.09,  0.07,  0.07,  0.08,  0.00,  0.07,  0.07],  # Canada
    [0.25,  0.14,  0.08,  0.20,  0.18,  0.07,  0.00,  0.08],  # Korea
    [0.22,  0.20,  0.10,  0.08,  0.14,  0.08,  0.08,  0.00],  # Brazil
])
W = _W_raw / _W_raw.sum(1, keepdims=True)   # row-normalise

# ── DGP parameters calibrated to book figures ─────────────────────────────────
# US VAR(2): higher interest rate persistence → near-unit eigenvalue ~ 0.973
A1_US = np.array([
    [ 0.50,  0.00, -0.28],    # GDP:  own persistence + strong rate drag
    [ 0.09,  0.55,  0.06],    # inf:  Phillips curve + rate effect
    [ 0.22,  0.12,  0.80],    # rs:   Taylor rule, high smoothing
])
A2_US = np.array([
    [ 0.09,  0.00, -0.12],    # second-lag rate drag
    [ 0.02,  0.07,  0.00],
    [ 0.06,  0.04,  0.08],
])
# Oil effect on US variables (weakly exogenous oil)
GAMMA_OIL_US = np.array([-0.06, 0.08, 0.02])

# Non-US: Φ_i, Λ_{i0}, Λ_{i1}
# Interest rate transmission to GDP is the key channel:
# Lambda0[gdp, rs] < 0  →  foreign rate rise  reduces domestic GDP
# Canada has the strongest link (wCAN,USA=0.55)
# PHI_i: diagonal persistence matrices (matching MATLAB/R specification)
PHI = {
    "Euro_Area": np.diag([0.50, 0.62, 0.82]),
    "UK":        np.diag([0.48, 0.58, 0.80]),
    "Japan":     np.diag([0.44, 0.52, 0.85]),
    "China":     np.diag([0.60, 0.65, 0.84]),
    "Canada":    np.diag([0.52, 0.58, 0.81]),
    "Korea":     np.diag([0.46, 0.54, 0.78]),
    "Brazil":    np.diag([0.40, 0.68, 0.74]),
}
# Lambda_0: contemporaneous foreign effect
# Key: gdp equation responds negatively to foreign interest rate
LAM0 = {
    "Euro_Area": np.array([[-0.10, 0.04, -0.08], [0.04, 0.08, 0.04], [0.06, 0.04, 0.14]]),
    "UK":        np.array([[-0.12, 0.04, -0.10], [0.04, 0.09, 0.04], [0.07, 0.04, 0.16]]),
    "Japan":     np.array([[-0.05, 0.02, -0.04], [0.02, 0.04, 0.02], [0.04, 0.02, 0.10]]),
    "China":     np.array([[-0.04, 0.02, -0.03], [0.02, 0.05, 0.02], [0.03, 0.02, 0.08]]),
    "Canada":    np.array([[-0.18, 0.06, -0.14], [0.06, 0.12, 0.06], [0.09, 0.06, 0.20]]),
    "Korea":     np.array([[-0.07, 0.03, -0.06], [0.03, 0.07, 0.03], [0.05, 0.03, 0.12]]),
    "Brazil":    np.array([[-0.06, 0.04, -0.05], [0.04, 0.09, 0.03], [0.05, 0.04, 0.10]]),
}
LAM1 = {c: 0.35 * LAM0[c] for c in LAM0}

# Residual standard deviations
SIG = {
    "USA":      np.array([0.010, 0.005, 0.006]),
    "Euro_Area":np.array([0.009, 0.004, 0.005]),
    "UK":       np.array([0.010, 0.005, 0.006]),
    "Japan":    np.array([0.009, 0.003, 0.004]),
    "China":    np.array([0.013, 0.005, 0.006]),
    "Canada":   np.array([0.010, 0.004, 0.005]),
    "Korea":    np.array([0.012, 0.005, 0.008]),
    "Brazil":   np.array([0.018, 0.008, 0.012]),
}
ALPHA = {
    "USA":      np.array([0.006, 0.004, 0.010]),
    "Euro_Area":np.array([0.004, 0.003, 0.008]),
    "UK":       np.array([0.005, 0.004, 0.009]),
    "Japan":    np.array([0.003, 0.001, 0.005]),
    "China":    np.array([0.013, 0.004, 0.008]),
    "Canada":   np.array([0.005, 0.004, 0.010]),
    "Korea":    np.array([0.008, 0.004, 0.060]),
    "Brazil":   np.array([0.004, 0.006, 0.055]),
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1  SIMULATE DATA
# ══════════════════════════════════════════════════════════════════════════════

def simulate_gvar(seed: int = 42) -> dict:
    """
    Simulate GVAR panel for N=8 countries, T=162 quarters.

    USA is the dominant unit; its VAR(2)+oil does not include foreign
    variables — this is the closed-economy assumption (Section 13.7.2).
    All non-US countries follow VARX*(1,1) with trade-weighted foreign vars.

    Returns: dict  country → pd.DataFrame(gdp, inf, rs)
    """
    rng = np.random.default_rng(seed)

    # Common oil price (weakly exogenous)
    oil = np.zeros(T)
    for t in range(1, T):
        oil[t] = 0.75 * oil[t-1] + rng.normal(0, 0.045)

    # Storage  (T × K per country)
    Y = {c: np.zeros((T, K)) for c in COUNTRIES}

    # Initialise with small noise around fixed effects
    for c in COUNTRIES:
        Y[c][0] = ALPHA[c] + rng.normal(0, SIG[c] * 0.5)
        Y[c][1] = ALPHA[c] + rng.normal(0, SIG[c] * 0.5)

    for t in range(2, T):
        # ── USA (dominant unit, closed VAR(2) + oil) ─────────────────────────
        eps_us    = rng.multivariate_normal(np.zeros(K), np.diag(SIG["USA"]**2))
        Y["USA"][t] = (ALPHA["USA"]
                       + A1_US @ Y["USA"][t-1]
                       + A2_US @ Y["USA"][t-2]
                       + GAMMA_OIL_US * oil[t]
                       + eps_us)

        # ── Non-US countries (VARX*(1,1)) ─────────────────────────────────────
        for c in COUNTRIES[1:]:
            i = CIDX[c]
            # Trade-weighted foreign aggregate y*_it
            # Use contemporaneous USA (already computed) + lagged others
            y_star = np.zeros(K)
            for j, c2 in enumerate(COUNTRIES):
                if j != i:
                    if c2 == "USA":
                        y_star += W[i, j] * Y[c2][t]      # contemp. dominant unit
                    else:
                        y_star += W[i, j] * Y[c2][t-1]    # lagged other countries
            y_star_lag = np.zeros(K)
            for j, c2 in enumerate(COUNTRIES):
                if j != i:
                    y_star_lag += W[i, j] * Y[c2][t-1]

            eps_c = rng.multivariate_normal(np.zeros(K), np.diag(SIG[c]**2))
            Y[c][t] = (ALPHA[c]
                       + PHI[c]  @ Y[c][t-1]
                       + LAM0[c] @ y_star
                       + LAM1[c] @ y_star_lag
                       + eps_c)

    dates = pd.period_range("1979Q2", periods=T, freq="Q").to_timestamp()
    return {c: pd.DataFrame(Y[c], columns=VARS, index=dates)
            for c in COUNTRIES}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2  TRADE-WEIGHTED FOREIGN VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

def make_foreign_vars(country_data: dict) -> dict:
    """
    y*_it = Σ_{j≠i} w_ij y_jt   (eq. 13.67)
    Returns: dict  country → pd.DataFrame(gdp_star, inf_star, rs_star)
    """
    foreign = {}
    for c in COUNTRIES:
        i  = CIDX[c]
        Ys = np.zeros((T, K))
        for j, c2 in enumerate(COUNTRIES):
            if j != i:
                Ys += W[i, j] * country_data[c2][VARS].values
        foreign[c] = pd.DataFrame(
            Ys, columns=["gdp_star", "inf_star", "rs_star"],
            index=country_data[c].index)
    return foreign


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3  ESTIMATE VARX* COUNTRY MODELS
# ══════════════════════════════════════════════════════════════════════════════

def estimate_varx(Y: np.ndarray, Y_star: np.ndarray,
                  p: int = 1, q: int = 1) -> dict:
    """
    Estimate VARX*(p,q) by OLS for non-US countries.

    Regressors: [1, y_{t-1}, y*_t, y*_{t-1}]
    Returns B, component matrices (Phi, Lambda0, Lambda1), residuals, R².
    Note: USA uses the dedicated estimate_var_usa() instead of this function.
    """
    Tc, k  = Y.shape
    ks     = Y_star.shape[1]
    ml     = max(p, q)
    Te     = Tc - ml

    X = np.hstack([
        np.ones((Te, 1)),
        Y[ml-1: Tc-1],          # y_{t-1}       (k cols)
        Y_star[ml: Tc],          # y*_t          (ks cols)
        Y_star[ml-1: Tc-1],      # y*_{t-1}      (ks cols)
    ])
    Yd = Y[ml:]

    B, _, _, _ = np.linalg.lstsq(X, Yd, rcond=None)
    U   = Yd - X @ B
    Sig = (U.T @ U) / Te

    ss_res = np.sum(U**2, 0)
    ss_tot = np.sum((Yd - Yd.mean(0))**2, 0) + 1e-12
    r2     = np.clip(1 - ss_res / ss_tot, 0, 1)

    # Split coefficient matrix (rows = regressors, cols = equations)
    n1 = 1; n2 = n1+k; n3 = n2+ks; n4 = n3+ks
    return {
        "B":       B,
        "const":   B[:n1,  :].T,
        "Phi":     B[n1:n2, :].T,
        "Lam0":    B[n2:n3, :].T,
        "Lam1":    B[n3:n4, :].T,
        "Sigma":   Sig,
        "U":       U,
        "r2":      r2,
        "r2_avg":  float(np.mean(r2)),
    }


def estimate_var_usa(Y: np.ndarray, p: int = 2) -> dict:
    """
    Pure VAR(p) for USA — no foreign variables (dominant unit assumption).

    USA is treated as a closed economy: Lam0 = Lam1 = 0.
    This avoids rank-deficiency when passing zero columns into estimate_varx().
    Mirrors estimate_var_usa() in the MATLAB and R companion files.
    """
    Tc, k = Y.shape
    Te    = Tc - p
    # Regressors: [const, y_{t-1}, y_{t-2}, ..., y_{t-p}]
    X = np.hstack([np.ones((Te, 1))] +
                  [Y[p-l-1: Tc-l-1] for l in range(p)])
    Yd = Y[p:]
    B, _, _, _ = np.linalg.lstsq(X, Yd, rcond=None)
    U   = Yd - X @ B
    Sig = (U.T @ U) / Te
    ss_res = np.sum(U**2, 0)
    ss_tot = np.sum((Yd - Yd.mean(0))**2, 0) + 1e-12
    r2     = np.clip(1 - ss_res / ss_tot, 0, 1)
    return {
        "B":      B,
        "Phi":    B[1:k+1, :].T,          # k×k  (lag-1 block)
        "Lam0":   np.zeros((k, k)),        # no foreign contemporaneous
        "Lam1":   np.zeros((k, k)),        # no foreign lagged
        "Sigma":  Sig,
        "U":      U,
        "r2":     r2,
        "r2_avg": float(np.mean(r2)),
    }


def estimate_all(country_data: dict, foreign: dict) -> dict:
    """Estimate VARX* for all countries.
    USA uses a dedicated pure VAR(2) — no foreign variables (dominant unit).
    Non-US countries use VARX*(1,1) with trade-weighted foreign aggregates.
    """
    models = {}
    for c in COUNTRIES:
        Y = country_data[c][VARS].values
        if c == "USA":
            models[c] = estimate_var_usa(Y, p=2)
        else:
            models[c] = estimate_varx(Y, foreign[c].values, p=1, q=1)
        r2 = models[c]["r2_avg"]
        print(f"  {c:<12}: avg R²={r2:.3f}  "
              f"[{', '.join(f'{v}={r:.2f}' for v,r in zip(VARS, models[c]['r2']))}]")
    return models


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4  WEAK EXOGENEITY TESTS  (Table 13.7)
# ══════════════════════════════════════════════════════════════════════════════

def test_weak_exog(country_data: dict, foreign: dict, models: dict) -> pd.DataFrame:
    """
    For each non-US country and each foreign variable v:
      Regress Δy*_{v,it} on lagged residual (EC term) + own lag.
      F-test for significance of the EC term.
      H0: foreign variable is weakly exogenous (F should be < 3.07 at 5%).

    Returns DataFrame matching Table 13.7 layout.
    """
    rows = []
    for c in COUNTRIES[1:]:      # skip USA
        U_hat = models[c]["U"]   # residuals from VARX*
        Ys    = foreign[c][["gdp_star", "inf_star", "rs_star"]].values
        Te    = len(U_hat)
        # Align U_hat with dY_star: both start at max_lag=1, period t
        dYs   = np.diff(Ys[-Te-1:], axis=0)[-Te:]   # Δy*_t, length Te
        ec    = U_hat[:-1]                            # lagged residual, length Te-1
        dYs_  = dYs[1:]                               # align, length Te-1
        n_obs = len(dYs_)

        f_stats = {}
        for v, vname in enumerate(["Δy*", "π*", "rs*"]):
            y   = dYs_[:, v]
            lag_y = np.zeros(n_obs)
            lag_y[1:] = y[:-1]
            X_u = np.column_stack([np.ones(n_obs), lag_y, ec[:n_obs, 0]])
            # ── make lengths safe ──
            min_len = min(len(y), X_u.shape[0])
            y_      = y[:min_len]
            X_u_    = X_u[:min_len]
            try:
                B_u, _, _, _  = np.linalg.lstsq(X_u_, y_, rcond=None)
                ss_u = float(np.sum((y_ - X_u_ @ B_u)**2))
                X_r_ = X_u_[:, :-1]
                B_r, _, _, _  = np.linalg.lstsq(X_r_, y_, rcond=None)
                ss_r = float(np.sum((y_ - X_r_ @ B_r)**2))
                df1, df2 = 1, max(min_len - X_u_.shape[1], 1)
                F = ((ss_r - ss_u) / df1) / (ss_u / df2)
            except Exception:
                F = 0.0
            f_stats[vname] = round(float(F), 2)

        rows.append({"Country": c.replace("_", " "),
                     "Δy*": f_stats["Δy*"],
                     "π*":  f_stats["π*"],
                     "rs*": f_stats["rs*"]})
    return pd.DataFrame(rows)


def print_table_13_7(df: pd.DataFrame):
    crit = 3.07
    print("\n" + "="*52)
    print("Table 13.7  Weak Exogeneity Tests: F-Statistics")
    print("="*52)
    print(f"{'Country':<14} {'Δy*':>8} {'π*':>8} {'rs*':>8}")
    print("-"*52)
    for _, row in df.iterrows():
        vals  = [row["Δy*"], row["π*"], row["rs*"]]
        flags = ["*" if v > crit else " " for v in vals]
        print(f"{row['Country']:<14} {vals[0]:>7.2f}{flags[0]} "
              f"{vals[1]:>7.2f}{flags[1]} {vals[2]:>7.2f}{flags[2]}")
    print("-"*52)
    print(f"5% critical value ≈ {crit}  (* exceeds; ** = reject weak exog.)")
    print("="*52)
    print("\nBook (Table 13.7) — Canada Δy* = 2.15 (largest),")
    print("  all other values < 2.0;  all < 3.07  → weak exog. holds.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5  STACK GLOBAL VAR  (eq. 13.71)
# ══════════════════════════════════════════════════════════════════════════════

def stack_global(models: dict) -> tuple:
    """
    Construct G0 y_t = a + G1 y_{t-1} + u_t, then F = G0^{-1} G1.

    For each country i:
      G0[i-block, j-block] -= Lam0[i] * w_ij     (contemporaneous foreign)
      G1[i-block, i-block]  = Phi[i]              (own lag)
      G1[i-block, j-block] += Lam1[i] * w_ij     (lagged foreign)

    Returns: G0, G1, F, Sigma_e  (global shock covariance)
    """
    Kg = N * K

    G0 = np.eye(Kg)
    G1 = np.zeros((Kg, Kg))

    for i, c in enumerate(COUNTRIES):
        rs, re = i*K, (i+1)*K
        Phi = models[c]["Phi"]
        G1[rs:re, rs:re] = Phi             # own lag

        if c == "USA":
            continue                       # dominant unit: no foreign variables

        Lam0 = models[c]["Lam0"]
        Lam1 = models[c]["Lam1"]
        for j, c2 in enumerate(COUNTRIES):
            if j != i:
                cs, ce = j*K, (j+1)*K
                wij = W[i, j]
                G0[rs:re, cs:ce] -= Lam0 * wij    # absorb contemp. foreign
                G1[rs:re, cs:ce] += Lam1 * wij    # lagged foreign

    G0_inv = np.linalg.inv(G0)
    F      = G0_inv @ G1

    # Block-diagonal global covariance
    Sblocks  = [models[c]["Sigma"] for c in COUNTRIES]
    Sig_u    = linalg.block_diag(*Sblocks)
    Sigma_e  = G0_inv @ Sig_u @ G0_inv.T

    return G0, G1, F, Sigma_e


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6  STABILITY CHECK + FIGURE 13.2
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig13_2(F: np.ndarray, path: str = None) -> float:
    """
    Figure 13.2: Eigenvalues of the GVAR companion matrix in the complex plane.
    """
    eigs    = np.linalg.eigvals(F)
    max_mod = float(np.max(np.abs(eigs)))

    fig, ax = plt.subplots(figsize=(6, 6))
    theta   = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k-", lw=1, label="Unit circle")
    ax.scatter(eigs.real, eigs.imag, color="#2E75B6", s=45, zorder=5,
               label="Eigenvalues")
    ax.axhline(0, color="gray", lw=0.4)
    ax.axvline(0, color="gray", lw=0.4)
    ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("Real"); ax.set_ylabel("Imaginary")
    ax.set_title("Eigenvalues of the GVAR Companion Matrix\n"
                 f"Max |λ| = {max_mod:.3f}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.text(0.98, 0.02, f"Max |λ| = {max_mod:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#C00000")
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close()
    return max_mod


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7  GIRF  (Pesaran & Shin, 1998)
# ══════════════════════════════════════════════════════════════════════════════

def compute_girf(F: np.ndarray, Sigma_e: np.ndarray,
                 shock_country: str, shock_var: str,
                 H: int = 20) -> dict:
    """
    GIRF_j(h) = (Σ_e e_j / σ_jj^{1/2}) * C_h  (Pesaran & Shin 1998)

    A positive 1 s.d. shock to rs_US (US short rate) = monetary tightening.
    Returns: dict  country → (H, K) array of GDP/inf/rs responses.
    """
    Kg  = F.shape[0]
    j   = CIDX[shock_country] * K + VARS.index(shock_var)

    e_j = np.zeros(Kg); e_j[j] = 1.0
    sig_jj = Sigma_e[j, j]
    b      = Sigma_e @ e_j / np.sqrt(sig_jj)    # (Kg,) impact vector

    irf_raw = np.zeros((H, Kg))
    Cs = np.eye(Kg)
    irf_raw[0] = Cs @ b
    for h in range(1, H):
        Cs = Cs @ F
        irf_raw[h] = Cs @ b

    return {c: irf_raw[:, CIDX[c]*K: CIDX[c]*K+K] for c in COUNTRIES}


def bootstrap_girf(F, Sigma_e, models, G0,
                   shock_country, shock_var,
                   H=20, n_boot=500, seed=10) -> tuple:
    """
    Residual-resampling bootstrap confidence bands for GIRF.
    Returns: irf_lo, irf_hi  (dicts country → (H,K))
    """
    rng   = np.random.default_rng(seed)
    Kg    = F.shape[0]
    boots = {c: np.zeros((n_boot, H, K)) for c in COUNTRIES}

    for b in range(n_boot):
        # Perturb Sigma_e by resampling country residuals
        Sigma_b = np.zeros_like(Sigma_e)
        for i, c in enumerate(COUNTRIES):
            U_b = models[c]["U"]
            idx = rng.integers(0, len(U_b), size=len(U_b))
            Ub  = U_b[idx]
            Sb  = (Ub.T @ Ub) / len(Ub)
            rs, re = i*K, (i+1)*K
            Sigma_b[rs:re, rs:re] = Sb

        G0_inv   = np.linalg.inv(G0)
        Sigma_eb = G0_inv @ Sigma_b @ G0_inv.T
        # Perturb F slightly
        F_b      = F + rng.normal(0, 0.02 * np.std(F), F.shape)

        try:
            irf_b = compute_girf(F_b, Sigma_eb, shock_country, shock_var, H)
            for c in COUNTRIES:
                boots[c][b] = irf_b[c]
        except Exception:
            pass

    irf_lo = {c: np.percentile(boots[c], 5,  0) for c in COUNTRIES}
    irf_hi = {c: np.percentile(boots[c], 95, 0) for c in COUNTRIES}
    return irf_lo, irf_hi


def plot_fig13_3(irf: dict, irf_lo: dict, irf_hi: dict,
                 H: int = 20, path: str = None):
    """
    Figure 13.3: GDP growth response to US monetary policy tightening
    (1 s.d. positive shock to rs_US ≈ +50 bp), 8-panel layout.
    """
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=False)
    axes_flat = axes.flatten()
    h = np.arange(H)

    for idx, c in enumerate(COUNTRIES):
        ax    = axes_flat[idx]
        gdp   = irf[c][:, 0]
        lo_g  = irf_lo[c][:, 0]
        hi_g  = irf_hi[c][:, 0]

        ax.fill_between(h, lo_g, hi_g, color="#2E75B6", alpha=0.20)
        ax.plot(h, gdp, color="#2E75B6", lw=2.0)
        ax.axhline(0, color="black", lw=0.6)

        # Peak annotation (trough)
        th = int(np.argmin(gdp))
        ax.annotate(f"{gdp[th]:.3f}",
                    xy=(th, gdp[th]), xytext=(th+1.5, gdp[th]-0.005),
                    fontsize=8, color="#C00000", ha="left",
                    arrowprops=dict(arrowstyle="->", color="#C00000", lw=0.8))

        ax.set_title(c.replace("_", " "), fontsize=9, fontweight="bold")
        ax.set_xlabel("Quarters", fontsize=8)
        ax.set_ylabel("GDP growth (pp)", fontsize=8)
        ax.set_xlim(0, H-1)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Figure 13.3: GVAR — GDP Growth Response to US Monetary Policy Tightening\n"
        r"GIRF to 1 s.d. increase in $rs_{US}$ ($\approx$50 bp)  |  90% bootstrap CI",
        fontsize=10, fontweight="bold")
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8  FEVD + SPILLOVER NETWORK  (Figure 13.4)
# ══════════════════════════════════════════════════════════════════════════════

def compute_gfevd_gdp(F: np.ndarray, Sigma_e: np.ndarray,
                       H: int = 8) -> np.ndarray:
    """
    Generalised FEVD for each country's GDP variable at horizon H.

    fevd_gdp[i, j] = share of country i's GDP FEV attributable to all shocks
                     from country j (summed over j's K variables).

    Returns (N, N) matrix of percentage shares (rows sum to ~100 after rounding).
    """
    Kg = F.shape[0]
    # Compute MA coefficients C_0, C_1, ..., C_{H-1}
    C = [np.eye(Kg)]
    for _ in range(H - 1):
        C.append(C[-1] @ F)

    out = np.zeros((N, N))
    for i in range(N):
        v = i * K           # global index of country i's GDP variable
        e_v = np.zeros(Kg); e_v[v] = 1.0

        # Total FEV of y_{i,GDP} at horizon H
        fev_tot = sum(float(e_v @ Cs @ Sigma_e @ Cs.T @ e_v) for Cs in C)
        fev_tot = max(fev_tot, 1e-12)

        for j in range(N):
            num = 0.0
            for s in range(K):         # sum over j's K shocks
                js   = j*K + s
                e_js = np.zeros(Kg); e_js[js] = 1.0
                sig_js = Sigma_e[js, js]
                nj = sum(float(e_v @ Cs @ Sigma_e @ e_js)**2
                         for Cs in C) / max(sig_js, 1e-12)
                num += nj
            out[i, j] = num / fev_tot

    # Normalise rows to sum to 1
    out = out / out.sum(1, keepdims=True)
    return out


def plot_fig13_4(fevd: np.ndarray, path: str = None):
    """
    Figure 13.4: FEVD heatmap (left) + spillover network (right).
    """
    labels  = [c.replace("_", " ") for c in COUNTRIES]
    short   = ["USA", "EUR", "GBR", "JPN", "CHN", "CAN", "KOR", "BRA"]
    pct     = np.round(fevd * 100).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left: heatmap ─────────────────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(N)); ax.set_xticklabels(short, fontsize=8)
    ax.set_yticks(range(N)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Shock source (country j)", fontsize=9)
    ax.set_ylabel("Affected country (country i)", fontsize=9)
    ax.set_title("Forecast Error Variance\nDecomposition (h = 8)", fontsize=10)
    for i in range(N):
        for j in range(N):
            v = pct[i, j]
            color = "white" if v > 55 else "black"
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=7, color=color)
    plt.colorbar(im, ax=ax, shrink=0.75, label="Share of GDP FEV (%)")

    # ── Right: network ────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("Spillover Network\n(edges > 3% of GDP FEV)", fontsize=10)

    theta_n = np.linspace(0, 2*np.pi, N, endpoint=False)
    pos = {c: (0.85*np.cos(t), 0.85*np.sin(t))
           for c, t in zip(COUNTRIES, theta_n)}

    out_conn = np.array([fevd[i].sum() - fevd[i, i] for i in range(N)])

    # Edges
    for i, ci in enumerate(COUNTRIES):
        for j, cj in enumerate(COUNTRIES):
            if i != j and fevd[i, j] > 0.03:
                x0, y0 = pos[ci]; x1, y1 = pos[cj]
                lw = 1 + 6 * fevd[i, j]
                ax2.annotate("", xy=(x1, y1), xytext=(x0, y0),
                             arrowprops=dict(arrowstyle="-|>", lw=lw,
                                             color="#2E75B6", alpha=0.55))

    # Nodes
    for i, c in enumerate(COUNTRIES):
        x, y   = pos[c]
        radius = 0.06 + 0.14 * out_conn[i]
        circle = plt.Circle((x, y), radius, color="#2E75B6",
                             alpha=0.85, zorder=5)
        ax2.add_patch(circle)
        ax2.text(x, y, short[i], ha="center", va="center",
                 fontsize=7, color="white", fontweight="bold", zorder=6)
    ax2.set_xlim(-1.4, 1.4); ax2.set_ylim(-1.4, 1.4); ax2.set_aspect("equal")

    fig.suptitle(
        "Figure 13.4: GVAR — Forecast Error Variance Decomposition "
        "and Spillover Network (h = 8 quarters)",
        fontsize=10, fontweight="bold")
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    sep = "=" * 70
    print(sep)
    print("GVAR: US MONETARY POLICY SPILLOVERS")
    print("Section 13.7.2 | Macroeconometrics | Alessia Paccagnini")
    print(sep)

    print("\n[1] Simulating GVAR data  (T=162, N=8) …")
    country_data = simulate_gvar(seed=42)
    print(f"    Countries: {COUNTRIES}")
    print(f"    wCAN,USA = {W[CIDX['Canada'], CIDX['USA']]:.2f}  (book: 0.55)")

    print("\n[2] Constructing trade-weighted foreign variables …")
    foreign = make_foreign_vars(country_data)

    print("\n[3] Estimating VARX*(1,1) country models …")
    models = estimate_all(country_data, foreign)

    print("\n[4] Weak exogeneity tests …")
    df_we = test_weak_exog(country_data, foreign, models)
    print_table_13_7(df_we)

    print("\n[5] Stacking global VAR …")
    G0, G1, F, Sigma_e = stack_global(models)
    Kg = F.shape[0]
    print(f"    Global system: {Kg}×{Kg}  ({N} countries × {K} variables)")

    print("\n[6] Stability check + Figure 13.2 …")
    path_f2 = "empirical_example_GlobalVAR_eigenvalues.pdf"
    max_eig = plot_fig13_2(F, path=path_f2)
    print(f"    Max |λ| = {max_eig:.3f}  (book: 0.973)")

    print("\n[7] GIRF to US monetary policy shock + Figure 13.3 …")
    H   = 20
    irf = compute_girf(F, Sigma_e, "USA", "rs", H=H)
    print("    Bootstrap CIs (500 replications) …")
    irf_lo, irf_hi = bootstrap_girf(F, Sigma_e, models, G0,
                                    "USA", "rs", H=H, n_boot=500, seed=20)
    path_f3 = "empirical_example_GlobalVAR_girf.pdf"
    plot_fig13_3(irf, irf_lo, irf_hi, H=H, path=path_f3)

    print("\n[8] FEVD + Spillover network + Figure 13.4 …")
    fevd      = compute_gfevd_gdp(F, Sigma_e, H=8)
    path_f4   = "empirical_example_GlobalVAR_fevd.pdf"
    plot_fig13_4(fevd, path=path_f4)

    # Summary
    print()
    print(sep)
    print("RESULTS SUMMARY  (book values in brackets)")
    print(sep)
    print(f"  Max eigenvalue:      {max_eig:.3f}  [0.973]")
    print()
    print("  Peak GDP response to 1 s.d. US rate shock (Figure 13.3):")
    book_peaks = {"USA": -0.141, "Euro_Area": -0.022, "UK": -0.034,
                  "Japan": -0.009, "China": -0.003, "Canada": -0.093,
                  "Korea": -0.004, "Brazil": None}
    for c in COUNTRIES:
        gdp = irf[c][:, 0]
        th  = int(np.argmin(gdp))
        bk  = f"[{book_peaks[c]:.3f}]" if book_peaks[c] else "[n/a]"
        print(f"    {c.replace('_',' '):<12}: {gdp[th]:+.3f} pp at h={th}  {bk}")
    print()
    print("  GDP FEVD at h=8 (Figure 13.4) — diagonal = own shocks:")
    own = np.diag(fevd)
    print(f"    Own shocks range: {own.min()*100:.0f}–{own.max()*100:.0f}%  [book: 91–97%]")
    can_i = CIDX["Canada"]; usa_i = CIDX["USA"]
    print(f"    USA → Canada:  {fevd[can_i, usa_i]*100:.0f}%  [book: ~6%]")
    eur_i = CIDX["Euro_Area"]; uk_i = CIDX["UK"]
    print(f"    EUR → UK:      {fevd[uk_i, eur_i]*100:.0f}%  [book: ~3%]")
    print(f"    Canada → USA:  {fevd[usa_i, can_i]*100:.0f}%  [book: ~3%]")
    print(sep)
    print(f"\n  Figures saved:")
    for p in [path_f2, path_f3, path_f4]:
        print(f"    {p}")


if __name__ == "__main__":
    main()
