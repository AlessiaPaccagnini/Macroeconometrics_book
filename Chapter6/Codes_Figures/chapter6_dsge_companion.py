"""
================================================================================
CHAPTER 6: DSGE MODELS — Python Companion Code (Google Colab)
Macroeconometrics Textbook
Author: Alessia Paccagnini
================================================================================

This notebook contains three self-contained sections:

  SECTION 1: Table 6.1  — Business Cycle Statistics
             (HP filter on FRED-QD data, McCracken & Ng 2016)

  SECTION 2: Figure 6.1 — RBC Impulse Responses
             (Blanchard-Kahn solution via QZ decomposition)

  SECTION 3: Figure 6.2 — Prior vs Posterior Distributions
             (Smets-Wouters 2007, Table 1A)

HOW TO USE IN GOOGLE COLAB:
  1. Upload this file to Colab (or copy-paste into a notebook)
  2. For Section 1: upload the FRED-QD Excel file when prompted
  3. Sections 2 and 3 run without any data files

Calibration note:
  delta = 0.025 throughout (10% annual depreciation rate, consistent
  with the BK example, exercises, and calibration discussion in Chapter 6)
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import ordqz
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
#
#  SECTION 1: TABLE 6.1 — Business Cycle Statistics
#
# ============================================================================
# Data: FRED-QD dataset (McCracken & Ng, 2016)
# File: 2026-01-QD.xlsx (upload when prompted)
# Sample: 1960:Q1 to 2019:Q4 (pre-COVID, 240 observations)
#
# Series used:
#   GDP              — GDPC1    (Real GDP)
#   Consumption      — PCNDx + PCESVx  (Nondurables + Services)
#   Investment       — GPDIC1   (Real Gross Private Domestic Investment)
#   Hours worked     — HOANBS   (Nonfarm Business Sector: Hours)
#   Labor productiv. — OPHNFB   (Nonfarm Business: Real Output Per Hour)
#   Real wages       — COMPRNFB (Nonfarm Business: Real Compensation Per Hour)
# ============================================================================


# --- 1.1 HP Filter Implementation ---

def hp_filter(y, lamb=1600):
    """
    Apply Hodrick-Prescott filter to extract trend and cycle.

    Parameters
    ----------
    y : array-like
        Time series (in levels or logs).
    lamb : float
        Smoothing parameter (1600 for quarterly data).

    Returns
    -------
    trend : np.array
    cycle : np.array — deviations from trend
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    I_mat = np.eye(n)
    D2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        D2[i, i:i + 3] = [1, -2, 1]

    A = I_mat + lamb * D2.T @ D2
    trend = np.linalg.solve(A, y)
    cycle = y - trend

    return trend, cycle


# --- 1.2 Load FRED-QD Data ---

def load_fred_qd(file_path):
    """
    Load macroeconomic series from the McCracken-Ng FRED-QD Excel file.

    Consumption is measured as nondurable goods + services (excluding
    durables), which is the standard definition in business cycle research.

    Parameters
    ----------
    file_path : str
        Path to the FRED-QD Excel file.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with six macroeconomic series in levels.
    """
    raw = pd.read_excel(file_path, sheet_name='in', header=None)
    colnames = raw.iloc[0].values

    # Data starts at row index 5 (row 6 in Excel)
    dates = pd.date_range(start='1960-01-01',
                          periods=len(raw.iloc[5:]), freq='QE')

    def get_series(code):
        idx = np.where(colnames == code)[0][0]
        return pd.to_numeric(raw.iloc[5:, idx].values, errors='coerce')

    # Consumption = nondurables + services
    cons_nds = get_series('PCNDx') + get_series('PCESVx')

    df = pd.DataFrame({
        'GDP':                get_series('GDPC1'),
        'Consumption':        cons_nds,
        'Investment':         get_series('GPDIC1'),
        'Hours worked':       get_series('HOANBS'),
        'Labor productivity': get_series('OPHNFB'),
        'Real wages':         get_series('COMPRNFB'),
    }, index=dates)

    return df


# --- 1.3 Compute Business Cycle Statistics ---

def compute_business_cycle_stats(df, lamb=1600):
    """
    Compute business cycle statistics for all variables in df.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with macroeconomic time series (in levels).
    lamb : float
        HP filter smoothing parameter.

    Returns
    -------
    stats_df : pd.DataFrame
        Table with std dev, relative volatility, and correlation with GDP.
    cycles_df : pd.DataFrame
        Cyclical components (% deviations from trend).
    """
    df_log = df.apply(np.log)

    cycles = pd.DataFrame(index=df.index)
    for col in df_log.columns:
        _, cycle = hp_filter(df_log[col].values, lamb=lamb)
        cycles[col] = cycle * 100  # percentage deviations

    gdp_cycle = cycles['GDP']
    results = []
    for col in cycles.columns:
        std_dev = cycles[col].std()
        rel_vol = std_dev / gdp_cycle.std()
        corr_gdp = cycles[col].corr(gdp_cycle)
        results.append({
            'Variable': col,
            'Std. Dev. (%)': round(std_dev, 2),
            'Relative to GDP': round(rel_vol, 2),
            'Correlation with GDP': round(corr_gdp, 2),
        })

    stats_df = pd.DataFrame(results)
    return stats_df, cycles


# --- 1.4 Visualisation Functions ---

def plot_hp_decomposition(df, cycles, variable='GDP', lamb=1600):
    """Plot original series with HP trend and cyclical component."""
    y_log = np.log(df[variable])
    trend, cycle = hp_filter(y_log.values, lamb=lamb)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(df.index, y_log, label=f'Log {variable}', alpha=0.7)
    axes[0].plot(df.index, trend, label='HP Trend', linewidth=2, color='red')
    axes[0].set_ylabel('Log Level')
    axes[0].set_title(f'{variable}: Original Series and HP Trend '
                      f'(\u03bb = {lamb})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, cycle * 100, label='Cyclical Component',
                 color='blue')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_ylabel('Percent Deviation from Trend')
    axes[1].set_xlabel('Date')
    axes[1].set_title(f'{variable}: Cyclical Component (% deviation)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_all_cycles(cycles_df):
    """Plot all cyclical components together."""
    fig, ax = plt.subplots(figsize=(14, 8))

    for col in cycles_df.columns:
        ax.plot(cycles_df.index, cycles_df[col], label=col, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Percent Deviation from Trend')
    ax.set_xlabel('Date')
    ax.set_title('Business Cycle Components: All Variables')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_correlation_with_gdp(cycles_df):
    """Plot scatter plots of each variable's cycle against GDP cycle."""
    variables = [col for col in cycles_df.columns if col != 'GDP']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    gdp_cycle = cycles_df['GDP']

    for i, var in enumerate(variables):
        ax = axes[i]
        ax.scatter(gdp_cycle, cycles_df[var], alpha=0.5, s=20)

        z = np.polyfit(gdp_cycle, cycles_df[var], 1)
        p = np.poly1d(z)
        x_line = np.linspace(gdp_cycle.min(), gdp_cycle.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2)

        corr = cycles_df[var].corr(gdp_cycle)
        ax.set_xlabel('GDP Cycle (%)')
        ax.set_ylabel(f'{var} Cycle (%)')
        ax.set_title(f'{var} vs GDP\nCorrelation = {corr:.2f}')
        ax.grid(True, alpha=0.3)

    for j in range(len(variables), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig


# --- 1.5 Run Section 1 ---

def run_table_6_1():
    """
    Run the full Table 6.1 analysis.
    Prompts for file upload in Colab; uses file_path otherwise.
    """
    print("=" * 80)
    print("SECTION 1: TABLE 6.1 — BUSINESS CYCLE STATISTICS")
    print("=" * 80)
    print()

    # --- Data loading ---
    try:
        from google.colab import files
        print("Please upload the file: 2026-01-QD.xlsx")
        uploaded = files.upload()
        file_path = list(uploaded.keys())[0]
    except ImportError:
        # Not in Colab — set path manually
        file_path = '2026-01-QD.xlsx'
        print(f"Not in Colab. Using local file: {file_path}")

    print("\n1. Loading FRED-QD data...")
    df = load_fred_qd(file_path)

    # Sample: 1960Q1 to 2019Q4 (pre-COVID)
    df = df.loc['1960-01-01':'2019-12-31'].dropna()
    first_q = f"{df.index[0].year}:Q{df.index[0].quarter}"
    last_q = f"{df.index[-1].year}:Q{df.index[-1].quarter}"
    print(f"   Sample: {first_q} to {last_q}  ({len(df)} observations)")
    print()

    # Compute statistics
    print("2. Computing business cycle statistics (HP filter, "
          "\u03bb = 1600)...")
    stats_df, cycles_df = compute_business_cycle_stats(df, lamb=1600)
    print()

    # Display table
    print("3. TABLE 6.1: Business Cycle Statistics for the United States")
    print("=" * 80)
    print(stats_df.to_string(index=False))
    print("=" * 80)
    print()

    # GDP autocorrelation
    gdp_ac1 = cycles_df['GDP'].autocorr(lag=1)
    gdp_ac4 = cycles_df['GDP'].autocorr(lag=4)
    print(f"   GDP cycle autocorrelation: "
          f"\u03c1(1) = {gdp_ac1:.2f}, \u03c1(4) = {gdp_ac4:.2f}")
    print()

    # Key stylized facts
    print("Key Stylized Facts:")
    print("  - Consumption is LESS volatile than GDP (consumption smoothing)")
    print("  - Investment is ~4.5x more volatile than GDP (forward-looking)")
    print("  - Hours worked are PROCYCLICAL (positive correlation with GDP)")
    print("  - Labor productivity is PROCYCLICAL but less volatile than GDP")
    print("  - Real wages are WEAKLY PROCYCLICAL (low positive correlation)")
    print()

    # Figures
    print("4. Creating visualisations...")

    fig1 = plot_hp_decomposition(df, cycles_df, variable='GDP')
    plt.savefig('hp_decomposition_gdp.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig2 = plot_all_cycles(cycles_df)
    plt.savefig('all_cycles.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig3 = plot_correlation_with_gdp(cycles_df)
    plt.savefig('correlations_with_gdp.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("=" * 80)
    print("Section 1 complete.")
    print("=" * 80)
    return stats_df, cycles_df


# ============================================================================
#
#  SECTION 2: FIGURE 6.1 — RBC Impulse Responses
#
# ============================================================================
# Calibration:
#   beta = 0.99,  alpha = 0.33,  delta = 0.025,  rho_A = 0.95
#   sigma = 1 (log utility),  eta = 1 (unit Frisch elasticity)
#
# Model solved using first-order perturbation (Blanchard-Kahn).
# ============================================================================


class RBCModel:
    """Real Business Cycle Model — log-linearised solution."""

    def __init__(self):
        # Preferences
        self.beta  = 0.99       # Discount factor
        self.sigma = 1.0        # Inverse IES (log utility)
        self.eta   = 1.0        # Inverse Frisch elasticity

        # Technology
        self.alpha = 0.33       # Capital share
        self.delta = 0.025      # Depreciation rate (10% annual)

        # Shock process
        self.rho_a   = 0.95     # AR(1) persistence
        self.sigma_a = 0.01     # Std dev of innovation

        # Steady state (computed below)
        self.ss = {}

    def compute_steady_state(self):
        """Compute deterministic steady state."""
        alpha = self.alpha
        beta  = self.beta
        delta = self.delta
        sigma = self.sigma
        eta   = self.eta

        # From Euler equation: r_ss = 1/beta - 1 + delta
        r_ss = 1.0 / beta - 1.0 + delta

        # Great ratios
        KY = alpha / r_ss
        IY = delta * KY
        CY = 1.0 - IY

        # From production function and labour FOC
        K_per_N = KY ** (1.0 / (1.0 - alpha))
        YN = K_per_N ** alpha
        w_ss = (1.0 - alpha) * YN
        CN = CY * YN

        # Intratemporal condition: w C^{-sigma} = N^eta
        N_ss = (w_ss * CN ** (-sigma)) ** (1.0 / (eta + sigma))
        K_ss = K_per_N * N_ss
        Y_ss = K_ss ** alpha * N_ss ** (1.0 - alpha)
        C_ss = CY * Y_ss
        I_ss = IY * Y_ss

        self.ss = {
            'Y': Y_ss, 'C': C_ss, 'I': I_ss, 'K': K_ss, 'N': N_ss,
            'w': w_ss, 'r': r_ss, 'A': 1.0,
            'CY': CY, 'IY': IY, 'KY': KY,
        }
        return self.ss

    def solve_blanchard_kahn(self):
        """
        Solve the log-linearised RBC model via Blanchard-Kahn.

        State vector:  s_t = [k_hat_t, a_hat_t]   (2 predetermined)
        Control:       c_hat_t                     (1 forward-looking)

        System written as:  A_lhs E_t[z_{t+1}] = A_rhs z_t
        where z = [k, a, c].

        Returns policy matrices P, F such that:
          s_{t+1} = P s_t   (no-shock transition)
          c_hat_t = F s_t
        """
        ss = self.ss
        alpha = self.alpha
        delta = self.delta
        beta  = self.beta
        sigma = self.sigma
        eta   = self.eta
        rho_a = self.rho_a

        CY = ss['CY']
        IY = ss['IY']
        r_ss = ss['r']

        # --- Log-linearised output elasticities ---
        # After substituting intratemporal condition:
        #   n = (a + alpha*k - sigma*c) / (eta + alpha)
        # into production:  y = alpha*k + (1-alpha)*n + a

        phi_yk = alpha + alpha * (1 - alpha) / (eta + alpha)
        phi_ya = 1.0 + (1 - alpha) / (eta + alpha)
        phi_yc = -(1 - alpha) * sigma / (eta + alpha)

        # Capital accumulation: k' = (1-delta)k + delta*i
        # with i = (y - CY*c)/IY
        Akk = (1 - delta) + delta * phi_yk / IY
        Aka = delta * phi_ya / IY
        Akc = delta * (phi_yc - CY) / IY

        # Euler equation coefficients
        rb  = r_ss * beta       # = alpha
        pk1 = phi_yk - 1.0

        A_lhs = np.array([
            [1.0,    0.0,    0.0],
            [0.0,    1.0,    0.0],
            [0.0,    0.0,    sigma - rb * phi_yc],
        ])

        A_rhs = np.array([
            [Akk,
             Aka,
             Akc],
            [0.0,
             rho_a,
             0.0],
            [rb * pk1 * Akk,
             rb * pk1 * Aka + rb * phi_ya * rho_a,
             sigma + rb * pk1 * Akc],
        ])

        # QZ decomposition (sort: stable eigenvalues first)
        AA, BB, alpha_qz, beta_qz, Q, Z = ordqz(
            A_rhs, A_lhs, sort='iuc', output='real')

        eigs = np.abs(alpha_qz / beta_qz)

        n_states = 2    # k, a
        n_controls = 1  # c
        n_unstable = np.sum(eigs > 1.0)

        print(f"   BK eigenvalues: {np.sort(eigs)}")
        print(f"   Stable: {np.sum(eigs < 1)}, "
              f"Unstable: {n_unstable}  ", end="")

        assert n_unstable == n_controls, \
            (f"BK condition fails: {n_unstable} unstable roots, "
             f"need {n_controls}")
        print("\u2713")

        # Partition Z
        Z11 = Z[:n_states, :n_states]
        Z21 = Z[n_states:, :n_states]

        # Policy function: c = F @ s = Z21 @ Z11^{-1} @ s
        F = Z21 @ np.linalg.inv(Z11)

        # State transition: s' = P s
        P = np.array([
            [Akk + Akc * F[0, 0],  Aka + Akc * F[0, 1]],
            [0.0,                   rho_a],
        ])

        self.P = P
        self.F = F
        self.phi_yk = phi_yk
        self.phi_ya = phi_ya
        self.phi_yc = phi_yc

        return P, F

    def compute_irfs(self, periods=40, shock_size=1.0):
        """
        Compute IRFs to a technology shock of size shock_size (%).

        Returns
        -------
        irf : dict of arrays, each (periods,) — % deviations from SS.
        """
        P = self.P
        F = self.F
        alpha = self.alpha
        sigma = self.sigma
        eta   = self.eta
        IY = self.ss['IY']
        CY = self.ss['CY']

        phi_yk = self.phi_yk
        phi_ya = self.phi_ya
        phi_yc = self.phi_yc

        # State path: s = [k_hat, a_hat]
        s = np.zeros((periods + 1, 2))
        s[0, 1] = shock_size   # technology shock at t=0

        for t in range(periods):
            s[t + 1] = P @ s[t]

        k_hat = s[:periods, 0]
        a_hat = s[:periods, 1]
        c_hat = (F @ s[:periods].T).flatten()
        n_hat = (a_hat + alpha * k_hat - sigma * c_hat) / (eta + alpha)
        y_hat = phi_yk * k_hat + phi_ya * a_hat + phi_yc * c_hat
        i_hat = (y_hat - CY * c_hat) / IY

        return {
            'Y': y_hat,
            'C': c_hat,
            'I': i_hat,
            'K': s[1:periods + 1, 0],
            'N': n_hat,
            'A': a_hat,
        }


def plot_irf_panel(irf, periods=40):
    """Create the 2x3 IRF panel matching Figure 6.1."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Impulse Responses to a One Percent Technology Shock',
                 fontsize=16, fontweight='bold', y=0.995)

    quarters = np.arange(periods)

    plots = [
        ('Y', 'Output ($Y$)',       axes[0, 0], '(1)'),
        ('C', 'Consumption ($C$)',   axes[0, 1], '(2)'),
        ('I', 'Investment ($I$)',    axes[0, 2], '(3)'),
        ('N', 'Hours ($N$)',         axes[1, 0], '(4)'),
        ('K', 'Capital ($K$)',       axes[1, 1], '(5)'),
        ('A', 'Technology ($A$)',    axes[1, 2], '(6)'),
    ]

    for var, title, ax, label in plots:
        ax.plot(quarters, irf[var][:periods], linewidth=2.5,
                color='#1f77b4')
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.3)
        ax.set_xlabel('Quarters', fontsize=11)
        ax.set_ylabel('% deviation', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, periods - 1)
        ax.text(0.95, 0.95, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', color='red',
                ha='right', va='top')

    annotations = [
        "(1) Output rises immediately (~1.3%) due to the direct "
        "productivity effect.",
        "(2) Consumption rises but less than output, reflecting "
        "consumption smoothing.",
        "(3) Investment rises substantially on impact as firms "
        "exploit higher productivity.",
        "(4) Hours rise initially via intertemporal substitution "
        "of labour supply.",
        "(5) Capital accumulates gradually (hump-shaped), "
        "generating endogenous persistence.",
        "(6) Technology follows AR(1) with \u03c1 = 0.95; "
        "half-life \u2248 14 quarters."
    ]
    fig.text(0.5, 0.02, "\n\n".join(annotations),
             ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
             multialignment='left')

    plt.tight_layout(rect=[0, 0.12, 1, 0.97])
    plt.savefig('figure_6_1_rbc_irf.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.show()
    return fig


def run_figure_6_1():
    """Run the full Figure 6.1 analysis."""
    print("=" * 80)
    print("SECTION 2: FIGURE 6.1 — RBC IMPULSE RESPONSES")
    print("Calibration: \u03b2=0.99, \u03b1=0.33, \u03b4=0.025, "
          "\u03c1_A=0.95, \u03c3=1, \u03b7=1")
    print("=" * 80)

    model = RBCModel()

    print("\n[1/3] Computing steady state...")
    ss = model.compute_steady_state()
    print(f"   Y = {ss['Y']:.4f},  C = {ss['C']:.4f},  "
          f"I = {ss['I']:.4f}")
    print(f"   K = {ss['K']:.4f},  N = {ss['N']:.4f}")
    print(f"   C/Y = {ss['CY']:.3f},  I/Y = {ss['IY']:.3f},  "
          f"K/Y = {ss['KY']:.2f}")

    print("\n[2/3] Solving model (Blanchard-Kahn)...")
    P, F = model.solve_blanchard_kahn()

    print("\n[3/3] Computing IRFs (1% technology shock, 40 quarters)...")
    irf = model.compute_irfs(periods=40, shock_size=1.0)

    print("\n   Impact effects (t=0):")
    for var in ['Y', 'C', 'I', 'N', 'K', 'A']:
        print(f"     {var:3s}: {irf[var][0]:+.4f}%")

    print("\n   Peak effects:")
    for var in ['Y', 'C', 'I', 'K']:
        peak_val = np.max(irf[var])
        peak_q = np.argmax(irf[var])
        print(f"     {var:3s}: {peak_val:.4f}% at quarter {peak_q}")

    print("\nGenerating Figure 6.1...")
    fig = plot_irf_panel(irf, periods=40)

    print("\n" + "=" * 80)
    print("Section 2 complete. Figure saved: figure_6_1_rbc_irf.png")
    print("=" * 80)
    return irf


# ============================================================================
#
#  SECTION 3: FIGURE 6.2 — Prior vs Posterior Distributions
#
# ============================================================================
# Smets-Wouters (2007, AER), Table 1A
#
# Layout: 2x3 panel
#   Row 1: xi_p (price stickiness), xi_w (wage stickiness), h (habit)
#   Row 2: phi (inv. adj. cost), r_pi (Taylor: inflation), rho (smoothing)
#
# Grey = prior, Blue = posterior (normal approx around mode)
# ============================================================================

# --- 3.1 Parameter specifications from Table 6.4 ---

PARAMS = {
    'xi_p': {
        'label':      r'$\xi_p$ (Price stickiness)',
        'prior':      'B',
        'prior_mean': 0.50,
        'prior_std':  0.10,
        'post_mode':  0.65,
        'post_mean':  0.66,
        'post_5':     0.56,
        'post_95':    0.74,
    },
    'xi_w': {
        'label':      r'$\xi_w$ (Wage stickiness)',
        'prior':      'B',
        'prior_mean': 0.50,
        'prior_std':  0.10,
        'post_mode':  0.73,
        'post_mean':  0.70,
        'post_5':     0.60,
        'post_95':    0.81,
    },
    'h': {
        'label':      r'$h$ (Habit formation)',
        'prior':      'B',
        'prior_mean': 0.70,
        'prior_std':  0.10,
        'post_mode':  0.71,
        'post_mean':  0.71,
        'post_5':     0.64,
        'post_95':    0.78,
    },
    'phi': {
        'label':      r'$\varphi$ (Inv. adjustment)',
        'prior':      'N',
        'prior_mean': 4.00,
        'prior_std':  1.50,
        'post_mode':  5.48,
        'post_mean':  5.74,
        'post_5':     3.97,
        'post_95':    7.42,
    },
    'r_pi': {
        'label':      r'$r_\pi$ (Taylor: inflation)',
        'prior':      'N',
        'prior_mean': 1.50,
        'prior_std':  0.25,
        'post_mode':  2.03,
        'post_mean':  2.04,
        'post_5':     1.74,
        'post_95':    2.33,
    },
    'rho': {
        'label':      r'$\rho$ (Interest smoothing)',
        'prior':      'B',
        'prior_mean': 0.75,
        'prior_std':  0.10,
        'post_mode':  0.81,
        'post_mean':  0.81,
        'post_5':     0.77,
        'post_95':    0.85,
    },
}

PANEL_ORDER = ['xi_p', 'xi_w', 'h', 'phi', 'r_pi', 'rho']


# --- 3.2 Distribution Constructors ---

def beta_from_mean_std(mean, std):
    """Construct scipy Beta distribution from mean and std."""
    var = std ** 2
    k = mean * (1 - mean) / var - 1.0
    a = mean * k
    b = (1 - mean) * k
    return stats.beta(a, b)


def build_prior(p):
    """Build a scipy distribution for a parameter's prior."""
    if p['prior'] == 'B':
        return beta_from_mean_std(p['prior_mean'], p['prior_std'])
    elif p['prior'] == 'N':
        return stats.norm(loc=p['prior_mean'], scale=p['prior_std'])
    elif p['prior'] == 'G':
        var = p['prior_std'] ** 2
        shape = p['prior_mean'] ** 2 / var
        scale = var / p['prior_mean']
        return stats.gamma(a=shape, scale=scale)


def build_posterior_normal(p):
    """
    Approximate posterior as Normal around the mode.
    Posterior std inferred from the 5th-95th percentile (90% interval).
    """
    post_std = (p['post_95'] - p['post_5']) / (2 * 1.645)
    return stats.norm(loc=p['post_mode'], scale=post_std)


# --- 3.3 Plot Figure 6.2 ---

def plot_figure_6_2():
    """
    Create Figure 6.2: Prior and Posterior Distributions.

    Grey shaded areas = prior distributions
    Blue shaded areas = posterior distributions
    Dashed vertical lines = posterior modes
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Bayesian Estimation: Prior vs Posterior Distributions',
                 fontsize=15, fontweight='bold', y=0.98)

    for idx, key in enumerate(PANEL_ORDER):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        p = PARAMS[key]

        prior_dist = build_prior(p)
        post_dist  = build_posterior_normal(p)

        if p['prior'] == 'B':
            x = np.linspace(0.001, 0.999, 500)
        else:
            post_std = (p['post_95'] - p['post_5']) / 3.29
            lo = min(p['prior_mean'] - 4 * p['prior_std'],
                     p['post_mode'] - 4 * post_std)
            hi = max(p['prior_mean'] + 4 * p['prior_std'],
                     p['post_mode'] + 4 * post_std)
            x = np.linspace(lo, hi, 500)

        # Prior (grey)
        prior_pdf = prior_dist.pdf(x)
        ax.fill_between(x, prior_pdf, alpha=0.35, color='grey',
                         label='Prior')
        ax.plot(x, prior_pdf, color='grey', linewidth=1.2, alpha=0.7)

        # Posterior (blue)
        post_pdf = post_dist.pdf(x)
        ax.fill_between(x, post_pdf, alpha=0.45, color='#2171b5',
                         label='Posterior')
        ax.plot(x, post_pdf, color='#2171b5', linewidth=1.8)

        # Posterior mode
        ax.axvline(p['post_mode'], color='#08519c', linestyle='--',
                   linewidth=1.5, alpha=0.8)

        ax.set_title(p['label'], fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=9)
        ax.set_yticks([])
        ax.grid(True, alpha=0.2)

        if p['prior'] == 'B':
            ax.set_xlim(0, 1)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', fontsize=11,
               framealpha=0.9, bbox_to_anchor=(0.08, 0.95))

    fig.text(0.5, 0.01,
             'Notes: Grey shaded areas show prior distributions; '
             'blue areas show posterior distributions (approximated as '
             'normal around the mode).\n'
             'Dashed vertical lines indicate posterior modes. '
             'Source: Smets and Wouters (2007, AER), Table 1A.',
             ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig('figure_6_2_prior_posterior.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.show()
    return fig


def run_figure_6_2():
    """Run the full Figure 6.2 analysis."""
    print("=" * 80)
    print("SECTION 3: FIGURE 6.2 — PRIOR vs POSTERIOR DISTRIBUTIONS")
    print("Smets-Wouters (2007, AER)")
    print("=" * 80)
    print()

    # Print summary table
    print("TABLE 6.4 (selected): Prior and Posterior")
    print("=" * 90)
    header = (f"{'Parameter':30s} {'Dist':>5s} {'Prior \u03bc':>8s} "
              f"{'Prior \u03c3':>8s} {'Post mode':>10s} "
              f"{'Post mean':>10s} {'[5%':>6s} {'95%]':>6s}")
    print(header)
    print("-" * 90)

    for key in PANEL_ORDER:
        p = PARAMS[key]
        print(f"{p['label']:30s} {p['prior']:>5s} "
              f"{p['prior_mean']:8.2f} {p['prior_std']:8.2f} "
              f"{p['post_mode']:10.2f} {p['post_mean']:10.2f} "
              f"{p['post_5']:6.2f} {p['post_95']:6.2f}")

    print("-" * 90)
    print()

    # Learning diagnostics
    print("Prior-to-Posterior Learning:")
    print("-" * 90)
    for key in PANEL_ORDER:
        p = PARAMS[key]
        post_std = (p['post_95'] - p['post_5']) / 3.29
        shrinkage = 1.0 - post_std / p['prior_std']
        shift = p['post_mode'] - p['prior_mean']
        print(f"  {p['label']:30s}  shift = {shift:+.2f},  "
              f"variance reduction = {shrinkage * 100:.0f}%")
    print()

    print("Generating Figure 6.2...")
    fig = plot_figure_6_2()

    print("\n" + "=" * 80)
    print("Section 3 complete. Figure saved: figure_6_2_prior_posterior.png")
    print("=" * 80)
    return fig


# ============================================================================
#
#  MAIN EXECUTION
#
# ============================================================================

if __name__ == '__main__':

    print()
    print("*" * 80)
    print("*  CHAPTER 6: DSGE MODELS — PYTHON COMPANION CODE")
    print("*  Macroeconometrics Textbook")
    print("*  Author: Alessia Paccagnini")
    print("*" * 80)
    print()
    print("This script contains three sections that can be run independently.")
    print("In Colab, you can run each section by calling its function:\n")
    print("  run_table_6_1()   — Section 1: Table 6.1 (needs data file)")
    print("  run_figure_6_1()  — Section 2: Figure 6.1 (no data needed)")
    print("  run_figure_6_2()  — Section 3: Figure 6.2 (no data needed)")
    print()
    print("Running all sections now...\n")

    # --- Section 1: Table 6.1 ---
    # Requires FRED-QD data file; skip if not available
    try:
        stats_df, cycles_df = run_table_6_1()
    except Exception as e:
        print(f"\nSection 1 skipped: {e}")
        print("(Upload 2026-01-QD.xlsx and call run_table_6_1() manually)\n")

    # --- Section 2: Figure 6.1 ---
    irf = run_figure_6_1()

    print()

    # --- Section 3: Figure 6.2 ---
    fig = run_figure_6_2()

    print()
    print("*" * 80)
    print("*  ALL SECTIONS COMPLETE")
    print("*" * 80)
