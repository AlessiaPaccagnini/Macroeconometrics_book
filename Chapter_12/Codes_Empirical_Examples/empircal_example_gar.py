"""
Growth-at-Risk (GaR) with the Chicago Fed NFCI
================================================
Textbook : Macroeconometrics
Author   : Alessia Paccagnini
Chapter  : 12 — Quantile Regression and Growth-at-Risk
Section  : 12.7 — Empirical Application: US Growth-at-Risk with NFCI

Empirical specification  (eq. 12.9):
    Q_τ(Δy_{t+4} | Ω_t) = α(τ) + β_y(τ)·Δy_t + β_π(τ)·π_t + β_f(τ)·NFCI_t

Estimation sample : 1971Q1–2024Q3   (N = 215)
Forecast horizon  : h = 4 quarters ahead
Quantiles         : {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95} + OLS

Files needed (upload when prompted in Colab, or place in working directory):
    GDPC1.xlsx   – Real GDP, quarterly  (sheet: Quarterly)
    GDPDEF.xlsx  – GDP Deflator, qtrly  (sheet: Quarterly)
    NFCI.csv     – NFCI weekly, FRED    (cols: observation_date, NFCI)

Script structure — aligned with Chapter 12 exercises:
  Step 1 — Load data       (Exercise: data collection)
  Step 2 — Transform       (Exercise: stationarity, summary stats)
  Step 3 — Quantile reg.   (Exercise: estimate at multiple τ, Table 12.3)
  Step 4 — Asymmetry       (Exercise: coefficient asymmetry plot, Fig 12.1)
  Step 5 — Coverage        (Exercise: model evaluation, Table 12.4)
  Step 6 — Risk assessment (Section 12.7.5 current conditions)
  Step 7 — Figures         (Fig 12.1–12.4 + extras)

Note: FEDFUNDS is NOT part of the Chapter 12 GaR model (eq. 12.9).
"""

import os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'font.size': 10})

# ─────────────────────────────────────────────────────────────────────────────
# 0.  ENVIRONMENT DETECTION  (Colab / Jupyter / script)
# ─────────────────────────────────────────────────────────────────────────────
def _detect_env():
    try:
        import google.colab; return 'colab'        # noqa: E702
    except ImportError:
        pass
    try:
        get_ipython(); return 'jupyter'            # noqa: F821, E702
    except NameError:
        return 'script'

ENV        = _detect_env()
OUTPUT_DIR = '/content/'                     if ENV == 'colab'   else \
             './'                            if ENV == 'jupyter' else '/home/claude/'
DATA_DIR   = '/content/'                     if ENV == 'colab'   else \
             './'                            if ENV == 'jupyter' else '/mnt/user-data/uploads/'

def _data(f): return os.path.join(DATA_DIR, f)
def _out(f):  return os.path.join(OUTPUT_DIR, f)

print("=" * 70)
print("GROWTH-AT-RISK ANALYSIS WITH NFCI")
print("Chapter 12, Section 12.7  |  Macroeconometrics  |  Alessia Paccagnini")
print("=" * 70)
print(f"Environment : {ENV}  |  data: {DATA_DIR}  |  output: {OUTPUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  COLAB FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED = ['GDPC1.xlsx', 'GDPDEF.xlsx', 'NFCI.csv']

if ENV == 'colab':
    missing = [f for f in REQUIRED if not os.path.exists(_data(f))]
    if missing:
        print(f"\n[Colab] Please upload: {missing}")
        from google.colab import files
        for fname, data in files.upload().items():
            with open(f'/content/{fname}', 'wb') as fh:
                fh.write(data)
        print("Files uploaded ✓")
    else:
        print("[Colab] All required files already present ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOAD DATA  (Exercise step 1)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("Step 1 — Loading data")

gdp = pd.read_excel(_data('GDPC1.xlsx'), sheet_name='Quarterly')
gdp['date'] = pd.to_datetime(gdp['observation_date'])
gdp = gdp.set_index('date')[['GDPC1']]
print(f"  GDPC1   : {len(gdp)} quarterly obs  ({gdp.index[0].date()} → {gdp.index[-1].date()})")

defl = pd.read_excel(_data('GDPDEF.xlsx'), sheet_name='Quarterly')
defl['date'] = pd.to_datetime(defl['observation_date'])
defl = defl.set_index('date')[['GDPDEF']]
print(f"  GDPDEF  : {len(defl)} quarterly obs")

try:
    nfci_w = pd.read_csv(_data('NFCI.csv'))
except FileNotFoundError:
    sys.exit("[ERROR] NFCI.csv not found. Download from https://fred.stlouisfed.org/series/NFCI")

nfci_w['date'] = pd.to_datetime(nfci_w['observation_date'])
nfci_w = nfci_w.set_index('date')[['NFCI']]
nfci_q = nfci_w.resample('QS').mean()
nfci_q.columns = ['nfci']
print(f"  NFCI    : {len(nfci_w)} weekly → {len(nfci_q)} quarterly avg  "
      f"({nfci_w.index[0].date()} → {nfci_w.index[-1].date()})")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  MERGE AND TRANSFORM  (Exercise step 1–2)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("Step 2 — Transformations and sample restriction")

df = gdp.join(defl, how='inner').join(nfci_q, how='left')
df['gdp_growth'] = np.log(df['GDPC1']).diff(4) * 100    # Δy_t  (eq. 12.9)
df['inflation']  = np.log(df['GDPDEF']).diff(4) * 100   # π_t
H = 4
df['gdp_forward'] = df['gdp_growth'].shift(-H)          # target: t+4

# Restrict to textbook sample (Section 12.7.1, Table 12.3 footnote)
df       = df.loc['1971-01-01':'2024-09-30']
df_clean = df.dropna(subset=['gdp_growth', 'inflation', 'nfci', 'gdp_forward'])
print(f"  Sample  : 1971Q1 → 2024Q3   N = {len(df_clean)}")

print("\nDescriptive statistics:")
print(df_clean[['gdp_growth', 'inflation', 'nfci']].describe().round(2).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 4.  QUANTILE REGRESSION  (Exercise step 2, Table 12.3)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("Step 3 — Quantile regression  (eq. 12.9 / Table 12.3)")

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
FORMULA   = 'gdp_forward ~ gdp_growth + inflation + nfci'

results = {}
for q in QUANTILES:
    results[q] = smf.quantreg(FORMULA, df_clean).fit(q=q)
ols_res = smf.ols(FORMULA, df_clean).fit()

VARS = [('Intercept','Intercept'), ('gdp_growth','GDP_t'),
        ('inflation','Inflation_t'), ('nfci','NFCI_t')]
SHOW = [0.05, 0.10, 0.25, 0.50, 0.90]

print(f"\n{'Variable':<12}", end='')
for q in SHOW: print(f"  τ={q:.2f} ", end='')
print("     OLS")
print("─" * 72)

for var, label in VARS:
    print(f"{label:<12}", end='')
    for q in SHOW:
        r = results[q]
        t = abs(r.params[var] / r.bse[var])
        s = '***' if t > 2.58 else '**' if t > 1.96 else '*' if t > 1.64 else '   '
        print(f"  {r.params[var]:>5.2f}{s}", end='')
    t = abs(ols_res.params[var] / ols_res.bse[var])
    s = '***' if t > 2.58 else '**' if t > 1.96 else '*' if t > 1.64 else '   '
    print(f"  {ols_res.params[var]:>5.2f}{s}")
    print(f"{'':12}", end='')
    for q in SHOW:
        print(f"  ({results[q].bse[var]:>4.2f})  ", end='')
    print(f"  ({ols_res.bse[var]:>4.2f})")

print("─" * 72)
print("* p<0.10  ** p<0.05  *** p<0.01  |  SEs: asymptotic Huber sandwich")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  NFCI ASYMMETRY  (Section 12.7.2, Exercise step 3)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("Step 4 — NFCI asymmetry  (Section 12.7.2)")

b05 = results[0.05].params['nfci']
b50 = results[0.50].params['nfci']
b90 = results[0.90].params['nfci']
ratio = abs(b05) / abs(b50)

print("\nNFCI coefficient across quantiles:")
for q in QUANTILES:
    coef = results[q].params['nfci']
    se   = results[q].bse['nfci']
    t    = coef / se
    sig  = '***' if abs(t) > 2.58 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.64 else ''
    print(f"  τ = {q:.2f}: {coef:>7.3f}  (SE {se:.3f})  {sig}")

print(f"\n  β_f(0.05) = {b05:.3f}")
print(f"  β_f(0.50) = {b50:.3f}")
print(f"  β_f(0.90) = {b90:.3f}")
print(f"  Asymmetry : {ratio:.2f}×")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  FITTED QUANTILES
# ─────────────────────────────────────────────────────────────────────────────
for q in QUANTILES:
    df_clean = df_clean.copy()
    df_clean[f'gar_{int(q*100):02d}'] = results[q].predict(df_clean)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  COVERAGE EVALUATION  (Table 12.4, Exercise step 5)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("Step 5 — Coverage evaluation  (Table 12.4)")
print(f"\n{'τ':<8} {'Nominal%':>10} {'Empirical%':>12} {'|Diff|':>8}   OK?")
print("─" * 48)
for q in QUANTILES:
    emp  = (df_clean['gdp_forward'] < df_clean[f'gar_{int(q*100):02d}']).mean() * 100
    diff = abs(emp - q * 100)
    ok   = '✓' if diff < 2.0 else '!'
    print(f"{q:<8.2f} {q*100:>10.1f} {emp:>12.1f} {diff:>8.1f}   {ok}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  CURRENT RISK ASSESSMENT  (Section 12.7.5)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("Step 6 — Current risk assessment  (Section 12.7.5)")

last       = df_clean.iloc[-1]
q_idx      = df_clean.index[-1]
gar5_cur   = last['gar_05']
med_cur    = last['gar_50']
nfci_cur   = last['nfci']

# Quarter label helper
def _qlabel(ts):
    return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"

# P(GDP<0): interpolate CDF over the 7 estimated quantile values
# Apply Chernozhukov rearrangement first (Section 12.3.2) to avoid crossing
qs_arr   = np.array(QUANTILES)
v_arr    = np.array([last[f'gar_{int(q*100):02d}'] for q in QUANTILES])
sort_idx = np.argsort(v_arr)
v_sorted = v_arr[sort_idx]
q_sorted = qs_arr[sort_idx]
if v_sorted[0] < 0 < v_sorted[-1]:
    prob_neg = float(np.interp(0.0, v_sorted, q_sorted))
elif v_sorted[-1] <= 0:
    prob_neg = 1.0
else:
    prob_neg = 0.0

print(f"\n  Last obs  : {_qlabel(q_idx)}  (NFCI = {nfci_cur:.2f})")
print(f"  GaR (5%)  : {gar5_cur:.2f}%")
print(f"  Median    : {med_cur:.2f}%")
print(f"  P(GDP<0)  : {prob_neg*100:.1f}%")

X_stress      = pd.DataFrame({'gdp_growth': [last['gdp_growth']],
                               'inflation':  [last['inflation']],
                               'nfci':       [2.0]})
gar5_stress   = results[0.05].predict(X_stress).values[0]
print(f"\n  Stress (NFCI=2.0, as in 2008): GaR(5%) = {gar5_stress:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("Step 7 — Generating figures")

RECESSIONS = [('1973-11-01','1975-03-01'), ('1980-01-01','1980-07-01'),
              ('1981-07-01','1982-11-01'), ('1990-07-01','1991-03-01'),
              ('2001-03-01','2001-11-01'), ('2007-12-01','2009-06-01'),
              ('2020-02-01','2020-04-01')]

def _rec(ax):
    for s, e in RECESSIONS:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.15, color='gray', zorder=0)

with PdfPages(_out('gar_nfci_figures.pdf')) as pdf:

    # Figure 12.1 — coefficient asymmetry
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, (var, title) in enumerate([('gdp_growth','Current GDP Growth'),
                                       ('inflation', 'Inflation'),
                                       ('nfci',      'NFCI (Financial Conditions)')]):
        ax    = axes.flatten()[i]
        coefs = [results[q].params[var] for q in QUANTILES]
        ses   = [results[q].bse[var]    for q in QUANTILES]
        cols  = ['#C00000' if c < 0 else '#2E75B6' for c in coefs]
        ax.bar(range(len(QUANTILES)), coefs, color=cols, alpha=0.8, zorder=2)
        ax.errorbar(range(len(QUANTILES)), coefs, yerr=1.96*np.array(ses),
                    fmt='none', color='black', capsize=4, zorder=3)
        ax.axhline(0,                  color='black', ls='-',  lw=0.8)
        ax.axhline(ols_res.params[var], color='green', ls='--', lw=1.5, label='OLS')
        ax.set_xticks(range(len(QUANTILES)))
        ax.set_xticklabels([f'{int(q*100)}%' for q in QUANTILES])
        ax.set_xlabel('Quantile (τ)'); ax.set_ylabel('Coefficient')
        ax.set_title(title)
        if i == 0: ax.legend(fontsize=9)

    axes[1,1].axis('off')
    txt = (f"KEY FINDING: NFCI Asymmetry\n\n"
           f"β_f(τ):\n"
           f"  τ=0.05: {b05:>7.3f}  (book: −2.23)\n"
           f"  τ=0.50: {b50:>7.3f}  (book: −0.83)\n"
           f"  τ=0.90: {b90:>7.3f}  (book:  0.14)\n\n"
           f"  {ratio:.2f}× larger at 5th than median\n"
           f"  (book: ~2.7×)\n\n"
           f"Financial conditions predict\n"
           f"the SHAPE of the distribution,\n"
           f"not just its mean.\n"
           f"Adrian et al. (2019)")
    axes[1,1].text(0.05, 0.97, txt, transform=axes[1,1].transAxes,
                   fontsize=10, va='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.suptitle('Figure 12.1: Quantile Regression Coefficients', fontsize=12, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.savefig(_out('fig1_gar_coefficients.pdf'), bbox_inches='tight')
    plt.close(); print("  Figure 12.1 — coefficient asymmetry ✓")

    # Figure 12.2 — fan chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(df_clean.index, df_clean['gar_05'], df_clean['gar_95'],
                    alpha=0.12, color='#2E75B6', label='5–95%')
    ax.fill_between(df_clean.index, df_clean['gar_10'], df_clean['gar_90'],
                    alpha=0.22, color='#2E75B6', label='10–90%')
    ax.fill_between(df_clean.index, df_clean['gar_25'], df_clean['gar_75'],
                    alpha=0.38, color='#2E75B6', label='25–75%')
    ax.plot(df_clean.index, df_clean['gar_50'],      '#2E75B6', lw=2,   label='Median')
    ax.plot(df_clean.index, df_clean['gdp_forward'], 'k-',      lw=1, alpha=0.7, label='Actual')
    _rec(ax); ax.axhline(0, color='red', ls='--', lw=0.8, alpha=0.7)
    ax.set(xlabel='Date', ylabel='GDP Growth (%, YoY)',
           title=f'Figure 12.2: Growth-at-Risk Fan Chart\n{H}-Quarter-Ahead  (1971Q1–2024Q3)',
           ylim=(-10, 12))
    ax.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.savefig(_out('fig2_gar_fanchart.pdf'), bbox_inches='tight')
    plt.close(); print("  Figure 12.2 — fan chart ✓")

    # Figure 12.3 — three panels
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax = axes[0]
    ax.plot(df_clean.index, df_clean['nfci'], '#2E75B6', lw=1.5)
    ax.fill_between(df_clean.index, 0, df_clean['nfci'],
                    where=df_clean['nfci'] > 0, alpha=0.35, color='#C00000', label='Tight')
    ax.fill_between(df_clean.index, 0, df_clean['nfci'],
                    where=df_clean['nfci'] < 0, alpha=0.25, color='#70AD47', label='Loose')
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.set(ylabel='NFCI', title='Panel A: Financial Conditions (NFCI > 0 = Tight)')
    ax.legend(loc='upper right', fontsize=9); _rec(ax)

    ax = axes[1]
    ax.plot(df_clean.index, df_clean['inflation'], '#C00000', lw=1.5)
    ax.axhline(2, color='#70AD47', ls='--', lw=1, label='2% target')
    ax.set(ylabel='Inflation (%, YoY)', title='Panel B: Inflation (GDP Deflator)')
    ax.legend(loc='upper right', fontsize=9); _rec(ax)

    ax = axes[2]
    ax.plot(df_clean.index, df_clean['gar_05'],      '#C00000', lw=2,   label='GaR (5th pct)')
    ax.plot(df_clean.index, df_clean['gar_50'],      '#2E75B6', lw=1.5, alpha=0.8, label='Median')
    ax.plot(df_clean.index, df_clean['gdp_forward'], 'k-',      lw=0.8, alpha=0.5, label='Actual')
    ax.fill_between(df_clean.index, df_clean['gar_05'], 0,
                    where=df_clean['gar_05'] < 0, alpha=0.25, color='#C00000')
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set(ylabel='GDP Growth (%)', xlabel='Date',
           title='Panel C: Growth-at-Risk (5th Pct) vs Median')
    ax.legend(loc='lower left', fontsize=9); _rec(ax)
    plt.suptitle('Figure 12.3: Financial Conditions and Growth-at-Risk',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.savefig(_out('fig3_gar_panels.pdf'), bbox_inches='tight')
    plt.close(); print("  Figure 12.3 — three panels ✓")

    # Figure 12.4 — predictive CDF
    # FIX: interpolate over already-estimated quantiles (no re-running 99 regressions)
    fine_q   = np.linspace(0.01, 0.99, 200)
    kv_raw   = np.array([last[f'gar_{int(q*100):02d}'] for q in QUANTILES])
    sort_idx = np.argsort(kv_raw)
    kv       = kv_raw[sort_idx]   # Chernozhukov rearrangement (Section 12.3.2)
    kq       = np.array(QUANTILES)[sort_idx]
    preds    = np.interp(fine_q, kq, kv)
    gar5_pt  = float(np.interp(0.05, fine_q, preds))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(preds, fine_q, '#2E75B6', lw=2.5)
    ax.fill_betweenx(fine_q, preds.min()-1, preds, alpha=0.15, color='#2E75B6')
    ax.axvline(0,    color='#C00000', ls='--', lw=1.5, label='Zero growth')
    ax.axhline(0.05, color='orange',  ls=':',  lw=1.5, label='5th pct (GaR)')
    ax.axhline(0.50, color='#70AD47', ls=':',  lw=1.5, label='Median')
    ax.scatter([gar5_pt], [0.05], color='#C00000', s=80, zorder=5)
    ax.annotate(f'GaR(5%) = {gar5_pt:.2f}%', xy=(gar5_pt, 0.05),
                xytext=(gar5_pt-2.5, 0.18), fontsize=11,
                arrowprops=dict(arrowstyle='->', color='#C00000'))
    ax.text(0.02, 0.98,
            f"GaR (5%)  : {gar5_pt:.2f}%\nP(GDP<0) : {prob_neg*100:.1f}%\nNFCI      : {nfci_cur:.2f}",
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set(xlabel='GDP Growth (%, YoY)', ylabel='Cumulative Probability',
           title=f'Figure 12.4: Predictive CDF  ({H}Q Ahead)  |  {_qlabel(q_idx)}  |  NFCI = {nfci_cur:.2f}',
           xlim=(preds.min()-1, preds.max()+1), ylim=(0, 1))
    ax.legend(loc='lower right')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.savefig(_out('fig4_gar_cdf.pdf'), bbox_inches='tight')
    plt.close(); print("  Figure 12.4 — predictive CDF ✓")

    # Extra: NFCI vs GaR scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(df_clean['nfci'], df_clean['gar_05'],
                    c=df_clean.index.year, cmap='viridis', alpha=0.6, s=30)
    z  = np.polyfit(df_clean['nfci'], df_clean['gar_05'], 1)
    xl = np.linspace(df_clean['nfci'].min(), df_clean['nfci'].max(), 100)
    ax.plot(xl, np.poly1d(z)(xl), '#C00000', lw=2, label=f'Slope = {z[0]:.2f}')
    ax.axhline(0, color='gray', ls='--', lw=1); ax.axvline(0, color='gray', ls='--', lw=1)
    ax.set(xlabel='NFCI', ylabel='GaR 5th Percentile (%)',
           title='NFCI vs GaR: Tighter Conditions → Lower Downside Threshold')
    ax.legend(); plt.colorbar(sc, ax=ax, label='Year')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.savefig(_out('fig5_gar_scatter.pdf'), bbox_inches='tight')
    plt.close(); print("  Figure 5 — NFCI vs GaR scatter ✓")

    # Extra: early warning
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_clean.index, df_clean['gar_05'], '#C00000', lw=2,   label='GaR 5th pct')
    ax.plot(df_clean.index, df_clean['gar_10'], '#ED7D31', lw=1.5, label='GaR 10th pct')
    ax.fill_between(df_clean.index, df_clean['gar_05'], df_clean['gar_10'],
                    alpha=0.25, color='#ED7D31')
    ax.axhline(0, color='black', ls='-', lw=0.5)
    ax.axhline(-2, color='gray', ls=':', lw=1, label='−2% reference')
    _rec(ax)
    ax.set(xlabel='Date', ylabel='GDP Growth (%)',
           title='Growth-at-Risk as Early Warning  (Gray = NBER recessions)')
    ax.legend(loc='lower left')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.savefig(_out('fig6_gar_warning.pdf'), bbox_inches='tight')
    plt.close(); print("  Figure 6 — early warning ✓")

print(f"\n✓ All figures → {_out('gar_nfci_figures.pdf')}")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"""
Growth-at-Risk with NFCI  |  Chapter 12, Section 12.7
======================================================
Sample        : 1971Q1–2024Q3   N = {len(df_clean)}
Horizon       : h = {H} quarters

Key finding — NFCI asymmetry (Table 12.3):
  β_f(0.05) = {b05:.3f}
  β_f(0.50) = {b50:.3f}
  β_f(0.90) = {b90:.3f}
  Asymmetry  = {ratio:.2f}×

Current assessment (Section 12.7.5 / {_qlabel(q_idx)}):
  NFCI = {nfci_cur:.2f}  |  GaR(5%) = {gar5_cur:.2f}%  |  Median = {med_cur:.2f}%
  P(GDP<0) = {prob_neg*100:.1f}%

Output: {OUTPUT_DIR}
  gar_nfci_figures.pdf  +  fig1–fig6 individual PDFs
""")
