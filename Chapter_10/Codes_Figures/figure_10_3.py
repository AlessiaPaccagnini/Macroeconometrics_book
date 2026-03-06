# =============================================================================
# figure_10_3.py
# Replication code for Figure 10.3
# US Macroeconomic Data with Regime Indicators (1960-2019)
#
# Author: Alessia Paccagnini
# Textbook: Macroeconometrics, Chapter 10
#
# Data file required (same folder):
#   macro_data_1960_2019.csv
#   Columns: date, gdp_growth, inflation, ffr, nber_rec
#
# Install: pip install numpy pandas matplotlib
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# ── 0. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('macro_data_1960_2019.csv')
df['date']       = pd.to_datetime(df['date'])
df['gdp_growth'] = pd.to_numeric(df['gdp_growth'], errors='coerce')
df['inflation']  = pd.to_numeric(df['inflation'],  errors='coerce')

dates = df['date'].values
gdp   = df['gdp_growth'].values   # annualised quarterly GDP growth (%)
infl  = df['inflation'].values    # annualised quarterly inflation (%)
ffr   = df['ffr'].values          # federal funds rate (%)
rec   = df['nber_rec'].values     # NBER recession dummy

# ── 1. Regime indicators ──────────────────────────────────────────────────────
# High-volatility: rolling 8-quarter std of GDP growth > 75th percentile
roll_std = pd.Series(gdp).rolling(8, min_periods=4).std().values
p75      = np.nanpercentile(roll_std, 75)
high_vol = (roll_std > p75).astype(int)
stress   = np.maximum(rec, high_vol)   # NBER OR high-vol

gm_date  = pd.Timestamp('1984-01-01')  # Great Moderation break
ffr_med  = np.median(ffr)

# ── 2. Colours ────────────────────────────────────────────────────────────────
C_REC='#C0392B'; C_HVOL='#2471A3'; C_STR='#7D3C98'; C_GM='#1E8449'
C_GDP='#1A1A2E'; C_INF='#B7600A'; C_FFR='#154360'

# ── 3. Helpers ────────────────────────────────────────────────────────────────
def shade_bands(ax, dates, ind, color, alpha=0.22):
    in_b = False
    for t in range(len(dates)):
        if ind[t] and not in_b:   s = dates[t]; in_b = True
        elif not ind[t] and in_b: ax.axvspan(s, dates[t], color=color, alpha=alpha, lw=0); in_b=False
    if in_b: ax.axvspan(s, dates[-1], color=color, alpha=alpha, lw=0)

def regime_bars(ax, dates, ind, y, h, color, alpha=0.75):
    in_b = False
    for t in range(len(dates)):
        if ind[t] and not in_b:   s = dates[t]; in_b = True
        elif not ind[t] and in_b:
            ax.barh(y, (dates[t]-s)/np.timedelta64(1,'D'), left=s,
                    height=h, color=color, alpha=alpha, align='center'); in_b=False
    if in_b:
        ax.barh(y, (dates[-1]-s)/np.timedelta64(1,'D'), left=s,
                height=h, color=color, alpha=alpha, align='center')

# ── 4. Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 11))
gs  = GridSpec(4, 1, figure=fig, hspace=0.06, height_ratios=[1.8,1.8,1.8,1.1])
ax1=fig.add_subplot(gs[0]); ax2=fig.add_subplot(gs[1],sharex=ax1)
ax3=fig.add_subplot(gs[2],sharex=ax1); ax4=fig.add_subplot(gs[3],sharex=ax1)
for ax in [ax1,ax2,ax3,ax4]:
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Panel 1 — GDP Growth
shade_bands(ax1,dates,rec,C_REC)
ax1.axhline(0,color='black',lw=0.6,ls='--',alpha=0.35)
ax1.plot(dates,gdp,color=C_GDP,lw=1.1)
ax1.axvline(gm_date,color=C_GM,lw=1.6,ls='--',alpha=0.85)
ax1.set_ylabel('GDP Growth (%)',fontsize=9)
ax1.set_title('US Macroeconomic Data with Regime Indicators (1960\u20132019)',
              fontsize=12,fontweight='bold',pad=9)
ax1.tick_params(labelbottom=False,labelsize=8); ax1.set_ylim(-11,11); ax1.grid(True,alpha=0.12)
ax1.legend(handles=[mpatches.Patch(color=C_REC,alpha=0.22,label='NBER Recession')],
           loc='upper right',fontsize=8,frameon=False)

# Panel 2 — Inflation
shade_bands(ax2,dates,rec,C_REC)
ax2.plot(dates,infl,color=C_INF,lw=1.1)
ax2.axvline(gm_date,color=C_GM,lw=1.6,ls='--',alpha=0.85)
ax2.set_ylabel('Inflation (%)',fontsize=9)
ax2.tick_params(labelbottom=False,labelsize=8); ax2.set_ylim(-0.5,10); ax2.grid(True,alpha=0.12)

# Panel 3 — Fed Funds Rate
shade_bands(ax3,dates,rec,C_REC)
ax3.plot(dates,ffr,color=C_FFR,lw=1.1)
ax3.axhline(ffr_med,color='grey',lw=1.0,ls=':',alpha=0.85,label=f'Median ({ffr_med:.1f}%)')
ax3.axvline(gm_date,color=C_GM,lw=1.6,ls='--',alpha=0.85,label='Great Moderation (1984)')
ax3.set_ylabel('Interest Rate (%)',fontsize=9)
ax3.tick_params(labelbottom=False,labelsize=8); ax3.set_ylim(-0.5,20.5)
ax3.legend(loc='upper right',fontsize=8,frameon=False); ax3.grid(True,alpha=0.12)

# Panel 4 — Regime indicators
y_r,y_h,y_s,h = 0.67,0.37,0.07,0.20
regime_bars(ax4,dates,rec,     y_r,h,C_REC)
regime_bars(ax4,dates,high_vol,y_h,h,C_HVOL)
regime_bars(ax4,dates,stress,  y_s,h,C_STR)
ax4.axvline(gm_date,color=C_GM,lw=1.6,ls='--',alpha=0.85)
ax4.set_yticks([y_r,y_h,y_s]); ax4.set_yticklabels(['Recession','High Vol','Stress'],fontsize=8)
ax4.set_ylim(-0.08,0.88); ax4.set_xlabel('Date',fontsize=9); ax4.tick_params(labelsize=8)
ax4.grid(True,alpha=0.10,axis='x')
ax4.xaxis.set_major_locator(mdates.YearLocator(10))
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax4.legend(handles=[
    mpatches.Patch(color=C_REC, alpha=0.75,label='NBER Recession'),
    mpatches.Patch(color=C_HVOL,alpha=0.75,label='High Uncertainty'),
    mpatches.Patch(color=C_STR, alpha=0.75,label='Combined Stress'),
    plt.Line2D([0],[0],color=C_GM,lw=1.6,ls='--',label='Great Moderation'),
], loc='lower right',fontsize=7.5,frameon=True,framealpha=0.9,ncol=2)

# ── 5. Save ───────────────────────────────────────────────────────────────────
plt.savefig('Figure_10_3.pdf', bbox_inches='tight')
plt.savefig('Figure_10_3.png', dpi=300, bbox_inches='tight')
print('Saved Figure_10_3.pdf / .png')
plt.show()
