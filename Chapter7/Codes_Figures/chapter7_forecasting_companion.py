"""
================================================================================
CHAPTER 7: FORECASTING — Figures 7.1–7.4
Macroeconometrics Textbook
Author: Alessia Paccagnini
================================================================================

This script generates four publication-ready figures for Chapter 7:

  Figure 7.1: bias_variance_tradeoff.pdf
              The Bias-Variance Tradeoff in Forecasting

  Figure 7.2: bias_variance_examples.pdf
              Bias-Variance Decomposition in Practice (two panels)

  Figure 7.3: forecast_errors_example.pdf
              Visualising Forecast Performance (actual vs forecast + errors)

  Figure 7.4: giacomini_rossi_fluctuation_test.pdf
              Giacomini-Rossi Fluctuation Test

HOW TO USE IN GOOGLE COLAB:
  1. Upload GDPC1.xlsx (from FRED) for Figure 7.3, OR the script
     will fall back to simulated data.
  2. Run the whole file, or call individual functions:
       run_figure_7_1()
       run_figure_7_2()
       run_figure_7_3()   # prompts for GDPC1.xlsx upload in Colab
       run_figure_7_4()
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Publication-quality defaults
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10


# ============================================================================
#
#  FIGURE 7.1 — The Bias-Variance Tradeoff in Forecasting
#  LaTeX: \includegraphics{.../bias_variance_tradeoff.pdf}
#
# ============================================================================

def bias_variance_tradeoff_plot():
    """
    Create a publication-quality bias-variance tradeoff plot for forecasting.

    The plot shows how model complexity affects bias, variance, and total error.
    """
    # Model complexity axis (from simple to complex)
    complexity = np.linspace(0, 10, 100)

    # Bias decreases with complexity (diminishing returns)
    bias_squared = 8 * np.exp(-0.5 * complexity) + 0.5

    # Variance increases with complexity (accelerating)
    variance = 0.1 + 0.15 * complexity**1.5

    # Irreducible error (constant)
    irreducible = np.ones_like(complexity) * 2.0

    # Total error = Bias^2 + Variance + Irreducible
    total_error = bias_squared + variance + irreducible

    # Find optimal complexity (minimum total error)
    optimal_idx = np.argmin(total_error)
    optimal_complexity = complexity[optimal_idx]
    optimal_error = total_error[optimal_idx]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(complexity, bias_squared, 'b-', linewidth=2.5,
            label='Bias²', alpha=0.8)
    ax.plot(complexity, variance, 'r-', linewidth=2.5,
            label='Variance', alpha=0.8)
    ax.plot(complexity, irreducible, 'k--', linewidth=1.5,
            label='Irreducible Error', alpha=0.6)
    ax.plot(complexity, total_error, 'g-', linewidth=3,
            label='Total Error', alpha=0.9)

    # Mark optimal point
    ax.plot(optimal_complexity, optimal_error, 'go', markersize=12,
            markeredgewidth=2, markeredgecolor='darkgreen',
            label='Optimal Complexity', zorder=5)
    ax.axvline(x=optimal_complexity, color='gray', linestyle=':',
               linewidth=1.5, alpha=0.7)

    # Annotations
    ax.annotate('Underfitting\n(High Bias)',
                xy=(1.5, 7), fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='lightblue', alpha=0.7))
    ax.annotate('Optimal\nComplexity',
                xy=(optimal_complexity, optimal_error - 1.5),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='lightgreen', alpha=0.7))
    ax.annotate('Overfitting\n(High Variance)',
                xy=(8, 8), fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='lightcoral', alpha=0.7))

    # Labels and formatting
    ax.set_xlabel('Model Complexity', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontsize=13, fontweight='bold')
    ax.set_title('The Bias-Variance Tradeoff in Forecasting',
                 fontsize=14, fontweight='bold', pad=15)

    ax.set_xticks([0, 2.5, 5, 7.5, 10])
    ax.set_xticklabels(['Random\nWalk', 'AR(1)', 'AR(4)\nBVAR',
                        'Unrestricted\nVAR', 'High-Dim\nModel'])
    ax.set_ylim([0, 12])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper center', ncol=5, framealpha=0.95,
              bbox_to_anchor=(0.5, -0.15))

    plt.tight_layout()
    return fig, ax


def run_figure_7_1():
    """Generate and save Figure 7.1."""
    print("=" * 70)
    print("FIGURE 7.1 — The Bias-Variance Tradeoff in Forecasting")
    print("=" * 70)

    fig, ax = bias_variance_tradeoff_plot()
    fig.savefig('bias_variance_tradeoff.pdf', bbox_inches='tight', dpi=300)
    fig.savefig('bias_variance_tradeoff.png', bbox_inches='tight', dpi=300)
    plt.show()

    print("Saved: bias_variance_tradeoff.pdf / .png\n")
    return fig


# ============================================================================
#
#  FIGURE 7.2 — Bias-Variance Decomposition in Practice
#  LaTeX: \includegraphics{.../bias_variance_examples.pdf}
#
# ============================================================================

def bias_variance_examples_plot():
    """
    Create two-panel figure:

    Left panel:  Distribution of forecast errors for three model classes.
    Right panel: Bias-variance decomposition by model (stacked bars + RMSE).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    np.random.seed(42)

    # --- Left panel: error distributions ---
    # Simple model: low variance, high bias
    simple_errors = np.random.normal(loc=1.5, scale=0.5, size=1000)
    # Complex model: high variance, low bias
    complex_errors = np.random.normal(loc=0.3, scale=1.8, size=1000)
    # Optimal model: balanced
    optimal_errors = np.random.normal(loc=0.5, scale=0.9, size=1000)

    ax1.hist(simple_errors, bins=40, alpha=0.6, color='blue',
             label=(f'Simple Model (RW)\n'
                    f'Bias={np.mean(simple_errors):.2f}, '
                    f'SD={np.std(simple_errors):.2f}'),
             density=True)
    ax1.hist(complex_errors, bins=40, alpha=0.6, color='red',
             label=(f'Complex Model (VAR(8))\n'
                    f'Bias={np.mean(complex_errors):.2f}, '
                    f'SD={np.std(complex_errors):.2f}'),
             density=True)
    ax1.hist(optimal_errors, bins=40, alpha=0.6, color='green',
             label=(f'Optimal Model (AR(4))\n'
                    f'Bias={np.mean(optimal_errors):.2f}, '
                    f'SD={np.std(optimal_errors):.2f}'),
             density=True)

    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2,
                label='Target (zero error)')
    ax1.set_xlabel('Forecast Error', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Forecast Errors',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Right panel: stacked bars + RMSE line ---
    models = ['RW', 'AR(1)', 'AR(2)', 'AR(4)',
              'VAR(2)', 'VAR(4)', 'BVAR', 'VAR(8)']
    rmse_values = [3.2, 2.9, 2.6, 2.3, 2.4, 2.5, 2.35, 2.8]
    bias_values = [2.8, 2.2, 1.5, 0.8, 0.7, 0.5, 0.6, 0.3]
    variance_values = [0.4, 0.7, 1.1, 1.5, 1.7, 2.0, 1.75, 2.5]

    x = np.arange(len(models))
    width = 0.35

    ax2.bar(x, bias_values, width, label='Bias²',
            color='steelblue', alpha=0.8)
    ax2.bar(x, variance_values, width, bottom=bias_values,
            label='Variance', color='coral', alpha=0.8)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, rmse_values, 'go-', linewidth=2.5, markersize=8,
                  label='Total RMSE', markeredgewidth=2,
                  markeredgecolor='darkgreen')

    optimal_idx = np.argmin(rmse_values)
    ax2_twin.plot(x[optimal_idx], rmse_values[optimal_idx], 'g*',
                  markersize=20, markeredgewidth=2,
                  markeredgecolor='darkgreen')

    ax2.set_xlabel('Model Specification', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error Components', fontsize=12, fontweight='bold')
    ax2_twin.set_ylabel('Total RMSE', fontsize=12,
                        fontweight='bold', color='green')
    ax2.set_title('Bias-Variance Decomposition by Model',
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend(loc='upper left', fontsize=10)
    ax2_twin.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2_twin.tick_params(axis='y', labelcolor='green')

    plt.tight_layout()
    return fig, (ax1, ax2)


def run_figure_7_2():
    """Generate and save Figure 7.2."""
    print("=" * 70)
    print("FIGURE 7.2 — Bias-Variance Decomposition in Practice")
    print("=" * 70)

    fig, axes = bias_variance_examples_plot()
    fig.savefig('bias_variance_examples.pdf', bbox_inches='tight', dpi=300)
    fig.savefig('bias_variance_examples.png', bbox_inches='tight', dpi=300)
    plt.show()

    print("Saved: bias_variance_examples.pdf / .png\n")
    return fig


# ============================================================================
#
#  FIGURE 7.3 — Visualising Forecast Performance
#  LaTeX: \includegraphics{.../forecast_errors_example.pdf}
#
#  Panel (a): Actual GDP growth vs one-step-ahead AR(2) forecast
#  Panel (b): Forecast errors with ±1 RMSE bands
#
#  Uses real FRED GDPC1 data (upload GDPC1.xlsx in Colab).
#  Falls back to simulated data if the file is unavailable.
#
# ============================================================================

def load_gdp_growth(file_path=None):
    """
    Load quarterly annualised GDP growth from FRED GDPC1 Excel file.

    Parameters
    ----------
    file_path : str or None
        Path to GDPC1.xlsx. If None, tries Colab upload.

    Returns
    -------
    dates : array of datetime64
    growth : array of float (annualised quarterly growth, %)
    """
    import pandas as pd

    # --- Attempt to load real data ---
    if file_path is None:
        try:
            from google.colab import files
            print("Please upload GDPC1.xlsx (from FRED):")
            uploaded = files.upload()
            file_path = list(uploaded.keys())[0]
        except ImportError:
            file_path = 'GDPC1.xlsx'

    try:
        df = pd.read_excel(file_path, sheet_name='Quarterly', header=0)
        df.columns = ['date', 'gdpc1']
        df['date'] = pd.to_datetime(df['date'])
        df['gdpc1'] = pd.to_numeric(df['gdpc1'], errors='coerce')
        df = df.dropna()
        df['growth'] = 400 * np.log(df['gdpc1'] / df['gdpc1'].shift(1))
        df = df.dropna().reset_index(drop=True)
        print(f"   Loaded GDPC1: {df.date.iloc[0].date()} to "
              f"{df.date.iloc[-1].date()} ({len(df)} obs)")
        return df['date'].values, df['growth'].values

    except Exception as e:
        print(f"   Could not load GDPC1.xlsx: {e}")
        print("   Using simulated GDP growth instead.")
        return None, None


def simulate_gdp_growth():
    """
    Generate simulated quarterly GDP growth (2000Q1–2022Q4)
    including Great Recession and COVID-like shocks.
    """
    import pandas as pd

    np.random.seed(2026)
    dates = pd.date_range('2000-01-01', '2022-12-31', freq='QS')
    T = len(dates)

    phi0, phi1, phi2 = 0.5, 0.32, 0.12
    sigma_e = 1.3

    y = np.zeros(T)
    y[0] = 3.0
    y[1] = phi0 + phi1 * y[0] + np.random.randn() * sigma_e
    for t in range(2, T):
        y[t] = phi0 + phi1*y[t-1] + phi2*y[t-2] + np.random.randn()*sigma_e

    # Great Recession
    gr_idx = np.where(dates >= pd.Timestamp('2008-10-01'))[0][0]
    y[gr_idx]     = -6.0
    y[gr_idx + 1] = -4.5
    y[gr_idx + 2] = -0.5
    y[gr_idx + 3] =  1.5

    # COVID shock
    covid_idx = np.where(dates >= pd.Timestamp('2020-04-01'))[0][0]
    y[covid_idx]     = -31.0
    y[covid_idx + 1] =  33.0
    y[covid_idx + 2] =   4.5

    return dates.values, y


def run_figure_7_3(file_path=None):
    """
    Generate and save Figure 7.3: Visualising Forecast Performance.

    Panel (a): Quarterly GDP growth (solid) and one-step-ahead AR(2)
               forecasts (dashed).
    Panel (b): Forecast errors with ±1 RMSE bands.

    Parameters
    ----------
    file_path : str or None
        Path to GDPC1.xlsx. In Colab, prompts for upload if None.
    """
    import pandas as pd

    print("=" * 70)
    print("FIGURE 7.3 — Visualising Forecast Performance")
    print("=" * 70)

    # --- Load data ---
    dates_full, growth_full = load_gdp_growth(file_path)

    if dates_full is not None:
        # Use real data: 2000Q1 to 2022Q4
        # Estimation starts from 2000Q1; forecasts shown from ~2010
        sel = ((dates_full >= np.datetime64('2000-01-01')) &
               (dates_full <= np.datetime64('2022-12-31')))
        dates = dates_full[sel]
        y = growth_full[sel]
        data_source = "FRED GDPC1"
    else:
        dates, y = simulate_gdp_growth()
        data_source = "Simulated"

    T = len(y)
    print(f"   Sample: {pd.Timestamp(dates[0]).date()} to "
          f"{pd.Timestamp(dates[-1]).date()} ({T} obs)")
    print(f"   Data source: {data_source}")

    # --- Expanding-window AR(2) one-step-ahead forecasts ---
    R = 40   # initial estimation window (~10 years: 2000–2009)
    forecasts = np.full(T, np.nan)

    for t in range(R, T - 1):
        # Estimate AR(2) on {y_0, ..., y_t}
        Y = y[2:t+1]
        X = np.column_stack([np.ones(t - 1), y[1:t], y[0:t-1]])
        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            forecasts[t + 1] = beta[0] + beta[1]*y[t] + beta[2]*y[t-1]
        except np.linalg.LinAlgError:
            forecasts[t + 1] = y[t]

    # Forecast errors
    fmask = ~np.isnan(forecasts)
    errors = np.where(fmask, y - forecasts, np.nan)
    valid_errors = errors[fmask]
    rmse = np.sqrt(np.nanmean(valid_errors**2))
    mae = np.nanmean(np.abs(valid_errors))

    # Pre-COVID RMSE (excluding 2020Q1–Q4)
    covid_start = np.datetime64('2020-01-01')
    covid_end   = np.datetime64('2020-12-31')
    no_covid = fmask & ~((dates >= covid_start) & (dates <= covid_end))
    if np.sum(no_covid) > 0:
        err_nc = y[no_covid] - forecasts[no_covid]
        rmse_nc = np.sqrt(np.mean(err_nc**2))
        mae_nc = np.mean(np.abs(err_nc))
    else:
        rmse_nc, mae_nc = rmse, mae

    print(f"   Forecast evaluation: {np.sum(fmask)} forecasts")
    print(f"   RMSE = {rmse:.2f}  (excl. COVID: {rmse_nc:.2f})")
    print(f"   MAE  = {mae:.2f}  (excl. COVID: {mae_nc:.2f})")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel (a): Actual vs Forecast
    ax = axes[0]
    ax.plot(dates, y, 'b-', lw=1.8, label='Actual', alpha=0.9)
    ax.plot(dates[fmask], forecasts[fmask], 'r--', lw=1.5,
            label='Forecast', alpha=0.8)
    ax.set_ylabel('GDP Growth (%)', fontsize=12)
    ax.set_title('(a) Actual vs. Forecast', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', lw=0.5, alpha=0.3)

    # Panel (b): Forecast Errors with RMSE bands
    ax = axes[1]
    bar_dates = dates[fmask]
    bar_errors = errors[fmask]
    bar_width = 70 if data_source == "FRED GDPC1" else 60
    ax.bar(bar_dates, bar_errors, width=bar_width,
           color='steelblue', alpha=0.7)
    ax.axhline(0, color='black', lw=0.8)

    # Use pre-COVID RMSE for the bands (more informative)
    ax.axhline(rmse_nc, color='red', ls='--', lw=1.5,
               label=f'+RMSE = {rmse_nc:.2f}')
    ax.axhline(-rmse_nc, color='red', ls='--', lw=1.5,
               label=f'\u2212RMSE = \u2212{rmse_nc:.2f}')

    ax.set_ylabel('Forecast Error (%)', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_title('(b) Forecast Errors', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # RMSE / MAE text box (upper-right, away from legend)
    textstr = f'RMSE = {rmse_nc:.2f}\nMAE = {mae_nc:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Caption
    fig.text(0.5, -0.01,
             'Notes: Panel (a) shows quarterly GDP growth (solid line) and '
             'one-step-ahead forecasts from an AR(2) model\n'
             '(dashed line). Panel (b) shows the corresponding forecast '
             'errors. The horizontal dashed lines in panel (b)\n'
             'indicate \u00b11 RMSE bands.',
             ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    fig.savefig('forecast_errors_example.pdf', bbox_inches='tight', dpi=300)
    fig.savefig('forecast_errors_example.png', bbox_inches='tight', dpi=300)
    plt.show()

    print("Saved: forecast_errors_example.pdf / .png\n")
    return fig


# ============================================================================
#
#  FIGURE 7.4 — Giacomini-Rossi Fluctuation Test
#  LaTeX: \includegraphics{.../giacomini_rossi_fluctuation_test.pdf}
#
# ============================================================================

def giacomini_rossi_fluctuation_test(loss_diff, window_size=40, alpha=0.05):
    """
    Implement Giacomini-Rossi fluctuation test for time-varying
    forecast performance.

    Parameters
    ----------
    loss_diff : array-like
        Time series of loss differentials d_t = L(e1_t) - L(e2_t).
    window_size : int
        Size of rolling window for local test (default: 40).
    alpha : float
        Significance level (default: 0.05).

    Returns
    -------
    dict with keys:
        t_stats        : array of rolling t-statistics
        max_stat       : maximum absolute t-statistic (test statistic)
        critical_value : approximate critical value
        p_value        : approximate p-value
        periods        : array of time periods corresponding to t_stats
        local_means    : rolling window means
        local_stds     : rolling window standard deviations
    """
    loss_diff = np.asarray(loss_diff)
    T = len(loss_diff)
    n_windows = T - window_size + 1

    t_stats = np.zeros(n_windows)
    local_means = np.zeros(n_windows)
    local_stds = np.zeros(n_windows)

    for i in range(n_windows):
        window = loss_diff[i:i + window_size]
        local_mean = np.mean(window)
        local_std = np.std(window, ddof=1)
        local_se = local_std / np.sqrt(window_size)

        local_means[i] = local_mean
        local_stds[i] = local_std
        t_stats[i] = local_mean / local_se if local_se > 0 else 0

    max_stat = np.max(np.abs(t_stats))

    # Approximate critical values (supremum of Brownian bridge)
    critical_values = {0.10: 1.73, 0.05: 1.95, 0.01: 2.37}
    critical_value = critical_values.get(alpha, 1.95)

    # Rough p-value approximation (bootstrap recommended in practice)
    p_value = 2 * (1 - stats.norm.cdf(max_stat))

    periods = np.arange(window_size // 2, window_size // 2 + n_windows)

    return {
        't_stats': t_stats,
        'max_stat': max_stat,
        'critical_value': critical_value,
        'p_value': p_value,
        'periods': periods,
        'local_means': local_means,
        'local_stds': local_stds,
    }


def plot_fluctuation_test(results, window_size=40, alpha=0.05,
                          model1_name="Model 1", model2_name="Model 2",
                          title=None, figsize=(12, 6)):
    """
    Create fluctuation plot for the Giacomini-Rossi test.

    Parameters
    ----------
    results : dict
        Output from giacomini_rossi_fluctuation_test().
    window_size : int
        Window size used in the test.
    alpha : float
        Significance level.
    model1_name, model2_name : str
        Names of models being compared.
    title : str
        Custom title (optional).
    figsize : tuple
        Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    t_stats = results['t_stats']
    periods = results['periods']
    critical_value = results['critical_value']

    # Plot the fluctuation statistic
    ax.plot(periods, t_stats, 'b-', linewidth=2,
            label='Rolling t-statistic')

    # Confidence bands
    ax.axhline(y=critical_value, color='r', linestyle='--', linewidth=1.5,
               label=f'{int((1 - alpha) * 100)}% critical value '
                     f'(+{critical_value:.2f})')
    ax.axhline(y=-critical_value, color='r', linestyle='--', linewidth=1.5,
               label=f'{int((1 - alpha) * 100)}% critical value '
                     f'(\u2212{critical_value:.2f})')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)

    # Shade regions where test rejects
    upper_reject = t_stats > critical_value
    lower_reject = t_stats < -critical_value

    if np.any(upper_reject):
        ax.fill_between(periods, critical_value, t_stats,
                        where=upper_reject, alpha=0.2, color='green',
                        label=f'{model2_name} significantly better')

    if np.any(lower_reject):
        ax.fill_between(periods, t_stats, -critical_value,
                        where=lower_reject, alpha=0.2, color='orange',
                        label=f'{model1_name} significantly better')

    # Labels and formatting
    ax.set_xlabel('Time Period (centered on rolling window)', fontsize=12)
    ax.set_ylabel('Rolling t-statistic', fontsize=12)

    if title is None:
        title = (f'Giacomini-Rossi Fluctuation Test\n'
                 f'(window size = {window_size})')
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Test result text box
    max_stat = results['max_stat']
    p_value = results['p_value']
    textstr = (f'Test statistic: {max_stat:.3f}\n'
               f'Approx. p-value: {p_value:.3f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    return fig, ax


def run_figure_7_4():
    """
    Generate and save Figure 7.4.

    Simulates loss differentials where Model 2 (AR(4)) is better early
    and Model 1 (AR(1)) is better late — demonstrating time-varying
    forecast performance.
    """
    print("=" * 70)
    print("FIGURE 7.4 — Giacomini-Rossi Fluctuation Test")
    print("=" * 70)

    # Simulate loss differentials with time-varying performance
    np.random.seed(42)
    T = 200

    # Create scenario: Model 2 better early, Model 1 better late
    t = np.arange(T)
    trend = 0.5 * (t - T / 2) / T
    loss_diff = trend + 0.3 * np.random.randn(T)

    # Run the test
    window_size = 40
    alpha = 0.05
    results = giacomini_rossi_fluctuation_test(
        loss_diff, window_size=window_size, alpha=alpha)

    # Create the plot
    fig, ax = plot_fluctuation_test(
        results,
        window_size=window_size,
        alpha=alpha,
        model1_name="AR(1)",
        model2_name="AR(4)",
        title="Giacomini-Rossi Fluctuation Test: AR(1) vs AR(4)")

    fig.savefig('giacomini_rossi_fluctuation_test.pdf',
                bbox_inches='tight', dpi=300)
    fig.savefig('giacomini_rossi_fluctuation_test.png',
                bbox_inches='tight', dpi=300)
    plt.show()

    # Print results
    print(f"   Maximum absolute statistic: {results['max_stat']:.3f}")
    print(f"   Critical value (5%): {results['critical_value']:.3f}")
    print(f"   Approximate p-value: {results['p_value']:.3f}")

    if results['max_stat'] > results['critical_value']:
        print("   => REJECT null of equal predictive ability")
        print("      Evidence of time-varying forecast performance")
    else:
        print("   => Do not reject null of equal predictive ability")

    print("   Note: Bootstrap is recommended for accurate critical values")
    print("Saved: giacomini_rossi_fluctuation_test.pdf / .png\n")
    return fig


# ============================================================================
#
#  MAIN EXECUTION
#
# ============================================================================

if __name__ == '__main__':

    print()
    print("*" * 70)
    print("*  CHAPTER 7: FORECASTING — Figures 7.1 to 7.4")
    print("*  Macroeconometrics Textbook")
    print("*  Author: Alessia Paccagnini")
    print("*" * 70)
    print()
    print("This script generates four figures. In Colab, call individually:")
    print("  run_figure_7_1()   — Bias-variance tradeoff (no data)")
    print("  run_figure_7_2()   — Bias-variance examples (no data)")
    print("  run_figure_7_3()   — Forecast errors (upload GDPC1.xlsx)")
    print("  run_figure_7_4()   — Giacomini-Rossi test (no data)")
    print()

    fig1 = run_figure_7_1()
    fig2 = run_figure_7_2()
    fig3 = run_figure_7_3()
    fig4 = run_figure_7_4()

    print("*" * 70)
    print("*  ALL FIGURES COMPLETE")
    print("*")
    print("*  LaTeX filenames (drop into chapters/ directory):")
    print("*    Figure 7.1: bias_variance_tradeoff.pdf")
    print("*    Figure 7.2: bias_variance_examples.pdf")
    print("*    Figure 7.3: forecast_errors_example.pdf")
    print("*    Figure 7.4: giacomini_rossi_fluctuation_test.pdf")
    print("*" * 70)
