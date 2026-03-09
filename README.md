#Macroeconometrics — Companion Code Repository
**Alessia Paccagnini**  
*Macroeconometrics*

---

## About this repository

This repository contains the companion code for the textbook *Macroeconometrics* (De Gruyter). It is organised into two types of material:

- **Figure replication codes** — scripts that reproduce every figure appearing in each chapter, using the simulated or calibrated datasets described in the text.
- **Empirical example codes** — scripts that implement the applied examples at the end of each chapter, where the methods are taken to real macroeconomic data.

All codes are provided in three languages: **Python**, **R**, and **MATLAB**. The only exception is Chapter 6 (DSGE Models), whose empirical examples use **Dynare** (a dedicated MATLAB/Octave toolbox for DSGE estimation), as is standard practice in the field.

**Data** required to run the empirical examples are included in the repository alongside the code.

> **This repository is actively maintained.** Further codes, datasets, and supplementary materials will be continuously added as the book goes to press.

---

## Repository structure

```
Macroeconometrics_book/
│
├── Chapter_02_Univariate/
│   ├── figures/          # Figure replication
│   └── empirical/        # Empirical examples
│       ├── example1_arima.{py,R,m}           # ARIMA: Box-Jenkins methodology
│       ├── example2_cointegration.{py,R,m}   # Spurious regression & ECM
│       └── example3_garch.{py,R,m}           # GARCH volatility dynamics
│
├── Chapter_03_VAR/
│   ├── figures/
│   └── empirical/        # VAR estimation, diagnostics, IRFs, VECM
│
├── Chapter_04_LocalProjections/
│   ├── figures/
│   └── empirical/        # Monetary policy shocks (LP vs smooth LP)
│                         # State-dependent fiscal multipliers
│
├── Chapter_05_ShockIdentification/
│   ├── figures/
│   └── empirical/        # Cholesky VAR & Proxy SVAR (US monetary policy)
│
├── Chapter_06_BayesianEstimation/
│   ├── figures/
│   └── empirical/        # BVAR: US monetary policy application
│
├── Chapter_07_DSGE/
│   ├── figures/
│   └── empirical/        # ⚠️  Dynare only (see note below)
│
├── Chapter_08_Forecasting/
│   ├── figures/
│   └── empirical/        # RW, VAR, BVAR forecast comparison (US GDP/inflation)
│
├── Chapter_09_HighDimensionality/
│   ├── figures/
│   └── empirical/        # FAVAR: US monetary policy with large information set
│
├── Chapter_10_MachineLearning/
│   ├── figures/
│   └── empirical/        # ML vs traditional forecasting (US GDP/inflation)
│
├── Chapter_11_TimeVaryingNonlinear/
│   ├── figures/
│   └── empirical/
│
├── Chapter_12_MixedFrequency/
│   ├── figures/
│   └── empirical/
│
├── Chapter_13_QuantileGaR/
│   ├── figures/
│   └── empirical/
│
├── Chapter_14_MultiCountry/
│   ├── figures/
│   └── empirical/
│       ├── empirical_example_PanelVAR.{py,R,m}   # Panel VAR: oil price shocks
│       └── empirical_example_GlobalVAR.{py,R,m}  # GVAR: US monetary policy spillovers
│
└── data/                 # All datasets used across chapters
```

---

## Empirical examples by chapter

| Chapter | Topic | Empirical example | Languages |
|---------|-------|-------------------|-----------|
| 2 | Univariate Time Series | ARIMA (Box-Jenkins), Cointegration & ECM, GARCH volatility | Python · R · MATLAB |
| 3 | VAR Models | VAR estimation & diagnostics, VECM, reduced-form IRFs | Python · R · MATLAB |
| 4 | Local Projections | Monetary policy shocks (LP vs smooth LP), fiscal multipliers | Python · R · MATLAB |
| 5 | Shock Identification | Cholesky VAR & Proxy SVAR for US monetary policy | Python · R · MATLAB |
| 6 | Bayesian Estimation | BVAR for US monetary policy | Python · R · MATLAB |
| 7 | DSGE Models | New Keynesian model estimation | **Dynare only** |
| 8 | Forecasting | RW vs VAR vs BVAR — US GDP, inflation, interest rates | Python · R · MATLAB |
| 9 | High Dimensionality | FAVAR — US monetary policy with 100+ variables | Python · R · MATLAB |
| 10 | Machine Learning | ML vs traditional methods — US GDP/inflation forecasting | Python · R · MATLAB |
| 11 | Time-Varying & Nonlinear | TVP-VAR, MS-VAR, STVAR | Python · R · MATLAB |
| 12 | Mixed-Frequency Data | MIDAS, MF-VAR | Python · R · MATLAB |
| 13 | Quantile Regression & GaR | Growth-at-Risk, quantile VAR | Python · R · MATLAB |
| 14 | Multi-Country Methods | Panel VAR (oil shocks), GVAR (US monetary policy spillovers) | Python · R · MATLAB |

---

## Software requirements

### Python
```
numpy, pandas, matplotlib, scipy, statsmodels, scikit-learn
```
Install all at once:
```bash
pip install numpy pandas matplotlib scipy statsmodels scikit-learn
```

### R
```
vars, MTS, rugarch, quantreg, forecast, ggplot2, tsDyn
```
Install all at once:
```r
install.packages(c("vars","MTS","rugarch","quantreg","forecast","ggplot2","tsDyn"))
```

### MATLAB
Requires the **Econometrics Toolbox** and **Statistics and Machine Learning Toolbox**.  
Chapter 14 multi-country codes run on base MATLAB with no additional toolboxes.

### Dynare (Chapter 7 only)
Dynare 6.x, freely available at [www.dynare.org](https://www.dynare.org).  
Compatible with MATLAB R2019b+ and Octave 7+.

---

## File naming conventions

Each script is self-contained and follows a consistent naming pattern:

| Pattern | Content |
|---------|---------|
| `figure_chXX_*.{py,R,m}` | Replicates a specific figure from Chapter XX |
| `empirical_example_*.{py,R,m}` | Implements a chapter empirical example |
| `ex*_data.csv` | Dataset used by the corresponding example |

Every script includes a header block with the chapter reference, textbook equation number, and a brief description of the method implemented.

---

## A note on Chapter 7 (DSGE)

DSGE model estimation requires Dynare, which provides a purpose-built preprocessor and solver that cannot be replicated straightforwardly in general-purpose Python or R scripts. The Chapter 7 empirical example files (`.mod` format) are Dynare-native and are documented separately in the chapter's software appendix. For installation and usage guidance see [www.dynare.org/documentation](https://www.dynare.org/documentation).

---

## Citation

If you use these codes in your research or teaching, please cite:

```
Paccagnini, A. (forthcoming). Macroeconometrics. De Gruyter.
```

---

## Contact and updates

Maintained by **Alessia Paccagnini** — https://sites.google.com/site/alessiapaccagnini/



> **Further materials — including additional datasets, solutions to exercises, and codes for the appendix chapters — will be continuously added to this repository.**
