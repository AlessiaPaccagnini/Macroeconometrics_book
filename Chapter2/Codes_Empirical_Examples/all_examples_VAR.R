# =============================================================================
# VAR EMPIRICAL EXAMPLES FOR MACROECONOMETRICS TEXTBOOK
# Chapter: VAR Models
# =============================================================================
#
# Textbook: Macroeconometrics
# Author: Alessia Paccagnini
#
#
# Data Source: FRED St. Louis McCracken & Ng Dataset (2026-01-QD.xlsx)
#
# This script provides three comprehensive examples using real quarterly US data:
# 1. VAR Estimation with Information Criteria, Residual Diagnostics, and Granger Causality
# 2. VECM Example with Cointegration Testing (Consumption-Income)
# 3. Reduced-Form Impulse Response Functions (motivating the need for identification)
#
# Period: 1960:Q2 to 2026:Q1 (264 observations for VAR, 265 for cointegration)
# Updated: March 3, 2026
#
# Required packages: readxl, vars, urca, tseries, lmtest
# Install with: install.packages(c("readxl", "vars", "urca", "tseries", "lmtest"))
# =============================================================================

library(readxl)
library(vars)
library(urca)
library(tseries)
library(lmtest)

cat(strrep("=", 80), "\n")
cat("MACROECONOMETRICS TEXTBOOK: VAR EMPIRICAL EXAMPLES\n")
cat("Author: Alessia Paccagnini, University College Dublin\n")
cat("Using REAL DATA: FRED McCracken-Ng Dataset (2026-01-QD.xlsx)\n")
cat(strrep("=", 80), "\n\n")

# ============================================================================
# FILE PATH — adjust for your environment
# ============================================================================

# --- Option A: Interactive file picker (uncomment this line) ---
# file_path <- file.choose()  # Opens a dialog to select the Excel file

# --- Option B: Set path manually ---
file_path <- "2026-01-QD.xlsx"

# ============================================================================
# LOAD DATA
# ============================================================================
# File structure (sheet 'in'):
#   Row 1: column names (sasdate, GDPC1, CPIAUCSL, ...)
#   Row 2: factor labels
#   Row 3: transformation codes
#   Rows 4-5: 1959:Q1-Q2 data
#   Row 6+: 1959:Q3 onward — we start here
#
# read_excel with col_names = FALSE reads row 1 as data.
# Row 6 in the file = row 6 in R (1-indexed) when col_names = FALSE.
# This matches Python's iloc[5] (0-indexed).
# ============================================================================

raw <- read_excel(file_path, sheet = "in", col_names = FALSE)
colnames_vec <- as.character(raw[1, ])

gdp_col    <- which(colnames_vec == "GDPC1")
cpi_col    <- which(colnames_vec == "CPIAUCSL")
ff_col     <- which(colnames_vec == "FEDFUNDS")
cons_col   <- which(colnames_vec == "PCECC96")
income_col <- which(colnames_vec == "DPIC96")

# Start from row 6 (= 1959:Q3, matching Python iloc[5])
start_row <- 6

gdp    <- as.numeric(raw[[gdp_col]][start_row:nrow(raw)])
cpi    <- as.numeric(raw[[cpi_col]][start_row:nrow(raw)])
ff     <- as.numeric(raw[[ff_col]][start_row:nrow(raw)])
cons   <- as.numeric(raw[[cons_col]][start_row:nrow(raw)])
income <- as.numeric(raw[[income_col]][start_row:nrow(raw)])

n_raw <- length(gdp)

# Annualised quarter-on-quarter growth rates
gdp_growth <- c(NA, 100 * 4 * diff(log(gdp)))
inflation  <- c(NA, 100 * 4 * diff(log(cpi)))

# Remove first observation (NaN from differencing) and build macro data
macro_mat <- data.frame(
  GDP_Growth = gdp_growth[-1],
  Inflation  = inflation[-1],
  FedFunds   = ff[-1]
)

# Remove any rows with NA
macro_mat <- macro_mat[complete.cases(macro_mat), ]

# Cointegration data — full series, no differencing needed
coint_mat <- data.frame(
  LogConsumption = log(cons),
  LogIncome      = log(income)
)
coint_mat <- coint_mat[complete.cases(coint_mat), ]

cat(sprintf("Macro data: %d observations (1960:Q2 to 2026:Q1)\n", nrow(macro_mat)))
cat(sprintf("Cointegration data: %d observations\n\n", nrow(coint_mat)))

# Create ts objects
var_ts <- ts(macro_mat, start = c(1960, 2), frequency = 4)

# ============================================================================
# EXAMPLE 1: VAR ESTIMATION
# ============================================================================

cat(strrep("=", 80), "\n")
cat("EXAMPLE 1: VAR ESTIMATION, DIAGNOSTICS, AND GRANGER CAUSALITY\n")
cat(strrep("=", 80), "\n\n")

# ---- Descriptive statistics ----
cat("Descriptive Statistics:\n")
print(summary(var_ts))

# ---- Unit root tests ----
cat("\n", strrep("-", 70), "\n")
cat("UNIT ROOT TESTS (Augmented Dickey-Fuller)\n")
cat(strrep("-", 70), "\n\n")

for (v in colnames(macro_mat)) {
  adf <- adf.test(macro_mat[[v]], alternative = "stationary")
  conclusion <- ifelse(adf$p.value < 0.05, "Stationary", "Non-stationary")
  cat(sprintf("  %-15s ADF = %7.3f  p = %5.3f  %s\n",
              v, adf$statistic, adf$p.value, conclusion))
}

# ---- Lag selection ----
cat("\n", strrep("-", 70), "\n")
cat("LAG ORDER SELECTION\n")
cat(strrep("-", 70), "\n\n")

lag_select <- VARselect(var_ts, lag.max = 12, type = "const")
print(lag_select$selection)
cat("\n")
print(round(t(lag_select$criteria[1:3, 1:8]), 3))

# NOTE: BIC(1) and BIC(2) differ by < 0.001. To ensure consistency
# across R, Python, and MATLAB, we fix p = 1 (the textbook specification).
optimal_lag <- 1
cat(sprintf("\nUsing lag order p = %d (textbook specification; BIC near-tied between p=1 and p=2)\n",
            optimal_lag))

# ---- VAR estimation ----
cat("\n", strrep("-", 70), "\n")
cat(sprintf("VAR(%d) ESTIMATION RESULTS\n", optimal_lag))
cat(strrep("-", 70), "\n\n")

var_model <- VAR(var_ts, p = optimal_lag, type = "const")
print(summary(var_model))

# ---- Residual diagnostics ----
cat("\n", strrep("-", 70), "\n")
cat("RESIDUAL DIAGNOSTICS\n")
cat(strrep("-", 70), "\n\n")

resid_mat <- residuals(var_model)

# Durbin-Watson
cat("Durbin-Watson:\n")
for (v in colnames(resid_mat)) {
  dw_val <- sum(diff(resid_mat[, v])^2) / sum(resid_mat[, v]^2)
  cat(sprintf("  %s: DW = %.3f\n", v, dw_val))
}

# Ljung-Box Q(12)
cat("\nLjung-Box Q(12):\n")
for (v in colnames(resid_mat)) {
  lb <- Box.test(resid_mat[, v], lag = 12, type = "Ljung-Box")
  sig <- ifelse(lb$p.value < 0.05, "**", "")
  cat(sprintf("  %s: Q = %.2f, p = %.4f %s\n", v, lb$statistic, lb$p.value, sig))
}

# Jarque-Bera
cat("\nJarque-Bera:\n")
for (v in colnames(resid_mat)) {
  jb <- jarque.bera.test(resid_mat[, v])
  sig <- ifelse(jb$p.value < 0.05, "**", "")
  cat(sprintf("  %s: JB = %.2f, p = %.4f %s\n", v, jb$statistic, jb$p.value, sig))
}

# Residual correlation
cat("\nResidual Correlation Matrix:\n")
corr_resid <- cor(resid_mat)
print(round(corr_resid, 3))

# ---- Granger causality ----
# Using lmtest::grangertest for pairwise bivariate tests
# Note: for exact match with Python/MATLAB (which also run bivariate tests),
# we use grangertest(response ~ predictor), which tests H0: predictor does NOT
# Granger-cause response.
cat("\n", strrep("-", 70), "\n")
cat("GRANGER CAUSALITY TESTS\n")
cat(strrep("-", 70), "\n\n")

var_names <- colnames(macro_mat)
cat(sprintf("%-45s %10s %10s\n", "Null Hypothesis", "F-stat", "p-value"))
cat(strrep("-", 70), "\n")

# Pairwise bivariate Granger tests (matches Python/MATLAB convention)
pairs <- list(
  c(1, 2), c(1, 3),  # what causes GDP_Growth? (Inflation, FedFunds)
  c(2, 1), c(2, 3),  # what causes Inflation? (GDP_Growth, FedFunds)
  c(3, 1), c(3, 2)   # what causes FedFunds? (GDP_Growth, Inflation)
)
for (p in pairs) {
  resp <- p[1]; pred <- p[2]
  # grangertest(response ~ predictor): tests H0: predictor does NOT cause response
  gc <- grangertest(var_ts[, resp] ~ var_ts[, pred], order = optimal_lag)
  f_stat <- gc$F[2]
  p_val  <- gc$`Pr(>F)`[2]
  sig <- ifelse(p_val < 0.01, "***", ifelse(p_val < 0.05, "**", ""))
  label <- sprintf("%s does not cause %s", var_names[pred], var_names[resp])
  cat(sprintf("%-45s %10.2f %10.4f %s\n", label, f_stat, p_val, sig))
}

# ---- Plot 1: Residual diagnostics ----
png("ex1_var_diagnostics.png", width = 1400, height = 1200, res = 150)
par(mfrow = c(3, 3), mar = c(4, 4, 3, 1))
for (i in 1:3) {
  v <- var_names[i]
  r <- resid_mat[, i]

  # Time series
  plot(r, type = "l", col = "blue", lwd = 0.8,
       main = paste0(v, ": Time Series"), ylab = "Residuals", xlab = "")
  abline(h = 0, col = "red", lty = 2)

  # Histogram
  hist(r, breaks = 30, probability = TRUE, col = rgb(0.27, 0.51, 0.71, 0.7),
       border = "white", main = paste0(v, ": Histogram"), xlab = "")
  curve(dnorm(x, mean(r), sd(r)), add = TRUE, col = "red", lwd = 2)

  # ACF
  acf(r, lag.max = 20, main = paste0(v, ": ACF"), col = "steelblue")
}
dev.off()
cat("\nSaved: ex1_var_diagnostics.png\n")

# ---- Plot 2: Lag selection ----
png("ex1_lag_selection.png", width = 1000, height = 600, res = 150)
ic_mat <- t(lag_select$criteria[1:3, ])
lags_vec <- 1:ncol(lag_select$criteria)
plot(lags_vec, ic_mat[, 1], type = "b", col = "green3", pch = 16, lwd = 2,
     ylim = range(ic_mat), xlab = "Number of Lags",
     ylab = "Information Criterion Value",
     main = "VAR Lag Order Selection\n(Real Data: 1960:Q2 - 2026:Q1)")
lines(lags_vec, ic_mat[, 2], type = "b", col = "red", pch = 15, lwd = 2)
lines(lags_vec, ic_mat[, 3], type = "b", col = "blue", pch = 17, lwd = 2)
abline(v = lag_select$selection["AIC(n)"], col = "green3", lty = 2)
abline(v = lag_select$selection["SC(n)"],  col = "red",    lty = 2)
abline(v = lag_select$selection["HQ(n)"],  col = "blue",   lty = 2)
legend("topright", legend = c("AIC", "BIC", "HQIC"),
       col = c("green3", "red", "blue"), pch = c(16, 15, 17), lwd = 2)
grid()
dev.off()
cat("Saved: ex1_lag_selection.png\n")

# ============================================================================
# EXAMPLE 2: COINTEGRATION
# ============================================================================

cat("\n", strrep("=", 80), "\n")
cat("EXAMPLE 2: COINTEGRATION AND VECM\n")
cat(strrep("=", 80), "\n\n")

cat(sprintf("Observations: %d\n\n", nrow(coint_mat)))

# ---- Unit root tests ----
cat("Unit Root Tests:\n")
coint_names <- c("LogConsumption", "LogIncome")
for (v in coint_names) {
  adf_lev  <- adf.test(coint_mat[[v]], alternative = "stationary")
  adf_diff <- adf.test(diff(coint_mat[[v]]), alternative = "stationary")
  cat(sprintf("  %s:\n", v))
  cat(sprintf("    Levels: ADF = %7.2f, p = %5.3f\n", adf_lev$statistic, adf_lev$p.value))
  cat(sprintf("    Diffs:  ADF = %7.2f, p = %5.3f\n", adf_diff$statistic, adf_diff$p.value))
}

# ---- Johansen cointegration test ----
# Case 4: linear trend in data, constant restricted to cointegrating equation.
# This is the correct specification when variables have deterministic trends
# (productivity growth). Using ecdet="const" (Case 2) spuriously finds r=2
# because the test confuses trend-stationarity with I(0).
cat("\n", strrep("-", 70), "\n")
cat("JOHANSEN COINTEGRATION TEST (Case 4: linear trend in data)\n")
cat(strrep("-", 70), "\n\n")

coint_ts <- ts(coint_mat, start = c(1960, 1), frequency = 4)

# urca::ca.jo with ecdet="trend" corresponds to Case 4 (det_order=1 in Python)
johansen <- ca.jo(coint_ts, type = "trace", ecdet = "trend", K = 2)
cat("Trace Test:\n")
print(summary(johansen))

johansen_eigen <- ca.jo(coint_ts, type = "eigen", ecdet = "trend", K = 2)
cat("\nMax Eigenvalue Test:\n")
print(summary(johansen_eigen))

# ---- OLS long-run relationship ----
ols_fit <- lm(LogConsumption ~ LogIncome, data = coint_mat)
ols_int   <- coef(ols_fit)[1]
ols_slope <- coef(ols_fit)[2]
cat(sprintf("\nOLS Long-run: LogC = %.4f + %.4f * LogY\n", ols_int, ols_slope))

# ---- Plot 3: Cointegration ----
dates_coint <- seq(as.Date("1960-01-01"), by = "quarter", length.out = nrow(coint_mat))

png("ex2_vecm_cointegration.png", width = 1200, height = 1000, res = 150)
par(mfrow = c(3, 1), mar = c(4, 4, 3, 1))

# Time series
plot(dates_coint, coint_mat$LogConsumption, type = "l", col = "blue", lwd = 1.5,
     main = "Time Series of Log Consumption and Log Income",
     ylab = "Log Scale", xlab = "")
lines(dates_coint, coint_mat$LogIncome, col = "orange", lwd = 1.5)
legend("topleft", legend = c("Log Consumption", "Log Income"),
       col = c("blue", "orange"), lwd = 1.5)
grid()

# Scatter plot with OLS line
plot(coint_mat$LogIncome, coint_mat$LogConsumption,
     pch = 20, col = rgb(0.27, 0.51, 0.71, 0.4), cex = 0.5,
     main = "Long-run Relationship",
     xlab = "Log Income", ylab = "Log Consumption")
abline(ols_fit, col = "red", lwd = 2)
legend("topleft",
       legend = sprintf("Long-run: C = %.2f + %.3f × Y", ols_int, ols_slope),
       col = "red", lwd = 2)
grid()

# Cointegrating residual
coint_resid <- residuals(ols_fit)
coint_resid_norm <- (coint_resid - mean(coint_resid)) / sd(coint_resid)
plot(dates_coint, coint_resid_norm, type = "l", col = "blue", lwd = 1,
     main = "Cointegrating Residual (Standardized)",
     xlab = "Quarter", ylab = "Deviation from Equilibrium")
abline(h = 0, col = "red", lty = 2)
polygon(c(dates_coint, rev(dates_coint)),
        c(rep(0, length(dates_coint)), rev(coint_resid_norm)),
        col = rgb(0, 0, 1, 0.2), border = NA)
grid()

dev.off()
cat("Saved: ex2_vecm_cointegration.png\n")

# ============================================================================
# EXAMPLE 3: IDENTIFICATION PROBLEM
# ============================================================================

cat("\n", strrep("=", 80), "\n")
cat("EXAMPLE 3: REDUCED-FORM IRFs AND IDENTIFICATION PROBLEM\n")
cat(strrep("=", 80), "\n\n")

cat("Residual Correlation Matrix:\n")
print(round(corr_resid, 3))

# ---- Plot 4: Correlation heatmap ----
png("ex3_innovation_correlation.png", width = 800, height = 600, res = 150)
par(mar = c(5, 5, 4, 6))

# Simple heatmap using image()
cols <- colorRampPalette(c("blue", "white", "red"))(100)
image(1:3, 1:3, corr_resid[3:1, ], col = cols, zlim = c(-1, 1),
      axes = FALSE, xlab = "", ylab = "",
      main = "Correlation of Reduced-Form Innovations\n(The Root of the Identification Problem)")
axis(1, at = 1:3, labels = var_names, las = 1)
axis(2, at = 1:3, labels = rev(var_names), las = 2)
box()

# Add text labels
for (i in 1:3) {
  for (j in 1:3) {
    val <- corr_resid[4 - j, i]
    clr <- ifelse(abs(val) > 0.5, "white", "black")
    text(i, j, sprintf("%.3f", val), col = clr, font = 2, cex = 1.2)
  }
}

dev.off()
cat("Saved: ex3_innovation_correlation.png\n")

# ---- Plot 5: Reduced-form IRFs ----
irf_result <- irf(var_model, n.ahead = 20, ortho = FALSE, boot = FALSE)

png("ex3_reduced_form_irf.png", width = 1400, height = 1200, res = 150)
par(mfrow = c(3, 3), mar = c(4, 4, 3, 1))

for (i in seq_along(var_names)) {   # impulse
  for (j in seq_along(var_names)) { # response
    imp_name <- var_names[i]
    resp_name <- var_names[j]

    vals <- irf_result$irf[[imp_name]][, resp_name]
    horizons <- 0:(length(vals) - 1)

    plot(horizons, vals, type = "l", col = "blue", lwd = 2,
         main = paste0("Response of ", resp_name, "\nto ", imp_name, " innovation"),
         xlab = ifelse(j == 3, "Quarters", ""), ylab = "Response")
    abline(h = 0, col = "black", lwd = 0.8)
    polygon(c(horizons, rev(horizons)),
            c(rep(0, length(horizons)), rev(vals)),
            col = rgb(0, 0, 1, 0.2), border = NA)
    grid()
  }
}

dev.off()
cat("Saved: ex3_reduced_form_irf.png\n")

# ============================================================================
# SUMMARY
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("ANALYSIS COMPLETE!\n")
cat(strrep("=", 80), "\n\n")
cat("Author: Alessia Paccagnini\n")
cat("University College Dublin, Smurfit Business School\n\n")
cat("Generated files:\n")
cat("  - ex1_var_diagnostics.png\n")
cat("  - ex1_lag_selection.png\n")
cat("  - ex2_vecm_cointegration.png\n")
cat("  - ex3_innovation_correlation.png\n")
cat("  - ex3_reduced_form_irf.png\n")
