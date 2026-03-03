# ===========================================================================
# Jordà (2005) Local Projections Example with Real U.S. Data — R Version
# ===========================================================================
# Comparing LP vs VAR impulse responses for monetary policy analysis
# Using FRED data: GDP (GDPC1), GDP Deflator (GDPDEF), Federal Funds Rate
#
# Instructions:
#   1. Place GDPC1.xlsx, GDPDEF.xlsx, and FEDFUNDS.xlsx in your working
#      directory (or adjust the paths below).
#   2. Install required packages if needed (see below).
#   3. Source this script: source("jorda_lp_real_data.R")
#. Author: Alessia Paccagnini
#. Textbook:Macroeconometrics
# ===========================================================================

# --- Install packages (uncomment if needed) --------------------------------
# install.packages(c("readxl", "vars", "sandwich", "lmtest", "mFilter"))

library(readxl)
library(vars)
library(sandwich)
library(lmtest)
library(mFilter)

# ===========================================================================
# 1. Load Data
# ===========================================================================
cat("Loading FRED data...\n")

# --- Set file paths (adjust if your files are elsewhere) -------------------
# For Google Colab / Jupyter-R you can upload files first, then point here.
gdp_file      <- "GDPC1.xlsx"
deflator_file <- "GDPDEF.xlsx"
fedfunds_file <- "FEDFUNDS.xlsx"

gdp_raw      <- read_excel(gdp_file,      sheet = "Quarterly")
deflator_raw <- read_excel(deflator_file,  sheet = "Quarterly")
fedfunds_raw <- read_excel(fedfunds_file,  sheet = "Monthly")

cat(sprintf("  GDP:       %d obs\n", nrow(gdp_raw)))
cat(sprintf("  Deflator:  %d obs\n", nrow(deflator_raw)))
cat(sprintf("  Fed Funds: %d obs\n", nrow(fedfunds_raw)))

# ===========================================================================
# 2. Process and Merge
# ===========================================================================

# Rename columns
colnames(gdp_raw)      <- c("date", "gdp")
colnames(deflator_raw) <- c("date", "deflator")
colnames(fedfunds_raw) <- c("date", "fedfunds")

# Convert dates
gdp_raw$date      <- as.Date(gdp_raw$date)
deflator_raw$date <- as.Date(deflator_raw$date)
fedfunds_raw$date <- as.Date(fedfunds_raw$date)

# Fed funds: monthly → quarterly average
fedfunds_raw$quarter <- as.Date(
  paste0(format(fedfunds_raw$date, "%Y"),
         "-", sprintf("%02d", (as.numeric(format(fedfunds_raw$date, "%m")) - 1) %/% 3 * 3 + 1),
         "-01"))
fedfunds_q <- aggregate(fedfunds ~ quarter, data = fedfunds_raw, FUN = mean)
colnames(fedfunds_q) <- c("date", "fedfunds")

# Align GDP and deflator to quarter-start dates
quarter_start <- function(d) {
  m <- as.numeric(format(d, "%m"))
  q_start <- (m - 1) %/% 3 * 3 + 1
  as.Date(paste0(format(d, "%Y"), "-", sprintf("%02d", q_start), "-01"))
}
gdp_raw$date      <- quarter_start(gdp_raw$date)
deflator_raw$date <- quarter_start(deflator_raw$date)

# Merge
df <- merge(gdp_raw, deflator_raw, by = "date")
df <- merge(df, fedfunds_q, by = "date")
df <- df[order(df$date), ]

cat(sprintf("\nMerged data: %d observations\n", nrow(df)))
cat(sprintf("Date range:  %s to %s\n", min(df$date), max(df$date)))

# ===========================================================================
# 3. Construct Variables
# ===========================================================================

# Output gap via HP filter (lambda = 1600)
df$log_gdp <- 100 * log(df$gdp)
hp <- hpfilter(ts(df$log_gdp, frequency = 4), freq = 1600)
df$output_gap <- as.numeric(hp$cycle)

# Inflation: 4-quarter change in log GDP deflator (annualised)
df$log_deflator <- 100 * log(df$deflator)
df$inflation <- c(rep(NA, 4), diff(df$log_deflator, lag = 4))

# Federal funds rate
df$fed_funds <- df$fedfunds

# Sample: 1960Q1 – 2007Q4
sample_idx <- df$date >= as.Date("1960-01-01") & df$date <= as.Date("2007-12-31")
df_sample  <- df[sample_idx, c("date", "output_gap", "inflation", "fed_funds")]
df_sample  <- na.omit(df_sample)

start_q <- sprintf("%sQ%d", format(min(df_sample$date), "%Y"),
                   (as.numeric(format(min(df_sample$date), "%m")) - 1) %/% 3 + 1)
end_q   <- sprintf("%sQ%d", format(max(df_sample$date), "%Y"),
                   (as.numeric(format(max(df_sample$date), "%m")) - 1) %/% 3 + 1)

cat(sprintf("\nEstimation sample: %s – %s  (T = %d)\n", start_q, end_q, nrow(df_sample)))
cat("\nDescriptive statistics:\n")
print(summary(df_sample[, c("output_gap", "inflation", "fed_funds")]))

# ===========================================================================
# 4. Estimate VAR Models
# ===========================================================================
cat("\n", strrep("=", 60), "\n")
cat("Estimating VAR models...\n")
cat(strrep("=", 60), "\n")

Y <- df_sample[, c("output_gap", "inflation", "fed_funds")]

var4 <- VAR(Y, p = 4, type = "const")
var1 <- VAR(Y, p = 1, type = "const")

cat(sprintf("  VAR(4)  AIC = %.2f   BIC = %.2f\n",
            AIC(var4), BIC(var4)))
cat(sprintf("  VAR(1)  AIC = %.2f   BIC = %.2f\n",
            AIC(var1), BIC(var1)))

# Cholesky IRFs (ordering: output_gap → inflation → fed_funds)
H <- 20
irf_var4 <- irf(var4, impulse = "fed_funds", n.ahead = H,
                 ortho = TRUE, ci = 0.95, boot = TRUE, runs = 500)
irf_var1 <- irf(var1, impulse = "fed_funds", n.ahead = H,
                 ortho = TRUE, ci = 0.95, boot = TRUE, runs = 500)

# ===========================================================================
# 5. Estimate Local Projections
# ===========================================================================
cat("\n", strrep("=", 60), "\n")
cat("Estimating Local Projections...\n")
cat(strrep("=", 60), "\n")

estimate_lp <- function(data, shock_var, response_var, H, n_lags) {
  # Jordà (2005) LP with Newey–West standard errors
  #
  # Args:
  #   data         : data.frame with variables (no date column)
  #   shock_var    : character, name of shock variable
  #   response_var : character, name of response variable
  #   H            : integer, maximum horizon
  #   n_lags       : integer, number of control lags
  #
  # Returns:
  #   list(irf, se, ci_lower, ci_upper) — each a numeric vector of length H+1

  vars_names <- colnames(data)
  n <- nrow(data)

  irf_vec <- numeric(H + 1)
  se_vec  <- numeric(H + 1)

  for (h in 0:H) {
    # Dependent variable: y_{t+h}
    y <- c(data[(1 + h):n, response_var], rep(NA, h))

    # Controls: lags 1..n_lags of every variable
    X <- data[, shock_var]  # contemporaneous shock
    for (v in vars_names) {
      for (lag in 1:n_lags) {
        lagged <- c(rep(NA, lag), data[1:(n - lag), v])
        X <- cbind(X, lagged)
      }
    }
    X <- cbind(1, X)  # add constant

    # Drop incomplete rows
    ok <- complete.cases(cbind(y, X))
    y_clean <- y[ok]
    X_clean <- as.matrix(X[ok, ])

    # OLS
    fit <- lm(y_clean ~ X_clean - 1)  # -1 because constant already in X

    # Newey–West standard errors
    nw_lags <- max(h + 1, 4)
    vcov_nw <- NeweyWest(fit, lag = nw_lags, prewhite = FALSE)

    # Coefficient on shock (column 2: after constant)
    irf_vec[h + 1] <- coef(fit)[2]
    se_vec[h + 1]  <- sqrt(vcov_nw[2, 2])
  }

  list(irf      = irf_vec,
       se       = se_vec,
       ci_lower = irf_vec - 1.96 * se_vec,
       ci_upper = irf_vec + 1.96 * se_vec)
}

Y_mat <- df_sample[, c("output_gap", "inflation", "fed_funds")]

cat("  Output gap response...\n")
lp_y  <- estimate_lp(Y_mat, "fed_funds", "output_gap", H, n_lags = 4)

cat("  Inflation response...\n")
lp_pi <- estimate_lp(Y_mat, "fed_funds", "inflation",  H, n_lags = 4)

cat("  Fed funds own response...\n")
lp_ff <- estimate_lp(Y_mat, "fed_funds", "fed_funds",  H, n_lags = 4)

cat("  Done!\n")

# ===========================================================================
# 6. Publication-Quality Figure
# ===========================================================================
cat("\n", strrep("=", 60), "\n")
cat("Creating figure...\n")
cat(strrep("=", 60), "\n")

# Extract VAR(4) IRFs
var4_y   <- irf_var4$irf$fed_funds[, "output_gap"]
var4_lo_y <- irf_var4$Lower$fed_funds[, "output_gap"]
var4_hi_y <- irf_var4$Upper$fed_funds[, "output_gap"]

var4_pi    <- irf_var4$irf$fed_funds[, "inflation"]
var4_lo_pi <- irf_var4$Lower$fed_funds[, "inflation"]
var4_hi_pi <- irf_var4$Upper$fed_funds[, "inflation"]

# VAR(1) IRFs
var1_y  <- irf_var1$irf$fed_funds[, "output_gap"]
var1_pi <- irf_var1$irf$fed_funds[, "inflation"]

horizons <- 0:H

# Colours
col_var4 <- "#2166AC"
col_lp   <- "#B2182B"
col_var1 <- "#4DAF4A"

# Save to file
png("jorda_lp_example_real_data.png", width = 14, height = 5.5,
    units = "in", res = 200)
par(mfrow = c(1, 2), mar = c(4.5, 4.5, 3, 1), family = "serif", cex = 1.05)

# --- Panel (a): Output Gap -------------------------------------------------
ym <- max(abs(c(var4_lo_y, var4_hi_y, lp_y$ci_lower, lp_y$ci_upper,
                var1_y))) * 1.15

plot(NA, xlim = c(0, H), ylim = c(-ym, ym),
     xlab = "Quarters after shock", ylab = "Percent",
     main = "(a) Response of Output Gap to Fed Funds Shock",
     xaxt = "n", font.main = 2)
axis(1, at = seq(0, 20, 4))
abline(h = 0, col = "black", lwd = 1)

# VAR(4) CI
polygon(c(horizons, rev(horizons)),
        c(var4_lo_y, rev(var4_hi_y)),
        col = adjustcolor(col_var4, alpha.f = 0.15), border = NA)
lines(horizons, var4_y, col = col_var4, lwd = 2.5)

# LP CI
polygon(c(horizons, rev(horizons)),
        c(lp_y$ci_lower, rev(lp_y$ci_upper)),
        col = adjustcolor(col_lp, alpha.f = 0.15), border = NA)
lines(horizons, lp_y$irf, col = col_lp, lwd = 2.5, lty = 2)

# VAR(1)
lines(horizons, var1_y, col = col_var1, lwd = 2, lty = 3)

legend("bottomright",
       legend = c("VAR(4)", "Local Projections", expression(paste("VAR(1) — misspecified"))),
       col = c(col_var4, col_lp, col_var1),
       lwd = c(2.5, 2.5, 2), lty = c(1, 2, 3),
       bg = "white", cex = 0.95)

# --- Panel (b): Inflation ---------------------------------------------------
ym_pi <- max(abs(c(var4_lo_pi, var4_hi_pi, lp_pi$ci_lower, lp_pi$ci_upper,
                    var1_pi))) * 1.15

plot(NA, xlim = c(0, H), ylim = c(-ym_pi, ym_pi),
     xlab = "Quarters after shock", ylab = "Percent",
     main = "(b) Response of Inflation to Fed Funds Shock",
     xaxt = "n", font.main = 2)
axis(1, at = seq(0, 20, 4))
abline(h = 0, col = "black", lwd = 1)

# VAR(4) CI
polygon(c(horizons, rev(horizons)),
        c(var4_lo_pi, rev(var4_hi_pi)),
        col = adjustcolor(col_var4, alpha.f = 0.15), border = NA)
lines(horizons, var4_pi, col = col_var4, lwd = 2.5)

# LP CI
polygon(c(horizons, rev(horizons)),
        c(lp_pi$ci_lower, rev(lp_pi$ci_upper)),
        col = adjustcolor(col_lp, alpha.f = 0.15), border = NA)
lines(horizons, lp_pi$irf, col = col_lp, lwd = 2.5, lty = 2)

# VAR(1)
lines(horizons, var1_pi, col = col_var1, lwd = 2, lty = 3)

legend("topright",
       legend = c("VAR(4)", "Local Projections", expression(paste("VAR(1) — misspecified"))),
       col = c(col_var4, col_lp, col_var1),
       lwd = c(2.5, 2.5, 2), lty = c(1, 2, 3),
       bg = "white", cex = 0.95)

# Main title
mtext(paste0("Impulse Responses to a Monetary Policy Shock: LP vs VAR\n",
             "U.S. Quarterly Data, ", start_q, "–", end_q),
      side = 3, outer = TRUE, line = -2, cex = 1.2, font = 2)

dev.off()
cat("Figure saved as 'jorda_lp_example_real_data.png'\n")

# ===========================================================================
# 7. Summary
# ===========================================================================
cat("\n", strrep("=", 60), "\n")
cat("SUMMARY: LP vs VAR Comparison (Real U.S. Data)\n")
cat(strrep("=", 60), "\n")
cat(sprintf("\nSample: %s – %s  (T = %d)\n", start_q, end_q, nrow(df_sample)))
cat("Variables: Output Gap, Inflation, Federal Funds Rate\n")
cat("Identification: Cholesky (output → inflation → fed funds)\n")

cat(sprintf("\n--- Output Gap Response (h = 8) ---\n"))
cat(sprintf("  VAR(4): %7.4f  [%7.4f, %7.4f]\n", var4_y[9], var4_lo_y[9], var4_hi_y[9]))
cat(sprintf("  LP:     %7.4f  [%7.4f, %7.4f]\n", lp_y$irf[9], lp_y$ci_lower[9], lp_y$ci_upper[9]))
cat(sprintf("  VAR(1): %7.4f  (misspecified)\n", var1_y[9]))

cat(sprintf("\n--- Inflation Response (h = 8) ---\n"))
cat(sprintf("  VAR(4): %7.4f  [%7.4f, %7.4f]\n", var4_pi[9], var4_lo_pi[9], var4_hi_pi[9]))
cat(sprintf("  LP:     %7.4f  [%7.4f, %7.4f]\n", lp_pi$irf[9], lp_pi$ci_lower[9], lp_pi$ci_upper[9]))
cat(sprintf("  VAR(1): %7.4f  (misspecified)\n", var1_pi[9]))

cat("\n", strrep("=", 60), "\n")
