# =========================================================================
#  EXAMPLE 2: PROXY SVAR (EXTERNAL INSTRUMENTS)
#  Complete Analysis with IRFs and Comparison to Cholesky
# =========================================================================
#  Data: U.S. FRED (GDPC1, GDPDEF, FEDFUNDS)
#  Instrument: Romer-Romer (2004) shocks, updated by Wieland-Yang (2020)
#  Sample: 1970:Q1 - 2007:Q4
#
#  Author: Alessia Paccagnini
#  Textbook: Macroeconometrics
# =========================================================================

rm(list = ls())
cat(strrep("=", 70), "\n")
cat("EXAMPLE 2: PROXY SVAR (EXTERNAL INSTRUMENTS)\n")
cat("Using Romer-Romer (2004) Monetary Policy Shocks\n")
cat(strrep("=", 70), "\n\n")

library(readxl)

# =========================================================================
#  SECTION 1: LOAD AND PREPARE DATA
# =========================================================================
cat("[1/7] Loading FRED data...\n")

# --- GDP ---
gdp_raw <- read_excel("GDPC1.xlsx", sheet = "Quarterly")
gdp_raw$observation_date <- as.Date(gdp_raw$observation_date)
gdp_raw$GDPC1 <- as.numeric(gdp_raw$GDPC1)

# --- GDP Deflator ---
def_raw <- read_excel("GDPDEF.xlsx", sheet = "Quarterly")
def_raw$observation_date <- as.Date(def_raw$observation_date)
def_raw$GDPDEF <- as.numeric(def_raw$GDPDEF)

# --- Federal Funds Rate (monthly) ---
ff_raw <- read_excel("FEDFUNDS.xlsx", sheet = "Monthly")
ff_raw$observation_date <- as.Date(ff_raw$observation_date)
ff_raw$FEDFUNDS <- as.numeric(ff_raw$FEDFUNDS)

# Convert monthly to quarterly (average)
ff_raw$quarter <- as.Date(cut(ff_raw$observation_date, breaks = "quarter"))
ff_q <- aggregate(FEDFUNDS ~ quarter, data = ff_raw, FUN = mean)
names(ff_q) <- c("date", "fedfunds")

# --- Compute annualized growth rates ---
gdp_growth <- 400 * diff(log(gdp_raw$GDPC1))
gdp_dates  <- gdp_raw$observation_date[-1]

inflation  <- 400 * diff(log(def_raw$GDPDEF))
inf_dates  <- def_raw$observation_date[-1]

cat("   Done.\n")

# =========================================================================
#  SECTION 2: LOAD ROMER-ROMER SHOCKS
# =========================================================================
cat("\n[2/7] Loading Romer-Romer monetary policy shocks...\n")

rr_raw <- read_excel("RR_monetary_shock_quarterly.xlsx")
rr_raw$date <- as.Date(rr_raw$date)

cat("   Source: Romer & Romer (2004), updated by Wieland & Yang (2020)\n")
cat(sprintf("   Observations: %d\n", nrow(rr_raw)))

# =========================================================================
#  SECTION 3: MERGE AND ALIGN DATA
# =========================================================================
cat("\n[3/7] Merging datasets...\n")

# Align all to quarter-end dates
# Use a common quarterly grid
df_gdp <- data.frame(date = as.Date(cut(gdp_dates, "quarter")),
                     gdp_growth = gdp_growth)
df_inf <- data.frame(date = as.Date(cut(inf_dates, "quarter")),
                     inflation = inflation)
df_rr  <- data.frame(date = as.Date(cut(rr_raw$date, "quarter")),
                     rr_shock = rr_raw$resid_romer)

data <- merge(df_gdp, df_inf, by = "date")
data <- merge(data, ff_q, by = "date")
data <- merge(data, df_rr, by = "date")

# Restrict to 1970:Q1 - 2007:Q4
data <- data[data$date >= as.Date("1970-01-01") &
             data$date <= as.Date("2007-12-31"), ]
data <- na.omit(data)

cat(sprintf("   Final sample: %s to %s\n", data$date[1], data$date[nrow(data)]))
cat(sprintf("   Observations: %d\n", nrow(data)))

# Extract matrices
Y      <- as.matrix(data[, c("gdp_growth", "inflation", "fedfunds")])
z_full <- data$rr_shock
K      <- ncol(Y)

var_names <- c("GDP Growth", "Inflation", "Federal Funds Rate")

# =========================================================================
#  SECTION 4: HELPER FUNCTIONS
# =========================================================================

estimate_var <- function(Y, p) {
  # Estimate VAR(p) by OLS
  # Returns: list(T_eff, B_hat, u, Sigma_u, X)
  T_full <- nrow(Y)
  K      <- ncol(Y)
  T_eff  <- T_full - p

  Y_dep <- Y[(p + 1):T_full, , drop = FALSE]

  # Build regressor matrix: constant + lags
  X <- matrix(1, nrow = T_eff, ncol = 1)
  for (lag in 1:p) {
    X <- cbind(X, Y[(p + 1 - lag):(T_full - lag), , drop = FALSE])
  }

  B_hat   <- solve(t(X) %*% X, t(X) %*% Y_dep)
  u       <- Y_dep - X %*% B_hat
  Sigma_u <- (t(u) %*% u) / (T_eff - K * p - 1)

  list(T_eff = T_eff, B_hat = B_hat, u = u, Sigma_u = Sigma_u, X = X)
}


proxy_svar_identification <- function(u, z) {
  # Proxy SVAR: b_k = Cov(u_k, z) / Cov(u_mp, z)
  # Returns: list(b, se, n, F_stat, R2)
  K       <- ncol(u)
  min_len <- min(nrow(u), length(z))
  u <- u[1:min_len, , drop = FALSE]
  z <- z[1:min_len]

  # Remove NaN
  valid <- !is.na(z)
  u <- u[valid, , drop = FALSE]
  z <- z[valid]
  n <- length(z)

  # Covariances
  cov_uz <- sapply(1:K, function(k) cov(u[, k], z))
  cov_mp_z <- cov_uz[K]

  # Impact coefficients
  b <- cov_uz / cov_mp_z

  # First-stage regression
  X_fs     <- cbind(1, z)
  beta_fs  <- solve(t(X_fs) %*% X_fs, t(X_fs) %*% u[, K])
  resid_fs <- u[, K] - X_fs %*% beta_fs

  TSS    <- sum((u[, K] - mean(u[, K]))^2)
  RSS    <- sum(resid_fs^2)
  R2     <- 1 - RSS / TSS
  F_stat <- (R2 / 1) / ((1 - R2) / (n - 2))

  # Bootstrap standard errors
  n_boot <- 500
  b_boot <- matrix(NA, n_boot, K)
  for (ib in 1:n_boot) {
    idx   <- sample(n, n, replace = TRUE)
    cov_b <- sapply(1:K, function(k) cov(u[idx, k], z[idx]))
    if (abs(cov_b[K]) > 1e-10) {
      b_boot[ib, ] <- cov_b / cov_b[K]
    }
  }
  se <- apply(b_boot, 2, sd, na.rm = TRUE)

  list(b = b, se = se, n = n, F_stat = F_stat, R2 = R2)
}


compute_ma_coefficients <- function(B_hat, p, K, H) {
  # Extract lag matrices B_1, ..., B_p from B_hat
  # B_hat: (1+K*p) x K, first row = constant
  B_list <- vector("list", p)
  for (j in 1:p) {
    rows <- (1 + (j - 1) * K + 1):(1 + j * K)
    B_list[[j]] <- t(B_hat[rows, , drop = FALSE])  # K x K
  }

  # Phi_h: list of (H+1) K x K matrices
  Phi <- vector("list", H + 1)
  Phi[[1]] <- diag(K)

  for (h in 1:H) {
    tmp <- matrix(0, K, K)
    for (j in 1:min(h, p)) {
      tmp <- tmp + B_list[[j]] %*% Phi[[h - j + 1]]
    }
    Phi[[h + 1]] <- tmp
  }
  Phi
}


compute_irfs_cholesky <- function(B_hat, P, p, K, H) {
  # Returns array: (H+1) x K x K
  Phi <- compute_ma_coefficients(B_hat, p, K, H)
  IRF <- array(0, dim = c(H + 1, K, K))
  for (h in 0:H) {
    IRF[h + 1, , ] <- Phi[[h + 1]] %*% P
  }
  IRF
}


compute_irfs_proxy <- function(B_hat, b, p, K, H) {
  # Returns matrix: (H+1) x K
  Phi <- compute_ma_coefficients(B_hat, p, K, H)
  IRF <- matrix(0, H + 1, K)
  for (h in 0:H) {
    IRF[h + 1, ] <- Phi[[h + 1]] %*% b
  }
  IRF
}


bootstrap_irfs_proxy <- function(Y, z_full, p, H, n_boot = 2000, ci = 0.90) {
  # Wild bootstrap for Proxy SVAR with Mammen (1993) weights
  # Same weights for residuals and instrument
  # Returns: list(median, lower, upper)  each (H+1) x K

  T_full <- nrow(Y)
  K      <- ncol(Y)

  # Baseline VAR
  base    <- estimate_var(Y, p)
  T_eff   <- base$T_eff
  B_base  <- base$B_hat
  u_base  <- base$u
  X_base  <- base$X

  # Align instrument
  z_aligned <- z_full[(p + 1):T_full]
  min_len   <- min(nrow(u_base), length(z_aligned))
  z_trim    <- z_aligned[1:min_len]

  # Mammen distribution parameters
  p_mammen <- (sqrt(5) + 1) / (2 * sqrt(5))
  w_lo     <- -(sqrt(5) - 1) / 2
  w_hi     <-  (sqrt(5) + 1) / 2

  irfs_boot <- array(NA, dim = c(n_boot, H + 1, K))
  n_fail    <- 0

  for (ib in 1:n_boot) {
    tryCatch({
      # Mammen weights
      w <- ifelse(runif(T_eff) < p_mammen, w_lo, w_hi)

      # Wild bootstrap residuals
      u_star <- u_base * w

      # Reconstruct data
      Y_star       <- matrix(0, T_full, K)
      Y_star[1:p, ] <- Y[1:p, ]
      for (t in 1:T_eff) {
        Y_star[p + t, ] <- X_base[t, ] %*% B_base + u_star[t, ]
      }

      # Re-estimate VAR
      fit_b <- estimate_var(Y_star, p)

      # Wild bootstrap instrument (same weights)
      w_z    <- w[1:min_len]
      z_star <- z_trim * w_z

      # Re-identify
      id_b <- proxy_svar_identification(fit_b$u, z_star)

      # Bootstrap IRFs
      irfs_boot[ib, , ] <- compute_irfs_proxy(fit_b$B_hat, id_b$b, p, K, H)
    }, error = function(e) {
      n_fail <<- n_fail + 1
    })
  }

  cat(sprintf("   Wild bootstrap: %d replications, %d failures\n", n_boot, n_fail))

  alpha  <- (1 - ci) / 2
  median <- apply(irfs_boot, c(2, 3), median, na.rm = TRUE)
  lower  <- apply(irfs_boot, c(2, 3), quantile, probs = alpha,     na.rm = TRUE)
  upper  <- apply(irfs_boot, c(2, 3), quantile, probs = 1 - alpha, na.rm = TRUE)

  list(median = median, lower = lower, upper = upper)
}


bootstrap_irfs_cholesky <- function(Y, p, H, n_boot = 500, ci = 0.90) {
  # Residual bootstrap for Cholesky IRFs
  # Returns: list(median, lower, upper)  each (H+1) x K (MP shock only)

  T_full <- nrow(Y)
  K      <- ncol(Y)

  base   <- estimate_var(Y, p)
  T_eff  <- base$T_eff
  B_orig <- base$B_hat
  u_orig <- base$u
  X_orig <- base$X

  irfs_boot <- array(NA, dim = c(n_boot, H + 1, K))

  for (ib in 1:n_boot) {
    tryCatch({
      idx    <- sample(T_eff, T_eff, replace = TRUE)
      u_boot <- u_orig[idx, , drop = FALSE]

      Y_boot       <- matrix(0, T_full, K)
      Y_boot[1:p, ] <- Y[1:p, ]
      Y_boot[(p + 1):T_full, ] <- X_orig %*% B_orig + u_boot

      fit_b   <- estimate_var(Y_boot, p)
      P_b     <- t(chol(fit_b$Sigma_u))   # lower triangular
      IRF_b   <- compute_irfs_cholesky(fit_b$B_hat, P_b, p, K, H)
      irfs_boot[ib, , ] <- IRF_b[, , K]   # MP shock = last column
    }, error = function(e) {
      # leave as NA
    })
  }

  alpha  <- (1 - ci) / 2
  median <- apply(irfs_boot, c(2, 3), median, na.rm = TRUE)
  lower  <- apply(irfs_boot, c(2, 3), quantile, probs = alpha,     na.rm = TRUE)
  upper  <- apply(irfs_boot, c(2, 3), quantile, probs = 1 - alpha, na.rm = TRUE)

  list(median = median, lower = lower, upper = upper)
}

# =========================================================================
#  SECTION 5: VAR ESTIMATION
# =========================================================================
cat("\n[4/7] Estimating VAR(4)...\n")

p <- 4
H <- 40

fit <- estimate_var(Y, p)
T_eff   <- fit$T_eff
B_hat   <- fit$B_hat
u_hat   <- fit$u
Sigma_u <- fit$Sigma_u
X_var   <- fit$X

cat(sprintf("   VAR(%d) estimated\n", p))
cat(sprintf("   Effective sample: %d observations\n", T_eff))

# =========================================================================
#  SECTION 6: STRUCTURAL IDENTIFICATION
# =========================================================================
cat("\n[5/7] Structural Identification...\n")

# --- A) Cholesky ---
P_chol <- t(chol(Sigma_u))   # lower triangular

cat("\n   A) CHOLESKY IDENTIFICATION\n")
cat("   ", strrep("-", 50), "\n")
cat(sprintf("              eps_GDP    eps_pi    eps_MP\n"))
row_labs <- c("GDP      ", "Inflation", "FedFunds ")
for (i in 1:K) {
  cat(sprintf("   %s  %9.4f  %9.4f  %9.4f\n", row_labs[i],
              P_chol[i, 1], P_chol[i, 2], P_chol[i, 3]))
}

# --- B) Proxy SVAR ---
cat("\n   B) PROXY SVAR IDENTIFICATION\n")
cat("   ", strrep("-", 50), "\n")
cat("   Instrument: Romer-Romer (2004) narrative shocks\n")

z <- z_full[(p + 1):length(z_full)]

set.seed(42)
id <- proxy_svar_identification(u_hat, z)
b_proxy  <- id$b
se_proxy <- id$se
n_valid  <- id$n
F_stat   <- id$F_stat
R2       <- id$R2

cat(sprintf("\n   First-stage diagnostics:\n"))
cat(sprintf("   F-statistic: %.1f\n", F_stat))
cat(sprintf("   R-squared: %.4f\n", R2))
cat(sprintf("   Observations: %d\n", n_valid))

if (F_stat > 10) {
  cat("   Strong instrument (F > 10)\n")
} else {
  cat("   WARNING: Weak instrument (F < 10)\n")
}

cat(sprintf("\n   Structural impact multipliers:\n"))
cat(sprintf("   %-20s %12s %12s %10s\n", "Variable", "Estimate", "Std. Error", "t-stat"))
cat("   ", strrep("-", 54), "\n")
for (i in 1:K) {
  t_stat <- ifelse(se_proxy[i] > 0, b_proxy[i] / se_proxy[i], NA)
  cat(sprintf("   %-20s %12.4f %12.4f %10.2f\n",
              var_names[i], b_proxy[i], se_proxy[i], t_stat))
}

# =========================================================================
#  SECTION 7: COMPUTE IRFs
# =========================================================================
cat("\n\n[6/7] Computing IRFs...\n")

# Cholesky IRFs
IRF_chol    <- compute_irfs_cholesky(B_hat, P_chol, p, K, H)
irf_chol_mp <- IRF_chol[, , K]   # MP shock

# Proxy SVAR IRFs
irf_proxy <- compute_irfs_proxy(B_hat, b_proxy, p, K, H)

# Bootstrap CIs
cat("   Computing bootstrap confidence intervals...\n")

set.seed(42)

cat("   Proxy SVAR: wild bootstrap (Mammen weights, 2000 reps, 90% CI)...\n")
boot_proxy <- bootstrap_irfs_proxy(Y, z_full, p, H, n_boot = 2000, ci = 0.90)

cat("   Cholesky: residual bootstrap (500 reps, 90% CI)...\n")
boot_chol <- bootstrap_irfs_cholesky(Y, p, H, n_boot = 500, ci = 0.90)

cat("   Done.\n")

# =========================================================================
#  SECTION 8: GENERATE FIGURES
# =========================================================================
cat("\n\n[7/7] Generating figures...\n")

col_proxy <- rgb(46, 134, 171, maxColorValue = 255)
col_band  <- rgb(46, 134, 171, alpha = 60, maxColorValue = 255)
col_chol  <- rgb(233, 79, 55, maxColorValue = 255)
horizons  <- 0:H

# Helper: draw a figure to screen (RStudio Plots pane), then save to PNG
# Usage: call plot_start(), draw your plot, then call plot_save(filename)
plot_start <- function(w = 14, h = 4) {
  # Draw to the active device (RStudio Plots pane or X11)
  dev.new(width = w, height = h, noRStudioGD = FALSE)
}

plot_save <- function(filename, w = 1400, h = 400, res = 150) {
  # Copy whatever is on screen to a PNG file
  dev.copy(png, filename = filename, width = w, height = h, res = res)
  dev.off()
  cat(sprintf("   Saved: %s\n", filename))
}

# --- FIGURE 1: Proxy SVAR IRFs ---
plot_start()
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0))

for (i in 1:K) {
  y_lo  <- boot_proxy$lower[, i]
  y_hi  <- boot_proxy$upper[, i]
  y_med <- boot_proxy$median[, i]

  plot(horizons, y_med, type = "n",
       ylim = range(c(y_lo, y_hi)),
       xlab = "Quarters after shock", ylab = "Percentage points",
       main = paste("Response of", var_names[i]))
  polygon(c(horizons, rev(horizons)), c(y_lo, rev(y_hi)),
          col = col_band, border = NA)
  lines(horizons, y_med, col = col_proxy, lwd = 2.5)
  abline(h = 0, lwd = 0.8)
  grid()
}

mtext("Impulse Responses to Monetary Policy Shock\n(Proxy SVAR, Romer-Romer Instrument, 90% Wild Bootstrap CI, U.S. 1970-2007)",
      outer = TRUE, cex = 1.0, font = 2)
plot_save("figure1_proxy_irf.png")

# --- FIGURE 2: Proxy vs Cholesky Comparison ---
plot_start()
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0))

for (i in 1:K) {
  y_lo  <- boot_proxy$lower[, i]
  y_hi  <- boot_proxy$upper[, i]
  y_med <- boot_proxy$median[, i]
  y_ch  <- boot_chol$median[, i]

  plot(horizons, y_med, type = "n",
       ylim = range(c(y_lo, y_hi, y_ch)),
       xlab = "Quarters after shock", ylab = "Percentage points",
       main = paste("Response of", var_names[i]))
  polygon(c(horizons, rev(horizons)), c(y_lo, rev(y_hi)),
          col = col_band, border = NA)
  lines(horizons, y_med, col = col_proxy, lwd = 2.5)
  lines(horizons, y_ch,  col = col_chol,  lwd = 2.5, lty = 2)
  abline(h = 0, lwd = 0.8)
  legend("topright", legend = c("Proxy SVAR (R-R)", "Cholesky"),
         col = c(col_proxy, col_chol), lwd = 2.5, lty = c(1, 2),
         cex = 0.8, bg = "white")
  grid()
}

mtext("Monetary Policy Shock: Proxy SVAR vs Cholesky\n(90% Bootstrap CI, U.S. Data 1970-2007)",
      outer = TRUE, cex = 1.0, font = 2)
plot_save("figure2_proxy_vs_cholesky.png")

cat("\n   All figures saved!\n")

# =========================================================================
#  SUMMARY
# =========================================================================
cat("\n"); cat(strrep("=", 70)); cat("\n")
cat("SUMMARY RESULTS\n")
cat(strrep("=", 70)); cat("\n")

cat("\n--- First-Stage Diagnostics ---\n")
cat(sprintf("F-statistic: %.1f\n", F_stat))
cat(sprintf("R-squared: %.4f\n", R2))

cat("\n--- Impact Multipliers Comparison ---\n")
cat(sprintf("%-25s %12s %12s\n", "Variable", "Cholesky", "Proxy SVAR"))
cat(strrep("-", 50)); cat("\n")
for (i in 1:K) {
  cat(sprintf("%-25s %12.4f %12.4f\n", var_names[i], irf_chol_mp[1, i], irf_proxy[1, i]))
}

cat("\n--- Comparison with Textbook Table 4.5 ---\n")
cat(sprintf("%-25s %10s %11s %10s %11s\n",
            "Variable", "Book Chol", "Book Proxy", "Code Chol", "Code Proxy"))
cat(strrep("-", 68)); cat("\n")
book_chol  <- c(0.00, 0.00, 0.90)
book_proxy <- c(0.78, 0.19, 1.00)
for (i in 1:K) {
  cat(sprintf("%-25s %10.2f %11.2f %10.4f %11.4f\n",
              var_names[i], book_chol[i], book_proxy[i],
              irf_chol_mp[1, i], irf_proxy[1, i]))
}

cat("\n"); cat(strrep("=", 70)); cat("\n")
cat("DONE\n")
cat(strrep("=", 70)); cat("\n")
