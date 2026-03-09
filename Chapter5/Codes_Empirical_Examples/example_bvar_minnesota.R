# ========================================================================
#  EXAMPLE: BAYESIAN VAR WITH MINNESOTA PRIOR
#  Comparison with Frequentist VAR (Chapter 5)
# ========================================================================
#  Data: U.S. FRED (GDPC1, GDPDEF, FEDFUNDS)
#  Sample: 1970:Q1 - 2007:Q4
#
#  This example demonstrates:
#  1. Minnesota prior implementation via dummy observations
#  2. Posterior simulation from Normal-Inverse Wishart
#  3. Bayesian IRFs with credible intervals
#  4. Comparison with frequentist (OLS) estimates
#  5. Prior sensitivity analysis
#
#  Author: Alessia Paccagnini
#  Textbook: Macroeconometrics
# ========================================================================

# Clear workspace
rm(list = ls())
cat("======================================================================\n")
cat("BAYESIAN VAR WITH MINNESOTA PRIOR\n")
cat("Comparison with Frequentist Estimation\n")
cat("======================================================================\n")

# ========================================================================
#  REQUIRED PACKAGES
# ========================================================================
required_packages <- c("readxl", "zoo", "MASS", "MCMCpack", "ggplot2",
                       "gridExtra", "reshape2")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

# ========================================================================
#  SECTION 1: LOAD DATA
# ========================================================================
cat("\n[1/7] Loading FRED data...\n")

# Load GDP (Real GDP, Quarterly)
gdp_raw   <- read_excel("GDPC1.xlsx", sheet = "Quarterly")
colnames(gdp_raw) <- c("date", "gdp")
gdp_raw$date <- as.Date(gdp_raw$date)
gdp_raw$gdp  <- as.numeric(gdp_raw$gdp)

# Load GDP Deflator
defl_raw  <- read_excel("GDPDEF.xlsx", sheet = "Quarterly")
colnames(defl_raw) <- c("date", "deflator")
defl_raw$date     <- as.Date(defl_raw$date)
defl_raw$deflator <- as.numeric(defl_raw$deflator)

# Load Federal Funds Rate (Monthly -> Quarterly)
ff_raw    <- read_excel("FEDFUNDS.xlsx", sheet = "Monthly")
colnames(ff_raw) <- c("date", "fedfunds")
ff_raw$date     <- as.Date(ff_raw$date)
ff_raw$fedfunds <- as.numeric(ff_raw$fedfunds)

# Convert Fed Funds to quarterly (average)
ff_raw$quarter <- as.yearqtr(ff_raw$date)
ff_q <- aggregate(fedfunds ~ quarter, data = ff_raw, FUN = mean)

# Compute growth rates (annualized)
gdp_growth <- 400 * diff(log(gdp_raw$gdp))
gdp_dates  <- gdp_raw$date[-1]

inflation  <- 400 * diff(log(defl_raw$deflator))
infl_dates <- defl_raw$date[-1]

# Convert to year-quarter for merging
gdp_df  <- data.frame(quarter = as.yearqtr(gdp_dates),  gdp_growth = gdp_growth)
infl_df <- data.frame(quarter = as.yearqtr(infl_dates),  inflation  = inflation)

# Merge all series
data_all <- merge(gdp_df, infl_df, by = "quarter")
data_all <- merge(data_all, ff_q,   by = "quarter")

# Sample: 1970:Q1 to 2007:Q4
start_q <- as.yearqtr("1970 Q1")
end_q   <- as.yearqtr("2007 Q4")
data_all <- data_all[data_all$quarter >= start_q & data_all$quarter <= end_q, ]

# Remove any NA rows
data_all <- na.omit(data_all)

cat(sprintf("   Data loaded: %d observations\n", nrow(data_all)))
cat(sprintf("   Sample: %s to %s\n",
            format(data_all$quarter[1]),
            format(data_all$quarter[nrow(data_all)])))

# ========================================================================
#  SECTION 2: OLS VAR ESTIMATION (BENCHMARK)
# ========================================================================
cat("\n[2/7] Estimating frequentist VAR(4) by OLS...\n")

var_names  <- c("gdp_growth", "inflation", "fedfunds")
Y          <- as.matrix(data_all[, var_names])
p          <- 4   # Lag order
K          <- ncol(Y)
TT         <- nrow(Y)
T_eff      <- TT - p

# Construct data matrices
Y_dep <- Y[(p + 1):TT, , drop = FALSE]
X_ols <- matrix(1, nrow = T_eff, ncol = 1)          # constant
for (lag in 1:p) {
  X_ols <- cbind(X_ols, Y[(p + 1 - lag):(TT - lag), ])
}

# OLS estimation
B_ols     <- solve(crossprod(X_ols), crossprod(X_ols, Y_dep))
u_ols     <- Y_dep - X_ols %*% B_ols
Sigma_ols <- crossprod(u_ols) / (T_eff - K * p - 1)

cat("   OLS estimation complete\n")
cat(sprintf("   Variables: %d, Lags: %d, Observations: %d\n", K, p, T_eff))

# ========================================================================
#  SECTION 3: MINNESOTA PRIOR SPECIFICATION
# ========================================================================
cat("\n[3/7] Setting up Minnesota prior...\n")

create_minnesota_dummies <- function(Y, p,
                                     lambda1 = 0.2,
                                     lambda2 = 0.5,
                                     lambda3 = 1.0,
                                     lambda4 = 1e5) {
  # -----------------------------------------------------------------------
  # Create Minnesota prior via dummy observations
  #
  # Arguments:
  #   Y        : matrix (T x K)  -- Data matrix
  #   p        : integer          -- Number of lags
  #   lambda1  : numeric          -- Overall tightness (smaller = more shrinkage)
  #   lambda2  : numeric          -- Cross-variable shrinkage (0 < lambda2 <= 1)
  #   lambda3  : numeric          -- Lag decay (higher = faster decay)
  #   lambda4  : numeric          -- Constant term tightness (large = diffuse)
  #
  # Returns:
  #   list with Y_d, X_d, sigma
  # -----------------------------------------------------------------------
  TT <- nrow(Y)
  K  <- ncol(Y)

  # Estimate AR(1) for each variable to get scaling factors
  sigma <- numeric(K)
  delta <- rep(1, K)   # Random walk prior

  for (i in 1:K) {
    y_i   <- Y[2:TT, i]
    y_lag <- Y[1:(TT - 1), i]
    X_ar  <- cbind(1, y_lag)
    beta_ar <- solve(crossprod(X_ar), crossprod(X_ar, y_i))
    resid   <- y_i - X_ar %*% beta_ar
    sigma[i] <- sd(resid)     # uses N-1 by default in R
  }

  # Dimensions
  n_regressors <- 1 + K * p   # constant + K*p lag coefficients
  n_dummies    <- K * p + K + 1

  Y_d <- matrix(0, nrow = n_dummies, ncol = K)
  X_d <- matrix(0, nrow = n_dummies, ncol = n_regressors)

  # ----- Block 1: Prior on VAR coefficients -----
  row <- 1
  for (l in 1:p) {
    for (i in 1:K) {
      Y_d[row, i] <- delta[i] * sigma[i] / (lambda1 * (l^lambda3))
      col_idx     <- 1 + (l - 1) * K + i
      X_d[row, col_idx] <- sigma[i] / (lambda1 * (l^lambda3))
      row <- row + 1
    }
  }

  # ----- Block 2: Sum-of-coefficients prior -----
  for (i in 1:K) {
    Y_d[row, i] <- delta[i] * sigma[i] / lambda1
    for (l in 1:p) {
      col_idx <- 1 + (l - 1) * K + i
      X_d[row, col_idx] <- sigma[i] / lambda1
    }
    row <- row + 1
  }

  # ----- Block 3: Prior on constant -----
  Y_d[row, ] <- 0
  X_d[row, 1] <- lambda4

  return(list(Y_d = Y_d, X_d = X_d, sigma = sigma))
}

# Hyperparameters
lambda1 <- 0.2    # Overall tightness
lambda2 <- 0.5    # Cross-variable shrinkage
lambda3 <- 1.0    # Lag decay

cat(sprintf("   Minnesota prior hyperparameters:\n"))
cat(sprintf("   lambda1 (overall tightness) = %.2f\n", lambda1))
cat(sprintf("   lambda2 (cross-variable)    = %.2f\n", lambda2))
cat(sprintf("   lambda3 (lag decay)         = %.2f\n", lambda3))

dummies <- create_minnesota_dummies(Y, p, lambda1, lambda2, lambda3)
Y_d     <- dummies$Y_d
X_d     <- dummies$X_d
sigma_scale <- dummies$sigma

cat(sprintf("   Created %d dummy observations\n", nrow(Y_d)))

# ========================================================================
#  SECTION 4: BAYESIAN ESTIMATION
# ========================================================================
cat("\n[4/7] Bayesian VAR estimation...\n")

estimate_bvar_minnesota <- function(Y, p,
                                    lambda1 = 0.2,
                                    lambda2 = 0.5,
                                    lambda3 = 1.0) {
  # -----------------------------------------------------------------------
  # Estimate BVAR with Minnesota prior using dummy observations.
  # Returns posterior parameters for Normal-Inverse Wishart distribution.
  # -----------------------------------------------------------------------
  TT    <- nrow(Y)
  K     <- ncol(Y)
  T_eff <- TT - p

  # Construct data matrices
  Y_dep <- Y[(p + 1):TT, , drop = FALSE]
  X     <- matrix(1, nrow = T_eff, ncol = 1)
  for (lag in 1:p) {
    X <- cbind(X, Y[(p + 1 - lag):(TT - lag), ])
  }

  # Create dummy observations
  dum <- create_minnesota_dummies(Y, p, lambda1, lambda2, lambda3)

  # Stack actual data with dummy observations
  Y_star <- rbind(dum$Y_d, Y_dep)
  X_star <- rbind(dum$X_d, X)

  # Posterior parameters (conjugate Normal-Inverse Wishart)
  XtX_star <- crossprod(X_star)
  XtY_star <- crossprod(X_star, Y_star)

  B_post <- solve(XtX_star, XtY_star)

  # Posterior scale matrix for Inverse Wishart
  resid_star <- Y_star - X_star %*% B_post
  S_post     <- crossprod(resid_star)

  # Posterior degrees of freedom
  nu_post <- nrow(Y_star) - ncol(X_star)

  # Posterior precision for coefficients
  V_post_inv <- XtX_star

  return(list(B_post     = B_post,
              S_post     = S_post,
              nu_post    = nu_post,
              V_post_inv = V_post_inv,
              X          = X))
}

bvar_est   <- estimate_bvar_minnesota(Y, p, lambda1)
B_bvar     <- bvar_est$B_post
S_bvar     <- bvar_est$S_post
nu_bvar    <- bvar_est$nu_post
V_bvar_inv <- bvar_est$V_post_inv

# Posterior mean of Sigma
Sigma_bvar <- S_bvar / (nu_bvar - K - 1)

cat("   BVAR estimation complete\n")
cat(sprintf("   Posterior degrees of freedom: %d\n", nu_bvar))

# Compare coefficient estimates
cat("\n   Comparison: First lag coefficients (own effects)\n")
cat("   --------------------------------------------------\n")
cat(sprintf("   %-15s %12s %12s %12s\n", "Variable", "OLS", "BVAR", "Shrinkage"))
cat("   --------------------------------------------------\n")
for (i in 1:K) {
  ols_coef  <- B_ols[1 + i, i]
  bvar_coef <- B_bvar[1 + i, i]
  shrink    <- ifelse(abs(ols_coef) > 0.01,
                      (1 - bvar_coef / ols_coef) * 100, 0)
  cat(sprintf("   %-15s %12.4f %12.4f %11.1f%%\n",
              var_names[i], ols_coef, bvar_coef, shrink))
}

# ========================================================================
#  SECTION 5: POSTERIOR SIMULATION
# ========================================================================
cat("\n[5/7] Drawing from posterior distribution...\n")

draw_posterior_niw <- function(B_post, S_post, nu_post, V_post_inv,
                               n_draws = 2000) {
  # -----------------------------------------------------------------------
  # Draw from Normal-Inverse Wishart posterior
  #
  #   Sigma | Y  ~ IW(nu_post, S_post)
  #   vec(B) | Sigma, Y ~ N(vec(B_post), Sigma (x) V_post_inv^{-1})
  # -----------------------------------------------------------------------
  K       <- ncol(B_post)
  n_coefs <- nrow(B_post)
  V_post  <- solve(V_post_inv)

  B_draws     <- array(NA, dim = c(n_coefs, K, n_draws))
  Sigma_draws <- array(NA, dim = c(K, K, n_draws))

  for (d in seq_len(n_draws)) {
    # Draw Sigma from Inverse Wishart  (MCMCpack::riwish)
    Sigma_draw <- riwish(v = nu_post, S = S_post)
    Sigma_draws[, , d] <- Sigma_draw

    # Draw B | Sigma from matrix normal
    L_sigma <- t(chol(Sigma_draw))       # lower Cholesky
    L_V     <- t(chol(V_post))           # lower Cholesky

    Z <- matrix(rnorm(n_coefs * K), nrow = n_coefs, ncol = K)
    B_draw <- B_post + L_V %*% Z %*% t(L_sigma)
    B_draws[, , d] <- B_draw
  }

  return(list(B_draws = B_draws, Sigma_draws = Sigma_draws))
}

n_draws     <- 2000
n_burn      <- 500
total_draws <- n_draws + n_burn

cat(sprintf("   Drawing %d samples (discarding %d burn-in)...\n",
            total_draws, n_burn))

set.seed(42)
post <- draw_posterior_niw(B_bvar, S_bvar, nu_bvar, V_bvar_inv, total_draws)

# Discard burn-in
B_draws     <- post$B_draws[, , (n_burn + 1):total_draws]
Sigma_draws <- post$Sigma_draws[, , (n_burn + 1):total_draws]

cat(sprintf("   %d posterior draws retained\n", n_draws))

# ========================================================================
#  SECTION 6: COMPUTE IRFs FROM POSTERIOR
# ========================================================================
cat("\n[6/7] Computing Bayesian IRFs with credible intervals...\n")

H <- 40   # Horizon

compute_irf_cholesky <- function(B, Sigma, p, K, H) {
  # -----------------------------------------------------------------------
  # Compute IRFs using Cholesky identification
  #
  # Arguments:
  #   B      : matrix (1+K*p) x K  -- VAR coefficient matrix
  #   Sigma  : matrix K x K        -- Covariance matrix
  #   p      : integer             -- Number of lags
  #   K      : integer             -- Number of variables
  #   H      : integer             -- Horizon
  #
  # Returns:
  #   IRF : array (H+1) x K x K
  # -----------------------------------------------------------------------
  P_chol <- t(chol(Sigma))    # lower Cholesky factor

  # Companion form
  A_comp <- matrix(0, nrow = K * p, ncol = K * p)
  for (l in 1:p) {
    A_comp[1:K, ((l - 1) * K + 1):(l * K)] <-
      t(B[(1 + (l - 1) * K + 1):(1 + l * K), ])
  }
  if (p > 1) {
    A_comp[(K + 1):(K * p), 1:(K * (p - 1))] <- diag(K * (p - 1))
  }

  # IRFs
  IRF <- array(0, dim = c(H + 1, K, K))
  IRF[1, , ] <- P_chol

  A_power <- diag(K * p)
  for (h in 1:H) {
    A_power   <- A_power %*% A_comp
    Phi_h     <- A_power[1:K, 1:K]
    IRF[h + 1, , ] <- Phi_h %*% P_chol
  }

  return(IRF)
}

# Compute IRFs for each posterior draw
cat("   Computing IRFs for each posterior draw...\n")
IRF_draws <- array(NA, dim = c(H + 1, K, K, n_draws))

for (d in seq_len(n_draws)) {
  IRF_draws[, , , d] <- compute_irf_cholesky(B_draws[, , d],
                                              Sigma_draws[, , d],
                                              p, K, H)
}

# Compute posterior median and credible intervals
IRF_median   <- apply(IRF_draws, c(1, 2, 3), median)
IRF_lower_68 <- apply(IRF_draws, c(1, 2, 3), quantile, probs = 0.16)
IRF_upper_68 <- apply(IRF_draws, c(1, 2, 3), quantile, probs = 0.84)
IRF_lower_90 <- apply(IRF_draws, c(1, 2, 3), quantile, probs = 0.05)
IRF_upper_90 <- apply(IRF_draws, c(1, 2, 3), quantile, probs = 0.95)

cat("   IRF computation complete\n")

# Compute OLS IRFs for comparison
IRF_ols <- compute_irf_cholesky(B_ols, Sigma_ols, p, K, H)

# ========================================================================
#  SECTION 7: GENERATE FIGURES
# ========================================================================
cat("\n[7/7] Generating figures...\n")

var_labels <- c("GDP Growth", "Inflation", "Federal Funds Rate")
shock_idx  <- 3   # Monetary policy shock (third variable)
horizons   <- 0:H

colors_var <- c("#2E86AB", "#A23B72", "#F18F01")
color_bayes <- "#E63946"
color_ols   <- "#457B9D"

# --- FIGURE 1: Bayesian IRFs to MP Shock with Credible Intervals ---
cat("   Figure 1: Bayesian IRFs to MP shock...\n")

png("figure1_bvar_irf.png", width = 1400, height = 400, res = 150)
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

for (i in 1:K) {
  med_i   <- IRF_median[, i, shock_idx]
  lo68_i  <- IRF_lower_68[, i, shock_idx]
  hi68_i  <- IRF_upper_68[, i, shock_idx]
  lo90_i  <- IRF_lower_90[, i, shock_idx]
  hi90_i  <- IRF_upper_90[, i, shock_idx]

  ylim_range <- range(c(lo90_i, hi90_i))

  plot(horizons, med_i, type = "n", ylim = ylim_range,
       xlab = "Quarters after shock", ylab = "Percentage points",
       main = paste("Response of", var_labels[i]))

  # 90% CI
  polygon(c(horizons, rev(horizons)), c(lo90_i, rev(hi90_i)),
          col = adjustcolor(colors_var[i], alpha.f = 0.15), border = NA)
  # 68% CI
  polygon(c(horizons, rev(horizons)), c(lo68_i, rev(hi68_i)),
          col = adjustcolor(colors_var[i], alpha.f = 0.30), border = NA)
  # Median
  lines(horizons, med_i, col = colors_var[i], lwd = 2.5)
  # Zero line
  abline(h = 0, col = "black", lwd = 0.8)
  grid()

  if (i == 1) {
    legend("topright", legend = c("90% CI", "68% CI", "Posterior median"),
           fill = c(adjustcolor(colors_var[i], 0.15),
                    adjustcolor(colors_var[i], 0.30), NA),
           border = NA, lty = c(NA, NA, 1), lwd = c(NA, NA, 2.5),
           col = c(NA, NA, colors_var[i]), cex = 0.7, bg = "white")
  }
}

mtext(paste0("Bayesian VAR: IRFs to Monetary Policy Shock\n",
             sprintf("(Minnesota Prior, lambda1=%.2f, U.S. Data 1970-2007)", lambda1)),
      outer = TRUE, cex = 0.9, font = 2)
dev.off()

# --- FIGURE 2: Comparison BVAR vs OLS ---
cat("   Figure 2: BVAR vs OLS comparison...\n")

png("figure2_bvar_vs_ols.png", width = 1400, height = 400, res = 150)
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

for (i in 1:K) {
  med_i  <- IRF_median[, i, shock_idx]
  lo68_i <- IRF_lower_68[, i, shock_idx]
  hi68_i <- IRF_upper_68[, i, shock_idx]
  ols_i  <- IRF_ols[, i, shock_idx]

  ylim_range <- range(c(lo68_i, hi68_i, ols_i))

  plot(horizons, med_i, type = "n", ylim = ylim_range,
       xlab = "Quarters after shock", ylab = "Percentage points",
       main = paste("Response of", var_labels[i]))

  # BVAR 68% CI
  polygon(c(horizons, rev(horizons)), c(lo68_i, rev(hi68_i)),
          col = adjustcolor(color_bayes, alpha.f = 0.30), border = NA)
  # BVAR median
  lines(horizons, med_i, col = color_bayes, lwd = 2.5)
  # OLS
  lines(horizons, ols_i, col = color_ols, lwd = 2.5, lty = 2)
  abline(h = 0, col = "black", lwd = 0.8)
  grid()

  if (i == 1) {
    legend("topright",
           legend = c("BVAR 68% CI", "BVAR median", "OLS"),
           fill   = c(adjustcolor(color_bayes, 0.30), NA, NA),
           border = NA, lty = c(NA, 1, 2), lwd = c(NA, 2.5, 2.5),
           col    = c(NA, color_bayes, color_ols),
           cex = 0.7, bg = "white")
  }
}

mtext("Comparison: Bayesian VAR (Minnesota Prior) vs OLS\n(U.S. Data 1970-2007, Cholesky Identification)",
      outer = TRUE, cex = 0.9, font = 2)
dev.off()

# --- FIGURE 3: Prior Sensitivity Analysis ---
cat("   Figure 3: Prior sensitivity analysis...\n")

lambda_values <- c(0.05, 0.1, 0.2, 0.5, 1.0)
n_lambda      <- length(lambda_values)
IRF_sensitivity <- array(NA, dim = c(H + 1, K, K, n_lambda))

for (j in seq_along(lambda_values)) {
  est_temp <- estimate_bvar_minnesota(Y, p, lambda1 = lambda_values[j])
  Sigma_temp <- est_temp$S_post / (est_temp$nu_post - K - 1)
  IRF_sensitivity[, , , j] <- compute_irf_cholesky(est_temp$B_post,
                                                    Sigma_temp, p, K, H)
}

png("figure3_bvar_sensitivity.png", width = 1400, height = 400, res = 150)
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

cmap <- colorRampPalette(c("#440154", "#31688e", "#35b779", "#fde725"))(n_lambda)

for (i in 1:K) {
  ylim_range <- range(c(sapply(1:n_lambda, function(j)
    range(IRF_sensitivity[, i, shock_idx, j])),
    range(IRF_ols[, i, shock_idx])))

  plot(horizons, IRF_sensitivity[, i, shock_idx, 1], type = "n",
       ylim = ylim_range,
       xlab = "Quarters after shock", ylab = "Percentage points",
       main = paste("Response of", var_labels[i]))

  for (j in seq_along(lambda_values)) {
    lines(horizons, IRF_sensitivity[, i, shock_idx, j],
          col = cmap[j], lwd = 2)
  }
  # OLS
  lines(horizons, IRF_ols[, i, shock_idx], col = "black", lwd = 1.5, lty = 2)
  abline(h = 0, col = "black", lwd = 0.8)
  grid()

  if (i == 1) {
    legend("topright",
           legend = c(sprintf("l1=%.2f", lambda_values), "OLS"),
           col    = c(cmap, "black"),
           lty    = c(rep(1, n_lambda), 2),
           lwd    = c(rep(2, n_lambda), 1.5),
           cex = 0.6, bg = "white")
  }
}

mtext("Prior Sensitivity: Effect of Overall Tightness (lambda1) on IRFs\n(Smaller lambda1 = More Shrinkage Toward Random Walk)",
      outer = TRUE, cex = 0.9, font = 2)
dev.off()

# --- FIGURE 4: Posterior Distribution of Impact Effects ---
cat("   Figure 4: Posterior distributions of impact effects...\n")

png("figure4_bvar_posterior.png", width = 1400, height = 400, res = 150)
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

for (i in 1:K) {
  impact_draws <- IRF_draws[1, i, shock_idx, ]

  post_mean   <- mean(impact_draws)
  post_median <- median(impact_draws)

  hist(impact_draws, breaks = 50, probability = TRUE,
       col = adjustcolor(colors_var[i], alpha.f = 0.7),
       border = "white",
       xlab = "Impact effect (h=0)", ylab = "Posterior density",
       main = var_labels[i])

  abline(v = post_mean,   col = "darkred",  lwd = 2, lty = 1)
  abline(v = post_median, col = "darkblue", lwd = 2, lty = 2)
  abline(v = IRF_ols[1, i, shock_idx], col = "black", lwd = 2, lty = 3)

  legend("topright",
         legend = c(sprintf("Mean: %.4f",   post_mean),
                    sprintf("Median: %.4f", post_median),
                    sprintf("OLS: %.4f",    IRF_ols[1, i, shock_idx])),
         col = c("darkred", "darkblue", "black"),
         lty = c(1, 2, 3), lwd = 2, cex = 0.65, bg = "white")
}

mtext("Posterior Distribution of Impact Effects (h=0)\n(Response to Monetary Policy Shock)",
      outer = TRUE, cex = 0.9, font = 2)
dev.off()

cat("\n   All figures saved!\n")

# ========================================================================
#  SUMMARY STATISTICS
# ========================================================================
cat("\n======================================================================\n")
cat("SUMMARY RESULTS\n")
cat("======================================================================\n")

cat("\n--- Coefficient Shrinkage (BVAR vs OLS) ---\n")
cat(sprintf("%-25s %12s %12s %12s\n", "Coefficient", "OLS", "BVAR", "Shrinkage"))
cat(paste(rep("-", 61), collapse = ""), "\n")

for (i in 1:K) {
  for (lag in 1:2) {
    ols_c  <- B_ols[1 + (lag - 1) * K + i, i]
    bvar_c <- B_bvar[1 + (lag - 1) * K + i, i]
    shrink <- ifelse(abs(ols_c) > 0.01,
                     abs(bvar_c - ols_c) / abs(ols_c) * 100, 0)
    cat(sprintf("%s lag %d                %12.4f %12.4f %11.1f%%\n",
                var_names[i], lag, ols_c, bvar_c, shrink))
  }
}

cat("\n--- Impact Effects of MP Shock (h=0) ---\n")
cat(sprintf("%-20s %12s %12s %20s\n", "Variable", "OLS", "BVAR Median", "68% CI"))
cat(paste(rep("-", 64), collapse = ""), "\n")
for (i in 1:K) {
  ols_impact  <- IRF_ols[1, i, shock_idx]
  bvar_impact <- IRF_median[1, i, shock_idx]
  ci_low      <- IRF_lower_68[1, i, shock_idx]
  ci_high     <- IRF_upper_68[1, i, shock_idx]
  cat(sprintf("%-20s %12.4f %12.4f [%7.4f, %7.4f]\n",
              var_labels[i], ols_impact, bvar_impact, ci_low, ci_high))
}

cat("\n--- Peak Effects of MP Shock ---\n")
for (i in 1:K) {
  if (i < 3) {
    peak_idx <- which.min(IRF_median[, i, shock_idx])
  } else {
    peak_idx <- which.max(IRF_median[1:20, i, shock_idx])
  }
  peak_val <- IRF_median[peak_idx, i, shock_idx]
  ci_low   <- IRF_lower_68[peak_idx, i, shock_idx]
  ci_high  <- IRF_upper_68[peak_idx, i, shock_idx]
  cat(sprintf("%s: Peak at h=%d, value = %.4f [%.4f, %.4f]\n",
              var_labels[i], peak_idx - 1, peak_val, ci_low, ci_high))
}

cat("\n--- Price Puzzle Check ---\n")
infl_irf_median <- IRF_median[1:20, 2, shock_idx]
puzzle_quarters <- which(infl_irf_median > 0) - 1
if (length(puzzle_quarters) > 0) {
  cat(sprintf("Price puzzle in posterior median: quarters %s\n",
              paste(puzzle_quarters, collapse = ", ")))
} else {
  cat("No price puzzle in posterior median\n")
}

# Probability of positive inflation response
prob_positive <- apply(IRF_draws[1:20, 2, shock_idx, , drop = FALSE], 1,
                       function(x) mean(x > 0))
max_prob   <- max(prob_positive)
max_prob_h <- which.max(prob_positive) - 1
cat(sprintf("Max probability of positive inflation response: %.2f%% at h=%d\n",
            max_prob * 100, max_prob_h))

cat("\n======================================================================\n")
cat("FILES GENERATED:\n")
cat("======================================================================\n")
cat("   figure1_bvar_irf.png         - Bayesian IRFs with credible intervals\n")
cat("   figure2_bvar_vs_ols.png      - Comparison BVAR vs OLS\n")
cat("   figure3_bvar_sensitivity.png - Prior sensitivity analysis\n")
cat("   figure4_bvar_posterior.png   - Posterior distributions\n")
