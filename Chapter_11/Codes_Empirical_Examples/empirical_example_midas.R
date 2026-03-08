# =============================================================================
# Author:   Alessia Paccagnini
# Textbook: Macroeconometrics
# MIDAS (Mixed Data Sampling) Estimation in R
# =============================================================================
# Translation of midas_estimation.py for the Macroeconometrics Textbook
# Chapter 11: Mixed-Frequency Data, Section 11.9
#
# Key References:
#   Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). "The MIDAS Touch"
#   Ghysels, E., & Marcellino, M. (2018). "Applied Economic Forecasting Using
#     Time Series Methods"
#
# Required packages:
#   readxl, dplyr, lubridate, ggplot2, gridExtra, scales, nloptr
#
# Install if needed:
#   install.packages(c("readxl","dplyr","lubridate","ggplot2","gridExtra",
#                      "scales","nloptr"))
# =============================================================================

library(readxl)
library(dplyr)
library(lubridate)
library(ggplot2)
library(gridExtra)
library(scales)
library(nloptr)

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================

load_and_prepare_data <- function() {
  #' Load FRED data and prepare for MIDAS estimation.
  #' Quarterly GDP (low frequency) regressed on monthly Fed Funds (high frequency).

  fedfunds <- read_excel("FEDFUNDS.xlsx", sheet = "Monthly")
  gdp      <- read_excel("GDPC1.xlsx",    sheet = "Quarterly")
  gdpdef   <- read_excel("GDPDEF.xlsx",   sheet = "Quarterly")

  colnames(fedfunds) <- c("date", "fedfunds")
  colnames(gdp)      <- c("date", "gdp")
  colnames(gdpdef)   <- c("date", "gdpdef")

  fedfunds$date <- as.Date(fedfunds$date)
  gdp$date      <- as.Date(gdp$date)
  gdpdef$date   <- as.Date(gdpdef$date)

  # Annualised quarterly GDP growth: 400 * log(GDP_t / GDP_{t-1})
  gdp <- gdp %>%
    arrange(date) %>%
    mutate(gdp_growth = 400 * log(gdp / lag(gdp)))

  # Annualised inflation from GDP deflator
  gdpdef <- gdpdef %>%
    arrange(date) %>%
    mutate(inflation = 400 * log(gdpdef / lag(gdpdef)))

  # Merge
  quarterly <- inner_join(gdp, gdpdef[, c("date", "inflation")], by = "date")

  list(fedfunds = fedfunds, quarterly = quarterly)
}


align_mixed_frequency <- function(quarterly_df, monthly_df, m = 3, K = 4) {
  #' Align quarterly and monthly data for MIDAS regression.
  #'
  #' @param quarterly_df  Data frame with quarterly data
  #' @param monthly_df    Data frame with monthly data (fedfunds column)
  #' @param m             High-frequency observations per low-frequency period
  #' @param K             Number of quarterly lags (each with m monthly obs)
  #' @return List: y (vector), X_hf (matrix T x m*K), dates (Date vector)

  quarterly_df <- quarterly_df %>% arrange(date)
  monthly_df   <- monthly_df   %>% arrange(date) %>%
    mutate(quarter = floor_date(date, "quarter"))

  y_list    <- numeric(0)
  X_hf_list <- list()
  dates_list <- as.Date(character(0))

  n_q <- nrow(quarterly_df)

  for (idx in (K + 1):n_q) {
    q_date <- quarterly_df$date[idx]
    y_val  <- quarterly_df$gdp_growth[idx]

    if (is.na(y_val)) next

    hf_values <- numeric(0)
    valid <- TRUE

    for (lag in 0:(K - 1)) {
      lag_q_date <- quarterly_df$date[idx - lag]
      q_monthly  <- monthly_df %>%
        filter(quarter == lag_q_date) %>%
        arrange(date)

      if (nrow(q_monthly) < m) { valid <- FALSE; break }

      # Take last m monthly values, most recent first (reverse)
      monthly_vals <- rev(tail(q_monthly$fedfunds, m))
      hf_values    <- c(hf_values, monthly_vals)
    }

    if (valid && length(hf_values) == m * K) {
      y_list     <- c(y_list, y_val)
      X_hf_list  <- c(X_hf_list, list(hf_values))
      dates_list <- c(dates_list, q_date)
    }
  }

  y    <- y_list
  X_hf <- do.call(rbind, X_hf_list)
  dates <- dates_list

  list(y = y, X_hf = X_hf, dates = dates)
}


# =============================================================================
# 2. MIDAS WEIGHTING SCHEMES
# =============================================================================

exponential_almon_weights <- function(K, theta1, theta2) {
  #' Exponential Almon lag polynomial weights.
  #' w(k; theta) = exp(theta1*k + theta2*k^2) / sum_j exp(theta1*j + theta2*j^2)

  k       <- 1:K
  weights <- exp(theta1 * k + theta2 * k^2)
  weights / sum(weights)
}


beta_weights <- function(K, theta1, theta2, eps = 1e-6) {
  #' Beta polynomial weights (Ghysels et al., 2007).
  #' w(k; theta1, theta2) proportional to (k/K)^(theta1-1) * (1-k/K)^(theta2-1)

  theta1 <- max(theta1, eps)
  theta2 <- max(theta2, eps)

  k <- 1:K
  x <- k / (K + 1)   # normalise to (0,1)

  weights <- x^(theta1 - 1) * (1 - x)^(theta2 - 1)
  weights <- pmax(weights, eps)
  weights / sum(weights)
}


pdl_weights <- function(K, degree = 2) {
  #' Polynomial Distributed Lag (PDL) basis matrix.
  #' Returns matrix (K x degree+1).

  k <- 1:K
  sapply(0:degree, function(d) k^d)
}


step_function_weights <- function(K, n_steps = 3) {
  #' Step function weight basis matrix (K x n_steps).

  step_size   <- K %/% n_steps
  step_matrix <- matrix(0, nrow = K, ncol = n_steps)

  for (s in 1:n_steps) {
    start_idx <- (s - 1) * step_size + 1
    end_idx   <- if (s < n_steps) s * step_size else K
    step_matrix[start_idx:end_idx, s] <- 1
  }
  step_matrix
}


# =============================================================================
# 3. MIDAS REGRESSION
# =============================================================================

#' Fit parametric MIDAS regression (Beta or Exp. Almon weights)
#'
#' @param y          Numeric vector — low-frequency dependent variable
#' @param X_hf       Matrix (T x K) — high-frequency regressors
#' @param weight_type "beta" or "exp_almon"
#' @param theta_init Initial values for weight parameters (length 2)
#' @return List with params, weights, fitted, residuals, and fit statistics
fit_midas <- function(y, X_hf, weight_type = "beta", theta_init = NULL) {

  T <- length(y)
  K <- ncol(X_hf)

  # Helper: get weights from parameter vector
  get_weights <- function(theta) {
    if (weight_type == "beta")
      beta_weights(K, theta[1], theta[2])
    else
      exponential_almon_weights(K, theta[1], theta[2])
  }

  # Objective: sum of squared residuals
  objective <- function(params) {
    alpha   <- params[1]
    beta    <- params[2]
    theta   <- params[3:4]
    weights <- get_weights(theta)
    X_w     <- X_hf %*% weights
    y_hat   <- alpha + beta * X_w
    sum((y - y_hat)^2)
  }

  # OLS initialisation with equal weights
  X_eq  <- rowMeans(X_hf)
  X_ols <- cbind(1, X_eq)
  b_ols <- solve(t(X_ols) %*% X_ols, t(X_ols) %*% y)

  if (is.null(theta_init)) {
    theta_init <- if (weight_type == "beta") c(1.0, 1.0) else c(0.0, 0.0)
  }

  params_init <- c(b_ols, theta_init)

  # Bounds
  if (weight_type == "beta") {
    lb <- c(-Inf, -Inf, 0.01, 0.01)
    ub <- c( Inf,  Inf, 10.0, 10.0)
  } else {
    lb <- c(-Inf, -Inf, -5, -5)
    ub <- c( Inf,  Inf,  5,  5)
  }

  # Optimise with L-BFGS-B via nloptr
  res <- nloptr(
    x0          = params_init,
    eval_f      = objective,
    lb          = lb,
    ub          = ub,
    opts        = list(algorithm   = "NLOPT_LD_LBFGS",
                       maxeval     = 2000,
                       xtol_rel    = 1e-8,
                       ftol_rel    = 1e-10,
                       check_derivatives = FALSE),
    eval_grad_f = function(p) {
      # Numerical gradient
      eps  <- 1e-5
      grad <- numeric(length(p))
      f0   <- objective(p)
      for (i in seq_along(p)) {
        pp      <- p; pp[i] <- pp[i] + eps
        grad[i] <- (objective(pp) - f0) / eps
      }
      grad
    }
  )

  params  <- res$solution
  weights <- get_weights(params[3:4])

  X_w      <- X_hf %*% weights
  fitted   <- params[1] + params[2] * X_w
  residuals <- y - fitted

  # Numerical Hessian for standard errors
  k_params <- length(params)
  H        <- matrix(0, k_params, k_params)
  eps      <- 1e-5

  for (i in 1:k_params) {
    for (j in 1:k_params) {
      pp <- pm <- mp <- mm <- params
      pp[i] <- pp[i] + eps; pp[j] <- pp[j] + eps
      pm[i] <- pm[i] + eps; pm[j] <- pm[j] - eps
      mp[i] <- mp[i] - eps; mp[j] <- mp[j] + eps
      mm[i] <- mm[i] - eps; mm[j] <- mm[j] - eps
      H[i,j] <- (objective(pp) - objective(pm) - objective(mp) + objective(mm)) /
                (4 * eps^2)
    }
  }

  sigma2 <- sum(residuals^2) / (T - k_params)
  vcov   <- tryCatch(2 * sigma2 * solve(H), error = function(e) matrix(NA, k_params, k_params))
  se     <- sqrt(abs(diag(vcov)))

  ss_tot      <- sum((y - mean(y))^2)
  ss_res      <- sum(residuals^2)
  r_squared   <- 1 - ss_res / ss_tot
  adj_r2      <- 1 - (1 - r_squared) * (T - 1) / (T - k_params)
  aic         <- T * log(ss_res / T) + 2 * k_params
  bic         <- T * log(ss_res / T) + k_params * log(T)

  list(
    params      = params,        # c(alpha, beta, theta1, theta2)
    weights     = weights,
    fitted      = as.numeric(fitted),
    residuals   = as.numeric(residuals),
    std_errors  = se,
    vcov        = vcov,
    r_squared   = r_squared,
    adj_r2      = adj_r2,
    sigma       = sqrt(sigma2),
    aic         = aic,
    bic         = bic,
    ssr         = res$objective,
    T           = T,
    K           = K,
    weight_type = weight_type,
    converged   = (res$status > 0)
  )
}


#' Fit Unrestricted MIDAS (U-MIDAS) via OLS
fit_umidas <- function(y, X_hf) {

  T <- length(y)
  K <- ncol(X_hf)
  X <- cbind(1, X_hf)

  params    <- solve(t(X) %*% X, t(X) %*% y)
  fitted    <- as.numeric(X %*% params)
  residuals <- y - fitted

  sigma2    <- sum(residuals^2) / (T - K - 1)
  vcov      <- sigma2 * solve(t(X) %*% X)
  se        <- sqrt(diag(vcov))

  ss_tot    <- sum((y - mean(y))^2)
  ss_res    <- sum(residuals^2)
  r2        <- 1 - ss_res / ss_tot
  adj_r2    <- 1 - (1 - r2) * (T - 1) / (T - K - 1)
  aic       <- T * log(ss_res / T) + 2 * (K + 1)
  bic       <- T * log(ss_res / T) + (K + 1) * log(T)

  list(
    params    = as.numeric(params),
    fitted    = fitted,
    residuals = residuals,
    se        = as.numeric(se),
    r2        = r2,
    adj_r2    = adj_r2,
    sigma     = sqrt(sigma2),
    aic       = aic,
    bic       = bic,
    T         = T,
    K         = K
  )
}


# =============================================================================
# 4. PRINT SUMMARIES
# =============================================================================

print_midas_summary <- function(m, label = NULL) {
  label <- if (is.null(label)) toupper(m$weight_type) else label
  cat("\n", strrep("=", 70), "\n")
  cat("MIDAS REGRESSION RESULTS —", label, "\n")
  cat(strrep("=", 70), "\n")
  cat(sprintf("Weight function: %s\n", label))
  cat(sprintf("Observations:    %d\n", m$T))
  cat(sprintf("HF lags (K):     %d\n", m$K))
  cat(strrep("-", 70), "\n")
  cat(sprintf("%-15s %12s %12s %12s\n", "Parameter", "Estimate", "Std.Err", "t-stat"))
  cat(strrep("-", 70), "\n")
  nms <- c("alpha", "beta", "theta1", "theta2")
  for (i in seq_along(m$params)) {
    t_stat <- m$params[i] / m$std_errors[i]
    cat(sprintf("%-15s %12.4f %12.4f %12.2f\n",
                nms[i], m$params[i], m$std_errors[i], t_stat))
  }
  cat(strrep("-", 70), "\n")
  cat(sprintf("  R-squared:      %.4f\n", m$r_squared))
  cat(sprintf("  Adj. R-squared: %.4f\n", m$adj_r2))
  cat(sprintf("  Sigma:          %.4f\n", m$sigma))
  cat(sprintf("  AIC:            %.2f\n",  m$aic))
  cat(sprintf("  BIC:            %.2f\n",  m$bic))
  cat(strrep("=", 70), "\n")
}


print_umidas_summary <- function(m) {
  cat("\n", strrep("=", 70), "\n")
  cat("UNRESTRICTED MIDAS (U-MIDAS) REGRESSION RESULTS\n")
  cat(strrep("=", 70), "\n")
  cat(sprintf("Observations:    %d\n", m$T))
  cat(sprintf("HF lags (K):     %d\n", m$K))
  cat(strrep("-", 70), "\n")
  cat(sprintf("%-15s %12s %12s %12s\n", "Parameter", "Estimate", "Std.Err", "t-stat"))
  cat(strrep("-", 70), "\n")
  # Print constant + first 5 + last 5
  show_idx <- unique(c(1, 2:min(6, length(m$params)), max(1,length(m$params)-4):length(m$params)))
  for (i in show_idx) {
    nm     <- if (i == 1) "alpha" else paste0("beta_lag", i - 1)
    t_stat <- m$params[i] / m$se[i]
    cat(sprintf("%-15s %12.4f %12.4f %12.2f\n", nm, m$params[i], m$se[i], t_stat))
    if (i == 6 && length(m$params) > 10) cat("  ...\n")
  }
  cat(strrep("-", 70), "\n")
  cat(sprintf("  R-squared:      %.4f\n", m$r2))
  cat(sprintf("  Adj. R-squared: %.4f\n", m$adj_r2))
  cat(sprintf("  Sigma:          %.4f\n", m$sigma))
  cat(sprintf("  AIC:            %.2f\n",  m$aic))
  cat(sprintf("  BIC:            %.2f\n",  m$bic))
  cat(strrep("=", 70), "\n")
}


# =============================================================================
# 5. VISUALISATION
# =============================================================================

plot_midas_weights <- function(weights, title = "MIDAS Weights") {
  K    <- length(weights)
  df   <- data.frame(lag = 1:K, weight = weights)
  ggplot(df, aes(x = lag, y = weight)) +
    geom_col(fill = "steelblue", alpha = 0.7, colour = "black", linewidth = 0.3) +
    labs(x = "Lag (months)", y = "Weight", title = title) +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(hjust = 0.5))
}


plot_fitted_vs_actual <- function(y, fitted, dates, title = "MIDAS Fitted vs Actual") {
  df <- data.frame(date = dates, actual = y, fitted = fitted)

  p1 <- ggplot(df, aes(x = date)) +
    geom_line(aes(y = actual, colour = "Actual"), linewidth = 1.2) +
    geom_line(aes(y = fitted, colour = "Fitted"), linewidth = 1.2, linetype = "dashed") +
    geom_hline(yintercept = 0, colour = "black", linewidth = 0.4) +
    scale_colour_manual(values = c("Actual" = "steelblue", "Fitted" = "firebrick")) +
    labs(x = "Date", y = "GDP Growth (%)", title = title, colour = NULL) +
    theme_bw(base_size = 11) +
    theme(legend.position = "top", plot.title = element_text(hjust = 0.5))

  p2 <- ggplot(df, aes(x = fitted, y = actual)) +
    geom_point(alpha = 0.5, colour = "steelblue") +
    geom_abline(slope = 1, intercept = 0, colour = "firebrick", linetype = "dashed") +
    labs(x = "Fitted Values", y = "Actual Values", title = "Fitted vs Actual") +
    theme_bw(base_size = 11) +
    theme(plot.title = element_text(hjust = 0.5))

  grid.arrange(p1, p2, nrow = 2)
}


compare_weight_functions <- function(K = 12) {
  lags <- 1:K

  # --- Beta weights ---
  configs_beta <- list(
    list(t1 = 1,  t2 = 1,  label = "Uniform (1,1)"),
    list(t1 = 1,  t2 = 5,  label = "Declining (1,5)"),
    list(t1 = 2,  t2 = 2,  label = "Hump (2,2)"),
    list(t1 = 5,  t2 = 1,  label = "Increasing (5,1)")
  )
  df_beta <- do.call(rbind, lapply(configs_beta, function(cfg) {
    data.frame(lag = lags, weight = beta_weights(K, cfg$t1, cfg$t2), label = cfg$label)
  }))

  p1 <- ggplot(df_beta, aes(x = lag, y = weight, colour = label)) +
    geom_line(linewidth = 0.9) + geom_point(size = 2) +
    labs(x = "Lag", y = "Weight", title = "Beta Polynomial Weights", colour = NULL) +
    theme_bw(base_size = 10) + theme(legend.position = "bottom")

  # --- Exp Almon weights ---
  configs_almon <- list(
    list(t1 =  0.0, t2 =  0.00, label = "Uniform (0,0)"),
    list(t1 = -0.1, t2 =  0.00, label = "Declining (-0.1,0)"),
    list(t1 =  0.1, t2 = -0.02, label = "Hump (0.1,-0.02)"),
    list(t1 =  0.1, t2 =  0.00, label = "Increasing (0.1,0)")
  )
  df_almon <- do.call(rbind, lapply(configs_almon, function(cfg) {
    data.frame(lag = lags, weight = exponential_almon_weights(K, cfg$t1, cfg$t2), label = cfg$label)
  }))

  p2 <- ggplot(df_almon, aes(x = lag, y = weight, colour = label)) +
    geom_line(linewidth = 0.9) + geom_point(size = 2) +
    labs(x = "Lag", y = "Weight", title = "Exponential Almon Weights", colour = NULL) +
    theme_bw(base_size = 10) + theme(legend.position = "bottom")

  # --- PDL basis ---
  P  <- pdl_weights(K, degree = 2)
  df_pdl <- data.frame(
    lag    = rep(lags, 3),
    value  = c(P[,1], P[,2]/K, P[,3]/K^2),
    basis  = rep(c("Constant","Linear (scaled)","Quadratic (scaled)"), each = K)
  )
  p3 <- ggplot(df_pdl, aes(x = lag, y = value, colour = basis)) +
    geom_line(linewidth = 0.9) + geom_point(size = 2) +
    labs(x = "Lag", y = "Basis Value", title = "PDL Polynomial Basis", colour = NULL) +
    theme_bw(base_size = 10) + theme(legend.position = "bottom")

  grid.arrange(p1, p2, p3, nrow = 2,
               top = "Comparison of MIDAS Weighting Schemes")
}


# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

cat("\n", strrep("=", 70), "\n")
cat("MIDAS ESTIMATION: Nowcasting GDP Growth with Fed Funds Rate\n")
cat(strrep("=", 70), "\n")

# Load data
cat("\n>>> Loading and preparing data...\n")
data_list <- load_and_prepare_data()
fedfunds  <- data_list$fedfunds
quarterly <- data_list$quarterly

cat(sprintf("Monthly Fed Funds:  %d observations\n",  nrow(fedfunds)))
cat(sprintf("  Range: %s to %s\n",
            format(min(fedfunds$date), "%Y-%m"),
            format(max(fedfunds$date), "%Y-%m")))
cat(sprintf("Quarterly GDP:      %d observations\n",  nrow(quarterly)))

# Align data (m=3 months per quarter, K=12 monthly lags)
cat("\n>>> Aligning mixed-frequency data...\n")
aligned <- align_mixed_frequency(quarterly, fedfunds, m = 3L, K = 12L)
y    <- aligned$y
X_hf <- aligned$X_hf
dates <- aligned$dates

cat(sprintf("Aligned sample:     %d quarterly observations\n", length(y)))
cat(sprintf("HF matrix:          %d x %d\n", nrow(X_hf), ncol(X_hf)))

# ── MIDAS Beta ────────────────────────────────────────────────────────────────
cat("\n>>> Estimating MIDAS with Beta polynomial weights...\n")
midas_beta <- fit_midas(y, X_hf, weight_type = "beta", theta_init = c(1.0, 3.0))
print_midas_summary(midas_beta, "BETA")

# ── MIDAS Exp. Almon ─────────────────────────────────────────────────────────
cat("\n>>> Estimating MIDAS with Exponential Almon weights...\n")
midas_almon <- fit_midas(y, X_hf, weight_type = "exp_almon", theta_init = c(-0.05, -0.01))
print_midas_summary(midas_almon, "EXP. ALMON")

# ── U-MIDAS ──────────────────────────────────────────────────────────────────
cat("\n>>> Estimating Unrestricted MIDAS (U-MIDAS)...\n")
umidas <- fit_umidas(y, X_hf)
print_umidas_summary(umidas)

# ── Model Comparison ─────────────────────────────────────────────────────────
cat("\n", strrep("=", 70), "\n")
cat("MODEL COMPARISON\n")
cat(strrep("=", 70), "\n")
cat(sprintf("%-25s %10s %10s %12s %12s\n", "Model", "R2", "Adj.R2", "AIC", "BIC"))
cat(strrep("-", 70), "\n")
cat(sprintf("%-25s %10.4f %10.4f %12.2f %12.2f\n",
            "MIDAS (Beta)",       midas_beta$r_squared, midas_beta$adj_r2,  midas_beta$aic,  midas_beta$bic))
cat(sprintf("%-25s %10.4f %10.4f %12.2f %12.2f\n",
            "MIDAS (Exp. Almon)", midas_almon$r_squared, midas_almon$adj_r2, midas_almon$aic, midas_almon$bic))
cat(sprintf("%-25s %10.4f %10.4f %12.2f %12.2f\n",
            "U-MIDAS",           umidas$r2, umidas$adj_r2, umidas$aic, umidas$bic))
cat(strrep("=", 70), "\n")

# ── Plots ─────────────────────────────────────────────────────────────────────
cat("\n>>> Generating visualisations...\n")

# Plot 1: Weight function comparison
png("midas_weight_comparison.png", width = 1400, height = 900, res = 120)
compare_weight_functions(K = 12)
dev.off()
cat("  Saved: midas_weight_comparison.png\n")

# Plot 2: Estimated Beta weights
p_beta <- plot_midas_weights(midas_beta$weights, "Estimated Beta Polynomial Weights")
ggsave("midas_beta_weights.png", p_beta, width = 8, height = 4, dpi = 150)
cat("  Saved: midas_beta_weights.png\n")

# Plot 3: Fitted vs Actual
png("midas_fitted_vs_actual.png", width = 1200, height = 800, res = 120)
plot_fitted_vs_actual(y, midas_beta$fitted, dates,
                      "MIDAS (Beta): GDP Growth – Fitted vs Actual")
dev.off()
cat("  Saved: midas_fitted_vs_actual.png\n")

# Plot 4: Parametric vs U-MIDAS weights
K_val         <- ncol(X_hf)
umidas_coefs  <- umidas$params[-1]           # exclude constant
umidas_norm   <- abs(umidas_coefs) / sum(abs(umidas_coefs))

df_comp <- rbind(
  data.frame(lag = 1:K_val, weight = umidas_norm,          type = "U-MIDAS (norm. |beta|)"),
  data.frame(lag = 1:K_val, weight = midas_beta$weights,   type = "Beta weights"),
  data.frame(lag = 1:K_val, weight = midas_almon$weights,  type = "Exp. Almon weights")
)

p_comp <- ggplot() +
  geom_col(data = subset(df_comp, type == "U-MIDAS (norm. |beta|)"),
           aes(x = lag, y = weight), fill = "grey70", alpha = 0.6, width = 0.5) +
  geom_line(data = subset(df_comp, type != "U-MIDAS (norm. |beta|)"),
            aes(x = lag, y = weight, colour = type), linewidth = 1.2) +
  geom_point(data = subset(df_comp, type != "U-MIDAS (norm. |beta|)"),
             aes(x = lag, y = weight, colour = type, shape = type), size = 3) +
  scale_colour_manual(values = c("Beta weights" = "firebrick", "Exp. Almon weights" = "darkgreen")) +
  labs(x = "Lag (months)", y = "Weight",
       title = "Comparison: Parametric vs Unrestricted MIDAS Weights",
       colour = NULL, shape = NULL) +
  theme_bw(base_size = 12) + theme(legend.position = "top")

ggsave("midas_weights_comparison.png", p_comp, width = 9, height = 5, dpi = 150)
cat("  Saved: midas_weights_comparison.png\n")

cat("\n>>> MIDAS estimation complete!\n")
cat(strrep("=", 70), "\n")
