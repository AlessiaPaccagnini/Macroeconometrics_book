# =============================================================================
# Author:   Alessia Paccagnini
# Textbook: Macroeconometrics
# Mixed-Frequency VAR (MF-VAR) Estimation in R
# =============================================================================
# Translation of mfvar_estimation.py for the Macroeconometrics Textbook
# Chapter 11: Mixed-Frequency Data, Section 11.9
#
# Following Ghysels (2016), Journal of Econometrics
# "Macroeconomics and the Reality of Mixed Frequency Data"
#
# Key References:
#   Ghysels, E. (2016). JoE.
#   Ghysels, E., Hill, J., & Motegi, K. (2016). Granger causality tests.
#   Foroni, C. & Marcellino, M. (2014). Mixed frequency structural VARs.
#
# Required packages:
#   readxl, dplyr, lubridate, ggplot2, gridExtra
#
# Install if needed:
#   install.packages(c("readxl","dplyr","lubridate","ggplot2","gridExtra"))
# =============================================================================

library(readxl)
library(dplyr)
library(lubridate)
library(ggplot2)
library(gridExtra)

# =============================================================================
# 1. DATA LOADING AND STACKING
# =============================================================================

load_data <- function() {
  fedfunds <- read_excel("FEDFUNDS.xlsx", sheet = "Monthly")
  gdp      <- read_excel("GDPC1.xlsx",    sheet = "Quarterly")
  gdpdef   <- read_excel("GDPDEF.xlsx",   sheet = "Quarterly")

  colnames(fedfunds) <- c("date", "fedfunds")
  colnames(gdp)      <- c("date", "gdp")
  colnames(gdpdef)   <- c("date", "gdpdef")

  fedfunds$date <- as.Date(fedfunds$date)
  gdp$date      <- as.Date(gdp$date)
  gdpdef$date   <- as.Date(gdpdef$date)

  list(fedfunds = fedfunds, gdp = gdp, gdpdef = gdpdef)
}


create_stacked_mfvar_data <- function(monthly_df, quarterly_df,
                                      monthly_var    = "fedfunds",
                                      quarterly_var  = "gdp",
                                      transform_quarterly = "growth",
                                      m = 3L) {
  #' Create the stacked MF-VAR dataset following Ghysels (2016).
  #'
  #' Z_tau = [x_{tau,1}, x_{tau,2}, x_{tau,3}, y_tau]'
  #'
  #' @param monthly_df   Monthly data frame (must have columns: date, <monthly_var>)
  #' @param quarterly_df Quarterly data frame (must have columns: date, <quarterly_var>)
  #' @param monthly_var  Name of the monthly variable
  #' @param quarterly_var Name of the quarterly variable
  #' @param transform_quarterly "growth" for 400*log-diff, "level" for raw
  #' @param m  Months per quarter
  #' @return List: Z (data frame), var_names (character vector)

  monthly_df   <- monthly_df   %>% arrange(date)
  quarterly_df <- quarterly_df %>% arrange(date)

  # Transform quarterly variable
  if (transform_quarterly == "growth") {
    quarterly_df <- quarterly_df %>%
      mutate(y = 400 * log(.data[[quarterly_var]] / lag(.data[[quarterly_var]])))
  } else {
    quarterly_df <- quarterly_df %>% mutate(y = .data[[quarterly_var]])
  }

  # Quarter label for monthly data
  monthly_df <- monthly_df %>%
    mutate(quarter = floor_date(date, "quarter"))

  stacked <- list()

  for (i in seq_len(nrow(quarterly_df))) {
    q_date <- quarterly_df$date[i]
    y_val  <- quarterly_df$y[i]

    if (is.na(y_val)) next

    q_monthly <- monthly_df %>%
      filter(quarter == q_date) %>%
      arrange(date)

    if (nrow(q_monthly) < m) next

    row_data <- list(
      date = q_date
    )
    row_data[[paste0(monthly_var, "_m1")]] <- q_monthly[[monthly_var]][1]
    row_data[[paste0(monthly_var, "_m2")]] <- q_monthly[[monthly_var]][2]
    row_data[[paste0(monthly_var, "_m3")]] <- q_monthly[[monthly_var]][3]
    row_data[[quarterly_var]]              <- y_val

    stacked[[length(stacked) + 1]] <- as.data.frame(row_data)
  }

  Z <- do.call(rbind, stacked)
  var_names <- c(paste0(monthly_var, c("_m1","_m2","_m3")), quarterly_var)

  list(Z = Z, var_names = var_names)
}


# =============================================================================
# 2. MF-VAR MODEL FUNCTIONS
# =============================================================================

create_var_matrices <- function(data_mat, p) {
  #' Build Y (dependent) and X (lagged regressors + constant) matrices.
  T_full <- nrow(data_mat)
  k      <- ncol(data_mat)

  Y <- data_mat[(p + 1):T_full, , drop = FALSE]   # (T-p) x k

  # X: [1, Z_{t-1}, Z_{t-2}, ..., Z_{t-p}]
  X <- matrix(1, nrow = T_full - p, ncol = 1)
  for (lag in 1:p) {
    X <- cbind(X, data_mat[(p - lag + 1):(T_full - lag), , drop = FALSE])
  }

  list(Y = Y, X = X)
}


fit_mfvar <- function(Z_df, var_names, p = 1L) {
  #' Estimate MF-VAR by OLS (equation-by-equation).
  #'
  #' @param Z_df      Data frame with columns: date, <var_names>
  #' @param var_names Character vector of variable names
  #' @param p         Lag order
  #' @return List with coefficient matrices, residuals, Sigma, std errors, etc.

  data_mat <- as.matrix(Z_df[, var_names])
  k        <- length(var_names)

  mats <- create_var_matrices(data_mat, p)
  Y    <- mats$Y
  X    <- mats$X
  T_eff <- nrow(Y)

  # OLS: B = (X'X)^{-1} X'Y   shape: (1 + k*p) x k
  XtX_inv <- solve(t(X) %*% X)
  B        <- XtX_inv %*% t(X) %*% Y

  # Extract intercept and coefficient matrices
  c_vec <- B[1, ]           # k-vector
  A_list <- vector("list", p)
  for (lag in 1:p) {
    idx_start <- 2 + (lag - 1) * k
    idx_end   <- 1 + lag * k
    A_list[[lag]] <- t(B[idx_start:idx_end, , drop = FALSE])  # k x k
  }

  fitted    <- X %*% B
  residuals <- Y - fitted
  Sigma     <- (t(residuals) %*% residuals) / T_eff   # ML estimate

  # Standard errors (Sigma ⊗ (X'X)^{-1} for vec(B))
  se_c <- numeric(k)
  se_A <- lapply(1:p, function(l) matrix(0, k, k))

  for (eq in 1:k) {
    var_b <- Sigma[eq, eq] * diag(XtX_inv)
    se_b  <- sqrt(var_b)
    se_c[eq] <- se_b[1]
    for (lag in 1:p) {
      idx_start <- 2 + (lag - 1) * k
      idx_end   <- 1 + lag * k
      se_A[[lag]][eq, ] <- se_b[idx_start:idx_end]
    }
  }

  list(
    A          = A_list,
    c          = c_vec,
    Sigma      = Sigma,
    residuals  = residuals,
    fitted     = fitted,
    se_c       = se_c,
    se_A       = se_A,
    T          = T_eff,
    k          = k,
    p          = p,
    var_names  = var_names,
    data_mat   = data_mat
  )
}


compute_ma_coefs <- function(model, periods) {
  #' Compute MA-representation coefficients Psi_0, Psi_1, ..., Psi_h.

  k   <- model$k
  Psi <- vector("list", periods + 1)
  Psi[[1]] <- diag(k)   # Psi_0 = I

  for (h in 1:periods) {
    Psi_h <- matrix(0, k, k)
    for (j in 1:min(h, model$p)) {
      Psi_h <- Psi_h + Psi[[h - j + 1]] %*% model$A[[j]]
    }
    Psi[[h + 1]] <- Psi_h
  }
  Psi
}


compute_irf <- function(model, shock_var, periods = 20,
                        orthogonalized = TRUE, shock_size = 1.0) {
  #' Compute impulse response functions.
  #'
  #' @param model        MF-VAR model list (from fit_mfvar)
  #' @param shock_var    Integer (1-based) or variable name string
  #' @param periods      IRF horizon
  #' @param orthogonalized Use Cholesky decomposition
  #' @param shock_size   Shock magnitude
  #' @return Matrix (periods+1) x k

  if (is.character(shock_var))
    shock_var <- which(model$var_names == shock_var)

  k   <- model$k
  Psi <- compute_ma_coefs(model, periods)

  # Cholesky factor (lower triangular)
  P <- if (orthogonalized) t(chol(model$Sigma)) else diag(k)

  impulse <- P[, shock_var] * shock_size

  irf_mat <- matrix(0, periods + 1, k)
  for (h in 0:periods) {
    irf_mat[h + 1, ] <- Psi[[h + 1]] %*% impulse
  }
  irf_mat
}


bootstrap_irf_ci <- function(model, shock_var, periods = 20, n_boot = 500,
                              ci_level = 0.90, seed = 42,
                              orthogonalized = TRUE, shock_size = 1.0) {
  #' Bootstrap confidence intervals for IRFs (residual bootstrap).

  set.seed(seed)
  if (is.character(shock_var))
    shock_var <- which(model$var_names == shock_var)

  k      <- model$k
  p      <- model$p
  T_eff  <- model$T

  irf_pt   <- compute_irf(model, shock_var, periods, orthogonalized, shock_size)
  boot_irfs <- array(0, dim = c(n_boot, periods + 1, k))

  for (b in 1:n_boot) {
    # Resample residuals
    idx_boot    <- sample(T_eff, T_eff, replace = TRUE)
    boot_resid  <- model$residuals[idx_boot, ]

    # Reconstruct data
    Z_orig  <- model$data_mat
    Z_boot  <- matrix(0, nrow(Z_orig) - p, k)
    Z_hist  <- Z_orig[1:p, , drop = FALSE]

    for (t in 1:T_eff) {
      Z_new <- model$c  # k-vector
      for (lag in 1:p)
        Z_new <- Z_new + as.numeric(model$A[[lag]] %*% Z_hist[nrow(Z_hist) - lag + 1, ])
      Z_new <- Z_new + as.numeric(boot_resid[t, ])
      Z_boot[t, ] <- Z_new
      Z_hist <- rbind(Z_hist, matrix(Z_new, nrow = 1))
    }

    # Re-estimate
    Z_full  <- rbind(Z_orig[1:p, ], Z_boot)
    Z_df_b  <- as.data.frame(Z_full)
    colnames(Z_df_b) <- model$var_names
    Z_df_b$date <- 1:nrow(Z_df_b)  # dummy date for fit_mfvar

    tryCatch({
      m_b <- fit_mfvar(Z_df_b, model$var_names, p = p)
      boot_irfs[b, , ] <- compute_irf(m_b, shock_var, periods, orthogonalized, shock_size)
    }, error = function(e) {
      boot_irfs[b, , ] <<- irf_pt
    })
  }

  alpha <- 1 - ci_level
  lower <- apply(boot_irfs, c(2, 3), quantile, probs = alpha / 2)
  upper <- apply(boot_irfs, c(2, 3), quantile, probs = 1 - alpha / 2)

  list(irf = irf_pt, lower = lower, upper = upper,
       all_irfs = boot_irfs, n_boot = n_boot, ci_level = ci_level)
}


compute_fevd <- function(model, periods = 20) {
  #' Forecast Error Variance Decomposition.
  #' Returns array (periods+1) x k x k.

  k   <- model$k
  P   <- t(chol(model$Sigma))
  Psi <- compute_ma_coefs(model, periods)

  # Orthogonalised MA coefficients: Theta_h = Psi_h %*% P
  Theta <- lapply(Psi, function(psi) psi %*% P)

  fevd <- array(0, dim = c(periods + 1, k, k))

  for (h in 0:periods) {
    mse      <- matrix(0, k, k)
    for (s in 0:h)
      mse <- mse + Theta[[s + 1]] %*% t(Theta[[s + 1]])
    mse_diag <- diag(mse)

    for (j in 1:k) {
      contrib <- numeric(k)
      for (s in 0:h)
        contrib <- contrib + Theta[[s + 1]][, j]^2
      fevd[h + 1, , j] <- contrib / mse_diag
    }
  }
  fevd
}


granger_causality_test <- function(model, cause_vars, effect_vars) {
  #' Wald test for Granger non-causality.

  if (is.character(cause_vars))
    cause_idx  <- match(cause_vars,  model$var_names)
  else
    cause_idx  <- cause_vars

  if (is.character(effect_vars))
    effect_idx <- match(effect_vars, model$var_names)
  else
    effect_idx <- effect_vars

  k              <- model$k
  p              <- model$p
  n_restrictions <- length(cause_idx) * length(effect_idx) * p

  mats <- create_var_matrices(model$data_mat, p)
  Y    <- mats$Y
  X    <- mats$X
  T_eff <- nrow(Y)

  # Unrestricted SSR for effect equations
  ssr_u <- sum(model$residuals[, effect_idx]^2)

  # Restricted X (remove cause variables from all lags)
  keep_cols <- 1L  # constant
  for (lag in 1:p)
    for (v in 1:k)
      if (!(v %in% cause_idx))
        keep_cols <- c(keep_cols, 1L + (lag - 1L) * k + v)

  X_r   <- X[, keep_cols, drop = FALSE]
  B_r   <- solve(t(X_r) %*% X_r, t(X_r) %*% Y[, effect_idx, drop = FALSE])
  resid_r <- Y[, effect_idx, drop = FALSE] - X_r %*% B_r
  ssr_r <- sum(resid_r^2)

  df1    <- n_restrictions
  df2    <- T_eff * length(effect_idx) - ncol(X) * length(effect_idx)
  F_stat <- ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
  W_stat <- F_stat * df1
  p_val  <- 1 - pchisq(W_stat, df1)

  list(
    F_statistic   = F_stat,
    Wald_statistic = W_stat,
    p_value       = p_val,
    df            = c(df1, df2),
    H0            = sprintf("No Granger causality from %s to %s",
                            paste(cause_vars, collapse = ","),
                            paste(effect_vars, collapse = ","))
  )
}


# =============================================================================
# 3. PRINT SUMMARY
# =============================================================================

print_mfvar_summary <- function(model) {
  cat("\n", strrep("=", 75), "\n")
  cat("MIXED-FREQUENCY VAR ESTIMATION RESULTS\n")
  cat("Following Ghysels (2016, Journal of Econometrics)\n")
  cat(strrep("=", 75), "\n")
  cat(sprintf("Sample size (T):      %d\n", model$T))
  cat(sprintf("Variables (k):        %d\n", model$k))
  cat(sprintf("Lags (p):             %d\n", model$p))
  cat(sprintf("Parameters per eq.:   %d\n", 1 + model$k * model$p))
  cat(sprintf("Total parameters:     %d\n", model$k * (1 + model$k * model$p)))

  cat("\nA1 matrix (first lag):\n")
  A1  <- model$A[[1]]
  SE1 <- model$se_A[[1]]
  nms <- model$var_names

  header <- sprintf("%-16s", "")
  for (nm in nms) header <- paste0(header, sprintf("%14s", substr(nm, 1, 13)))
  cat(header, "\n", strrep("-", nchar(header)), "\n")

  for (i in seq_len(model$k)) {
    row_str <- sprintf("%-16s", substr(nms[i], 1, 15))
    for (j in seq_len(model$k)) {
      coef   <- A1[i, j]
      se_val <- SE1[i, j]
      t_val  <- if (se_val > 0) abs(coef / se_val) else 0
      stars  <- if (t_val > 2.576) "***" else if (t_val > 1.960) "**" else if (t_val > 1.645) "*" else ""
      row_str <- paste0(row_str, sprintf("%10.4f%-3s", coef, stars))
    }
    cat(row_str, "\n")
  }
  cat("Note: *** p<0.01, ** p<0.05, * p<0.10\n")

  # Intercepts
  cat("\nIntercepts:\n", strrep("-", 50), "\n")
  for (i in seq_len(model$k))
    cat(sprintf("  %-20s: %8.4f (SE: %.4f)\n", nms[i], model$c[i], model$se_c[i]))

  # Information criteria
  det_S  <- det(model$Sigma)
  k      <- model$k
  T_eff  <- model$T
  ll     <- -0.5 * T_eff * (k * log(2 * pi) + log(det_S) + k)
  n_par  <- k * (1 + k * model$p)
  aic_ic <- -2 * ll + 2 * n_par
  bic_ic <- -2 * ll + n_par * log(T_eff)

  cat(sprintf("\n  Log-likelihood: %12.2f\n", ll))
  cat(sprintf("  AIC:            %12.2f\n",  aic_ic))
  cat(sprintf("  BIC:            %12.2f\n",  bic_ic))
  cat(strrep("=", 75), "\n")
}


# =============================================================================
# 4. VISUALISATION
# =============================================================================

plot_irf_simple <- function(model, shock_var, periods = 20) {
  if (is.character(shock_var)) shock_name <- shock_var
  else                          shock_name <- model$var_names[shock_var]

  irf_mat <- compute_irf(model, shock_var, periods)
  k       <- model$k
  h_vec   <- 0:periods

  plots <- lapply(1:k, function(i) {
    df <- data.frame(h = h_vec, response = irf_mat[, i])
    ggplot(df, aes(x = h, y = response)) +
      geom_ribbon(aes(ymin = 0, ymax = response), alpha = 0.2, fill = "steelblue") +
      geom_line(colour = "steelblue", linewidth = 1.5) +
      geom_hline(yintercept = 0, linewidth = 0.5) +
      labs(x = "Quarters", y = "Response",
           title = paste("Response of", model$var_names[i])) +
      theme_bw(base_size = 10)
  })
  grid.arrange(grobs = plots, ncol = 2,
               top = paste("Impulse Responses to", shock_name, "Shock (1 Std Dev)"))
}


plot_irf_with_ci_r <- function(ci_results, var_names, shock_name,
                                periods = 20, ci_level = 0.90) {
  k     <- ncol(ci_results$irf)
  h_vec <- 0:periods
  ci_pct <- round(ci_level * 100)

  plots <- lapply(1:k, function(i) {
    df <- data.frame(
      h      = h_vec,
      irf    = ci_results$irf[, i],
      lower  = ci_results$lower[, i],
      upper  = ci_results$upper[, i]
    )
    ggplot(df, aes(x = h)) +
      geom_ribbon(aes(ymin = lower, ymax = upper), fill = "steelblue", alpha = 0.25) +
      geom_line(aes(y = irf), colour = "steelblue", linewidth = 1.5) +
      geom_hline(yintercept = 0, linewidth = 0.5) +
      labs(x = "Quarters", y = "Response",
           title = paste("Response of", var_names[i])) +
      theme_bw(base_size = 10)
  })
  grid.arrange(grobs = plots, ncol = 2,
               top = paste0("Impulse Responses to ", shock_name,
                             "\n", ci_pct, "% Bootstrap CI"))
}


plot_multiple_irfs_ci <- function(model, shock_vars, response_var,
                                   periods = 20, n_boot = 500,
                                   ci_level = 0.90, seed = 42) {
  if (is.character(response_var))
    resp_idx <- which(model$var_names == response_var)
  else
    resp_idx <- response_var

  ci_pct  <- round(ci_level * 100)
  colours <- c("firebrick", "darkgreen", "steelblue")

  plots <- lapply(seq_along(shock_vars), function(idx) {
    sv  <- shock_vars[[idx]]
    snm <- if (is.character(sv)) sv else model$var_names[sv]
    cat(sprintf("  Computing IRF for shock to %s...\n", snm))

    ci <- bootstrap_irf_ci(model, sv, periods, n_boot, ci_level, seed + idx - 1)

    df <- data.frame(
      h     = 0:periods,
      irf   = ci$irf[, resp_idx],
      lower = ci$lower[, resp_idx],
      upper = ci$upper[, resp_idx]
    )
    ggplot(df, aes(x = h)) +
      geom_ribbon(aes(ymin = lower, ymax = upper),
                  fill = colours[idx], alpha = 0.3) +
      geom_line(aes(y = irf), colour = colours[idx], linewidth = 1.5) +
      geom_hline(yintercept = 0, linewidth = 0.5) +
      labs(x = "Quarters", y = "Response",
           title = paste("Shock to", snm),
           subtitle = paste0(ci_pct, "% Bootstrap CI")) +
      theme_bw(base_size = 10)
  })

  grid.arrange(grobs = plots, nrow = 1,
               top = paste("Response of", model$var_names[resp_idx], "to Different Shocks"))
}


plot_fevd_r <- function(model, periods = 20) {
  fevd_arr <- compute_fevd(model, periods)
  k        <- model$k
  h_vec    <- 0:periods
  nms      <- model$var_names

  plots <- lapply(1:k, function(i) {
    df <- do.call(rbind, lapply(1:k, function(j) {
      data.frame(h = h_vec, share = fevd_arr[, i, j], shock = nms[j])
    }))
    ggplot(df, aes(x = h, y = share, fill = shock)) +
      geom_area(position = "stack", alpha = 0.8) +
      scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
      labs(x = "Quarters", y = "Share",
           title = paste("FEVD of", nms[i]), fill = "Shock") +
      theme_bw(base_size = 10) + theme(legend.position = "right")
  })
  grid.arrange(grobs = plots, ncol = 2,
               top = "Forecast Error Variance Decomposition")
}


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

cat("\n", strrep("=", 75), "\n")
cat("MF-VAR ESTIMATION: Following Ghysels (2016, JoE)\n")
cat(strrep("=", 75), "\n")

# Load data
cat("\n>>> Loading data...\n")
raw <- load_data()
fedfunds <- raw$fedfunds
gdp      <- raw$gdp

cat(sprintf("Monthly Fed Funds:  %d observations\n", nrow(fedfunds)))
cat(sprintf("Quarterly GDP:      %d observations\n", nrow(gdp)))

# Create stacked MF-VAR data
cat("\n>>> Creating stacked MF-VAR data...\n")
stacked_res <- create_stacked_mfvar_data(fedfunds, gdp,
                                          monthly_var    = "fedfunds",
                                          quarterly_var  = "gdp",
                                          transform_quarterly = "growth")
Z         <- stacked_res$Z
var_names <- stacked_res$var_names

cat(sprintf("Stacked data:       %d rows x %d cols\n", nrow(Z), ncol(Z)))
cat("Variables:", paste(var_names, collapse = ", "), "\n")
cat("First observations:\n")
print(head(Z, 5))

# ── Estimate MF-VAR(1) ────────────────────────────────────────────────────────
cat("\n>>> Estimating MF-VAR(1)...\n")
mfvar1 <- fit_mfvar(Z, var_names, p = 1L)
print_mfvar_summary(mfvar1)

# ── Granger Causality Tests ───────────────────────────────────────────────────
cat("\n>>> Granger Causality Tests:\n", strrep("-", 75), "\n")

gc_ffr_gdp <- granger_causality_test(mfvar1,
                                      cause_vars  = c("fedfunds_m1","fedfunds_m2","fedfunds_m3"),
                                      effect_vars = "gdp")
cat("\nTest: Monthly Fed Funds => GDP Growth\n")
cat(sprintf("  Wald statistic: %.4f\n",  gc_ffr_gdp$Wald_statistic))
cat(sprintf("  p-value:        %.4f\n",  gc_ffr_gdp$p_value))
cat(sprintf("  Conclusion:     %s at 5%% level\n",
            if (gc_ffr_gdp$p_value < 0.05) "Reject H0" else "Fail to reject H0"))

gc_gdp_ffr <- granger_causality_test(mfvar1,
                                      cause_vars  = "gdp",
                                      effect_vars = c("fedfunds_m1","fedfunds_m2","fedfunds_m3"))
cat("\nTest: GDP Growth => Monthly Fed Funds\n")
cat(sprintf("  Wald statistic: %.4f\n",  gc_gdp_ffr$Wald_statistic))
cat(sprintf("  p-value:        %.4f\n",  gc_gdp_ffr$p_value))
cat(sprintf("  Conclusion:     %s at 5%% level\n",
            if (gc_gdp_ffr$p_value < 0.05) "Reject H0" else "Fail to reject H0"))

# ── Plots (no CI — quick) ────────────────────────────────────────────────────
cat("\n>>> Generating visualisations...\n")

png("mfvar_irf_ffr_m1.png", width = 1400, height = 900, res = 120)
plot_irf_simple(mfvar1, "fedfunds_m1", periods = 16)
dev.off()
cat("  Saved: mfvar_irf_ffr_m1.png\n")

png("mfvar_irf_gdp.png", width = 1400, height = 900, res = 120)
plot_irf_simple(mfvar1, "gdp", periods = 16)
dev.off()
cat("  Saved: mfvar_irf_gdp.png\n")

png("mfvar_fevd.png", width = 1400, height = 900, res = 120)
plot_fevd_r(mfvar1, periods = 16)
dev.off()
cat("  Saved: mfvar_fevd.png\n")

# ── Bootstrap CI for FFR_m1 shock ─────────────────────────────────────────────
cat("\n>>> Computing IRFs with 90% bootstrap CI (500 reps)...\n")
ci_ffr_m1 <- bootstrap_irf_ci(mfvar1, "fedfunds_m1", periods = 16,
                                n_boot = 500, ci_level = 0.90, seed = 42)

png("mfvar_irf_ffr_m1_ci.png", width = 1400, height = 900, res = 120)
plot_irf_with_ci_r(ci_ffr_m1, var_names, "fedfunds_m1", periods = 16, ci_level = 0.90)
dev.off()
cat("  Saved: mfvar_irf_ffr_m1_ci.png\n")

# ── GDP response by month of shock ────────────────────────────────────────────
cat("\n>>> Comparing GDP response to shocks at different months...\n")
png("mfvar_gdp_response_by_month.png", width = 1400, height = 500, res = 120)
plot_multiple_irfs_ci(mfvar1,
                       shock_vars   = c("fedfunds_m1","fedfunds_m2","fedfunds_m3"),
                       response_var = "gdp",
                       periods      = 16,
                       n_boot       = 500,
                       ci_level     = 0.90,
                       seed         = 123)
dev.off()
cat("  Saved: mfvar_gdp_response_by_month.png\n")

# ── IRF Significance Summary ──────────────────────────────────────────────────
cat("\n>>> IRF Significance Summary (90% CI):\n", strrep("-", 75), "\n")
gdp_idx <- which(var_names == "gdp")
for (sv in c("fedfunds_m1","fedfunds_m2","fedfunds_m3")) {
  ci_res <- bootstrap_irf_ci(mfvar1, sv, periods = 8,
                              n_boot = 300, ci_level = 0.90, seed = 42)
  cat(sprintf("\n  Shock to %s:\n", sv))
  for (h in c(0,1,2,4,8)) {
    pt  <- ci_res$irf[h + 1, gdp_idx]
    lo  <- ci_res$lower[h + 1, gdp_idx]
    hi  <- ci_res$upper[h + 1, gdp_idx]
    sig <- if (lo > 0 | hi < 0) "*" else ""
    cat(sprintf("    h=%d: %7.3f [%7.3f, %7.3f] %s\n", h, pt, lo, hi, sig))
  }
}

# ── MF-VAR vs LF-VAR ─────────────────────────────────────────────────────────
cat("\n>>> Comparing MF-VAR with traditional LF-VAR...\n")

# Quarterly average of monthly Fed Funds
fedfunds_q <- fedfunds %>%
  mutate(quarter = floor_date(date, "quarter")) %>%
  group_by(quarter) %>%
  summarise(fedfunds = mean(fedfunds), .groups = "drop") %>%
  rename(date = quarter)

gdp_growth_df <- raw$gdp %>%
  arrange(date) %>%
  mutate(gdp_growth = 400 * log(gdp / lag(gdp))) %>%
  select(date, gdp_growth) %>%
  filter(!is.na(gdp_growth))

lf_data <- inner_join(fedfunds_q, gdp_growth_df, by = "date") %>% arrange(date)
lf_var_names <- c("fedfunds", "gdp_growth")

lfvar <- fit_mfvar(lf_data, lf_var_names, p = 1L)

cat("\nLF-VAR vs MF-VAR coefficient comparison:\n")
cat(sprintf("  A1[FFR->FFR]: LF=%.4f  vs  MF(m1->m1)=%.4f\n",
            lfvar$A[[1]][1,1], mfvar1$A[[1]][1,1]))
cat(sprintf("  A1[FFR->GDP]: LF=%.4f  vs  MF(m1->GDP)=%.4f\n",
            lfvar$A[[1]][2,1], mfvar1$A[[1]][4,1]))
cat(sprintf("  A1[GDP->FFR]: LF=%.4f  vs  MF(GDP->m1)=%.4f\n",
            lfvar$A[[1]][1,2], mfvar1$A[[1]][1,4]))
cat(sprintf("  A1[GDP->GDP]: LF=%.4f  vs  MF(GDP->GDP)=%.4f\n",
            lfvar$A[[1]][2,2], mfvar1$A[[1]][4,4]))

# Comparison plot
irf_mf <- compute_irf(mfvar1, "fedfunds_m1", periods = 16)
irf_lf <- compute_irf(lfvar,  "fedfunds",    periods = 16)
h_vec  <- 0:16

df_comp <- data.frame(
  h       = rep(h_vec, 2),
  gdp_resp = c(irf_mf[, 4], irf_lf[, 2]),
  ffr_resp = c(irf_mf[, 1], irf_lf[, 1]),
  model   = rep(c("MF-VAR", "LF-VAR"), each = 17)
)

p_gdp <- ggplot(df_comp, aes(x = h, y = gdp_resp, colour = model, linetype = model)) +
  geom_line(linewidth = 1.3) +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  scale_colour_manual(values = c("MF-VAR" = "steelblue", "LF-VAR" = "firebrick")) +
  scale_linetype_manual(values = c("MF-VAR" = "solid", "LF-VAR" = "dashed")) +
  labs(x = "Quarters", y = "Response", title = "Response of GDP Growth",
       colour = NULL, linetype = NULL) +
  theme_bw(base_size = 11)

p_ffr <- ggplot(df_comp, aes(x = h, y = ffr_resp, colour = model, linetype = model)) +
  geom_line(linewidth = 1.3) +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  scale_colour_manual(values = c("MF-VAR" = "steelblue", "LF-VAR" = "firebrick")) +
  scale_linetype_manual(values = c("MF-VAR" = "solid", "LF-VAR" = "dashed")) +
  labs(x = "Quarters", y = "Response", title = "Response of Interest Rate",
       colour = NULL, linetype = NULL) +
  theme_bw(base_size = 11)

png("mfvar_vs_lfvar_comparison.png", width = 1400, height = 500, res = 120)
grid.arrange(p_gdp, p_ffr, ncol = 2,
             top = "MF-VAR vs LF-VAR: Impulse Response Comparison")
dev.off()
cat("  Saved: mfvar_vs_lfvar_comparison.png\n")

# ── MF-VAR(2) Lag Selection ───────────────────────────────────────────────────
cat("\n>>> Estimating MF-VAR(2) for comparison...\n")
mfvar2 <- fit_mfvar(Z, var_names, p = 2L)

info_criteria <- function(model) {
  det_S <- det(model$Sigma)
  k     <- model$k
  T_eff <- model$T
  ll    <- -0.5 * T_eff * (k * log(2 * pi) + log(det_S) + k)
  n_par <- k * (1 + k * model$p)
  c(ll = ll, aic = -2*ll + 2*n_par, bic = -2*ll + n_par*log(T_eff))
}

ic1 <- info_criteria(mfvar1)
ic2 <- info_criteria(mfvar2)

cat("\nLag Selection:\n", strrep("-", 60), "\n")
cat(sprintf("%-12s %12s %12s %12s\n", "Model", "Log-Lik", "AIC", "BIC"))
cat(strrep("-", 60), "\n")
cat(sprintf("%-12s %12.2f %12.2f %12.2f\n", "MF-VAR(1)", ic1["ll"], ic1["aic"], ic1["bic"]))
cat(sprintf("%-12s %12.2f %12.2f %12.2f\n", "MF-VAR(2)", ic2["ll"], ic2["aic"], ic2["bic"]))
cat(sprintf("BIC prefers: %s\n", if (ic1["bic"] < ic2["bic"]) "MF-VAR(1)" else "MF-VAR(2)"))

cat("\n>>> MF-VAR estimation complete!\n")
cat(strrep("=", 75), "\n")
