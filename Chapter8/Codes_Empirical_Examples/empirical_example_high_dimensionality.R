# =============================================================================
# Replicating Bernanke, Boivin & Eliasz (2005) FAVAR Analysis
# =============================================================================
# Textbook: Macroeconometrics
# Author:   Alessia Paccagnini
#
# This script:
#   1. Loads FRED-MD dataset (McCracken & Ng, 2016)
#   2. Applies FRED-MD transformation codes for stationarity
#   3. Extracts 5 factors via PCA (excluding FFR)
#   4. Estimates 3-variable VAR (IP, CPI, FFR)
#   5. Estimates FAVAR (5 factors + FFR) — two-step approach
#   6. Compares IRFs across subsamples (Pre-Volcker / Great Moderation / Full)
#   7. Rolling out-of-sample forecasting: FAVAR vs VAR vs Random Walk
#   8. Clark-West (2007) test for nested models
#   9. Giacomini-Rossi (2010) fluctuation test for forecast stability
#
# References:
#   Bernanke, Boivin & Eliasz (2005), QJE
#   McCracken & Ng (2016), JBES
#   Clark & West (2007), JoE
#   Giacomini & Rossi (2010), ReStud
#
# Data: 2025-12-MD.csv  (FRED-MD monthly)
# Sample: 1962-01-01 to 2007-12-01 (pre-crisis, consistent with textbook)
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(purrr)
})

# =============================================================================
# SECTION 1 — DATA LOADING AND TRANSFORMATION
# =============================================================================

load_fred_md <- function(filepath) {
  raw         <- read.csv(filepath, check.names = FALSE, stringsAsFactors = FALSE)
  tcodes      <- as.numeric(raw[1, -1])
  var_names   <- colnames(raw)[-1]
  dates       <- as.Date(raw[-1, 1], format = "%m/%d/%Y")
  data_mat    <- apply(raw[-1, -1], 2, as.numeric)
  rownames(data_mat) <- as.character(dates)
  list(data = data_mat, tcodes = tcodes, var_names = var_names, dates = dates)
}

transform_series <- function(x, tcode) {
  # FRED-MD transformation codes:
  #  1=levels  2=diff  3=diff2  4=log  5=logdiff  6=logdiff2  7=Δ(pct chg)
  small <- 1e-6
  n     <- length(x)
  y     <- rep(NA_real_, n)
  switch(as.character(tcode),
    "1" = { y <- x },
    "2" = { y[2:n] <- diff(x) },
    "3" = { y[3:n] <- diff(diff(x)) },
    "4" = { y <- log(pmax(x, small)) },
    "5" = { y[2:n] <- diff(log(pmax(x, small))) },
    "6" = { y[3:n] <- diff(diff(log(pmax(x, small)))) },
    "7" = { pct <- x[-1] / x[-n] - 1; y[3:n] <- diff(pct) },
    { y <- x }
  )
  y
}

prepare_data <- function(fred, start_date, end_date) {
  dates <- fred$dates
  idx   <- dates >= as.Date(start_date) & dates <= as.Date(end_date)
  mat   <- fred$data[idx, ]
  d     <- dates[idx]

  # Transform each column
  trans <- matrix(NA_real_, nrow = nrow(mat), ncol = ncol(mat),
                  dimnames = dimnames(mat))
  for (j in seq_len(ncol(mat)))
    trans[, j] <- transform_series(mat[, j], fred$tcodes[j])

  # Drop columns with > 10% missing
  miss_pct  <- colMeans(is.na(trans))
  trans     <- trans[, miss_pct < 0.10, drop = FALSE]
  list(data = trans, dates = d)
}

# =============================================================================
# SECTION 2 — FACTOR EXTRACTION
# =============================================================================

extract_factors <- function(data_list, n_factors = 5,
                             exclude_vars = "FEDFUNDS") {
  mat   <- data_list$data
  keep  <- !colnames(mat) %in% exclude_vars
  X     <- mat[, keep, drop = FALSE]

  # Drop rows with ANY missing (explicit — avoids silent sample shrinkage)
  X     <- X[complete.cases(X), , drop = FALSE]

  # Standardise
  mu    <- colMeans(X)
  sigma <- apply(X, 2, sd)
  Xs    <- sweep(sweep(X, 2, mu), 2, sigma, "/")

  # PCA via SVD
  sv    <- svd(Xs, nu = n_factors, nv = n_factors)
  F_hat <- sv$u[, 1:n_factors, drop = FALSE] %*%
             diag(sv$d[1:n_factors], n_factors, n_factors)
  # svd() strips rownames — restore from clean data matrix so date alignment works
  rownames(F_hat) <- rownames(X)
  loadings <- sv$v[, 1:n_factors, drop = FALSE]

  # Explained variance
  var_exp <- sv$d[1:n_factors]^2 / sum(sv$d^2)

  cat(sprintf("    Variables in panel : %d\n",    ncol(X)))
  cat(sprintf("    Observations       : %d\n",    nrow(X))  )
  cat(sprintf("    Variance explained : %.1f%%\n", sum(var_exp) * 100))

  list(F = F_hat, loadings = loadings, var_exp = var_exp,
       mu = mu, sigma = sigma, X_clean = X,
       var_names_factor = colnames(X))
}

# =============================================================================
# SECTION 3 — VAR ESTIMATION
# =============================================================================

estimate_var <- function(Y, p) {
  T <- nrow(Y); K <- ncol(Y)
  # Construct regressors: intercept + lags 1..p
  X <- matrix(1, T - p, 1)
  for (i in seq_len(p))
    X <- cbind(X, Y[(p + 1 - i):(T - i), , drop = FALSE])
  Y_dep <- Y[(p + 1):T, , drop = FALSE]
  B     <- t(solve(crossprod(X), crossprod(X, Y_dep)))   # K x (Kp+1)
  resid <- Y_dep - X %*% t(B)
  Sigma <- crossprod(resid) / (T - p - K * p - 1)
  list(B = B, Sigma = Sigma, resid = resid)
}

var_companion <- function(B, K, p) {
  C <- matrix(0, K * p, K * p)
  C[1:K, ] <- B[, -1]      # drop intercept column
  if (p > 1)
    C[(K + 1):(K * p), 1:(K * (p - 1))] <- diag(K * (p - 1))
  C
}

# =============================================================================
# SECTION 4 — IMPULSE RESPONSES
# =============================================================================

compute_irf <- function(B, Sigma, K, p, horizon = 48,
                        shock_var = NULL, normalize_to = NULL) {
  if (is.null(shock_var)) shock_var <- K
  P   <- t(chol(Sigma))           # lower Cholesky
  C   <- var_companion(B, K, p)
  e   <- numeric(K); e[shock_var] <- 1
  imp <- P %*% e
  if (!is.null(normalize_to))
    imp <- imp * (normalize_to / imp[shock_var])

  irf   <- matrix(0, horizon, K)
  state <- numeric(K * p)
  state[1:K] <- imp
  for (h in seq_len(horizon)) {
    irf[h, ] <- state[1:K]
    state    <- C %*% state
  }
  irf
}

bootstrap_var_irf <- function(Y, p, horizon, n_boot = 500,
                               shock_var = NULL, alpha = 0.10,
                               normalize_to = 0.25) {
  T <- nrow(Y); K <- ncol(Y)
  fit       <- estimate_var(Y, p)
  irf_point <- compute_irf(fit$B, fit$Sigma, K, p, horizon,
                             shock_var, normalize_to)
  irf_store <- array(0, c(n_boot, horizon, K))

  for (b in seq_len(n_boot)) {
    idx     <- sample(nrow(fit$resid), replace = TRUE)
    resid_b <- fit$resid[idx, , drop = FALSE]
    Y_b     <- matrix(0, T, K)
    Y_b[1:p, ] <- Y[1:p, ]
    for (t in (p + 1):T) {
      # Explicit lag stacking — avoids ordering bugs
      Y_lag <- as.vector(t(Y_b[(t - 1):(t - p), , drop = FALSE]))
      Y_b[t, ] <- fit$B[, 1] + fit$B[, -1] %*% Y_lag + resid_b[t - p, ]
    }
    tryCatch({
      fb <- estimate_var(Y_b, p)
      irf_store[b, , ] <- compute_irf(fb$B, fb$Sigma, K, p, horizon,
                                       shock_var, normalize_to)
    }, error = function(e) {
      irf_store[b, , ] <<- irf_point
    })
  }

  irf_lo <- apply(irf_store, c(2, 3), quantile, probs = alpha / 2)
  irf_hi <- apply(irf_store, c(2, 3), quantile, probs = 1 - alpha / 2)
  list(point = irf_point, lo = irf_lo, hi = irf_hi)
}

recover_favar_irf <- function(irf_factors, loadings, sigma_vec,
                               var_names_factor, n_factors,
                               response_vars) {
  # eq. (8.29): ∂X_{i,t+h}/∂ε_t = σ_i · λ_i' · ∂F_{t+h}/∂ε_t
  irf_list <- list()
  for (var in response_vars) {
    idx <- which(var_names_factor == var)
    if (length(idx) == 1) {
      lam <- loadings[idx, , drop = TRUE]   # (n_factors,)
      irf_list[[var]] <- irf_factors[, 1:n_factors, drop = FALSE] %*%
                           lam * sigma_vec[idx]
    }
  }
  irf_list
}

# =============================================================================
# SECTION 5 — IRF ANALYSIS PER SUBSAMPLE
# =============================================================================

analyze_subsample <- function(fred, start_date, end_date, sample_name,
                               n_factors = 5, p = 12, horizon = 48,
                               n_boot = 500) {
  cat(sprintf("\n%s\n  %s\n  %s  →  %s\n%s\n",
              strrep("=", 60), sample_name, start_date, end_date,
              strrep("=", 60)))

  dl   <- prepare_data(fred, start_date, end_date)
  fact <- extract_factors(dl, n_factors = n_factors)

  VAR_VARS <- c("INDPRO", "CPIAUCSL", "FEDFUNDS")

  # Align factors with VAR variables by date string (rownames).
  # extract_factors() runs complete.cases() on the FULL panel MINUS FFR, so
  # fact$F rows are a subset of dl$data rows. We intersect the two date sets
  # to find the common complete observations, then subset both consistently.
  fact_dates <- rownames(fact$F)          # dates present after complete.cases
  var_dates  <- rownames(                 # dates where all 3 VAR series exist
                  na.omit(dl$data[, VAR_VARS, drop = FALSE]))
  common_dates <- intersect(fact_dates, var_dates)

  if (length(common_dates) == 0)
    stop(sprintf("No overlapping dates between factor matrix and VAR variables in %s",
                 sample_name))

  fact_idx <- match(common_dates, fact_dates)
  var_idx  <- match(common_dates, rownames(dl$data))

  F     <- fact$F[fact_idx, , drop = FALSE]
  Y_var <- dl$data[var_idx, VAR_VARS, drop = FALSE]
  cat(sprintf("    Final T = %d\n", nrow(Y_var)))

  Y_favar <- cbind(F, Y_var[, "FEDFUNDS", drop = FALSE])

  # VAR IRFs with bootstrap
  boot    <- bootstrap_var_irf(Y_var, p, horizon, n_boot,
                                shock_var = 3, normalize_to = 0.25)
  irf_var <- boot$point; irf_lo <- boot$lo; irf_hi <- boot$hi

  # FAVAR IRFs
  K_fav    <- ncol(Y_favar)
  fit_fav  <- estimate_var(Y_favar, p)
  irf_fav_raw <- compute_irf(fit_fav$B, fit_fav$Sigma, K_fav, p, horizon,
                              shock_var = K_fav, normalize_to = 0.25)
  irf_favar <- recover_favar_irf(irf_fav_raw, fact$loadings, fact$sigma,
                                  fact$var_names_factor, n_factors,
                                  c("INDPRO", "CPIAUCSL"))
  irf_favar[["FEDFUNDS"]] <- irf_fav_raw[, K_fav, drop = FALSE]

  var_cpi_pos   <- sum(irf_var[1:12, 2] > 0)
  favar_cpi_pos <- sum(irf_favar[["CPIAUCSL"]][1:12] > 0)
  cat(sprintf("    Price puzzle — VAR: %d/12 pos  FAVAR: %d/12 pos\n",
              var_cpi_pos, favar_cpi_pos))

  list(sample_name = sample_name, T = nrow(Y_var),
       irf_var = irf_var, irf_lo = irf_lo, irf_hi = irf_hi,
       irf_favar = irf_favar,
       var_cpi_pos = var_cpi_pos, favar_cpi_pos = favar_cpi_pos,
       Y_var = Y_var, F = F, Y_favar = Y_favar,
       fact = fact, p = p, n_factors = n_factors)
}

# =============================================================================
# SECTION 6 — ROLLING FORECASTING
# =============================================================================

rolling_forecasts <- function(Y_var, F, p = 12, h = 1,
                               initial_window = 120, n_factors = 5) {
  T       <- nrow(Y_var)
  Y_favar <- cbind(F, Y_var[, 3, drop = FALSE])
  K_var   <- ncol(Y_var)
  K_fav   <- ncol(Y_favar)

  fc_rw <- fc_var <- fc_favar <- actuals <- numeric(0)

  for (t in seq(initial_window, T - h)) {
    actual <- Y_var[t + h, 3]
    actuals <- c(actuals, actual)

    # Random Walk
    fc_rw <- c(fc_rw, Y_var[t, 3])

    # VAR
    fc_v <- tryCatch({
      fit   <- estimate_var(Y_var[1:t, ], p)
      state <- numeric(K_var * p)
      state[1:K_var] <- Y_var[t, ]
      fc_h <- Y_var[t, ]
      for (s in seq_len(h)) {
        fc_h  <- fit$B[, 1] + fit$B[, -1] %*% state
        state <- c(fc_h, state[1:(K_var * (p - 1))])
      }
      fc_h[3]
    }, error = function(e) Y_var[t, 3])
    fc_var <- c(fc_var, fc_v)

    # FAVAR
    fc_f <- tryCatch({
      fit   <- estimate_var(Y_favar[1:t, ], p)
      state <- numeric(K_fav * p)
      state[1:K_fav] <- Y_favar[t, ]
      fc_h  <- Y_favar[t, ]
      for (s in seq_len(h)) {
        fc_h  <- fit$B[, 1] + fit$B[, -1] %*% state
        state <- c(fc_h, state[1:(K_fav * (p - 1))])
      }
      fc_h[K_fav]
    }, error = function(e) Y_var[t, 3])
    fc_favar <- c(fc_favar, fc_f)
  }

  list(rw = fc_rw, var = fc_var, favar = fc_favar, actual = actuals)
}

# =============================================================================
# SECTION 7 — FORECAST EVALUATION TESTS
# =============================================================================

clark_west_test <- function(actual, fc_bench, fc_model) {
  # Clark & West (2007) MSPE-adjusted test for nested models.
  # H₀: benchmark forecasts as well as the larger model.
  # Reference: Clark & West (2007), Journal of Econometrics 138, 291-311.
  e1 <- actual - fc_bench
  e2 <- actual - fc_model
  c_t   <- e1^2 - (e2^2 - (fc_bench - fc_model)^2)
  c_bar <- mean(c_t)
  se    <- sd(c_t) / sqrt(length(c_t))
  cw    <- if (se > 0) c_bar / se else NA_real_
  pval  <- if (!is.na(cw)) 1 - pnorm(cw) else NA_real_   # one-sided
  list(stat = cw, pval = pval,
       mspe_bench = mean(e1^2), mspe_model = mean(e2^2))
}

giacomini_rossi_test <- function(actual, fc1, fc2,
                                  window = NULL, alpha = 0.10) {
  # Giacomini & Rossi (2010) fluctuation test for forecast stability.
  # Reference: Giacomini & Rossi (2010), Review of Economic Studies 77, 530-561.
  e1 <- actual - fc1
  e2 <- actual - fc2
  d  <- e1^2 - e2^2
  n  <- length(d)
  if (is.null(window)) window <- max(floor(n * 0.2), 10)

  # Long-run variance (Newey-West, bandwidth = window-1)
  d_dm <- d - mean(d)
  bw   <- window - 1
  lrv  <- var(d)
  for (k in seq_len(bw)) {
    w   <- 1 - k / (bw + 1)
    lrv <- lrv + 2 * w * mean(d_dm[(k + 1):n] * d_dm[1:(n - k)])
  }
  lrv <- max(lrv, 1e-12)

  # Rolling means and GR series
  rolling  <- sapply(seq(window, n),
                     function(t) mean(d[(t - window + 1):t]))
  gr_s     <- sqrt(window) * rolling / sqrt(lrv)
  gr_stat  <- max(abs(gr_s))

  # Critical values from G-R (2010) Table 1 (two-sided, sup norm)
  cv_table <- c("0.10" = 2.49, "0.05" = 2.80, "0.01" = 3.40)
  cv <- cv_table[as.character(alpha)]

  list(stat = gr_stat, cv = cv, gr_series = gr_s,
       time_idx = seq(window, n))
}

compute_forecast_metrics <- function(fc_list) {
  actual  <- fc_list$actual
  metrics <- lapply(c(rw = "rw", var = "var", favar = "favar"),
                    function(m) {
                      e <- actual - fc_list[[m]]
                      list(RMSE = sqrt(mean(e^2)), MAE = mean(abs(e)))
                    })

  cw_var_rw    <- clark_west_test(actual, fc_list$rw,  fc_list$var)
  cw_favar_rw  <- clark_west_test(actual, fc_list$rw,  fc_list$favar)
  cw_favar_var <- clark_west_test(actual, fc_list$var, fc_list$favar)

  gr_var_rw    <- giacomini_rossi_test(actual, fc_list$rw,  fc_list$var)
  gr_favar_rw  <- giacomini_rossi_test(actual, fc_list$rw,  fc_list$favar)
  gr_favar_var <- giacomini_rossi_test(actual, fc_list$var, fc_list$favar)

  list(metrics = metrics,
       CW = list(VAR_vs_RW    = cw_var_rw,
                 FAVAR_vs_RW  = cw_favar_rw,
                 FAVAR_vs_VAR = cw_favar_var),
       GR = list(VAR_vs_RW    = gr_var_rw,
                 FAVAR_vs_RW  = gr_favar_rw,
                 FAVAR_vs_VAR = gr_favar_var))
}

# =============================================================================
# SECTION 8 — PRINT TABLE
# =============================================================================

print_forecast_table <- function(eval_res) {
  cat("\n", strrep("=", 65), "\n")
  cat("OUT-OF-SAMPLE FORECAST EVALUATION  (target: FFR, h=1)\n")
  cat(strrep("=", 65), "\n")
  cat(sprintf("%-10s %10s %10s\n", "Model", "RMSE", "MAE"))
  cat(strrep("-", 32), "\n")
  for (m in c("rw", "var", "favar")) {
    cat(sprintf("%-10s %10.4f %10.4f\n",
                toupper(m),
                eval_res$metrics[[m]]$RMSE,
                eval_res$metrics[[m]]$MAE))
  }

  cat("\n--- Clark-West (2007) [H₁: model > benchmark, one-sided] ---\n")
  cat(sprintf("  %-20s %10s %10s %5s\n",
              "Comparison", "CW stat", "p-value", "sig"))
  cat("  ", strrep("-", 50), "\n")
  cw_rows <- list(c("VAR vs RW",    "VAR_vs_RW"),
                  c("FAVAR vs RW",  "FAVAR_vs_RW"),
                  c("FAVAR vs VAR", "FAVAR_vs_VAR"))
  for (row in cw_rows) {
    r <- eval_res$CW[[row[2]]]
    stars <- ifelse(r$pval < 0.01, "***",
               ifelse(r$pval < 0.05, "**",
                 ifelse(r$pval < 0.10, "*", "")))
    cat(sprintf("  %-20s %10.3f %10.3f %5s\n",
                row[1], r$stat, r$pval, stars))
  }

  cat("\n--- Giacomini-Rossi (2010) Fluctuation Test ---\n")
  cat(sprintf("  %-20s %10s %10s %8s\n",
              "Comparison", "GR stat", "CV (10%)", "stable?"))
  cat("  ", strrep("-", 52), "\n")
  gr_rows <- list(c("VAR vs RW",    "VAR_vs_RW"),
                  c("FAVAR vs RW",  "FAVAR_vs_RW"),
                  c("FAVAR vs VAR", "FAVAR_vs_VAR"))
  for (row in gr_rows) {
    r <- eval_res$GR[[row[2]]]
    stable <- ifelse(r$stat < r$cv, "YES", "NO")
    cat(sprintf("  %-20s %10.3f %10.3f %8s\n",
                row[1], r$stat, r$cv, stable))
  }
  cat(strrep("=", 65), "\n")
}

# =============================================================================
# SECTION 9 — PLOTTING
# =============================================================================

plot_irf_comparison <- function(results_list, prefix = "bbe_favar") {
  months <- seq_len(nrow(results_list[[1]]$irf_var)) - 1
  df_list <- lapply(results_list, function(res) {
    data.frame(
      Month      = months,
      VAR_CPI    = res$irf_var[, 2] * 100,
      VAR_CPI_lo = res$irf_lo[, 2]  * 100,
      VAR_CPI_hi = res$irf_hi[, 2]  * 100,
      FAV_CPI    = as.vector(res$irf_favar[["CPIAUCSL"]]) * 100,
      VAR_IP     = res$irf_var[, 1] * 100,
      FAV_IP     = as.vector(res$irf_favar[["INDPRO"]]) * 100,
      Sample     = res$sample_name
    )
  })
  df <- bind_rows(df_list)
  df$Sample <- factor(df$Sample,
                       levels = sapply(results_list, `[[`, "sample_name"))

  p_cpi <- ggplot(df, aes(x = Month)) +
    facet_wrap(~Sample, nrow = 1) +
    geom_ribbon(aes(ymin = VAR_CPI_lo, ymax = VAR_CPI_hi),
                fill = "#1f77b4", alpha = 0.20) +
    geom_line(aes(y = VAR_CPI, colour = "VAR"), lwd = 1.0) +
    geom_line(aes(y = FAV_CPI, colour = "FAVAR"), lwd = 1.0, lty = 2) +
    geom_hline(yintercept = 0, colour = "black", lwd = 0.5) +
    annotate("rect", xmin = 0, xmax = 12,
             ymin = -Inf, ymax = Inf, alpha = 0.06, fill = "red") +
    scale_colour_manual(values = c(VAR = "#1f77b4", FAVAR = "#d62728")) +
    coord_cartesian(clip = "off") +
    labs(y = "CPI Response (%)", x = NULL, colour = NULL,
         title = "CPI Response (Price Puzzle Test)") +
    theme_bw(base_size = 11) +
    theme(legend.position = "bottom",
          strip.text = element_text(face = "bold"))

  p_ip <- ggplot(df, aes(x = Month)) +
    facet_wrap(~Sample, nrow = 1) +
    geom_ribbon(aes(ymin = VAR_CPI_lo, ymax = VAR_CPI_hi),
                fill = "#1f77b4", alpha = 0.20) +
    geom_line(aes(y = VAR_IP, colour = "VAR"), lwd = 1.0) +
    geom_line(aes(y = FAV_IP, colour = "FAVAR"), lwd = 1.0, lty = 2) +
    geom_hline(yintercept = 0, colour = "black", lwd = 0.5) +
    scale_colour_manual(values = c(VAR = "#1f77b4", FAVAR = "#d62728")) +
    coord_cartesian(clip = "off") +
    labs(y = "IP Response (%)", x = "Months", colour = NULL,
         title = "Industrial Production Response") +
    theme_bw(base_size = 11) +
    theme(legend.position = "bottom",
          strip.text = element_text(face = "bold"))

  suppressMessages({
    library(gridExtra)
    g <- gridExtra::grid.arrange(p_cpi, p_ip, nrow = 2,
           top = "VAR vs FAVAR Across Monetary Policy Regimes\n\
Response to 25bp Federal Funds Rate Increase")
    ggsave(paste0(prefix, "_irf.pdf"), g, width = 14, height = 9)
    ggsave(paste0(prefix, "_irf.png"), g, width = 14, height = 9, dpi = 150)
  })
  cat("  Saved:", paste0(prefix, "_irf.pdf/.png"), "\n")
}

plot_forecast_evaluation <- function(eval_res, fc_list, prefix = "bbe_favar") {
  metrics <- eval_res$metrics
  df_bar  <- data.frame(
    Model = factor(c("RW","VAR","FAVAR"), levels = c("RW","VAR","FAVAR")),
    RMSE  = c(metrics$rw$RMSE, metrics$var$RMSE, metrics$favar$RMSE)
  )

  # CW annotations
  cw_ann <- do.call(rbind, lapply(
    list(c("VAR vs RW", "VAR_vs_RW", 2),
         c("FAVAR vs RW", "FAVAR_vs_RW", 3),
         c("FAVAR vs VAR", "FAVAR_vs_VAR", 2.5)),
    function(row) {
      r  <- eval_res$CW[[row[2]]]
      st <- ifelse(r$pval < 0.01, "***",
              ifelse(r$pval < 0.05, "**",
                ifelse(r$pval < 0.10, "*", "")))
      data.frame(label = paste0("CW: ", round(r$stat, 2), st),
                 x = as.numeric(row[3]),
                 y = max(df_bar$RMSE) * 1.12,
                 stringsAsFactors = FALSE)
    }))

  p1 <- ggplot(df_bar, aes(x = Model, y = RMSE, fill = Model)) +
    geom_col(alpha = 0.85, colour = "black", lwd = 0.4) +
    geom_text(aes(label = round(RMSE, 4)), vjust = -0.5, size = 3.5) +
    geom_text(data = cw_ann, aes(x = x, y = y, label = label),
              inherit.aes = FALSE, size = 3, colour = "dimgrey") +
    scale_fill_manual(values = c(RW = "#1f77b4", VAR = "#2ca02c",
                                  FAVAR = "#d62728")) +
    coord_cartesian(ylim = c(0, max(df_bar$RMSE) * 1.25), clip = "off") +
    labs(y = "RMSE (FFR)", title = "Forecast Accuracy") +
    theme_bw(base_size = 11) + theme(legend.position = "none")

  # GR fluctuation paths
  gr_list <- eval_res$GR
  df_gr   <- bind_rows(
    data.frame(t    = gr_list$VAR_vs_RW$time_idx,
               gr   = gr_list$VAR_vs_RW$gr_series,
               comp = "VAR vs RW"),
    data.frame(t    = gr_list$FAVAR_vs_RW$time_idx,
               gr   = gr_list$FAVAR_vs_RW$gr_series,
               comp = "FAVAR vs RW"),
    data.frame(t    = gr_list$FAVAR_vs_VAR$time_idx,
               gr   = gr_list$FAVAR_vs_VAR$gr_series,
               comp = "FAVAR vs VAR")
  )
  cv_val <- gr_list$FAVAR_vs_RW$cv

  gr_ylim <- c(min(df_gr$gr, -cv_val) * 1.15, max(df_gr$gr, cv_val) * 1.15)

  p2 <- ggplot(df_gr, aes(x = t, y = gr, colour = comp, lty = comp)) +
    geom_line(lwd = 1.2) +
    geom_hline(yintercept =  cv_val, lty = 2, colour = "black", lwd = 1) +
    geom_hline(yintercept = -cv_val, lty = 2, colour = "black", lwd = 1) +
    geom_hline(yintercept = 0, colour = "grey50", lwd = 0.6) +
    scale_colour_manual(values = c("VAR vs RW"    = "#1f77b4",
                                    "FAVAR vs RW"  = "#d62728",
                                    "FAVAR vs VAR" = "#2ca02c")) +
    labs(x = "Rolling window end", y = "GR statistic",
         colour = NULL, lty = NULL,
         title = "Giacomini-Rossi (2010) Fluctuation Test",
         subtitle = paste0("Dashed lines: ±", cv_val, " critical value (10%)")) +
    coord_cartesian(ylim = gr_ylim, clip = "off") +
    theme_bw(base_size = 11) +
    theme(legend.position = "bottom")

  suppressMessages({
    library(gridExtra)
    g <- gridExtra::grid.arrange(p1, p2, nrow = 1,
           top = "Out-of-Sample Forecast Comparison: FFR\n\
RW vs VAR vs FAVAR  |  Clark-West & Giacomini-Rossi Tests")
    ggsave(paste0(prefix, "_forecast_eval.pdf"), g, width = 14, height = 6)
    ggsave(paste0(prefix, "_forecast_eval.png"), g, width = 14, height = 6,
           dpi = 150)
  })
  cat("  Saved:", paste0(prefix, "_forecast_eval.pdf/.png"), "\n")
}

# =============================================================================
# MAIN
# =============================================================================

main <- function() {
  cat(strrep("=", 70), "\n")
  cat("BBE (2005) FAVAR Replication with FRED-MD Data\n")
  cat("Section 8.13.5  |  Macroeconometrics  |  Alessia Paccagnini\n")
  cat(strrep("=", 70), "\n")

  # Load data
  DATA_FILE <- "2025-12-MD.csv"
  cat(sprintf("\n1. Loading %s ...\n", DATA_FILE))
  fred <- load_fred_md(DATA_FILE)
  cat(sprintf("   Raw data: %d obs, %d variables\n",
              nrow(fred$data), ncol(fred$data)))
  cat(sprintf("   Range  : %s — %s\n",
              min(fred$dates), max(fred$dates)))

  # Subsample IRF analysis
  SUBSAMPLES <- list(
    list("1962-01-01", "1984-12-01", "Pre-Volcker Era (1962-1984)"),
    list("1985-01-01", "2007-12-01", "Great Moderation (1985-2007)"),
    list("1962-01-01", "2007-12-01", "Full Sample (1962-2007)")
  )

  cat("\n2. Subsample IRF analysis ...\n")
  results_irf <- lapply(SUBSAMPLES, function(s)
    analyze_subsample(fred, s[[1]], s[[2]], s[[3]],
                      n_factors = 5, p = 12, horizon = 48, n_boot = 500))

  plot_irf_comparison(results_irf)

  # Price puzzle summary
  cat("\n", strrep("=", 65), "\n")
  cat("PRICE PUZZLE SUMMARY\n")
  cat(strrep("=", 65), "\n")
  cat(sprintf("%-35s %5s %9s %10s\n", "Sample", "T", "VAR pos", "FAVAR pos"))
  cat(strrep("-", 65), "\n")
  for (res in results_irf)
    cat(sprintf("%-35s %5d %6d/12 %7d/12\n",
                res$sample_name, res$T,
                res$var_cpi_pos, res$favar_cpi_pos))

  # Rolling forecast comparison on Full Sample
  cat("\n3. Rolling forecasting on Full Sample (h=1, initial_window=120) ...\n")
  full   <- results_irf[[3]]
  fc_out <- rolling_forecasts(full$Y_var, full$F,
                               p = full$p, h = 1,
                               initial_window = 120,
                               n_factors = full$n_factors)
  cat(sprintf("   Forecast origins: %d\n", length(fc_out$actual)))

  # Tests
  cat("\n4. Forecast metrics and test statistics ...\n")
  eval_res <- compute_forecast_metrics(fc_out)
  print_forecast_table(eval_res)

  # Figures
  cat("\n5. Saving figures ...\n")
  plot_forecast_evaluation(eval_res, fc_out)

  # Save CSVs
  months <- seq_len(48) - 1
  for (res in results_irf) {
    tag <- gsub("[ ()\\-]", "_", res$sample_name)
    write.csv(data.frame(
      Month     = months,
      VAR_IP    = res$irf_var[, 1] * 100,
      VAR_CPI   = res$irf_var[, 2] * 100,
      VAR_FFR   = res$irf_var[, 3],
      FAVAR_IP  = as.vector(res$irf_favar[["INDPRO"]]) * 100,
      FAVAR_CPI = as.vector(res$irf_favar[["CPIAUCSL"]]) * 100
    ), file = paste0("bbe_irf_", tag, ".csv"), row.names = FALSE)
  }
  write.csv(data.frame(Actual = fc_out$actual, RW = fc_out$rw,
                        VAR = fc_out$var, FAVAR = fc_out$favar),
            file = "bbe_forecast_results.csv", row.names = FALSE)

  cat("\n", strrep("=", 70), "\n")
  cat("Done. Files generated:\n")
  cat("  bbe_favar_irf.pdf/.png           — Figure 8.1 (IRF comparison)\n")
  cat("  bbe_favar_forecast_eval.pdf/.png — Forecast evaluation figure\n")
  cat("  bbe_irf_*.csv                    — Numerical IRF results\n")
  cat("  bbe_forecast_results.csv         — Rolling forecast series\n")
  cat(strrep("=", 70), "\n")
}

main()
