# =============================================================================
# Empirical Application: Forecasting U.S. GDP Growth and Inflation
# Traditional Methods vs. Machine Learning
# =============================================================================
#
# Author   : Alessia Paccagnini
# Textbook : Macroeconometrics
#
# DATA  (FRED):
#   GDP.xlsx      — Real GDP (GDPC1), quarterly
#   GDPDEFL.xlsx  — GDP Deflator (GDPDEF), quarterly
#   FFR.xlsx      — Federal Funds Rate (FEDFUNDS), monthly -> quarterly avg
#
# SAMPLE: 1954:Q4 - 2025:Q3  (284 quarterly observations)
# TRANSFORMATIONS:
#   GDP Growth = 400 * diff(log(GDPC1))   [annualised %]
#   Inflation  = 400 * diff(log(GDPDEF))  [annualised %]
#   FedFunds   = level                    [%]
#
# METHODS:
#   Traditional : VAR(4), BVAR(4) with Minnesota prior (lambda1 = 0.2)
#   ML          : LASSO, Ridge, Elastic Net (5-fold CV), Random Forest
#
# EVALUATION:
#   Window    : expanding
#   OOS       : final 60 observations ~2010:Q1-2025:Q3
#   Horizons  : h = 1, 4 quarters ahead
#   Point     : RMSE, MAE
#   Density   : 90% PI coverage, width
#   Sig.      : Diebold-Mariano vs VAR
#
# PACKAGES REQUIRED:
#   readxl    — Excel file loading
#   dplyr     — data manipulation
#   lubridate — date handling
#   glmnet    — LASSO, Ridge, Elastic Net
#   ranger    — Random Forest (fast, recommended over randomForest)
#   ggplot2   — figures
#   gridExtra — multi-panel figures
#
# Install once with:
#   install.packages(c("readxl","dplyr","lubridate","glmnet","ranger",
#                      "ggplot2","gridExtra"))
# =============================================================================

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(lubridate)
  library(glmnet)
  library(ranger)
  library(ggplot2)
  library(gridExtra)
})

set.seed(42)

cat(paste(rep("=", 70), collapse = ""), "\n")
cat("Forecasting U.S. GDP Growth and Inflation\n")
cat("Traditional Methods vs. Machine Learning\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")


# =============================================================================
# SECTION 1: DATA LOADING AND PREPARATION
# =============================================================================

load_fred_data <- function(gdp_path, deflator_path, fedfunds_path) {
  # ---------------------------------------------------------------------------
  # Load FRED Excel files and return a quarterly data.frame with columns:
  #   GDP_growth  — 400 * diff(log(GDPC1))  [annualised %]
  #   Inflation   — 400 * diff(log(GDPDEF)) [annualised %]
  #   FedFunds    — Federal Funds Rate level [%]
  # Sample restricted to 1954:Q4 – 2025:Q3 (284 observations).
  # ---------------------------------------------------------------------------

  # --- Real GDP (quarterly) ---
  gdp <- read_excel(gdp_path, col_names = FALSE, skip = 1)
  colnames(gdp) <- c("date", "GDPC1")
  gdp$date  <- as.Date(gdp$date)
  gdp$GDPC1 <- as.numeric(gdp$GDPC1)
  gdp$quarter <- floor_date(gdp$date, unit = "quarter")

  # --- GDP Deflator (quarterly) ---
  defl <- read_excel(deflator_path, col_names = FALSE, skip = 1)
  colnames(defl) <- c("date", "GDPDEF")
  defl$date   <- as.Date(defl$date)
  defl$GDPDEF <- as.numeric(defl$GDPDEF)
  defl$quarter <- floor_date(defl$date, unit = "quarter")

  # --- Federal Funds Rate (monthly -> quarterly average) ---
  ffr <- read_excel(fedfunds_path, col_names = FALSE, skip = 1)
  colnames(ffr) <- c("date", "FEDFUNDS")
  ffr$date     <- as.Date(ffr$date)
  ffr$FEDFUNDS <- as.numeric(ffr$FEDFUNDS)
  ffr$quarter  <- floor_date(ffr$date, unit = "quarter")
  ffr_q <- ffr %>%
    group_by(quarter) %>%
    summarise(FedFunds = mean(FEDFUNDS, na.rm = TRUE), .groups = "drop")

  # --- Merge on quarterly dates ---
  df <- gdp %>%
    select(quarter, GDPC1) %>%
    inner_join(defl %>% select(quarter, GDPDEF), by = "quarter") %>%
    inner_join(ffr_q, by = "quarter") %>%
    arrange(quarter)

  # --- Transformations ---
  df <- df %>%
    mutate(
      GDP_growth = c(NA, diff(log(GDPC1)))  * 400,
      Inflation  = c(NA, diff(log(GDPDEF))) * 400
    ) %>%
    filter(!is.na(GDP_growth)) %>%
    select(quarter, GDP_growth, Inflation, FedFunds)

  # --- Restrict to book sample: 1954:Q4 – 2025:Q3 ---
  df <- df %>%
    filter(quarter >= as.Date("1954-10-01") &
           quarter <= as.Date("2025-07-01"))

  cat(sprintf("Sample: %s – %s  (%d observations)\n",
              format(min(df$quarter), "%Y:Q%q"),
              format(max(df$quarter), "%Y:Q%q"),
              nrow(df)))
  cat("Variables: GDP_growth [ann. %], Inflation [ann. %], FedFunds [%]\n\n")

  return(df)
}


prepare_forecast_data <- function(DATA, target_col, h, p) {
  # ---------------------------------------------------------------------------
  # Build direct h-step-ahead forecast matrices.
  #
  #   y[t]   = DATA[t + h, target_col]
  #   X[t, ] = c(DATA[t, ], DATA[t-1, ], ..., DATA[t-p+1, ])  (all variables)
  #
  # Valid rows: t = p+1, ..., T-h  (1-based)
  # Returns list(y, X) as plain matrices (no dates needed in loops).
  # ---------------------------------------------------------------------------
  T      <- nrow(DATA)
  n_vars <- ncol(DATA)

  t_start <- p + 1          # first valid row (1-based)
  t_end   <- T - h          # last valid row
  N       <- t_end - t_start + 1

  # Target: value h steps ahead
  y <- DATA[(t_start + h - 1):(t_end + h - 1), target_col]

  # Feature matrix: p lags of all variables
  X <- matrix(0, nrow = N, ncol = p * n_vars)
  for (lag in 1:p) {
    col_start <- (lag - 1) * n_vars + 1
    col_end   <- lag * n_vars
    rows_from <- (t_start - lag):(t_end - lag)
    X[, col_start:col_end] <- DATA[rows_from, ]
  }

  list(y = y, X = X)
}


# =============================================================================
# SECTION 2: MODEL FUNCTIONS
# =============================================================================

# ---- VAR (OLS with intercept) -----------------------------------------------
fit_var <- function(y, X) {
  # Returns list(beta, sigma)
  # beta  : (k+1) vector  [intercept; slopes]
  # sigma : residual std  matching Python ddof = ncol(X_aug)
  X_aug <- cbind(1, X)
  beta  <- as.vector(qr.solve(X_aug, y))
  resid <- y - X_aug %*% beta
  k     <- ncol(X_aug)
  sigma <- sqrt(sum(resid^2) / (length(resid) - k))
  list(beta = beta, sigma = sigma)
}

predict_var <- function(fit, X_te) {
  x_aug <- c(1, X_te)
  as.numeric(x_aug %*% fit$beta)
}

predict_interval_var <- function(fit, X_te, alpha = 0.1) {
  fc <- predict_var(fit, X_te)
  z  <- qnorm(1 - alpha / 2)
  c(lb = fc - z * fit$sigma, ub = fc + z * fit$sigma)
}


# ---- BVAR (Minnesota prior) -------------------------------------------------
fit_bvar <- function(y, X, p, n_vars,
                     lam1 = 0.2, lam2 = 0.5, lam3 = 1.0) {
  # Conjugate normal-normal posterior.
  # Prior variance for coefficient j:
  #   lag     = ceiling(j / n_vars)
  #   var_idx = (j - 1) %% n_vars        (0-based)
  #   v_j     = (lam1 / lag^lam3)^2  * lam2^2  if cross-variable
  #           = (lam1 / lag^lam3)^2             if own-variable
  # Intercept prior variance = 100 (diffuse).
  T <- length(y)
  k <- ncol(X)

  prior_var <- numeric(k)
  for (j in seq_len(k)) {
    lag     <- ceiling(j / n_vars)
    var_idx <- (j - 1) %% n_vars          # 0-based
    prior_var[j] <- (lam1 / lag^lam3)^2
    if (var_idx > 0) prior_var[j] <- prior_var[j] * lam2^2
  }

  X_aug          <- cbind(1, X)
  prior_var_full <- c(100, prior_var)
  prior_mean     <- rep(0, k + 1)

  V_prior_inv <- diag(1 / prior_var_full)
  V_post_inv  <- V_prior_inv + crossprod(X_aug)
  V_post      <- solve(V_post_inv)
  beta        <- as.vector(V_post %*% (V_prior_inv %*% prior_mean +
                                       t(X_aug) %*% y))

  resid <- y - X_aug %*% beta
  sigma <- sd(resid)            # ddof = 1, matches Python

  list(beta = beta, sigma = sigma, V_post = V_post)
}

predict_bvar <- function(fit, X_te) {
  x_aug <- c(1, X_te)
  as.numeric(x_aug %*% fit$beta)
}

predict_interval_bvar <- function(fit, X_te, alpha = 0.1) {
  x_aug    <- c(1, X_te)
  fc       <- as.numeric(x_aug %*% fit$beta)
  pred_var <- as.numeric(x_aug %*% fit$V_post %*% x_aug) + fit$sigma^2
  z        <- qnorm(1 - alpha / 2)
  c(lb = fc - z * sqrt(pred_var), ub = fc + z * sqrt(pred_var))
}


# ---- Standardise / unstandardise --------------------------------------------
standardise <- function(X) {
  mu <- colMeans(X)
  sd <- apply(X, 2, sd)
  sd[sd == 0] <- 1
  list(X_s = scale(X, center = mu, scale = sd), mu = mu, sd = sd)
}

apply_scale <- function(X_te, mu, sd) {
  as.numeric((X_te - mu) / sd)
}


# ---- LASSO (5-fold CV via glmnet) -------------------------------------------
fit_lasso_cv <- function(y, X_s) {
  fit   <- cv.glmnet(X_s, y, alpha = 1, nfolds = 5,
                     standardize = FALSE)   # already standardised
  lam   <- fit$lambda.min
  coef_v <- as.vector(coef(fit, s = lam))  # (intercept, slopes)
  resid  <- y - cbind(1, X_s) %*% coef_v
  list(beta = coef_v, sigma = sd(resid))
}


# ---- Ridge (5-fold CV via glmnet) -------------------------------------------
fit_ridge_cv <- function(y, X_s) {
  fit   <- cv.glmnet(X_s, y, alpha = 0, nfolds = 5,
                     standardize = FALSE)
  lam   <- fit$lambda.min
  coef_v <- as.vector(coef(fit, s = lam))
  resid  <- y - cbind(1, X_s) %*% coef_v
  list(beta = coef_v, sigma = sd(resid))
}


# ---- Elastic Net (5-fold CV, grid over alpha) --------------------------------
fit_elasticnet_cv <- function(y, X_s) {
  alphas   <- c(0.1, 0.5, 0.7, 0.9, 0.95, 0.99)
  best_mse <- Inf
  best_fit <- NULL

  for (a in alphas) {
    fit <- tryCatch(
      cv.glmnet(X_s, y, alpha = a, nfolds = 5, standardize = FALSE),
      error = function(e) NULL
    )
    if (!is.null(fit)) {
      mse_min <- min(fit$cvm)
      if (mse_min < best_mse) {
        best_mse <- mse_min
        best_fit <- fit
      }
    }
  }

  lam    <- best_fit$lambda.min
  coef_v <- as.vector(coef(best_fit, s = lam))
  resid  <- y - cbind(1, X_s) %*% coef_v
  list(beta = coef_v, sigma = sd(resid))
}

# Shared predict for all glmnet-based models
predict_penalised <- function(fit, X_te_s) {
  as.numeric(c(1, X_te_s) %*% fit$beta)
}

predict_interval_penalised <- function(fit, X_te_s, alpha = 0.1) {
  fc <- predict_penalised(fit, X_te_s)
  z  <- qnorm(1 - alpha / 2)
  c(lb = fc - z * fit$sigma, ub = fc + z * fit$sigma)
}


# ---- Random Forest (ranger, 500 trees) --------------------------------------
fit_random_forest <- function(y, X_s) {
  df_tr  <- as.data.frame(X_s)
  df_tr$.y <- y
  mdl    <- ranger(
    formula        = .y ~ .,
    data           = df_tr,
    num.trees      = 500,
    min.node.size  = 5,
    mtry           = max(1, floor(sqrt(ncol(X_s)))),
    seed           = 42,
    num.threads    = 1
  )
  y_fitted <- mdl$predictions
  resid    <- y - y_fitted
  sigma_rf <- sd(resid)
  list(mdl = mdl, sigma = sigma_rf, resid = resid)
}

predict_interval_rf <- function(fit, X_te_s, alpha = 0.1) {
  # Tree-level predictions via ranger predict with predict.all = TRUE
  df_te  <- as.data.frame(matrix(X_te_s, nrow = 1))
  colnames(df_te) <- fit$mdl$forest$independent.variable.names
  pred   <- predict(fit$mdl, data = df_te, predict.all = TRUE)
  tree_preds <- as.vector(pred$predictions)
  fc         <- mean(tree_preds)
  tree_var   <- var(tree_preds)
  total_std  <- sqrt(tree_var + fit$sigma^2)
  z          <- qnorm(1 - alpha / 2)
  c(fc = fc, lb = fc - z * total_std, ub = fc + z * total_std)
}


# =============================================================================
# SECTION 3: EVALUATION METRICS
# =============================================================================

rmse_fn     <- function(y, yhat) sqrt(mean((y - yhat)^2))
mae_fn      <- function(y, yhat) mean(abs(y - yhat))
coverage_fn <- function(y, lb, ub) mean(y >= lb & y <= ub)
width_fn    <- function(lb, ub) mean(ub - lb)


# =============================================================================
# DIEBOLD-MARIANO TEST
# =============================================================================

diebold_mariano <- function(e1, e2, h) {
  # H0: E[d_t] = 0,  d_t = e1_t^2 - e2_t^2
  # HAC variance: Newey-West, bandwidth = h - 1
  # Negative DM -> model 1 outperforms model 2.
  d     <- e1^2 - e2^2
  d     <- d[!is.na(d)]
  Tn    <- length(d)
  d_bar <- mean(d)

  gamma0    <- var(d)                   # ddof = 1
  gamma_sum <- 0
  bw        <- h - 1
  if (bw > 0) {
    for (k in seq_len(bw)) {
      dc  <- d - d_bar
      gk  <- mean(dc[(k + 1):Tn] * dc[1:(Tn - k)])
      w   <- 1 - k / (bw + 1)          # Bartlett kernel
      gamma_sum <- gamma_sum + 2 * w * gk
    }
  }

  var_dbar <- (gamma0 + gamma_sum) / Tn
  if (!is.na(var_dbar) && var_dbar > 0) {
    DM      <- d_bar / sqrt(var_dbar)
    p_value <- 2 * (1 - pnorm(abs(DM)))
  } else {
    DM <- NA; p_value <- NA
  }
  list(DM = DM, p_value = p_value)
}

sig_stars <- function(p) {
  if (is.na(p))    return("")
  if (p < 0.01)    return("***")
  if (p < 0.05)    return("**")
  if (p < 0.10)    return("*")
  return("")
}


# =============================================================================
# SECTION 4: EXPANDING-WINDOW FORECASTING LOOP
# =============================================================================

run_forecasting_comparison <- function(df, target_var,
                                       horizons  = c(1, 4),
                                       test_size = 60,
                                       p         = 4,
                                       alpha     = 0.10,
                                       verbose   = TRUE) {
  methods   <- c("VAR", "BVAR", "LASSO", "Ridge", "ElasticNet", "RandomForest")
  n_methods <- length(methods)

  # Column index of target variable in DATA matrix
  var_names  <- c("GDP_growth", "Inflation", "FedFunds")
  target_col <- which(var_names == target_var)
  DATA       <- as.matrix(df[, var_names])
  n_vars     <- ncol(DATA)

  # Minnesota prior params
  lam1 <- 0.2; lam2 <- 0.5; lam3 <- 1.0

  results <- list()

  for (h in horizons) {
    if (verbose) {
      cat(sprintf("\n%s\nForecasting %s at horizon h=%d\n%s\n",
                  strrep("=", 60), target_var, h, strrep("=", 60)))
    }

    fd  <- prepare_forecast_data(DATA, target_col, h, p)
    y_full <- fd$y
    X_full <- fd$X
    T      <- length(y_full)
    T0     <- T - test_size

    # Storage matrices
    fc_mat <- matrix(0, test_size, n_methods,
                     dimnames = list(NULL, methods))
    lb_mat <- fc_mat
    ub_mat <- fc_mat
    act    <- numeric(test_size)

    for (t in seq_len(test_size)) {
      train_end <- T0 + t - 1          # last training index (1-based)

      y_tr <- y_full[1:train_end]
      X_tr <- X_full[1:train_end, , drop = FALSE]
      y_te <- y_full[train_end + 1]
      X_te <- X_full[train_end + 1, ]
      act[t] <- y_te

      if (verbose && (t - 1) %% 20 == 0)
        cat(sprintf("  Forecast origin %d/%d\n", t, test_size))

      # ---- VAR ----
      fv <- fit_var(y_tr, X_tr)
      fc_mat[t, "VAR"] <- predict_var(fv, X_te)
      pi <- predict_interval_var(fv, X_te, alpha)
      lb_mat[t, "VAR"] <- pi["lb"]; ub_mat[t, "VAR"] <- pi["ub"]

      # ---- BVAR ----
      fb <- fit_bvar(y_tr, X_tr, p, n_vars, lam1, lam2, lam3)
      fc_mat[t, "BVAR"] <- predict_bvar(fb, X_te)
      pi <- predict_interval_bvar(fb, X_te, alpha)
      lb_mat[t, "BVAR"] <- pi["lb"]; ub_mat[t, "BVAR"] <- pi["ub"]

      # ---- Standardise features for ML methods ----
      sc    <- standardise(X_tr)
      X_tr_s <- sc$X_s
      X_te_s <- apply_scale(X_te, sc$mu, sc$sd)

      # ---- LASSO ----
      fl <- fit_lasso_cv(y_tr, X_tr_s)
      fc_mat[t, "LASSO"] <- predict_penalised(fl, X_te_s)
      pi <- predict_interval_penalised(fl, X_te_s, alpha)
      lb_mat[t, "LASSO"] <- pi["lb"]; ub_mat[t, "LASSO"] <- pi["ub"]

      # ---- Ridge ----
      fr <- fit_ridge_cv(y_tr, X_tr_s)
      fc_mat[t, "Ridge"] <- predict_penalised(fr, X_te_s)
      pi <- predict_interval_penalised(fr, X_te_s, alpha)
      lb_mat[t, "Ridge"] <- pi["lb"]; ub_mat[t, "Ridge"] <- pi["ub"]

      # ---- Elastic Net ----
      fe <- fit_elasticnet_cv(y_tr, X_tr_s)
      fc_mat[t, "ElasticNet"] <- predict_penalised(fe, X_te_s)
      pi <- predict_interval_penalised(fe, X_te_s, alpha)
      lb_mat[t, "ElasticNet"] <- pi["lb"]; ub_mat[t, "ElasticNet"] <- pi["ub"]

      # ---- Random Forest ----
      frf <- fit_random_forest(y_tr, X_tr_s)
      prf <- predict_interval_rf(frf, X_te_s, alpha)
      fc_mat[t, "RandomForest"] <- prf["fc"]
      lb_mat[t, "RandomForest"] <- prf["lb"]
      ub_mat[t, "RandomForest"] <- prf["ub"]
    }

    # ---- Metrics ----
    metrics <- data.frame(
      Method     = methods,
      RMSE       = sapply(methods, function(m) rmse_fn(act, fc_mat[, m])),
      MAE        = sapply(methods, function(m) mae_fn(act, fc_mat[, m])),
      Coverage90 = sapply(methods, function(m)
                     coverage_fn(act, lb_mat[, m], ub_mat[, m])),
      AvgWidth   = sapply(methods, function(m)
                     width_fn(lb_mat[, m], ub_mat[, m])),
      stringsAsFactors = FALSE,
      row.names = NULL
    )

    if (verbose) {
      cat(sprintf("\n  Results for h=%d:\n", h))
      cat(sprintf("  %-15s %8s %8s %10s %10s\n",
                  "Method", "RMSE", "MAE", "Coverage", "AvgWidth"))
      cat(sprintf("  %s\n", strrep("-", 53)))
      for (i in seq_len(nrow(metrics))) {
        cat(sprintf("  %-15s %8.3f %8.3f %9.1f%% %10.2f\n",
                    metrics$Method[i], metrics$RMSE[i], metrics$MAE[i],
                    metrics$Coverage90[i] * 100, metrics$AvgWidth[i]))
      }
    }

    # Reconstruct forecast-origin dates
    T_full      <- nrow(DATA)
    t_start_idx <- p + 1                    # first row of y_full in DATA
    origin_rows <- (t_start_idx + T0 - 1 + 1):(t_start_idx + T0 - 1 + test_size)
    origin_rows <- pmin(origin_rows, nrow(df))
    fc_dates    <- df$quarter[origin_rows]

    results[[as.character(h)]] <- list(
      forecasts = fc_mat,
      actuals   = act,
      lb        = lb_mat,
      ub        = ub_mat,
      metrics   = metrics,
      dates     = fc_dates
    )
  }

  results
}


# =============================================================================
# SECTION 5: RESULTS TABLES
# =============================================================================

print_table96 <- function(res_gdp, res_inf) {
  cat("\n", strrep("=", 72), "\n", sep = "")
  cat("Point Forecast Accuracy: RMSE and MAE\n")
  cat(strrep("=", 72), "\n", sep = "")
  cat(sprintf("%-15s  %16s  %16s\n", "", "h=1", "h=4"))
  cat(sprintf("%-15s  %8s %8s  %8s %8s\n",
              "Method", "RMSE", "MAE", "RMSE", "MAE"))
  cat(strrep("-", 55), "\n", sep = "")

  for (info in list(list("GDP Growth", res_gdp),
                    list("Inflation",  res_inf))) {
    cat(sprintf("\n  %s\n", info[[1]]))
    res <- info[[2]]
    for (m in c("VAR","BVAR","LASSO","Ridge","ElasticNet","RandomForest")) {
      r1 <- res[["1"]]$metrics[res[["1"]]$metrics$Method == m, ]
      r4 <- res[["4"]]$metrics[res[["4"]]$metrics$Method == m, ]
      cat(sprintf("  %-13s  %8.2f %8.2f  %8.2f %8.2f\n",
                  m, r1$RMSE, r1$MAE, r4$RMSE, r4$MAE))
    }
  }
}

print_table97 <- function(res_gdp, res_inf) {
  cat("\n", strrep("=", 72), "\n", sep = "")
  cat("Density Forecast Accuracy: Coverage and Interval Width\n")
  cat("(Nominal coverage = 90%)\n")
  cat(strrep("=", 72), "\n", sep = "")
  cat(sprintf("%-15s  %22s  %22s\n", "", "h=1", "h=4"))
  cat(sprintf("%-15s  %8s %10s  %8s %10s\n",
              "Method", "Cov.", "Width", "Cov.", "Width"))
  cat(strrep("-", 60), "\n", sep = "")

  for (info in list(list("GDP Growth", res_gdp),
                    list("Inflation",  res_inf))) {
    cat(sprintf("\n  %s\n", info[[1]]))
    res <- info[[2]]
    for (m in c("VAR","BVAR","LASSO","Ridge","ElasticNet","RandomForest")) {
      r1 <- res[["1"]]$metrics[res[["1"]]$metrics$Method == m, ]
      r4 <- res[["4"]]$metrics[res[["4"]]$metrics$Method == m, ]
      cat(sprintf("  %-13s  %7.1f%%  %10.2f  %7.1f%%  %10.2f\n",
                  m,
                  r1$Coverage90 * 100, r1$AvgWidth,
                  r4$Coverage90 * 100, r4$AvgWidth))
    }
  }
}

print_table98 <- function(res_gdp, res_inf) {
  cat("\n", strrep("=", 95), "\n", sep = "")
  cat("Diebold-Mariano Tests: Comparison to VAR Benchmark\n")
  cat("(Negative DM -> method outperforms VAR; HAC bandwidth = h-1)\n")
  cat(strrep("=", 95), "\n", sep = "")
  cat(sprintf("%-15s  %30s  %30s\n", "", "GDP Growth", "Inflation"))
  cat(sprintf("%-15s  %9s %8s  %9s %8s  %9s %8s  %9s %8s\n",
              "Method",
              "h=1 DM","p-val","h=4 DM","p-val",
              "h=1 DM","p-val","h=4 DM","p-val"))
  cat(strrep("-", 95), "\n", sep = "")

  non_var <- c("BVAR","LASSO","Ridge","ElasticNet","RandomForest")

  for (m in non_var) {
    row_str <- sprintf("%-15s", m)
    for (res in list(res_gdp, res_inf)) {
      for (h_chr in c("1", "4")) {
        r     <- res[[h_chr]]
        e_var <- r$actuals - r$forecasts[, "VAR"]
        e_m   <- r$actuals - r$forecasts[, m]
        dm    <- diebold_mariano(e_m, e_var, as.integer(h_chr))
        stars <- sig_stars(dm$p_value)
        row_str <- paste0(row_str,
                          sprintf("  %+8.2f%-3s %7.3f",
                                  dm$DM, stars, dm$p_value))
      }
    }
    cat(row_str, "\n")
  }
  cat("\n  * p<0.10  ** p<0.05  *** p<0.01\n")
}


# =============================================================================
# SECTION 6: FIGURES
# =============================================================================

plot_forecast_comparison <- function(res, target_label, h,
                                     save_path = NULL) {
  r       <- res[[as.character(h)]]
  methods <- c("VAR","BVAR","LASSO","Ridge","ElasticNet","RandomForest")
  colors  <- c("#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b")
  dates   <- r$dates

  plot_list <- vector("list", 6)
  for (i in seq_along(methods)) {
    m   <- methods[i]
    col <- colors[i]

    df_plot <- data.frame(
      date   = dates,
      actual = r$actuals,
      fc     = r$forecasts[, m],
      lb     = r$lb[, m],
      ub     = r$ub[, m]
    )

    rmse_val <- rmse_fn(r$actuals, r$forecasts[, m])
    cov_val  <- coverage_fn(r$actuals, r$lb[, m], r$ub[, m])
    ann_text <- sprintf("RMSE: %.3f\nCov: %.1f%%", rmse_val, cov_val * 100)

    p <- ggplot(df_plot, aes(x = date)) +
      geom_ribbon(aes(ymin = lb, ymax = ub),
                  fill = col, alpha = 0.2) +
      geom_line(aes(y = actual), colour = "black", linewidth = 0.8) +
      geom_line(aes(y = fc), colour = col,
                linewidth = 0.7, linetype = "dashed") +
      annotate("text", x = min(dates), y = Inf,
               label = ann_text, hjust = 0, vjust = 1.2,
               size = 3, colour = "black",
               fontface = "plain") +
      labs(title = m, x = "Date", y = target_label) +
      theme_bw(base_size = 10) +
      theme(plot.title = element_text(face = "bold"))

    plot_list[[i]] <- p
  }

  combined <- arrangeGrob(
    grobs = plot_list, ncol = 3,
    top  = sprintf("%s Forecasts (h=%d)", target_label, h)
  )

  if (!is.null(save_path)) {
    ggsave(save_path, combined, width = 15, height = 10, dpi = 150)
    cat(sprintf("  Saved: %s\n", save_path))
  }

  invisible(combined)
}


plot_metrics_comparison <- function(res_list, target_label,
                                    save_path = NULL) {
  methods  <- c("VAR","BVAR","LASSO","Ridge","ElasticNet","RandomForest")
  horizons <- c(1, 4)

  rmse_df <- do.call(rbind, lapply(horizons, function(h) {
    met <- res_list[[as.character(h)]]$metrics
    data.frame(Method  = met$Method,
               RMSE    = met$RMSE,
               Horizon = factor(paste0("h=", h)))
  }))
  rmse_df$Method <- factor(rmse_df$Method, levels = methods)

  cov_df <- do.call(rbind, lapply(horizons, function(h) {
    met <- res_list[[as.character(h)]]$metrics
    data.frame(Method   = met$Method,
               Coverage = met$Coverage90 * 100,
               Horizon  = factor(paste0("h=", h)))
  }))
  cov_df$Method <- factor(cov_df$Method, levels = methods)

  p1 <- ggplot(rmse_df, aes(x = Method, y = RMSE, fill = Horizon)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = "Point Forecast Accuracy (RMSE)", x = NULL, y = "RMSE") +
    theme_bw(base_size = 10) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          plot.title  = element_text(face = "bold"))

  p2 <- ggplot(cov_df, aes(x = Method, y = Coverage, fill = Horizon)) +
    geom_bar(stat = "identity", position = "dodge") +
    geom_hline(yintercept = 90, colour = "red",
               linetype = "dashed", linewidth = 0.8) +
    ylim(0, 110) +
    labs(title = "Density Forecast Accuracy (Coverage)",
         x = NULL, y = "Coverage (%)") +
    theme_bw(base_size = 10) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          plot.title  = element_text(face = "bold"))

  combined <- arrangeGrob(p1, p2, ncol = 2,
                          top = sprintf("Forecasting Comparison: %s",
                                        target_label))

  if (!is.null(save_path)) {
    ggsave(save_path, combined, width = 14, height = 5, dpi = 150)
    cat(sprintf("  Saved: %s\n", save_path))
  }

  invisible(combined)
}


# =============================================================================
# SECTION 7: MAIN EXECUTION
# =============================================================================

cat("[1/4] Loading data...\n")

# Update paths if files are in a different directory
GDP_PATH      <- "GDP.xlsx"
DEFLATOR_PATH <- "GDPDEFL.xlsx"
FEDFUNDS_PATH <- "FFR.xlsx"

df <- load_fred_data(GDP_PATH, DEFLATOR_PATH, FEDFUNDS_PATH)

cat("Summary Statistics:\n")
print(summary(df[, c("GDP_growth", "Inflation", "FedFunds")]))

# ---------------------------------------------------------------------------
cat("\n[2/4] Running forecasting comparison (expanding window, 60 OOS obs)...\n")

results_gdp <- run_forecasting_comparison(
  df,
  target_var = "GDP_growth",
  horizons   = c(1, 4),
  test_size  = 60,
  p          = 4,
  verbose    = TRUE
)

results_inf <- run_forecasting_comparison(
  df,
  target_var = "Inflation",
  horizons   = c(1, 4),
  test_size  = 60,
  p          = 4,
  verbose    = TRUE
)

# ---------------------------------------------------------------------------
cat("\n[3/4] Building results tables...\n")

print_table96(results_gdp, results_inf)
print_table97(results_gdp, results_inf)
print_table98(results_gdp, results_inf)

# ---------------------------------------------------------------------------
cat("\n[4/4] Generating figures...\n")

plot_forecast_comparison(results_gdp, "GDP Growth", h = 1,
                         save_path = "figure9_gdp_h1.png")
plot_forecast_comparison(results_gdp, "GDP Growth", h = 4,
                         save_path = "figure9_gdp_h4.png")
plot_forecast_comparison(results_inf,  "Inflation",  h = 1,
                         save_path = "figure9_inf_h1.png")
plot_forecast_comparison(results_inf,  "Inflation",  h = 4,
                         save_path = "figure9_inf_h4.png")
plot_metrics_comparison(results_gdp, "GDP Growth",
                        save_path = "figure9_metrics_gdp.png")
plot_metrics_comparison(results_inf,  "Inflation",
                        save_path = "figure9_metrics_inf.png")

cat("\n", strrep("=", 70), "\n", sep = "")
cat("Replication complete.\n")
cat("Outputs: figure9_*.png\n")
cat(strrep("=", 70), "\n", sep = "")
