#==========================================================================
# EXAMPLE 1: CHOLESKY IDENTIFICATION - 4-VARIABLE VAR
# Textbook Section 4.18.1
#
# Author: Alessia Paccagnini
# Textbook: Macroeconometrics
#
# Variables: GDP Growth, Oil Price Growth, Inflation, Federal Funds Rate
# Sample: 1971:Q1 - 2007:Q4
# Identification: Cholesky (Recursive)
#
# Produces 4 figures
#   1. IRFs_MP_shock_R.png
#   2. Cumulative_responses_R.png
#   3. FEVD_R.png
#   4. Historical_decomposition_R.png
#
# Dependencies: readxl, MASS (both standard)
# Usage: source("example1_cholesky_VAR.R")  with 2026-01-QD.xlsx in working dir
#==========================================================================

rm(list = ls())
cat("========================================\n")
cat("4-VARIABLE CHOLESKY IDENTIFICATION\n")
cat("========================================\n\n")

# Load required packages
required_pkgs <- c("readxl", "MASS")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cran.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

# Colors matching Python COLORS dict
col_vec <- c(
  rgb(0.180, 0.525, 0.671),   # #2E86AB - GDP (blue)
  rgb(0.024, 0.655, 0.490),   # #06A77D - Oil (green)
  rgb(0.635, 0.231, 0.447),   # #A23B72 - Inflation (purple)
  rgb(0.945, 0.561, 0.004)    # #F18F01 - Fed Funds (orange)
)

# Transparent color helper
col_alpha <- function(hex_col, alpha) {
  r <- col2rgb(hex_col) / 255
  rgb(r[1], r[2], r[3], alpha = alpha)
}

# Matrix power helper (avoids expm dependency)
mat_power <- function(M, n) {
  if (n == 0) return(diag(nrow(M)))
  result <- diag(nrow(M))
  for (ii in 1:n) result <- result %*% M
  return(result)
}

# Safe numeric extraction
safe_numeric <- function(x) {
  if (is.numeric(x)) return(x)
  return(as.numeric(as.character(x)))
}

#==========================================================================
# [1/7] LOAD DATA
#==========================================================================
cat("[1/7] Loading data...\n")

data_raw <- read_excel("2026-01-QD.xlsx")

# Skip first 2 rows (factor loadings and transformations)
data_df <- data_raw[3:nrow(data_raw), ]

# Extract date column - handle Excel serial numbers robustly
dates <- tryCatch(
  as.Date(data_df$sasdate),
  error = function(e) {
    tryCatch(
      as.Date(as.numeric(data_df$sasdate), origin = "1899-12-30"),
      error = function(e2) {
        as.Date(as.POSIXct(as.numeric(data_df$sasdate), origin = "1899-12-30"))
      }
    )
  }
)

cat(sprintf("   Dates: %d values, range %s to %s\n",
            length(dates), min(dates, na.rm = TRUE), max(dates, na.rm = TRUE)))

# Extract variables
gdp      <- safe_numeric(data_df$GDPC1)
deflator <- safe_numeric(data_df$GDPCTPI)
fedfunds <- safe_numeric(data_df$FEDFUNDS)
oil      <- safe_numeric(data_df$OILPRICEx)

cat(sprintf("   GDP: %d non-NA (of %d)\n", sum(!is.na(gdp)), length(gdp)))
cat(sprintf("   Oil: %d non-NA (of %d)\n", sum(!is.na(oil)), length(oil)))

# Compute growth rates
n_obs <- length(gdp)
gdp_growth     <- 100 * diff(log(gdp))
oil_growth     <- 100 * diff(log(oil))
inflation_rate <- 100 * diff(log(deflator))
fedfunds_level <- fedfunds[2:n_obs]
dates_growth   <- dates[2:n_obs]

# Remove NAs and filter to 1971:Q1 - 2007:Q4
valid <- !is.na(gdp_growth) & !is.na(oil_growth) &
         !is.na(inflation_rate) & !is.na(fedfunds_level)
Y_full     <- cbind(gdp_growth[valid], oil_growth[valid],
                    inflation_rate[valid], fedfunds_level[valid])
dates_full <- dates_growth[valid]

idx <- dates_full >= as.Date("1971-01-01") & dates_full <= as.Date("2007-12-31")
Y            <- Y_full[idx, ]
dates_sample <- dates_full[idx]

T_obs <- nrow(Y)
K     <- ncol(Y)

if (T_obs == 0) stop("No data in sample period! Check date parsing.")

cat(sprintf("   Sample: %d observations\n", T_obs))
cat(sprintf("   Period: %s to %s\n\n", dates_sample[1], dates_sample[T_obs]))

#==========================================================================
# [2/7] VAR ESTIMATION
#==========================================================================
cat("[2/7] Estimating VAR(4)...\n")

n_lags <- 4L   # Use n_lags (not p) to avoid ggplot name collision
H      <- 20L

# Build lagged matrix
X_mat <- matrix(1, nrow = T_obs - n_lags, ncol = 1)
for (lag in 1:n_lags) {
  X_mat <- cbind(X_mat, Y[(n_lags - lag + 1):(T_obs - lag), ])
}
Y_est <- Y[(n_lags + 1):T_obs, ]

B_hat <- solve(t(X_mat) %*% X_mat, t(X_mat) %*% Y_est)
U     <- Y_est - X_mat %*% B_hat

T_eff   <- T_obs - n_lags
Sigma_u <- (t(U) %*% U) / (T_eff - K * n_lags - 1)

cat(sprintf("   Effective sample: %d observations\n\n", T_eff))

#==========================================================================
# [3/7] CHOLESKY IDENTIFICATION
#==========================================================================
cat("[3/7] Cholesky identification...\n")

P_chol   <- t(chol(Sigma_u))
mp_shock <- P_chol[4, 4]

cat(sprintf("   MP shock: %.2f pp\n\n", mp_shock))

#==========================================================================
# [4/7] IMPULSE RESPONSE FUNCTIONS
#==========================================================================
cat("[4/7] Computing impulse responses...\n")

# Companion form
Fp <- matrix(0, K * n_lags, K * n_lags)
Fp[1:K, ] <- t(B_hat[2:nrow(B_hat), ])
if (n_lags > 1) {
  Fp[(K + 1):(K * n_lags), 1:(K * (n_lags - 1))] <- diag(K * (n_lags - 1))
}
J_mat <- cbind(diag(K), matrix(0, K, K * (n_lags - 1)))

# IRF computation function
compute_irf <- function(B, Pmat, H, K, n_lags) {
  Fc <- matrix(0, K * n_lags, K * n_lags)
  Fc[1:K, ] <- t(B[2:nrow(B), ])
  if (n_lags > 1) Fc[(K + 1):(K * n_lags), 1:(K * (n_lags - 1))] <- diag(K * (n_lags - 1))
  Jc <- cbind(diag(K), matrix(0, K, K * (n_lags - 1)))

  irf_out <- array(0, dim = c(H + 1, K, K))
  irf_out[1, , ] <- Pmat
  Fh <- diag(K * n_lags)
  for (hh in 1:H) {
    Fh <- Fh %*% Fc
    irf_out[hh + 1, , ] <- Jc %*% Fh %*% t(Jc) %*% Pmat
  }
  return(irf_out)
}

IRF    <- compute_irf(B_hat, P_chol, H, K, n_lags)
irf_mp <- IRF[, , 4]

cat("   IRFs computed\n\n")

#==========================================================================
# [5/7] BOOTSTRAP CONFIDENCE INTERVALS
#==========================================================================
cat("[5/7] Bootstrap confidence intervals (300 reps)...\n")

B_sim      <- 300L
IRF_boot   <- array(0, dim = c(B_sim, H + 1, K, K))
U_centered <- sweep(U, 2, colMeans(U))

set.seed(42)

for (b in 1:B_sim) {
  if (b %% 100 == 0) cat(sprintf("   Progress: %d/%d\n", b, B_sim))

  idx_b  <- sample(1:T_eff, T_eff, replace = TRUE)
  U_star <- U_centered[idx_b, ]

  Y_star        <- matrix(0, T_obs, K)
  Y_star[1:n_lags, ] <- Y[1:n_lags, ]

  for (tt in (n_lags + 1):T_obs) {
    lags <- c()
    for (lag in 1:n_lags) lags <- c(lags, Y_star[tt - lag, ])
    X_t <- c(1, lags)
    Y_star[tt, ] <- X_t %*% B_hat + U_star[tt - n_lags, ]
  }

  Y_boot <- Y_star[(n_lags + 1):T_obs, ]
  X_boot <- matrix(1, T_obs - n_lags, 1)
  for (lag in 1:n_lags) {
    X_boot <- cbind(X_boot, Y_star[(n_lags - lag + 1):(T_obs - lag), ])
  }

  B_boot <- tryCatch(
    solve(t(X_boot) %*% X_boot, t(X_boot) %*% Y_boot),
    error = function(e) NULL
  )
  if (is.null(B_boot)) { IRF_boot[b, , , ] <- IRF; next }

  U_boot     <- Y_boot - X_boot %*% B_boot
  Sigma_boot <- (t(U_boot) %*% U_boot) / (T_eff - K * n_lags - 1)

  tryCatch({
    P_boot <- t(chol(Sigma_boot))
    IRF_boot[b, , , ] <- compute_irf(B_boot, P_boot, H, K, n_lags)
  }, error = function(e) {
    IRF_boot[b, , , ] <<- IRF
  })
}

# Bias correction
bias    <- apply(IRF_boot, c(2, 3, 4), mean) - IRF
IRF_ctr <- sweep(IRF_boot, c(2, 3, 4), bias)

# Confidence intervals
CI_68 <- array(0, dim = c(2, H + 1, K, K))
CI_90 <- array(0, dim = c(2, H + 1, K, K))
for (h_i in 1:(H + 1)) {
  for (ii in 1:K) {
    for (jj in 1:K) {
      vals <- IRF_ctr[, h_i, ii, jj]
      CI_68[, h_i, ii, jj] <- quantile(vals, c(0.16, 0.84))
      CI_90[, h_i, ii, jj] <- quantile(vals, c(0.05, 0.95))
    }
  }
}

cat("   Bootstrap complete!\n\n")

#==========================================================================
# [6/7] FORECAST ERROR VARIANCE DECOMPOSITION
#==========================================================================
cat("[6/7] Computing FEVD...\n")

FEVD <- array(0, dim = c(H + 1, K, K))
for (hh in 0:H) {
  mse <- matrix(0, K, K)
  for (jj in 0:hh) mse <- mse + IRF[jj + 1, , ] %*% t(IRF[jj + 1, , ])
  for (ii in 1:K) {
    for (jj in 1:K) {
      sc <- sum(IRF[1:(hh + 1), ii, jj]^2)
      if (mse[ii, ii] > 0) FEVD[hh + 1, ii, jj] <- sc / mse[ii, ii]
    }
  }
}

cat("   FEVD computed\n\n")

#==========================================================================
# [7/7] HISTORICAL DECOMPOSITION
#==========================================================================
cat("[7/7] Computing historical decomposition...\n")

eps <- U %*% t(solve(P_chol))

HD <- array(0, dim = c(T_eff, K, K))
for (tt in 1:T_eff) {
  for (shock in 1:K) {
    contrib <- rep(0, K)
    for (ss in 1:tt) {
      Fs     <- mat_power(Fp, tt - ss)
      impact <- J_mat %*% Fs %*% t(J_mat) %*% P_chol[, shock]
      contrib <- contrib + impact * eps[ss, shock]
    }
    HD[tt, , shock] <- contrib
  }
}

cat("   Historical decomposition computed\n\n")

#==========================================================================
# PLOTTING
#==========================================================================
cat("Creating figures...\n\n")

var_names   <- c("GDP Growth", "Oil Price Growth", "Inflation", "Federal Funds")
shock_names <- c("GDP Shock", "Oil Shock", "Inflation Shock", "MP Shock")
h_vec       <- 0:H
dates_hd    <- dates_sample[(n_lags + 1):T_obs]

# -------------------------------------------------------------------------
# FIGURE 1: IRFs to Monetary Policy Shock
# -------------------------------------------------------------------------
png("IRFs_MP_shock_R.png", width = 1600, height = 400, res = 150)
par(mfrow = c(1, 4), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0), bg = "white")

for (ii in 1:K) {
  ci90_lo <- CI_90[1, , ii, 4]
  ci90_hi <- CI_90[2, , ii, 4]
  ci68_lo <- CI_68[1, , ii, 4]
  ci68_hi <- CI_68[2, , ii, 4]

  ylims <- range(ci90_lo, ci90_hi, irf_mp[, ii])

  plot(h_vec, irf_mp[, ii], type = "n", xlab = "Quarters",
       ylab = if (ii == 1) "Percentage Points" else "",
       main = var_names[ii], font.main = 2,
       ylim = ylims, las = 1, panel.first = grid(col = gray(0.85)))

  polygon(c(h_vec, rev(h_vec)), c(ci90_lo, rev(ci90_hi)),
          col = col_alpha(col_vec[ii], 0.15), border = NA)
  polygon(c(h_vec, rev(h_vec)), c(ci68_lo, rev(ci68_hi)),
          col = col_alpha(col_vec[ii], 0.30), border = NA)
  lines(h_vec, irf_mp[, ii], col = col_vec[ii], lwd = 2.5)
  abline(h = 0, lty = 2, lwd = 0.8)

  if (ii == 4) {
    legend("topright", legend = c("IRF", "68%", "90%"),
           col = c(col_vec[ii], col_alpha(col_vec[ii], 0.30),
                   col_alpha(col_vec[ii], 0.15)),
           lwd = c(2.5, 8, 8), cex = 0.7, bg = "white")
  }
}

mtext("Impulse Responses to Monetary Policy Shock",
      outer = TRUE, cex = 1.1, font = 2)
dev.off()
cat("  Saved: IRFs_MP_shock_R.png\n")

# -------------------------------------------------------------------------
# FIGURE 2: Cumulative Responses
# -------------------------------------------------------------------------
png("Cumulative_responses_R.png", width = 1200, height = 450, res = 150)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0), bg = "white")

irf_cum   <- apply(irf_mp, 2, cumsum)
CI_68_cum <- array(0, dim = dim(CI_68))
for (qq in 1:2) {
  for (i2 in 1:K) {
    for (j2 in 1:K) {
      CI_68_cum[qq, , i2, j2] <- cumsum(CI_68[qq, , i2, j2])
    }
  }
}

# GDP Level
ci_lo <- CI_68_cum[1, , 1, 4]
ci_hi <- CI_68_cum[2, , 1, 4]
ylims <- range(ci_lo, ci_hi, irf_cum[, 1])
plot(h_vec, irf_cum[, 1], type = "n", xlab = "Quarters",
     ylab = "Percentage Points", main = "GDP Level Response",
     font.main = 2, ylim = ylims, las = 1,
     panel.first = grid(col = gray(0.85)))
polygon(c(h_vec, rev(h_vec)), c(ci_lo, rev(ci_hi)),
        col = col_alpha(col_vec[1], 0.3), border = NA)
lines(h_vec, irf_cum[, 1], col = col_vec[1], lwd = 2.5)
abline(h = 0, lty = 2, lwd = 0.8)

# Price Level
ci_lo <- CI_68_cum[1, , 3, 4]
ci_hi <- CI_68_cum[2, , 3, 4]
ylims <- range(ci_lo, ci_hi, irf_cum[, 3])
plot(h_vec, irf_cum[, 3], type = "n", xlab = "Quarters",
     ylab = "", main = "Price Level Response",
     font.main = 2, ylim = ylims, las = 1,
     panel.first = grid(col = gray(0.85)))
polygon(c(h_vec, rev(h_vec)), c(ci_lo, rev(ci_hi)),
        col = col_alpha(col_vec[3], 0.3), border = NA)
lines(h_vec, irf_cum[, 3], col = col_vec[3], lwd = 2.5)
abline(h = 0, lty = 2, lwd = 0.8)

mtext("Cumulative Responses: GDP and Price Level Effects",
      outer = TRUE, cex = 1.1, font = 2)
dev.off()
cat("  Saved: Cumulative_responses_R.png\n")

# -------------------------------------------------------------------------
# FIGURE 3: FEVD (stacked area)
# -------------------------------------------------------------------------
png("FEVD_R.png", width = 1400, height = 1000, res = 150)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0), bg = "white")

for (ii in 1:K) {
  stack <- matrix(0, nrow = H + 1, ncol = K)
  for (jj in 1:K) stack[, jj] <- FEVD[, ii, jj] * 100
  cum_stack <- t(apply(stack, 1, cumsum))

  plot(h_vec, rep(0, H + 1), type = "n", xlab = "Quarters",
       ylab = "Percent", main = var_names[ii], font.main = 2,
       ylim = c(0, 100), las = 1,
       panel.first = grid(col = gray(0.85)))

  for (jj in K:1) {
    top_j    <- cum_stack[, jj]
    bottom_j <- if (jj == 1) rep(0, H + 1) else cum_stack[, jj - 1]
    polygon(c(h_vec, rev(h_vec)), c(bottom_j, rev(top_j)),
            col = col_alpha(col_vec[jj], 0.8), border = NA)
  }

  legend("right", legend = shock_names, fill = col_vec,
         cex = 0.7, bg = "white", border = NA)
}

mtext("Forecast Error Variance Decomposition",
      outer = TRUE, cex = 1.2, font = 2)
dev.off()
cat("  Saved: FEVD_R.png\n")

# -------------------------------------------------------------------------
# FIGURE 4: Historical Decomposition (stacked positive/negative)
# NOTE: Convert dates to numeric so polygon(c(x, rev(x)), ...) works.
#       Date class is silently dropped by c() concatenation, making polygons
#       invisible when plot() used Date-typed x-axis.
# -------------------------------------------------------------------------
png("Historical_decomposition_R.png", width = 1400, height = 1000, res = 150)
par(mfrow = c(2, 2), mar = c(5, 4, 3, 1), oma = c(0, 0, 2, 0), bg = "white")

dates_hd_num <- as.numeric(dates_hd)

for (ii in 1:K) {
  actual_dm <- Y[(n_lags + 1):T_obs, ii] - mean(Y[(n_lags + 1):T_obs, ii])
  pos_total <- rep(0, T_eff)
  neg_total <- rep(0, T_eff)
  for (jj in 1:K) {
    vals <- HD[, ii, jj]
    pos_total <- pos_total + pmax(vals, 0)
    neg_total <- neg_total + pmin(vals, 0)
  }
  ylims <- range(pos_total, neg_total, actual_dm) * 1.05

  plot(dates_hd_num, actual_dm, type = "n", xlab = "", xaxt = "n",
       ylab = "Percentage Points",
       main = var_names[ii], font.main = 2, ylim = ylims, las = 1,
       panel.first = grid(col = gray(0.85)))

  # Manual date axis
  yr_seq <- seq(as.Date("1975-01-01"), as.Date("2007-01-01"), by = "5 years")
  axis(1, at = as.numeric(yr_seq), labels = format(yr_seq, "%Y"),
       las = 2, cex.axis = 0.8)

  bottom_pos <- rep(0, T_eff)
  bottom_neg <- rep(0, T_eff)
  for (jj in 1:K) {
    vals     <- HD[, ii, jj]
    pos_vals <- pmax(vals, 0)
    neg_vals <- pmin(vals, 0)

    top_pos <- bottom_pos + pos_vals
    polygon(c(dates_hd_num, rev(dates_hd_num)),
            c(bottom_pos, rev(top_pos)),
            col = col_alpha(col_vec[jj], 0.7), border = NA)
    bottom_pos <- top_pos

    top_neg <- bottom_neg + neg_vals
    polygon(c(dates_hd_num, rev(dates_hd_num)),
            c(bottom_neg, rev(top_neg)),
            col = col_alpha(col_vec[jj], 0.7), border = NA)
    bottom_neg <- top_neg
  }

  lines(dates_hd_num, actual_dm, col = "black", lwd = 1.5)
  abline(h = 0, col = rgb(0, 0, 0, 0.3), lwd = 0.8)

  legend("topleft", legend = c(shock_names, "Actual"),
         fill   = c(col_vec, NA),
         border = c(rep(NA, 4), NA),
         lwd    = c(NA, NA, NA, NA, 1.5),
         col    = c(rep(NA, 4), "black"),
         cex = 0.6, bg = "white", ncol = 2)
}

mtext("Historical Decomposition", outer = TRUE, cex = 1.2, font = 2)
dev.off()
cat("  Saved: Historical_decomposition_R.png\n")


#==========================================================================
# SUMMARY
#==========================================================================
cat("\n========================================\n")
cat("  ANALYSIS COMPLETE!\n")
cat("========================================\n\n")

cat(sprintf("Key Results:\n"))
cat(sprintf("   MP shock:        %.2f pp\n", mp_shock))
cat(sprintf("   GDP peak:        %.2f at Q%d\n", min(irf_mp[, 1]),
            which.min(irf_mp[, 1]) - 1))
cat(sprintf("   Oil peak:        %.2f at Q%d\n", min(irf_mp[, 2]),
            which.min(irf_mp[, 2]) - 1))
cat(sprintf("   Price puzzle:    %d quarters\n", sum(irf_mp[1:6, 3] > 0)))

cat("\nFEVD at Horizon 20 (5 years):\n")
cat(sprintf("   %-15s %7s %7s %7s %7s\n", "Variable", "GDP", "Oil", "Infl", "MP"))
cat(sprintf("   %s\n", paste(rep("-", 50), collapse = "")))
for (ii in 1:K) {
  fevd_h20 <- FEVD[21, ii, ] * 100
  cat(sprintf("   %-15s %6.1f%% %6.1f%% %6.1f%% %6.1f%%\n",
              var_names[ii], fevd_h20[1], fevd_h20[2], fevd_h20[3], fevd_h20[4]))
}

cat("\nFigures saved:\n")
cat("   1. IRFs_MP_shock_R.png\n")
cat("   2. Cumulative_responses_R.png\n")
cat("   3. FEVD_R.png\n")
cat("   4. Historical_decomposition_R.png\n")
cat("========================================\n")
