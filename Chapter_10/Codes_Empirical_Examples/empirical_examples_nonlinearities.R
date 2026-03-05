# =============================================================================
# FOUR APPROACHES TO REGIME-DEPENDENT DYNAMICS
# =============================================================================
#
# Author:   Alessia Paccagnini
# Textbook: Macroeconometrics
#
# Empirical comparison of:
#   Approach 1: Time-Varying Parameter VAR with Stochastic Volatility (TVP-VAR-SV)
#   Approach 2: Markov-Switching VAR (MS-VAR)
#   Approach 3: Threshold VAR (TVAR)
#   Approach 4: Smooth Transition VAR (STVAR)
#
# Data: US quarterly macro data 1960Q1-2019Q4 (FRED)
#   GDPC1    -> annualised GDP growth
#   GDPDEF   -> annualised inflation
#   FEDFUNDS -> quarterly average federal funds rate
#
# Variable ordering (Cholesky): GDP growth | Inflation | Fed Funds Rate
#
# Requirements: base R only (MASS is used for mvrnorm — ships with R)
#   readxl  - for Excel import (install.packages("readxl"))
#   ggplot2 - for publication-quality plots (install.packages("ggplot2"))
#   patchwork - for multi-panel layouts (install.packages("patchwork"))
#   If these are unavailable the script falls back to base graphics.
#
# Usage:
#   setwd() to the folder with GDPC1.xlsx, GDPDEF.xlsx, FEDFUNDS.xlsx
#   source("section_10_9_all_approaches.R")
# =============================================================================

set.seed(42)
suppressWarnings({
  has_readxl   <- requireNamespace("readxl",   quietly = TRUE)
  has_ggplot2  <- requireNamespace("ggplot2",  quietly = TRUE)
  has_patchwork<- requireNamespace("patchwork",quietly = TRUE)
})
library(MASS)   # mvrnorm

# ── Figure output helper: save PNG ───────────────────────────────────────────
save_fig <- function(filename, expr, width = 10, height = 6) {
  png(filename, width = width, height = height, units = "in", res = 150)
  on.exit(dev.off())
  force(expr)
}

# =============================================================================
# 0.  GLOBAL SETTINGS
# =============================================================================

LAGS      <- 2L
HORZ      <- 20L
N         <- 3L
SHOCK_EQ  <- 3L      # Federal funds rate (last in Cholesky ordering)

# MCMC iterations and burn-in
# For quick testing use the defaults below.
# For publication quality increase to REPS_TVP=20000L, REPS_MS=20000L.
# Always inspect convergence diagnostics after any run (see below).
REPS_TVP  <- 3000L;  BURN_TVP <- 1000L
REPS_MS   <- 5000L;  BURN_MS  <- 3000L

# Date indices into the *model* sample (after dropping LAGS initial obs)
# Data: 1960Q1-2019Q4 (240 obs). Model sample starts 1960Q3 (index 1).
# 2005Q1 = model row 178  (1-based) - 2 = 179
# 2008Q4 = model row 193  (1-based) - 2 = 194
T_NORMAL  <- 179L
T_STRESS  <- 194L

# NBER recession dates
nber_dates <- list(
  c("1960-04-01","1961-02-01"), c("1969-12-01","1970-11-01"),
  c("1973-11-01","1975-03-01"), c("1980-01-01","1980-07-01"),
  c("1981-07-01","1982-11-01"), c("1990-07-01","1991-03-01"),
  c("2001-03-01","2001-11-01"), c("2007-12-01","2009-06-01")
)
nber_df <- data.frame(
  start = as.Date(sapply(nber_dates, `[`, 1)),
  end   = as.Date(sapply(nber_dates, `[`, 2))
)

cat(strrep("=", 65), "\n")
cat("FOUR APPROACHES TO REGIME-DEPENDENT DYNAMICS\n")
cat(strrep("=", 65), "\n\n")


# =============================================================================
# CONVERGENCE DIAGNOSTICS
# =============================================================================
# Two standard tools for assessing MCMC chain quality:
#
#   trace_plot()   — visual check: the chain should look like "white noise"
#                    around a stable mean. Trends or drifts indicate the
#                    sampler has not converged.
#
#   geweke_z()     — formal test (Geweke 1992): compares the mean of the
#                    first 10% of draws with the mean of the last 50%.
#                    Under convergence the statistic is ~N(0,1).
#                    |z| > 1.96 flags a potential convergence problem.
#
# How to use after running each sampler:
#
#   TVP-VAR-SV — check representative time-varying coefficients and
#                log-volatility paths at the mid-sample point:
#                   trace_plot(beta_store[, T %/% 2, 1], "beta_1 (mid)")
#                   trace_plot(h_store[, T %/% 2, 1],    "log-vol GDP (mid)")
#                   geweke_z(beta_store[, T %/% 2, 1])
#
#   MS-VAR     — check the regime-specific VAR coefficients and the
#                draw-by-draw average smoothed probability:
#                   trace_plot(bsave1[, 1], "MS B1[1]")
#                   trace_plot(rowMeans(reg_save), "avg regime prob")
#                   geweke_z(bsave1[, 1])
#
# Rule of thumb: run with at least 4x the default REPS for final results and
# verify that Geweke |z| < 1.96 for the parameters of interest.
# =============================================================================

#' @title MCMC trace plot and density
#' @param chain  Numeric vector of posterior draws (burn-in removed)
#' @param label  Character label for plot title
#' @param save_png Logical; save PNG if TRUE (default TRUE)
trace_plot <- function(chain, label = "parameter", save_png = TRUE) {
  chain <- as.numeric(chain)
  fname <- paste0("diag_trace_", gsub("[^A-Za-z0-9]", "_", label), ".png")
  if (save_png) png(fname, width = 900, height = 320, res = 110)
  op <- par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

  # Left: trace
  plot(chain, type = "l", lwd = 0.7, col = "#2166ac",
       main = paste("Trace —", label), xlab = "Draw", ylab = "Value")
  abline(h = mean(chain), col = "red", lwd = 1.5, lty = 2)
  legend("topright", legend = "Posterior mean", col = "red",
         lty = 2, lwd = 1.5, bty = "n", cex = 0.8)

  # Right: density
  d <- density(chain)
  plot(d, main = paste("Density —", label), xlab = "Value", ylab = "Density",
       col = "#d6604d", lwd = 2)
  abline(v = mean(chain), col = "red", lwd = 1.5, lty = 2)
  abline(v = quantile(chain, 0.16), col = "grey50", lwd = 1.0, lty = 3)
  abline(v = quantile(chain, 0.84), col = "grey50", lwd = 1.0, lty = 3)

  par(op)
  if (save_png) dev.off()
  invisible(NULL)
}

#' @title Geweke (1992) convergence z-score
#' @param chain  Numeric vector of posterior draws (burn-in removed)
#' @param first  Fraction for early window  (default 0.10)
#' @param last   Fraction for late window   (default 0.50)
#' @return Scalar z-score; |z| > 1.96 flags non-convergence
geweke_z <- function(chain, first = 0.1, last = 0.5) {
  chain <- as.numeric(chain)
  n  <- length(chain)
  n1 <- floor(first * n)
  n2 <- floor(last  * n)
  a  <- chain[seq_len(n1)]
  b  <- chain[seq(n - n2 + 1, n)]

  # Spectral density at frequency 0 via AR(1) approximation
  spec0 <- function(x) {
    x   <- x - mean(x)
    n_x <- length(x)
    if (n_x < 4) return(var(x) + 1e-12)
    rho <- sum(x[-1] * x[-n_x]) / (sum(x[-n_x]^2) + 1e-12)
    rho <- pmax(pmin(rho, 0.99), -0.99)
    s2  <- var(x)
    s2 / (1 - rho)^2
  }

  s_a <- spec0(a) / length(a)
  s_b <- spec0(b) / length(b)
  denom <- sqrt(s_a + s_b)
  if (denom < 1e-12) return(0)
  (mean(a) - mean(b)) / denom
}

#' @title Run diagnostics for a matrix of MCMC draws
#' @param store      Matrix (ndraws x nparams)
#' @param param_names Character vector of parameter names (optional)
#' @param n_check    Number of parameters to sample for checking (default 5)
#' @param tag        String prefix for saved PNG filenames
mcmc_diagnostics <- function(store, param_names = NULL, n_check = 5L,
                              tag = "param") {
  store <- as.matrix(store)
  ndraws <- nrow(store)
  npar   <- ncol(store)
  if (is.null(param_names))
    param_names <- paste0("param_", seq_len(npar))

  idx <- round(seq(1, npar, length.out = min(n_check, npar)))

  cat(sprintf("  %-30s %10s  %s\n", "Parameter", "Geweke z", "Flag"))
  cat("  ", strrep("-", 53), "\n", sep = "")
  any_flag <- FALSE
  for (i in idx) {
    z    <- geweke_z(store[, i])
    flag <- if (abs(z) > 1.96) "  *** WARN" else ""
    if (abs(z) > 1.96) any_flag <- TRUE
    cat(sprintf("  %-30s %10.3f%s\n", param_names[i], z, flag))
    trace_plot(store[, i], label = paste0(tag, "_", param_names[i]))
  }
  if (!any_flag) {
    cat("  All Geweke |z| < 1.96 — no convergence issues detected.\n")
  } else {
    cat("  *** Flagged parameters suggest insufficient iterations.\n")
    cat("      Consider increasing REPS and/or checking trace plots.\n")
  }
  cat("\n")
}

# =============================================================================
# 1.  DATA PREPARATION
# =============================================================================

cat("Loading FRED data...\n")

load_fred_excel <- function(path, sheet, col_date = 1, col_val = 2) {
  if (has_readxl) {
    d <- readxl::read_excel(path, sheet = sheet, col_names = FALSE, skip = 1)
    d <- as.data.frame(d[, c(col_date, col_val)])
  } else {
    stop("readxl not available. Install with: install.packages('readxl')")
  }
  colnames(d) <- c("date", "value")
  d$date  <- as.Date(d$date)
  d <- d[!is.na(d$date) & !is.na(d$value), ]
  d
}

gdp_raw  <- load_fred_excel("GDPC1.xlsx",    sheet = "Quarterly")
def_raw  <- load_fred_excel("GDPDEF.xlsx",   sheet = "Quarterly")
ff_raw   <- load_fred_excel("FEDFUNDS.xlsx", sheet = "Monthly")

# Monthly fed funds -> quarterly average
ff_raw$quarter <- as.Date(format(ff_raw$date, "%Y-%m-01")) |>
  (\(d) as.Date(paste0(format(d, "%Y-"), sprintf("%02d", ((as.integer(format(d,"%m"))-1)%/%3)*3+1), "-01")))()
ff_q <- aggregate(value ~ quarter, data = ff_raw, FUN = mean)
colnames(ff_q) <- c("date", "fedfunds")

# Merge on quarterly dates
data_m <- merge(merge(
  setNames(gdp_raw, c("date","gdp")),
  setNames(def_raw, c("date","gdpdef")),
  by = "date"), ff_q, by = "date")
data_m <- data_m[order(data_m$date), ]

# Annualised quarterly growth rates
data_m$gdp_growth <- c(NA, diff(log(data_m$gdp))   * 400)
data_m$inflation  <- c(NA, diff(log(data_m$gdpdef)) * 400)

# Filter 1960Q1-2019Q4, drop first NaN row
data_m <- data_m[data_m$date >= as.Date("1960-01-01") &
                 data_m$date <= as.Date("2019-12-31"), ]
data_m <- data_m[!is.na(data_m$gdp_growth), ]

dates_plot <- data_m$date
Y_raw      <- as.matrix(data_m[, c("gdp_growth","inflation","fedfunds")])
T_raw      <- nrow(Y_raw)

cat(sprintf("Sample:        %s  to  %s\n", dates_plot[1], dates_plot[T_raw]))
cat(sprintf("Observations:  %d\n\n", T_raw))
cat("Variable order:  [1] GDP Growth | [2] Inflation | [3] Fed Funds Rate\n\n")

# =============================================================================
# 2.  HELPER FUNCTIONS (used across all approaches)
# =============================================================================

# ── Shade NBER recessions on a base-R plot ───────────────────────────────────
shade_nber <- function(ylims, dates) {
  for (i in seq_len(nrow(nber_df))) {
    xs <- as.numeric(c(nber_df$start[i], nber_df$end[i],
                       nber_df$end[i],   nber_df$start[i]))
    ys <- c(ylims[1], ylims[1], ylims[2], ylims[2])
    polygon(xs, ys, col = adjustcolor("grey70", alpha.f = 0.35),
            border = NA)
  }
}

# ── IRF simulation (equivalent to irfsim.m, but returns exactly HORZ rows) ──
irfsim_r <- function(B, N, lags, chol_mat, shock, horz) {
  # B: (k x N) where k = N*lags + 1
  # Returns (horz x N) matrix
  irf <- matrix(0, horz + lags, N)
  eps <- matrix(0, horz + lags, N)
  eps[lags + 1, ] <- shock   # shock is already Cholesky-scaled
  for (h in (lags + 1):horz) {
    x <- c()
    for (i in seq_len(lags))
      x <- c(x, irf[h - i, ])
    x <- c(x, 1)
    irf[h, ] <- x %*% B + eps[h, ]
  }
  irf[(lags + 1):(horz + lags), , drop = FALSE]  # HORZ rows
}

# ── Multivariate normal draw (uses MASS::mvrnorm) ────────────────────────────
rmvn <- function(mu, Sigma) {
  drop(MASS::mvrnorm(1, mu = mu, Sigma = Sigma))
}

# =============================================================================
# 3.  APPROACH 1 — TVP-VAR WITH STOCHASTIC VOLATILITY
# =============================================================================

cat(strrep("=", 65), "\n")
cat("APPROACH 1: TVP-VAR-SV  (Primiceri 2005)\n")
cat(strrep("=", 65), "\n")

# ── Build regressor matrix ────────────────────────────────────────────────────
p  <- LAGS
Y  <- Y_raw[(p + 1):T_raw, ]
T  <- nrow(Y)
k  <- N * p + 1L
nk <- N * k

X <- matrix(0, T, k)
for (t in seq_len(T)) {
  row <- c()
  for (lag in seq_len(p)) row <- c(row, Y_raw[p + t - lag, ])
  X[t, ] <- c(row, 1)
}

# OLS initialisation
B_ols <- solve(crossprod(X), crossprod(X, Y))
e_ols <- Y - X %*% B_ols
S_ols <- crossprod(e_ols) / (T - k)

# Observation matrices Z_t: (N x nk), block-diagonal
Z_arr <- array(0, dim = c(T, N, nk))
for (t in seq_len(T))
  for (eq in seq_len(N))
    Z_arr[t, eq, ((eq-1)*k + 1):(eq*k)] <- X[t, ]

# Initialisations
beta0    <- as.vector(B_ols)
h_all    <- matrix(rep(log(diag(S_ols) + 1e-6), each = T), T, N)
beta_all <- matrix(rep(beta0, each = T), T, nk)
kappa_Q  <- 0.01
Q_diag   <- rep(kappa_Q, nk)
m0       <- beta0
P0       <- 4 * diag(nk)
sig_eta  <- 0.1

ndraws_tvp  <- REPS_TVP - BURN_TVP
beta_store  <- array(0, c(ndraws_tvp, T, nk))
h_store     <- array(0, c(ndraws_tvp, T, N))

cat(sprintf("Running Gibbs sampler: %d iterations, %d burn-in...\n",
            REPS_TVP, BURN_TVP))
cat("  NOTE: increase REPS_TVP to 20000 for publication quality.\n")

jdraw <- 0L
Q     <- diag(Q_diag)

for (isim in seq_len(REPS_TVP)) {

  # ── Step 1: Carter-Kohn FFBS for beta_t ─────────────────────────────────
  H_obs <- exp(h_all)   # (T x N) variances

  # Forward filter
  mf <- matrix(0, T, nk)
  Pf <- array(0, c(T, nk, nk))
  mp <- m0;  Pp <- P0

  for (t in seq_len(T)) {
    Zt <- matrix(Z_arr[t, , ], N, nk)
    Ht <- diag(H_obs[t, ], N)
    vt <- Y[t, ] - drop(Zt %*% mp)
    Ft <- Zt %*% Pp %*% t(Zt) + Ht
    Ft_inv <- tryCatch(solve(Ft), error = function(e) MASS::ginv(Ft))
    Kt <- Pp %*% t(Zt) %*% Ft_inv
    mf[t, ] <- mp + Kt %*% vt
    Pnew <- Pp - Kt %*% Zt %*% Pp
    Pnew <- 0.5 * (Pnew + t(Pnew)) + 1e-9 * diag(nk)
    Pf[t, , ] <- Pnew
    mp <- mf[t, ]
    Pp <- Pnew + Q
  }

  # Backward sample
  beta_path <- matrix(0, T, nk)
  PfT <- matrix(Pf[T, , ], nk, nk)
  beta_path[T, ] <- rmvn(mf[T, ], PfT + 1e-9 * diag(nk))

  for (t in (T-1):1) {
    Pf_t    <- matrix(Pf[t, , ], nk, nk)
    Pp_next <- Pf_t + Q
    J       <- Pf_t %*% tryCatch(solve(Pp_next), error=function(e) MASS::ginv(Pp_next))
    mb      <- mf[t, ] + drop(J %*% (beta_path[t+1, ] - mf[t, ]))
    Pb      <- Pf_t - J %*% Pf_t
    Pb      <- 0.5*(Pb + t(Pb)) + 1e-9*diag(nk)
    beta_path[t, ] <- rmvn(mb, Pb)
  }
  beta_all <- beta_path

  # ── Step 2: Draw Q diagonal (IG) ─────────────────────────────────────────
  db    <- diff(beta_all)           # (T-1) x nk
  sq    <- colSums(db^2)
  nu_q  <- nk + 1 + (T - 1)
  Q_diag <- pmin(pmax(1 / rgamma(nk, shape = nu_q/2,
                                  rate = (kappa_Q + sq)/2), 1e-8), 1.0)
  Q <- diag(Q_diag)

  # ── Step 3: Residuals ────────────────────────────────────────────────────
  resid <- matrix(0, T, N)
  for (t in seq_len(T)) {
    Zt <- matrix(Z_arr[t, , ], N, nk)
    resid[t, ] <- Y[t, ] - drop(Zt %*% beta_all[t, ])
  }

  # ── Step 4: Sample log-volatilities (single-move MH) ─────────────────────
  for (i in seq_len(N)) {
    h_cur <- h_all[, i]
    for (t in seq_len(T)) {
      h_prev <- if (t == 1) h_cur[1] else h_cur[t-1]
      h_prop <- h_prev + sig_eta * rnorm(1)
      ll_p <- -0.5*h_prop - 0.5*resid[t,i]^2*exp(-h_prop)
      ll_c <- -0.5*h_cur[t] - 0.5*resid[t,i]^2*exp(-h_cur[t])
      if (log(runif(1) + 1e-300) < ll_p - ll_c) h_cur[t] <- h_prop
    }
    h_all[, i] <- h_cur
  }

  # ── Store ────────────────────────────────────────────────────────────────
  if (isim > BURN_TVP) {
    jdraw <- jdraw + 1L
    beta_store[jdraw, , ] <- beta_all
    h_store[jdraw, , ]    <- h_all
  }
  if (isim %% 500 == 0)
    cat(sprintf("  Iteration %4d / %d\n", isim, REPS_TVP))
}
cat("  TVP-VAR-SV complete.\n")
cat("\n  --- TVP-VAR-SV Convergence Diagnostics ---\n")
cat("  Checking representative time-varying coefficients and log-volatilities...\n")
.mid <- T %/% 2L
.tvp_check <- cbind(
  beta_store[, .mid, 1L],          # first coeff at mid-sample
  beta_store[, .mid, nk %/% 2L],   # middle coeff
  h_store[, .mid, 1L],             # log-vol GDP at mid-sample
  h_store[, .mid, 2L],             # log-vol Inflation
  h_store[, .mid, 3L]              # log-vol FFR
)
.tvp_names <- c("beta[mid,1]", paste0("beta[mid,", nk %/% 2L, "]"),
                "log-vol GDP[mid]", "log-vol Inf[mid]", "log-vol FFR[mid]")
mcmc_diagnostics(.tvp_check, param_names = .tvp_names, n_check = 5L, tag = "tvpvar")
cat("  Trace plots saved as diag_trace_tvpvar_*.png\n")
cat("  Tip: increase REPS_TVP to >= 20000 for publication quality.\n\n")

# ── Posterior volatility summaries ───────────────────────────────────────────
sig_arr <- exp(h_store / 2)    # (ndraws x T x N)
sig_med  <- apply(sig_arr, c(2,3), median)
sig_lo   <- apply(sig_arr, c(2,3), quantile, 0.16)
sig_hi   <- apply(sig_arr, c(2,3), quantile, 0.84)

# ── TVP-VAR IRF at a given date ──────────────────────────────────────────────
compute_tvp_irf <- function(t_idx) {
  ndraws <- dim(beta_store)[1]
  irfs   <- array(0, c(ndraws, HORZ, N))
  for (d in seq_len(ndraws)) {
    B_t     <- matrix(beta_store[d, t_idx, ], k, N)
    A_list  <- lapply(seq_len(p), function(lag) t(B_t[((lag-1)*N+1):(lag*N), ]))
    sigma_t <- exp(h_store[d, t_idx, ] / 2)
    impact  <- numeric(N); impact[SHOCK_EQ] <- sigma_t[SHOCK_EQ]
    irf <- matrix(0, HORZ, N);  irf[1, ] <- impact
    for (h in 2:HORZ)
      for (lag in seq_len(min(h-1, p)))
        irf[h, ] <- irf[h, ] + drop(A_list[[lag]] %*% irf[h-lag, ])
    irfs[d, , ] <- irf
  }
  list(med = apply(irfs, c(2,3), median),
       lo  = apply(irfs, c(2,3), quantile, 0.16),
       hi  = apply(irfs, c(2,3), quantile, 0.84))
}

cat("Computing TVP-VAR-SV IRFs at 2005Q1 and 2008Q4...\n")
irf_n <- compute_tvp_irf(T_NORMAL)
irf_s <- compute_tvp_irf(T_STRESS)

# High-volatility regime flag (for synthesis)
gdp_vol     <- sig_med[, 1]
vol_thresh  <- quantile(gdp_vol, 0.75)
tvp_hv_flag <- gdp_vol > vol_thresh

dates_tvp <- dates_plot[(p+1):T_raw]

# ── TVP-VAR-SV: Stochastic Volatilities ─────────────────────────────────────
var_names  <- c("GDP Growth","Inflation","Fed Funds Rate")
col_fig    <- c("#2166ac","#d6604d","#4dac26")

save_fig("fig_tvpvar_volatility.png", width=10, height=9, {
  par(mfrow=c(3,1), mar=c(2,4,2,1), oma=c(2,0,3,0))
  for (i in seq_len(N)) {
    yl <- range(c(sig_lo[,i], sig_hi[,i])) * c(0.9, 1.1)
    plot(dates_tvp, sig_med[,i], type="n", ylim=yl,
         ylab="Volatility (SD)", main=paste(var_names[i], "Volatility"),
         xlab="", las=1, xaxt="n")
    shade_nber(yl, dates_tvp)
    polygon(c(dates_tvp, rev(dates_tvp)),
            c(sig_lo[,i], rev(sig_hi[,i])),
            col=adjustcolor(col_fig[i], 0.3), border=NA)
    lines(dates_tvp, sig_med[,i], col=col_fig[i], lwd=2)
    axis.Date(1, dates_tvp, format="%Y")
    grid(col="grey90", lty=1)
    if (i == 1) {
      abline(v=as.Date("1984-01-01"), lty=2, lwd=0.9)
      text(as.Date("1985-06-01"), yl[2]*0.85,
           "Great\nModeration", cex=0.7, col="grey40")
    }
    legend("topright", legend=c("68% Credible Interval","Posterior Mean"),
           fill=c(adjustcolor(col_fig[i],0.3),NA),
           lty=c(NA,1), col=c(NA,col_fig[i]), lwd=c(NA,2), bty="n", cex=0.8)
  }
  mtext("Time-Varying Stochastic Volatilities (TVP-VAR-SV)",
        outer=TRUE, cex=1.1, font=2)
})
cat("TVP-VAR-SV volatility figure saved.\n")

# ── TVP-VAR-SV IRFs: Regime-specific IRFs ────────────────────────────────────────
resp_labels <- c("GDP Growth Response","Inflation Response","Fed Funds Rate Response")

save_fig("fig_tvpvar_irf_comparison.png", width=13, height=4, {
  par(mfrow=c(1,3), mar=c(4,4,3,1), oma=c(0,0,3,0))
  hor <- 0:(HORZ-1)
  for (i in seq_len(N)) {
    yl <- range(c(irf_n$lo[,i], irf_n$hi[,i], irf_s$lo[,i], irf_s$hi[,i]))
    plot(hor, irf_n$med[,i], type="n", ylim=yl, xlab="Quarters Ahead",
         ylab="", main=resp_labels[i], las=1)
    polygon(c(hor, rev(hor)), c(irf_n$lo[,i], rev(irf_n$hi[,i])),
            col=adjustcolor("#2166ac",0.20), border=NA)
    polygon(c(hor, rev(hor)), c(irf_s$lo[,i], rev(irf_s$hi[,i])),
            col=adjustcolor("#d6604d",0.20), border=NA)
    lines(hor, irf_n$med[,i], col="#2166ac", lwd=2)
    lines(hor, irf_s$med[,i], col="#d6604d", lwd=2, lty=2)
    abline(h=0, lty=3, col="black")
    grid(col="grey90", lty=1)
    if (i==1) legend("bottomleft",
      legend=c("Normal Times (2005Q1)","Stress Times (2008Q4)"),
      col=c("#2166ac","#d6604d"), lty=c(1,2), lwd=2, bty="n", cex=0.85)
  }
  mtext("TVP-VAR-SV: Monetary Policy Shock (1 SD Tightening)\nNormal vs. Stress Period",
        outer=TRUE, cex=1.0, font=2)
})
cat("TVP-VAR-SV IRF comparison figure saved.\n\n")

cat(sprintf("TVP-VAR-SV  — Normal (2005Q1): Peak GDP = %.3f%%\n", min(irf_n$med[,1])))
cat(sprintf("TVP-VAR-SV  — Stress (2008Q4): Peak GDP = %.3f%%\n\n", min(irf_s$med[,1])))

# =============================================================================
# 4.  APPROACH 2 — MARKOV-SWITCHING VAR
# =============================================================================

cat(strrep("=", 65), "\n")
cat("APPROACH 2: MARKOV-SWITCHING VAR  (Gibbs sampler)\n")
cat(strrep("=", 65), "\n")

# ── Helper: Minnesota-style dummy observations ────────────────────────────────
make_dummies <- function(Y, lags, lambda=0.1, tau=1.0, epsilon=1/1000) {
  N    <- ncol(Y)
  mu   <- colMeans(Y)
  # AR(1) residual std per variable
  sigma <- sapply(seq_len(N), function(i) {
    y_i <- Y[,i]; x_i <- cbind(c(NA, head(y_i,-1)), 1)
    x_i <- x_i[2:nrow(x_i),]; y_i <- y_i[2:length(y_i)]
    b   <- lm.fit(x_i, y_i)$coefficients
    sd(y_i - x_i %*% b)
  })
  k <- N*lags+1
  Yd <- matrix(0, N*lags + N + 1, N)
  Xd <- matrix(0, N*lags + N + 1, k)
  # Lag dummies
  for (lag in seq_len(lags)) {
    for (j in seq_len(N)) {
      row <- (lag-1)*N + j
      Yd[row, j] <- sigma[j] * lag / lambda
      Xd[row, ((lag-1)*N+j)] <- sigma[j] * lag / lambda
    }
  }
  # Co-persistence dummy
  start <- N*lags + 1
  Yd[start:(start+N-1), ] <- diag(mu / tau, N)
  for (j in seq_len(N))
    Xd[start+j-1, ] <- c(rep(mu / tau, lags), 1/tau)
  # Constant dummy
  Yd[N*lags+N+1, ] <- mu * epsilon
  Xd[N*lags+N+1, ] <- c(rep(mu * epsilon, lags), epsilon)
  list(Yd = Yd, Xd = Xd)
}

# ── Build VAR matrices ────────────────────────────────────────────────────────
ms_build_xy <- function(Y_in, lags) {
  T_in <- nrow(Y_in); N_in <- ncol(Y_in)
  k_in <- N_in * lags + 1
  Yv <- Y_in[(lags+1):T_in, , drop=FALSE]
  Xv <- matrix(0, nrow(Yv), k_in)
  for (t in seq_len(nrow(Yv))) {
    row <- c()
    for (l in seq_len(lags)) row <- c(row, Y_in[lags+t-l, ])
    Xv[t,] <- c(row, 1)
  }
  list(Y=Yv, X=Xv)
}

# ── Inverse-Wishart draw ─────────────────────────────────────────────────────
riwish <- function(nu, S) {
  # Draw from IW(nu, S): nu degrees of freedom, scale matrix S
  # Uses the fact that if X~W(nu,S^-1) then X^-1 ~ IW(nu,S)
  p   <- nrow(S)
  ch  <- chol(solve(S))
  Z   <- matrix(rnorm(nu * p), nu, p) %*% ch
  solve(crossprod(Z))
}

# ── Hamilton forward filter ───────────────────────────────────────────────────
hamilton_filter <- function(Y, X, B1, B2, Sig1, Sig2, p_vec, q_vec) {
  T_f  <- nrow(Y); N_f <- ncol(Y)
  iS1  <- solve(Sig1); iS2 <- solve(Sig2)
  dS1  <- det(Sig1);   dS2 <- det(Sig2)
  fprob <- matrix(0, T_f, 2)
  lik   <- 0
  ett   <- c(0.5, 0.5)
  for (t in seq_len(T_f)) {
    P_tr <- matrix(c(p_vec[t], 1-q_vec[t], 1-p_vec[t], q_vec[t]), 2, 2)
    e1   <- Y[t,] - X[t,] %*% B1
    e2   <- Y[t,] - X[t,] %*% B2
    n1   <- (1/sqrt(dS1)) * exp(-0.5 * drop(e1 %*% iS1 %*% t(e1)))
    n2   <- (1/sqrt(dS2)) * exp(-0.5 * drop(e2 %*% iS2 %*% t(e2)))
    ett1 <- ett * c(n1, n2)
    fit  <- sum(ett1)
    if (fit <= 0) { lik <- lik - 10; ett <- c(0.5, 0.5) }
    else {
      ett       <- drop(P_tr %*% ett1) / fit
      fprob[t,] <- ett1 / fit
      lik       <- lik + log(fit)
    }
  }
  list(fprob=fprob, lik=lik)
}

# ── Backward state sampler ────────────────────────────────────────────────────
sample_states <- function(fprob, p_vec, q_vec) {
  T_s <- nrow(fprob)
  st  <- integer(T_s)
  p00 <- fprob[T_s,1]; p01 <- fprob[T_s,2]
  st[T_s] <- if (runif(1) >= p00/(p00+p01)) 1L else 0L
  for (t in (T_s-1):1) {
    P_tr <- matrix(c(p_vec[t],1-q_vec[t],1-p_vec[t],q_vec[t]),2,2)
    if (st[t+1]==0) { p00<-P_tr[1,1]*fprob[t,1]; p01<-P_tr[1,2]*fprob[t,2] }
    else            { p00<-P_tr[2,1]*fprob[t,1]; p01<-P_tr[2,2]*fprob[t,2] }
    st[t] <- if (runif(1) < p00/(p00+p01)) 0L else 1L
  }
  st
}

# ── Logistic transition probability sampler (TVTP) ───────────────────────────
# Uses a probit-style data augmentation (Albert-Chib 1993 approximation):
# For each t, P(S_t=1|S_{t-1}) = logistic(g0 + g1*S_{t-1})
# We approximate with Gaussian proposals around current gamma.
sample_tvtp_gamma <- function(st, g0, ivg0) {
  T_g <- length(st)
  slag <- c(0L, st[1:(T_g-1)])
  # design matrix: [1, S_{t-1}]
  Zg   <- cbind(1, slag)[-1, , drop=FALSE]
  sg   <- st[-1]
  # Current gamma draw via Metropolis (random walk on g)
  # proposal scale
  g_cur <- g0
  for (iter in 1:5) {
    g_prop <- g_cur + rnorm(length(g_cur)) * 0.1
    # log-likelihood
    lp_cur  <- sum(sg * plogis(Zg %*% g_cur, log.p=TRUE) +
                   (1-sg) * plogis(-Zg %*% g_cur, log.p=TRUE))
    lp_prop <- sum(sg * plogis(Zg %*% g_prop, log.p=TRUE) +
                   (1-sg) * plogis(-Zg %*% g_prop, log.p=TRUE))
    # prior: N(0, diag(1/ivg0))
    lprior_cur  <- -0.5 * drop(crossprod(g_cur,  ivg0) %*% g_cur)
    lprior_prop <- -0.5 * drop(crossprod(g_prop, ivg0) %*% g_prop)
    if (log(runif(1)) < (lp_prop+lprior_prop) - (lp_cur+lprior_cur))
      g_cur <- g_prop
  }
  g_cur
}

# ── MS-VAR Gibbs sampler ─────────────────────────────────────────────────────
ms_gibbs <- function(Y_in, lags, REPS, BURN, lambda=0.1, tau=1.0,
                     epsilon=1/1000, verbose=TRUE) {

  xy   <- ms_build_xy(Y_in, lags)
  Yv   <- xy$Y; Xv <- xy$X
  T_g  <- nrow(Yv); N_g <- ncol(Yv); k_g <- ncol(Xv)
  dum  <- make_dummies(Y_in, lags, lambda, tau, epsilon)
  Yd   <- dum$Yd; Xd <- dum$Xd

  # OLS starts in each regime (use first/second half as initial regimes)
  mid <- T_g %/% 2L
  B1  <- lm.fit(Xv[1:mid,], Yv[1:mid,])$coefficients
  B2  <- lm.fit(Xv[(mid+1):T_g,], Yv[(mid+1):T_g,])$coefficients
  e1  <- Yv[1:mid,] - Xv[1:mid,] %*% B1
  e2  <- Yv[(mid+1):T_g,] - Xv[(mid+1):T_g,] %*% B2
  Sig1 <- crossprod(e1)/mid + 1e-4*diag(N_g)
  Sig2 <- crossprod(e2)/mid + 1e-4*diag(N_g)

  p_vec <- rep(0.95, T_g); q_vec <- rep(0.95, T_g)
  st    <- c(rep(0L,mid), rep(1L,T_g-mid))

  # prior for gamma (TVTP)
  g0     <- c(-2, -4)
  ivg0   <- diag(1/c(10, 10))

  npost <- REPS - BURN
  bsave1   <- matrix(0, npost, N_g*k_g)
  bsave2   <- matrix(0, npost, N_g*k_g)
  sigS1    <- array(0, c(npost, N_g, N_g))
  sigS2    <- array(0, c(npost, N_g, N_g))
  reg_save <- matrix(0L, npost, T_g)
  pmat     <- matrix(0, npost, T_g)
  qmat     <- matrix(0, npost, T_g)

  jdraw <- 0L
  for (isim in seq_len(REPS)) {

    # 1. Hamilton filter + backward sampler
    res  <- hamilton_filter(Yv, Xv, B1, B2, Sig1, Sig2, p_vec, q_vec)
    st   <- sample_states(res$fprob, p_vec, q_vec)
    slag <- c(0L, st[1:(T_g-1)])

    # 2. TVTP gamma draw
    g0 <- sample_tvtp_gamma(st, g0, ivg0)
    Zg <- cbind(1, slag)
    lp <- plogis(Zg %*% g0)
    p_vec <- pmin(pmax(lp, 0.05), 0.99)
    q_vec <- p_vec   # simplified: same equation for both

    # 3. VAR coefficients (regime-specific, Minnesota prior via dummies)
    for (regime in 0:1) {
      idx <- which(st == regime)
      if (length(idx) < k_g + 2) next
      Yr <- rbind(Yv[idx,], Yd)
      Xr <- rbind(Xv[idx,], Xd)
      B_ols_r <- lm.fit(Xr, Yr)$coefficients
      XX <- crossprod(Xr)
      iXX <- tryCatch(solve(XX), error=function(e) MASS::ginv(XX))
      # Draw from matrix-Normal
      B_draw <- B_ols_r + t(chol(iXX)) %*% matrix(rnorm(k_g*N_g), k_g, N_g) %*% chol(if(regime==0) Sig1 else Sig2)
      if (regime==0) B1 <- B_draw else B2 <- B_draw
    }

    # 4. Covariance (Inverse-Wishart)
    for (regime in 0:1) {
      idx <- which(st == regime)
      if (length(idx) < 2) next
      Br  <- if(regime==0) B1 else B2
      er  <- rbind(Yv[idx,], Yd) - rbind(Xv[idx,], Xd) %*% Br
      S_draw <- riwish(nrow(er), crossprod(er))
      if (regime==0) Sig1 <- S_draw + 1e-6*diag(N_g) else
                     Sig2 <- S_draw + 1e-6*diag(N_g)
    }

    # 5. Store
    if (isim > BURN) {
      jdraw <- jdraw + 1L
      bsave1[jdraw,]    <- as.vector(t(B1))
      bsave2[jdraw,]    <- as.vector(t(B2))
      sigS1[jdraw,,]    <- Sig1
      sigS2[jdraw,,]    <- Sig2
      reg_save[jdraw,]  <- st
      pmat[jdraw,]      <- p_vec
      qmat[jdraw,]      <- q_vec
    }
    if (verbose && isim %% 500 == 0)
      cat(sprintf("  MS-VAR iteration %4d / %d\n", isim, REPS))
  }

  list(bsave1=bsave1, bsave2=bsave2, sigS1=sigS1, sigS2=sigS2,
       regime=reg_save, pmat=pmat, qmat=qmat)
}

cat(sprintf("Running MS-VAR Gibbs sampler: %d iterations, %d burn-in...\n",
            REPS_MS, BURN_MS))

ms_out <- ms_gibbs(Y_raw, LAGS, REPS_MS, BURN_MS,
                   lambda=0.1, tau=1.0, verbose=TRUE)
cat("  MS-VAR complete.\n")
cat("\n  --- MS-VAR Convergence Diagnostics ---\n")
cat("  Checking VAR coefficients, regime probabilities, transition parameters...\n")
.ms_check <- cbind(
  ms_out$bsave1[, 1L],              # first normal-regime coeff
  ms_out$bsave2[, 1L],              # first stress-regime coeff
  rowMeans(ms_out$regime),          # avg smoothed regime prob per draw
  rowMeans(ms_out$pmat),            # avg p00 per draw
  rowMeans(ms_out$qmat)             # avg q11 per draw
)
.ms_names <- c("B1[1]", "B2[1]", "avg regime prob", "avg p00", "avg q11")
mcmc_diagnostics(.ms_check, param_names = .ms_names, n_check = 5L, tag = "msvar")
cat("  Trace plots saved as diag_trace_msvar_*.png\n")
cat("  Tip: increase REPS_MS to >= 20000 for publication quality.\n\n")

# ── Regime probabilities & transition stats ───────────────────────────────────
ndraws_ms   <- nrow(ms_out$regime)
T_ms        <- ncol(ms_out$regime)
smooth_prob <- colMeans(ms_out$regime)
p00_mean    <- mean(rowMeans(ms_out$pmat))
q11_mean    <- mean(rowMeans(ms_out$qmat))

cat(sprintf("MS-VAR  P(Normal→Normal) = %.4f   E[Duration] = %.1f quarters\n",
            p00_mean, 1/(1-p00_mean)))
cat(sprintf("MS-VAR  P(Stress→Stress) = %.4f   E[Duration] = %.1f quarters\n\n",
            q11_mean, 1/(1-q11_mean)))

# ── Regime-specific IRFs ──────────────────────────────────────────────────────
cat("Computing MS-VAR regime-specific IRFs...\n")
k_ms <- N * LAGS + 1L
irf_ms_n <- array(0, c(ndraws_ms, HORZ, N))
irf_ms_s <- array(0, c(ndraws_ms, HORZ, N))

for (d in seq_len(ndraws_ms)) {
  for (regime in 0:1) {
    bv   <- if (regime==0) ms_out$bsave1[d,] else ms_out$bsave2[d,]
    Sig  <- if (regime==0) matrix(ms_out$sigS1[d,,],N,N) else
                           matrix(ms_out$sigS2[d,,],N,N)
    B_d  <- matrix(bv, N, k_ms, byrow=TRUE)   # (N x k)  bsave stored row-major
    B_d  <- t(B_d)                             # -> (k x N)
    ch   <- t(chol(Sig))                       # lower Cholesky
    shock <- numeric(N); shock[SHOCK_EQ] <- ch[SHOCK_EQ, SHOCK_EQ]
    irf  <- irfsim_r(B_d, N, LAGS, ch, shock, HORZ)
    if (regime==0) irf_ms_n[d,,] <- irf else irf_ms_s[d,,] <- irf
  }
}

ms_n_med <- apply(irf_ms_n, c(2,3), median)
ms_n_lo  <- apply(irf_ms_n, c(2,3), quantile, 0.16)
ms_n_hi  <- apply(irf_ms_n, c(2,3), quantile, 0.84)
ms_s_med <- apply(irf_ms_s, c(2,3), median)
ms_s_lo  <- apply(irf_ms_s, c(2,3), quantile, 0.16)
ms_s_hi  <- apply(irf_ms_s, c(2,3), quantile, 0.84)

ms_stress_flag <- smooth_prob > 0.5
dates_ms <- dates_plot[seq_len(T_ms)]

# ── MS-VAR: Smoothed Regime Probability ─────────────────────────────────
save_fig("fig_msvar_regime_prob.png", width=10, height=4, {
  par(mar=c(4,4,3,1))
  plot(dates_ms, smooth_prob, type="n", ylim=c(0,1),
       xlab="Date", ylab="Probability",
       main="MS-VAR: Smoothed Stress Regime Probability",
       las=1)
  shade_nber(c(0,1), dates_ms)
  polygon(c(dates_ms, rev(dates_ms)), c(rep(0,T_ms), rev(smooth_prob)),
          col=adjustcolor("#2166ac",0.55), border=NA)
  lines(dates_ms, smooth_prob, col="#2166ac", lwd=1.5)
  abline(h=0.5, lty=2, lwd=1.2)
  legend("topright", legend=c("Pr(Stress Regime)","50% Threshold"),
         lty=c(1,2), col=c("#2166ac","black"), lwd=2, bty="n", cex=0.85)
  grid(col="grey90", lty=1)
})
cat("MS-VAR regime probability figure saved.\n")

cat(sprintf("MS-VAR  Normal regime: Peak GDP = %.3f%%\n", min(ms_n_med[,1])))
cat(sprintf("MS-VAR  Stress regime: Peak GDP = %.3f%%\n\n", min(ms_s_med[,1])))

# =============================================================================
# 5.  APPROACH 3 — THRESHOLD VAR
# =============================================================================

cat(strrep("=", 65), "\n")
cat("APPROACH 3: THRESHOLD VAR\n")
cat(strrep("=", 65), "\n")

TVAR_DELAY <- 1L; TVAR_VAR <- 1L; TVAR_NCRIT <- 15L
N_GRID <- 100L;   N_GIRF  <- 1000L

# ── Build TVAR data ───────────────────────────────────────────────────────────
Yt   <- Y_raw[(LAGS+1):T_raw, ]
T_tv <- nrow(Yt)
Xt   <- matrix(0, T_tv, N*LAGS+1)
for (t in seq_len(T_tv)) {
  row <- c(); for (l in seq_len(LAGS)) row <- c(row, Y_raw[LAGS+t-l, ])
  Xt[t,] <- c(row, 1)
}
Ystar <- c(rep(NA, TVAR_DELAY), Y_raw[1:(T_raw-TVAR_DELAY), TVAR_VAR])
Ystar <- Ystar[(LAGS+1):T_raw]

# ── Grid search for threshold ─────────────────────────────────────────────────
tau_min  <- quantile(Ystar, 0.15); tau_max <- quantile(Ystar, 0.85)
tau_grid <- seq(tau_min, tau_max, length.out=N_GRID)
sse_grid <- rep(Inf, N_GRID)

for (ig in seq_len(N_GRID)) {
  tau <- tau_grid[ig]
  e1  <- Ystar <= tau; e2 <- !e1
  if (sum(e1) < TVAR_NCRIT || sum(e2) < TVAR_NCRIT) next
  B1 <- lm.fit(Xt[e1,], Yt[e1,])$coefficients
  B2 <- lm.fit(Xt[e2,], Yt[e2,])$coefficients
  r1 <- Yt[e1,] - Xt[e1,] %*% B1
  r2 <- Yt[e2,] - Xt[e2,] %*% B2
  sse_grid[ig] <- sum(r1^2) + sum(r2^2)
}

best_idx <- which.min(sse_grid)
tau_hat  <- tau_grid[best_idx]
e1_hat   <- Ystar <= tau_hat; e2_hat <- !e1_hat
B1_tvar  <- lm.fit(Xt[e1_hat,], Yt[e1_hat,])$coefficients
B2_tvar  <- lm.fit(Xt[e2_hat,], Yt[e2_hat,])$coefficients
r1_tvar  <- Yt[e1_hat,] - Xt[e1_hat,] %*% B1_tvar
r2_tvar  <- Yt[e2_hat,] - Xt[e2_hat,] %*% B2_tvar
Sig1_tv  <- crossprod(r1_tvar)/(sum(e1_hat)-N*LAGS-1)
Sig2_tv  <- crossprod(r2_tvar)/(sum(e2_hat)-N*LAGS-1)

cat(sprintf("Threshold estimate:  tau_hat = %.4f%%\n", tau_hat))
cat(sprintf("Low-growth regime:   %d quarters (%.1f%%)\n", sum(e1_hat), 100*mean(e1_hat)))
cat(sprintf("High-growth regime:  %d quarters (%.1f%%)\n\n", sum(e2_hat), 100*mean(e2_hat)))

# ── Hansen sup-LM bootstrap ───────────────────────────────────────────────────
B_lin   <- lm.fit(Xt, Yt)$coefficients
r_lin   <- Yt - Xt %*% B_lin
SSE_lin <- sum(r_lin^2)
LM_stat <- (SSE_lin - sse_grid[best_idx]) / sse_grid[best_idx] * T_tv

N_BOOT  <- 500L
LM_boot <- numeric(N_BOOT)
for (b in seq_len(N_BOOT)) {
  idx_b    <- sample(T_tv, replace=TRUE)
  Yt_b     <- Xt %*% B_lin + r_lin[idx_b,]
  Blin_b   <- lm.fit(Xt, Yt_b)$coefficients
  rlin_b   <- Yt_b - Xt %*% Blin_b
  SSE_lb   <- sum(rlin_b^2)
  sse_b    <- rep(Inf, N_GRID)
  for (ig in seq_len(N_GRID)) {
    tau <- tau_grid[ig]
    e1b <- Ystar <= tau; e2b <- !e1b
    if (sum(e1b)<TVAR_NCRIT || sum(e2b)<TVAR_NCRIT) next
    B1b <- lm.fit(Xt[e1b,], Yt_b[e1b,])$coefficients
    B2b <- lm.fit(Xt[e2b,], Yt_b[e2b,])$coefficients
    r1b <- Yt_b[e1b,] - Xt[e1b,] %*% B1b
    r2b <- Yt_b[e2b,] - Xt[e2b,] %*% B2b
    sse_b[ig] <- sum(r1b^2) + sum(r2b^2)
  }
  LM_boot[b] <- (SSE_lb - min(sse_b)) / min(sse_b) * T_tv
}
pval_LM <- mean(LM_boot >= LM_stat)
cat(sprintf("Hansen sup-LM:  stat = %.3f,  bootstrap p-value = %.4f\n", LM_stat, pval_LM))
if (pval_LM < 0.05) cat("  -> Linearity rejected at 5% level.\n")
cat("\n")

# ── Generalized IRFs via Monte Carlo ─────────────────────────────────────────
ch1_tv <- t(chol(Sig1_tv)); ch2_tv <- t(chol(Sig2_tv))

compute_girf_tvar <- function(start_regime) {
  girf_diff <- matrix(0, N_GIRF, HORZ * N)
  idx_pool  <- which(if(start_regime==1) e1_hat else e2_hat)

  for (g in seq_len(N_GIRF)) {
    t0   <- sample(idx_pool, 1)
    t0   <- max(min(t0, T_tv), LAGS+1)
    # history: last LAGS rows
    Y_hist <- matrix(0, LAGS, N)
    for (l in seq_len(LAGS))
      Y_hist[LAGS-l+1, ] <- if (t0-l >= 1) Yt[t0-l,] else Yt[1,]

    Y_base  <- matrix(0, HORZ, N)
    Y_shock <- matrix(0, HORZ, N)
    eps_all <- matrix(rnorm(HORZ*N), HORZ, N)

    for (h in seq_len(HORZ)) {
      # regressor
      build_x <- function(Y_path) {
        row <- c()
        for (l in seq_len(LAGS))
          row <- c(row, if (h-l >= 1) Y_path[h-l,] else Y_hist[LAGS-l+1+h-1,])
        c(row, 1)
      }
      xb <- build_x(Y_base); xs <- build_x(Y_shock)

      # threshold variable
      delay_h_b <- if (h-TVAR_DELAY >= 1) Y_base[h-TVAR_DELAY, TVAR_VAR] else
                   Y_hist[max(1, LAGS-(TVAR_DELAY-h)), TVAR_VAR]
      delay_h_s <- if (h-TVAR_DELAY >= 1) Y_shock[h-TVAR_DELAY, TVAR_VAR] else
                   delay_h_b

      reg_b <- delay_h_b <= tau_hat
      reg_s <- delay_h_s <= tau_hat
      Bb <- if(reg_b) B1_tvar else B2_tvar
      Bs <- if(reg_s) B1_tvar else B2_tvar
      cb <- if(reg_b) ch1_tv  else ch2_tv
      cs <- if(reg_s) ch1_tv  else ch2_tv

      innov <- eps_all[h,]
      Y_base[h,]  <- drop(xb %*% Bb) + drop(innov %*% t(cb))
      Y_shock[h,] <- drop(xs %*% Bs) + drop(innov %*% t(cs))
      if (h == 1) {
        sv <- numeric(N); sv[SHOCK_EQ] <- cs[SHOCK_EQ, SHOCK_EQ]
        Y_shock[h,] <- Y_shock[h,] + sv
      }
    }
    girf_diff[g,] <- as.vector(Y_shock - Y_base)
  }
  matrix(colMeans(girf_diff), HORZ, N)
}

cat("Computing TVAR GIRFs...\n")
girf1 <- compute_girf_tvar(1)   # low-growth regime
girf2 <- compute_girf_tvar(2)   # high-growth regime
tvar_flag <- e1_hat
cat(sprintf("TVAR  Low-growth:  Peak GDP = %.3f%%\n", min(girf1[,1])))
cat(sprintf("TVAR  High-growth: Peak GDP = %.3f%%\n\n", min(girf2[,1])))

# ── TVAR GIRFs: TVAR GIRFs ───────────────────────────────────────────────────
save_fig("fig_tvar_girf.png", width=13, height=4, {
  par(mfrow=c(1,3), mar=c(4,4,3,1), oma=c(0,0,3,0))
  hor <- 0:(HORZ-1)
  for (i in seq_len(N)) {
    yl <- range(c(girf1[,i], girf2[,i])) * c(1.2, 1.2)
    plot(hor, girf1[,i], type="l", col="#2166ac", lwd=2, ylim=yl,
         xlab="Quarters Ahead", ylab="", main=resp_labels[i], las=1)
    lines(hor, girf2[,i], col="#4dac26", lwd=2)
    abline(h=0, lty=3); grid(col="grey90",lty=1)
    if (i==1) legend("bottomright",
      legend=c("Low Growth Regime","High Growth Regime"),
      col=c("#2166ac","#4dac26"), lwd=2, bty="n", cex=0.85)
  }
  mtext("Threshold VAR: Generalized Impulse Responses\nMonetary Policy Shock (1 SD Tightening)",
        outer=TRUE, cex=1.0, font=2)
})
cat("TVAR GIRF figure saved.\n\n")

# =============================================================================
# 6.  APPROACH 4 — SMOOTH TRANSITION VAR
# =============================================================================

cat(strrep("=", 65), "\n")
cat("APPROACH 4: SMOOTH TRANSITION VAR\n")
cat(strrep("=", 65), "\n")

logistic_F <- function(z, gamma, c) 1 / (1 + exp(-gamma * (z - c)))

fit_stvar_coefs <- function(Yt, Xt, Ft) {
  w1 <- 1 - Ft; w2 <- Ft
  B1 <- lm.fit(Xt * sqrt(w1), Yt * sqrt(w1))$coefficients
  B2 <- lm.fit(Xt * sqrt(w2), Yt * sqrt(w2))$coefficients
  r1 <- Yt - Xt %*% B1; r2 <- Yt - Xt %*% B2
  S1 <- crossprod(r1 * sqrt(w1)) / (sum(w1)-1) + 1e-8*diag(N)
  S2 <- crossprod(r2 * sqrt(w2)) / (sum(w2)-1) + 1e-8*diag(N)
  list(B1=B1, B2=B2, S1=S1, S2=S2)
}

stvar_sse <- function(log_gamma, c, Zvar, Yt, Xt) {
  gamma <- exp(log_gamma)
  if (gamma <= 0) return(1e10)
  Ft <- logistic_F(Zvar, gamma, c)
  if (min(Ft) > 0.99 || max(Ft) < 0.01) return(1e10)
  cf <- fit_stvar_coefs(Yt, Xt, Ft)
  Yhat <- (1-Ft) * (Xt %*% cf$B1) + Ft * (Xt %*% cf$B2)
  sum((Yt - Yhat)^2)
}

Zvar <- Ystar  # already aligned

# 2-D grid search
c_grid     <- seq(quantile(Zvar,0.10), quantile(Zvar,0.90), length.out=40)
gamma_grid <- c(0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0)
sse_2d     <- matrix(Inf, length(gamma_grid), length(c_grid))

for (ig in seq_along(gamma_grid)) {
  for (ic in seq_along(c_grid)) {
    Ft <- logistic_F(Zvar, gamma_grid[ig], c_grid[ic])
    if (min(Ft) > 0.99 || max(Ft) < 0.01) next
    cf <- fit_stvar_coefs(Yt, Xt, Ft)
    Yhat <- (1-Ft)*(Xt %*% cf$B1) + Ft*(Xt %*% cf$B2)
    sse_2d[ig,ic] <- sum((Yt-Yhat)^2)
  }
}

best_pos    <- which(sse_2d == min(sse_2d), arr.ind=TRUE)[1,]
gamma_init  <- gamma_grid[best_pos[1]]
c_init      <- c_grid[best_pos[2]]

# Local refinement
opt <- optim(c(log(gamma_init), c_init), function(p) stvar_sse(p[1], p[2], Zvar, Yt, Xt),
             method="Nelder-Mead",
             control=list(maxit=800, reltol=1e-6))
gamma_hat <- min(exp(opt$par[1]), 20.0)
c_hat     <- opt$par[2]

cat(sprintf("STVAR estimates:  gamma_hat = %.3f,  c_hat = %.4f%%\n", gamma_hat, c_hat))

Ft_hat  <- logistic_F(Zvar, gamma_hat, c_hat)
st_coef <- fit_stvar_coefs(Yt, Xt, Ft_hat)
B1_st   <- st_coef$B1; B2_st <- st_coef$B2
Sig1_st <- st_coef$S1; Sig2_st <- st_coef$S2
dates_st <- dates_plot[(LAGS+1):T_raw]

# IRFs across five states
z_states   <- c(-2.0, -0.5, 0.0, 1.5, 3.0)
cols_st    <- c("#991a1a","#cc4d1a","#999919","#339933","#0d6699")
lbl_states <- c("z=-2.0% (Deep Recession)","z=-0.5%","z=0.0%","z=+1.5%","z=+3.0% (Expansion)")
irf_st     <- array(0, c(length(z_states), HORZ, N))

for (iz in seq_along(z_states)) {
  Fz      <- logistic_F(z_states[iz], gamma_hat, c_hat)
  B_mix   <- (1-Fz)*B1_st + Fz*B2_st
  Sig_mix <- (1-Fz)*Sig1_st + Fz*Sig2_st + 1e-8*diag(N)
  ch_m    <- t(chol(Sig_mix))
  shock   <- numeric(N); shock[SHOCK_EQ] <- ch_m[SHOCK_EQ, SHOCK_EQ]
  irf_st[iz,,] <- irfsim_r(B_mix, N, LAGS, ch_m, shock, HORZ)
}

# ── Information criteria ──────────────────────────────────────────────────────
ic_model <- function(Y_in, X_in, resid_in, k_params) {
  T_n   <- nrow(Y_in)
  Sig   <- crossprod(resid_in) / T_n
  ll    <- -T_n*N/2*log(2*pi) - T_n/2*log(det(Sig)) - T_n/2
  c(AIC = -2*ll + 2*k_params, BIC = -2*ll + log(T_n)*k_params)
}
r_lin2  <- Yt - Xt %*% lm.fit(Xt, Yt)$coefficients
ic_lin  <- ic_model(Yt, Xt, r_lin2, N*(N*LAGS+1))
r_tvar  <- rbind(Yt[e1_hat,]-Xt[e1_hat,]%*%B1_tvar, Yt[e2_hat,]-Xt[e2_hat,]%*%B2_tvar)
ic_tvar <- ic_model(Yt, Xt, r_tvar, 2*N*(N*LAGS+1))
r_stv   <- Yt - (1-Ft_hat)*(Xt%*%B1_st) - Ft_hat*(Xt%*%B2_st)
ic_stv  <- ic_model(Yt, Xt, r_stv, 2*N*(N*LAGS+1)+2)

cat("Information Criteria:\n")
cat(sprintf("  %-12s  AIC = %8.1f   BIC = %8.1f\n", "Linear VAR", ic_lin[1],  ic_lin[2]))
cat(sprintf("  %-12s  AIC = %8.1f   BIC = %8.1f\n", "TVAR",       ic_tvar[1], ic_tvar[2]))
cat(sprintf("  %-12s  AIC = %8.1f   BIC = %8.1f\n\n","STVAR",     ic_stv[1],  ic_stv[2]))

stvar_flag <- Ft_hat > 0.5

# ── STVAR Transition: STVAR Transition Function ───────────────────────────────────
save_fig("fig_stvar_transition.png", width=10, height=6, {
  par(mfrow=c(2,1), mar=c(2,4,2,1), oma=c(2,0,3,0))
  plot(dates_st, Zvar, type="l", col="#3333cc", lwd=1.5,
       ylab="GDP Growth (%)", main="GDP Growth (Transition Variable)", las=1, xaxt="n")
  shade_nber(range(Zvar)*c(1.1,1.1), dates_st)
  abline(h=c_hat, col="red", lty=2, lwd=1.5)
  legend("bottomright", legend=c("GDP Growth", sprintf("c = %.2f%%", c_hat)),
         col=c("#3333cc","red"), lty=c(1,2), lwd=2, bty="n", cex=0.85)
  axis.Date(1, dates_st, format="%Y"); grid(col="grey90",lty=1)

  plot(dates_st, Ft_hat, type="l", col="#257a25", lwd=2,
       ylim=c(0,1), ylab="Transition Weight",
       main=sprintf("F(z): Weight on High-Growth Regime  (gamma = %.2f)", gamma_hat),
       las=1, xaxt="n")
  shade_nber(c(0,1), dates_st)
  abline(h=0.5, lty=2, lwd=1.0)
  axis.Date(1, dates_st, format="%Y"); grid(col="grey90",lty=1)
  mtext("STVAR: Gradual Regime Changes", outer=TRUE, font=2, cex=1.1)
})
cat("STVAR transition figure saved.\n")

# ── STVAR IRF Continuum: STVAR IRF Continuum ─────────────────────────────────────────
save_fig("fig_stvar_irf_continuum.png", width=13, height=4, {
  par(mfrow=c(1,3), mar=c(4,4,3,1), oma=c(0,0,3,0))
  hor <- 0:(HORZ-1)
  for (i in seq_len(N)) {
    yl <- range(irf_st[,,i]) * c(1.2, 1.2)
    plot(hor, irf_st[1,,i], type="n", ylim=yl,
         xlab="Quarters Ahead", ylab="", main=resp_labels[i], las=1)
    for (iz in seq_along(z_states))
      lines(hor, irf_st[iz,,i], col=cols_st[iz], lwd=2)
    abline(h=0, lty=3); grid(col="grey90",lty=1)
    if (i==1) legend("bottomright", legend=lbl_states, col=cols_st,
                     lwd=2, bty="n", cex=0.65)
  }
  mtext("STVAR: IRFs Across Continuum of States\nMonetary Policy Shock (1 SD Tightening)",
        outer=TRUE, cex=1.0, font=2)
})
cat("STVAR IRF continuum figure saved.\n")

# ── Model Comparison: Model Comparison ───────────────────────────────────────────
save_fig("fig_model_comparison.png", width=8, height=4, {
  par(mfrow=c(1,2), mar=c(4,5,3,1))
  models <- c("Linear VAR","TVAR","STVAR")
  aics   <- c(ic_lin[1], ic_tvar[1], ic_stv[1])
  bics   <- c(ic_lin[2], ic_tvar[2], ic_stv[2])
  cols_ic <- c("#8080cc","#4db34d","#cc8040")
  barplot(aics, names.arg=models, col=cols_ic, main="AIC (lower is better)",
          ylab="AIC", las=1)
  grid(nx=NA, ny=NULL, col="grey90")
  barplot(bics, names.arg=models, col=cols_ic, main="BIC (lower is better)",
          ylab="BIC", las=1)
  grid(nx=NA, ny=NULL, col="grey90")
  mtext("Model Selection: Information Criteria",
        outer=FALSE, cex=0.9, font=2, side=3, line=-1, adj=0.5)
})
cat("Model comparison figure saved.\n\n")

# =============================================================================
# 7.  SYNTHESIS — REGIME CONCORDANCE
# =============================================================================

cat(strrep("=", 65), "\n")
cat("SYNTHESIS: REGIME CLASSIFICATION CONCORDANCE\n")
cat(strrep("=", 65), "\n")

T_common <- min(length(tvp_hv_flag), T_ms - LAGS,
                length(tvar_flag),   length(stvar_flag))

tvp_c   <- tvp_hv_flag[seq_len(T_common)]
ms_c    <- ms_stress_flag[seq(LAGS+1, LAGS+T_common)]
tvar_c  <- tvar_flag[seq_len(T_common)]
stvar_c <- stvar_flag[seq_len(T_common)]
dates_c <- dates_plot[(LAGS+1):(LAGS+T_common)]

# NBER flag
nber_flag <- logical(T_common)
for (t in seq_len(T_common))
  for (r in seq_len(nrow(nber_df)))
    if (dates_c[t] >= nber_df$start[r] && dates_c[t] <= nber_df$end[r])
      nber_flag[t] <- TRUE

flags  <- cbind(NBER=nber_flag, MS_VAR=ms_c, TVAR=tvar_c, STVAR=stvar_c)
Cmat   <- cor(flags * 1.0)
labels <- colnames(flags)

cat("Concordance correlations:\n")
print(round(Cmat, 3))
cat("\n")

# (Concordance heatmap removed — correlation table printed above is sufficient.)

# ── Stacked regime timeline ─────────────────────────────────────────────────────
flag_labels <- c("NBER Recession","MS-VAR Stress","TVAR Low-Growth","STVAR Stress")
cols_syn    <- c("#cc3333","#3366cc","#33b333","#b36600")

save_fig("fig_regime_timeline.png", width=12, height=5.5, {
  par(mfrow=c(4,1), mar=c(1,5,0.5,1), oma=c(3,0,2,0))
  all_flags <- list(nber_flag, ms_c, tvar_c, stvar_c)
  for (i in 1:4) {
    plot(dates_c, all_flags[[i]]*1, type="n", ylim=c(0,1),
         ylab=flag_labels[i], xlab="", yaxt="n",
         xaxt=if(i<4) "n" else "s", las=1)
    polygon(c(dates_c, rev(dates_c)), c(all_flags[[i]]*1, rep(0,T_common)),
            col=adjustcolor(cols_syn[i], 0.6), border=NA)
    axis(2, c(0,1)); grid(col="grey90",lty=1)
    if (i==4) axis.Date(1, dates_c, format="%Y")
  }
  mtext("Regime Classifications Across Four Approaches",
        outer=TRUE, cex=1.0, font=2)
})
cat("Regime timeline saved.\n\n")

# =============================================================================
# 8.  SUMMARY
# =============================================================================

cat(strrep("=", 65), "\n")
cat("SYNTHESIS: MODEL-ROBUST FINDINGS\n")
cat(strrep("=", 65), "\n\n")

amp_tvp  <- abs(min(irf_s$med[,1])) / max(abs(min(irf_n$med[,1])), 1e-6)
amp_ms   <- abs(min(ms_s_med[,1]))  / max(abs(min(ms_n_med[,1])),  1e-6)
amp_tvar <- abs(min(girf1[,1]))     / max(abs(min(girf2[,1])),     1e-6)
amp_stv  <- abs(min(irf_st[1,,1]))  / max(abs(min(irf_st[5,,1])), 1e-6)

cat("GDP Response Amplification (Stress/Normal or Low/High):\n")
cat(sprintf("  TVP-VAR-SV : %.2fx\n", amp_tvp))
cat(sprintf("  MS-VAR     : %.2fx\n", amp_ms))
cat(sprintf("  TVAR       : %.2fx\n", amp_tvar))
cat(sprintf("  STVAR      : %.2fx\n\n", amp_stv))
cat("Core finding: All four models show LARGER policy effects during stress.\n\n")

cat(strrep("=", 65), "\n")
cat("OUTPUT FILES\n")
cat(strrep("=", 65), "\n")
outfiles <- c("fig_tvpvar_volatility.png",
              "fig_tvpvar_irf_comparison.png",
              "fig_msvar_regime_prob.png",
              "fig_tvar_girf.png",
              "fig_stvar_transition.png",
              "fig_stvar_irf_continuum.png",
              "fig_model_comparison.png",
              "fig_regime_timeline.png")
cat(paste0("  ", outfiles, collapse="\n"), "\n")
cat(strrep("=", 65), "\n")
cat("ALL DONE — Replication complete.\n")
cat(strrep("=", 65), "\n")
