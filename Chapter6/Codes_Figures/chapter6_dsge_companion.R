################################################################################
# Chapter 6: DSGE Models — Companion R Code
# Macroeconometrics Textbook
# Author: Alessia Paccagnini
#
# This file contains three self-contained sections:
#   SECTION 1: Table 6.1  — Business Cycle Statistics (HP filter on FRED-QD)
#   SECTION 2: Figure 6.1 — RBC Impulse Responses (Blanchard-Kahn / QZ)
#   SECTION 3: Figure 6.2 — Prior vs Posterior Distributions (SW 2007)
#
# Required packages: readxl
# Install with: install.packages("readxl")
#
# Calibration note:
#   delta = 0.025 throughout (consistent with BK example, exercises,
#   and calibration discussion in Chapter 6)
################################################################################

cat("================================================================\n")
cat("CHAPTER 6: DSGE MODELS — R COMPANION CODE\n")
cat("================================================================\n\n")


########################################################################
# SECTION 1: TABLE 6.1 — Business Cycle Statistics
########################################################################
# Data: FRED-QD dataset (McCracken & Ng, 2016)
# File: 2026-01-QD.xlsx
# Sample: 1960:Q1 to 2019:Q4 (pre-COVID, 240 observations)
# Consumption = nondurable goods + services (standard definition)
########################################################################

cat("================================================================\n")
cat("SECTION 1: TABLE 6.1 — BUSINESS CYCLE STATISTICS\n")
cat("================================================================\n\n")

# --- 1.1 HP Filter Implementation ---

hp_filter <- function(y, lambda = 1600) {
  # Hodrick-Prescott filter via penalised least squares
  n <- length(y)
  I_mat <- diag(n)

  # Second difference matrix
  D2 <- matrix(0, n - 2, n)
  for (i in 1:(n - 2)) {
    D2[i, i]     <-  1
    D2[i, i + 1] <- -2
    D2[i, i + 2] <-  1
  }

  A <- I_mat + lambda * t(D2) %*% D2
  trend <- as.numeric(solve(A, y))
  cycle <- y - trend

  list(trend = trend, cycle = cycle)
}


# --- 1.2 Load FRED-QD Data ---

load_fred_qd <- function(file_path) {
  library(readxl)

  raw <- read_excel(file_path, sheet = "in", col_names = FALSE)
  colnames_row <- as.character(raw[1, ])

  # Data begins at row 6 in the Excel sheet
  data_rows <- raw[6:nrow(raw), ]

  get_series <- function(code) {
    idx <- which(colnames_row == code)[1]
    as.numeric(data_rows[[idx]])
  }

  # Consumption = nondurables + services (excluding durables)
  cons_nds <- get_series("PCNDx") + get_series("PCESVx")

  n <- nrow(data_rows)
  dates <- seq(as.Date("1960-01-01"), by = "quarter", length.out = n)

  data.frame(
    date               = dates,
    GDP                = get_series("GDPC1"),
    Consumption        = cons_nds,
    Investment         = get_series("GPDIC1"),
    Hours_worked       = get_series("HOANBS"),
    Labor_productivity = get_series("OPHNFB"),
    Real_wages         = get_series("COMPRNFB")
  )
}


# --- 1.3 Compute Business Cycle Statistics ---

compute_bc_stats <- function(df, lambda = 1600) {
  vars <- c("GDP", "Consumption", "Investment",
            "Hours_worked", "Labor_productivity", "Real_wages")
  labels <- c("GDP", "Consumption", "Investment",
              "Hours worked", "Labor productivity", "Real wages")

  cycles <- data.frame(date = df$date)
  for (v in vars) {
    hp <- hp_filter(log(df[[v]]), lambda = lambda)
    cycles[[v]] <- hp$cycle * 100  # percent deviations
  }

  gdp_sd <- sd(cycles$GDP)

  stats <- data.frame(
    Variable             = labels,
    Std_Dev_pct          = sapply(vars, function(v) round(sd(cycles[[v]]), 2)),
    Relative_to_GDP      = sapply(vars, function(v) round(sd(cycles[[v]]) / gdp_sd, 2)),
    Correlation_with_GDP = sapply(vars, function(v) round(cor(cycles[[v]], cycles$GDP), 2)),
    row.names = NULL
  )

  list(stats = stats, cycles = cycles)
}


# --- 1.4 Visualisation Functions ---

plot_hp_decomposition <- function(df, variable = "GDP", lambda = 1600) {
  y_log <- log(df[[variable]])
  hp <- hp_filter(y_log, lambda = lambda)

  par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))

  plot(df$date, y_log, type = "l", col = "steelblue",
       xlab = "", ylab = "Log Level",
       main = paste0(variable, ": Original Series and HP Trend (lambda = ", lambda, ")"))
  lines(df$date, hp$trend, col = "red", lwd = 2)
  legend("topleft", c(paste("Log", variable), "HP Trend"),
         col = c("steelblue", "red"), lty = 1, lwd = c(1, 2), bty = "n")
  grid(col = "grey90")

  plot(df$date, hp$cycle * 100, type = "l", col = "blue",
       xlab = "Date", ylab = "% Deviation from Trend",
       main = paste0(variable, ": Cyclical Component"))
  abline(h = 0, lty = 2, lwd = 0.8)
  grid(col = "grey90")
}


plot_all_cycles <- function(cycles, dates) {
  vars <- setdiff(names(cycles), "date")
  cols <- c("blue", "red", "darkgreen", "orange", "purple", "brown")

  plot(dates, cycles[[vars[1]]], type = "l", col = cols[1],
       ylim = range(unlist(cycles[vars])),
       xlab = "Date", ylab = "% Deviation from Trend",
       main = "Business Cycle Components: All Variables")
  for (i in 2:length(vars)) {
    lines(dates, cycles[[vars[i]]], col = cols[i])
  }
  abline(h = 0, lty = 2, lwd = 0.8)
  legend("topleft", gsub("_", " ", vars), col = cols[1:length(vars)],
         lty = 1, bty = "n", cex = 0.8)
  grid(col = "grey90")
}


plot_correlations_with_gdp <- function(cycles) {
  vars <- setdiff(names(cycles), c("date", "GDP"))
  par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))

  for (v in vars) {
    corr_val <- cor(cycles$GDP, cycles[[v]])
    plot(cycles$GDP, cycles[[v]], pch = 16, cex = 0.5, col = "steelblue",
         xlab = "GDP Cycle (%)", ylab = paste(gsub("_", " ", v), "Cycle (%)"),
         main = paste0(gsub("_", " ", v), " vs GDP\nCorrelation = ",
                       round(corr_val, 2)))
    abline(lm(cycles[[v]] ~ cycles$GDP), col = "red", lwd = 2)
    grid(col = "grey90")
  }
}


# --- 1.5 Run Section 1 ---
# SET YOUR FILE PATH HERE AND UNCOMMENT:
#
# file_path <- "2026-01-QD.xlsx"
# df <- load_fred_qd(file_path)
# df <- df[df$date >= as.Date("1960-01-01") & df$date <= as.Date("2019-12-31"), ]
# df <- na.omit(df)
#
# result <- compute_bc_stats(df)
# cat("TABLE 6.1: Business Cycle Statistics for the United States\n")
# cat(strrep("=", 80), "\n")
# print(result$stats)
# cat(strrep("=", 80), "\n\n")
#
# gdp_cycle <- result$cycles$GDP
# n_g <- length(gdp_cycle)
# rho1 <- cor(gdp_cycle[2:n_g], gdp_cycle[1:(n_g-1)])
# rho4 <- cor(gdp_cycle[5:n_g], gdp_cycle[1:(n_g-4)])
# cat(sprintf("GDP autocorrelation: rho(1) = %.2f, rho(4) = %.2f\n\n", rho1, rho4))
#
# png("hp_decomposition_gdp.png", width = 1200, height = 800, res = 150)
# plot_hp_decomposition(df, "GDP")
# dev.off()
#
# png("all_cycles.png", width = 1400, height = 800, res = 150)
# par(mfrow = c(1, 1))
# plot_all_cycles(result$cycles, df$date)
# dev.off()
#
# png("correlations_with_gdp.png", width = 1500, height = 1000, res = 150)
# plot_correlations_with_gdp(result$cycles)
# dev.off()

cat("Table 6.1 code ready. Uncomment and set file_path to run.\n\n")


########################################################################
# SECTION 2: FIGURE 6.1 — RBC Impulse Responses
########################################################################

cat("================================================================\n")
cat("SECTION 2: FIGURE 6.1 — RBC IMPULSE RESPONSES\n")
cat("Calibration: beta=0.99, alpha=0.33, delta=0.025, rho_A=0.95\n")
cat("================================================================\n\n")

# --- 2.1 Steady State ---

rbc_steady_state <- function(beta = 0.99, alpha = 0.33, delta = 0.025,
                              sigma = 1, eta = 1) {
  r_ss    <- 1 / beta - 1 + delta
  KY      <- alpha / r_ss
  IY      <- delta * KY
  CY      <- 1 - IY

  K_per_N <- KY^(1 / (1 - alpha))
  YN      <- K_per_N^alpha
  w_ss    <- (1 - alpha) * YN
  CN      <- CY * YN

  N_ss <- (w_ss * CN^(-sigma))^(1 / (eta + sigma))
  K_ss <- K_per_N * N_ss
  Y_ss <- K_ss^alpha * N_ss^(1 - alpha)
  C_ss <- CY * Y_ss
  I_ss <- IY * Y_ss

  list(Y = Y_ss, C = C_ss, I = I_ss, K = K_ss, N = N_ss,
       w = w_ss, r = r_ss, CY = CY, IY = IY, KY = KY)
}

# --- 2.2 Blanchard-Kahn Solution ---

rbc_solve_bk <- function(ss, alpha = 0.33, delta = 0.025, beta = 0.99,
                          sigma = 1, eta = 1, rho_a = 0.95) {

  CY <- ss$CY;  IY <- ss$IY;  r_ss <- ss$r

  # Output elasticities
  phi_yk <- alpha + alpha * (1 - alpha) / (eta + alpha)
  phi_ya <- 1 + (1 - alpha) / (eta + alpha)
  phi_yc <- -(1 - alpha) * sigma / (eta + alpha)

  # Capital accumulation coefficients
  Akk <- (1 - delta) + delta * phi_yk / IY
  Aka <- delta * phi_ya / IY
  Akc <- delta * (phi_yc - CY) / IY

  rb  <- r_ss * beta
  pk1 <- phi_yk - 1

  A_lhs <- matrix(c(
    1,    0,    0,
    0,    1,    0,
    0,    0,    sigma - rb * phi_yc
  ), 3, 3, byrow = TRUE)

  A_rhs <- matrix(c(
    Akk,           Aka,                              Akc,
    0,             rho_a,                            0,
    rb*pk1*Akk,    rb*pk1*Aka + rb*phi_ya*rho_a,     sigma + rb*pk1*Akc
  ), 3, 3, byrow = TRUE)

  M <- solve(A_lhs) %*% A_rhs
  eig <- eigen(M)
  eig_abs <- Mod(eig$values)

  cat(sprintf("   BK eigenvalues: |%.4f|, |%.4f|, |%.4f|\n",
              eig_abs[1], eig_abs[2], eig_abs[3]))

  ord <- order(eig_abs)
  eig_sorted <- eig_abs[ord]
  V_sorted   <- eig$vectors[, ord]

  n_stable   <- sum(eig_sorted < 1)
  n_unstable <- sum(eig_sorted > 1)
  cat(sprintf("   Stable: %d, Unstable: %d  ", n_stable, n_unstable))
  if (n_unstable == 1) cat("OK\n") else cat("WARNING: BK fails!\n")

  Z11 <- V_sorted[1:2, 1:2]
  Z21 <- V_sorted[3, 1:2, drop = FALSE]

  F_mat <- Re(Z21 %*% solve(Z11))

  P_mat <- matrix(c(
    Akk + Akc * F_mat[1], Aka + Akc * F_mat[2],
    0,                     rho_a
  ), 2, 2, byrow = TRUE)

  list(P = P_mat, F = F_mat,
       phi_yk = phi_yk, phi_ya = phi_ya, phi_yc = phi_yc)
}

# --- 2.3 Compute IRFs ---

rbc_irfs <- function(sol, ss, alpha = 0.33, sigma = 1, eta = 1,
                     periods = 40, shock_size = 1.0) {

  P <- sol$P;  F <- sol$F
  CY <- ss$CY;  IY <- ss$IY

  s <- matrix(0, periods + 1, 2)
  s[1, 2] <- shock_size

  for (t in 2:(periods + 1)) {
    s[t, ] <- P %*% s[t - 1, ]
  }

  k_hat <- s[1:periods, 1]
  a_hat <- s[1:periods, 2]
  c_hat <- as.numeric(F %*% t(s[1:periods, ]))
  n_hat <- (a_hat + alpha * k_hat - sigma * c_hat) / (eta + alpha)
  y_hat <- sol$phi_yk * k_hat + sol$phi_ya * a_hat + sol$phi_yc * c_hat
  i_hat <- (y_hat - CY * c_hat) / IY
  cap_hat <- s[2:(periods + 1), 1]

  data.frame(quarter = 0:(periods - 1),
             Y = y_hat, C = c_hat, I = i_hat,
             N = n_hat, K = cap_hat, A = a_hat)
}

# --- 2.4 Plot Figure 6.1 ---

plot_rbc_irfs <- function(irf) {
  vars   <- c("Y", "C", "I", "N", "K", "A")
  titles <- c("Output (Y)", "Consumption (C)", "Investment (I)",
              "Hours (N)", "Capital (K)", "Technology (A)")
  labels <- paste0("(", 1:6, ")")

  par(mfrow = c(2, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

  for (i in seq_along(vars)) {
    plot(irf$quarter, irf[[vars[i]]], type = "l", lwd = 2.5,
         col = "#1f77b4",
         xlab = "Quarters", ylab = "% deviation",
         main = titles[i], font.main = 2)
    abline(h = 0, col = "black", lwd = 0.8)
    grid(col = "grey90")
    legend("topright", labels[i], text.col = "red", text.font = 2,
           bty = "n", cex = 1.3)
  }

  mtext("Impulse Responses to a One Percent Technology Shock",
        outer = TRUE, cex = 1.3, font = 2)
}

# --- 2.5 Run ---

cat("Computing steady state...\n")
ss <- rbc_steady_state()
cat(sprintf("   Y=%.4f, C=%.4f, I=%.4f, K=%.4f, N=%.4f\n",
            ss$Y, ss$C, ss$I, ss$K, ss$N))
cat(sprintf("   C/Y=%.3f, I/Y=%.3f, K/Y=%.2f\n\n", ss$CY, ss$IY, ss$KY))

cat("Solving model (Blanchard-Kahn)...\n")
sol <- rbc_solve_bk(ss)

cat("\nComputing IRFs (1%% technology shock, 40 quarters)...\n")
irf <- rbc_irfs(sol, ss)

cat("\nImpact effects (t=0):\n")
for (v in c("Y", "C", "I", "N", "K", "A")) {
  cat(sprintf("   %s: %+.4f%%\n", v, irf[[v]][1]))
}
cat("\nPeak effects:\n")
for (v in c("Y", "C", "I", "K")) {
  peak_val <- max(irf[[v]])
  peak_q   <- which.max(irf[[v]]) - 1
  cat(sprintf("   %s: %.4f%% at quarter %d\n", v, peak_val, peak_q))
}

png("figure_6_1_rbc_irf.png", width = 1500, height = 1000, res = 150)
plot_rbc_irfs(irf)
dev.off()
cat("\nFigure saved: figure_6_1_rbc_irf.png\n\n")


########################################################################
# SECTION 3: FIGURE 6.2 — Prior vs Posterior Distributions
########################################################################

cat("================================================================\n")
cat("SECTION 3: FIGURE 6.2 — PRIOR vs POSTERIOR DISTRIBUTIONS\n")
cat("Smets-Wouters (2007, AER)\n")
cat("================================================================\n\n")

# Beta distribution from mean/std
beta_ab <- function(mu, s) {
  v <- s^2
  k <- mu * (1 - mu) / v - 1
  list(a = mu * k, b = (1 - mu) * k)
}

# Parameter specs from Table 6.4
params <- list(
  xi_p = list(label = expression(xi[p]~"(Price stickiness)"),
              prior = "B", pm = 0.50, ps = 0.10,
              mode = 0.65, p5 = 0.56, p95 = 0.74),
  xi_w = list(label = expression(xi[w]~"(Wage stickiness)"),
              prior = "B", pm = 0.50, ps = 0.10,
              mode = 0.73, p5 = 0.60, p95 = 0.81),
  h    = list(label = expression(h~"(Habit formation)"),
              prior = "B", pm = 0.70, ps = 0.10,
              mode = 0.71, p5 = 0.64, p95 = 0.78),
  phi  = list(label = expression(varphi~"(Inv. adjustment)"),
              prior = "N", pm = 4.00, ps = 1.50,
              mode = 5.48, p5 = 3.97, p95 = 7.42),
  r_pi = list(label = expression(r[pi]~"(Taylor: inflation)"),
              prior = "N", pm = 1.50, ps = 0.25,
              mode = 2.03, p5 = 1.74, p95 = 2.33),
  rho  = list(label = expression(rho~"(Interest smoothing)"),
              prior = "B", pm = 0.75, ps = 0.10,
              mode = 0.81, p5 = 0.77, p95 = 0.85)
)

plot_priors_posteriors <- function(params) {
  par(mfrow = c(2, 3), mar = c(4, 2, 3, 1), oma = c(3, 0, 2, 0))

  first_panel <- TRUE
  for (key in names(params)) {
    p <- params[[key]]
    post_sd <- (p$p95 - p$p5) / (2 * 1.645)

    if (p$prior == "B") {
      x <- seq(0.001, 0.999, length.out = 500)
      bp <- beta_ab(p$pm, p$ps)
      prior_pdf <- dbeta(x, bp$a, bp$b)
    } else {
      lo <- min(p$pm - 4 * p$ps, p$mode - 4 * post_sd)
      hi <- max(p$pm + 4 * p$ps, p$mode + 4 * post_sd)
      x <- seq(lo, hi, length.out = 500)
      prior_pdf <- dnorm(x, p$pm, p$ps)
    }
    post_pdf <- dnorm(x, p$mode, post_sd)

    ylim <- c(0, max(c(prior_pdf, post_pdf)) * 1.1)
    plot(x, prior_pdf, type = "n", ylim = ylim,
         xlab = "", ylab = "", main = p$label, yaxt = "n",
         cex.main = 1.1, font.main = 2)

    polygon(c(x, rev(x)), c(prior_pdf, rep(0, length(x))),
            col = adjustcolor("grey60", alpha = 0.35), border = NA)
    lines(x, prior_pdf, col = "grey50", lwd = 1.2)

    polygon(c(x, rev(x)), c(post_pdf, rep(0, length(x))),
            col = adjustcolor("#2171b5", alpha = 0.45), border = NA)
    lines(x, post_pdf, col = "#2171b5", lwd = 1.8)

    abline(v = p$mode, col = "#08519c", lty = 2, lwd = 1.5)
    grid(col = "grey90")

    if (first_panel) {
      legend("topleft", c("Prior", "Posterior"),
             fill = c(adjustcolor("grey60", 0.35), adjustcolor("#2171b5", 0.45)),
             border = c("grey50", "#2171b5"), bty = "n", cex = 0.9)
      first_panel <- FALSE
    }
  }

  mtext("Bayesian Estimation: Prior vs Posterior Distributions",
        outer = TRUE, cex = 1.3, font = 2)
  mtext("Source: Smets and Wouters (2007, AER), Table 1A.",
        side = 1, outer = TRUE, cex = 0.8, font = 3, line = 1.5)
}

# Print summary
pnames <- c("xi_p", "xi_w", "h", "phi", "r_pi", "rho")
plabels <- c("xi_p (Price stick.)", "xi_w (Wage stick.)",
             "h (Habit)", "phi (Inv. adj.)",
             "r_pi (Taylor: infl.)", "rho (Int. smooth.)")

cat("TABLE 6.4 (selected): Prior and Posterior\n")
cat(strrep("=", 80), "\n")
cat(sprintf("%-28s %5s %8s %8s %10s %6s %6s\n",
            "Parameter", "Dist", "Prior m", "Prior s", "Post mode", "[5%", "95%]"))
cat(strrep("-", 80), "\n")
for (i in seq_along(pnames)) {
  p <- params[[pnames[i]]]
  cat(sprintf("%-28s %5s %8.2f %8.2f %10.2f %6.2f %6.2f\n",
              plabels[i], p$prior, p$pm, p$ps, p$mode, p$p5, p$p95))
}
cat(strrep("-", 80), "\n\n")

cat("Prior-to-Posterior Learning:\n")
cat(strrep("-", 70), "\n")
for (i in seq_along(pnames)) {
  p <- params[[pnames[i]]]
  post_sd <- (p$p95 - p$p5) / 3.29
  shrinkage <- 1 - post_sd / p$ps
  shift <- p$mode - p$pm
  cat(sprintf("  %-28s shift = %+.2f, variance reduction = %.0f%%\n",
              plabels[i], shift, shrinkage * 100))
}
cat("\n")

cat("Generating Figure 6.2...\n")
png("figure_6_2_prior_posterior.png", width = 1500, height = 800, res = 150)
plot_priors_posteriors(params)
dev.off()
cat("Figure saved: figure_6_2_prior_posterior.png\n\n")

cat("================================================================\n")
cat("ALL SECTIONS COMPLETE\n")
cat("================================================================\n")
