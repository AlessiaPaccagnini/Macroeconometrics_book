# =========================================================
# Nonstationary Processes: Simulation and Visualization
# =========================================================
# This script generates artificial time series for:
#   - Deterministic Trend: y_t = alpha + beta*t + u_t (u_t is AR(1))
#   - Random Walk: y_t = y_{t-1} + epsilon_t
#   - Random Walk with Drift: y_t = delta + y_{t-1} + epsilon_t
#
# Author:   Alessia Paccagnini
# Textbook: Macroeconometrics
# =========================================================

library(ggplot2)
library(patchwork)

set.seed(42)

# Output directory: same folder as this script (portable)
out_dir <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)),
                    error = function(e) getwd())

# =========================================================
# Parameters
# =========================================================
T      <- 250
sigma  <- 1.0
alpha  <- 2.0
beta   <- 0.1
phi_u  <- 0.7
delta  <- 0.15
nlags  <- 30

# =========================================================
# Generation functions
# =========================================================

generate_deterministic_trend <- function(T, alpha, beta, phi, sigma = 1.0) {
  epsilon <- rnorm(T, 0, sigma)
  u       <- numeric(T)
  u[1]    <- rnorm(1, 0, sigma / sqrt(1 - phi^2))
  for (t in 2:T)
    u[t] <- phi * u[t-1] + epsilon[t]
  t_index <- 0:(T-1)
  y       <- alpha + beta * t_index + u
  list(y = y, u = u, t_index = t_index)
}

generate_random_walk <- function(T, sigma = 1.0, y0 = 0) {
  epsilon <- rnorm(T, 0, sigma)
  y       <- numeric(T)
  y[1]    <- y0 + epsilon[1]
  for (t in 2:T)
    y[t] <- y[t-1] + epsilon[t]
  list(y = y, epsilon = epsilon)
}

generate_random_walk_drift <- function(T, delta, sigma = 1.0, y0 = 0) {
  epsilon <- rnorm(T, 0, sigma)
  y       <- numeric(T)
  y[1]    <- y0 + delta + epsilon[1]
  for (t in 2:T)
    y[t] <- delta + y[t-1] + epsilon[t]
  list(y = y, epsilon = epsilon)
}

# =========================================================
# Generate all processes
# =========================================================
det_res <- generate_deterministic_trend(T, alpha, beta, phi_u, sigma)
y_det   <- det_res$y
t_index <- det_res$t_index

rw_res  <- generate_random_walk(T, sigma)
y_rw    <- rw_res$y

rwd_res <- generate_random_walk_drift(T, delta, sigma)
y_rwd   <- rwd_res$y

wn      <- rnorm(T, 0, sigma)

# =========================================================
# Helper: ACF bar plot
# =========================================================
acf_ggplot <- function(x, lags = 30, title = "ACF", ylim_range = c(-0.3, 1.1)) {
  n    <- length(x)
  vals <- acf(x, lag.max = lags, plot = FALSE)$acf[-1]
  df   <- data.frame(lag = 1:lags, value = vals)
  ci   <- 1.96 / sqrt(n)

  ggplot(df, aes(x = lag, y = value)) +
    geom_hline(yintercept = 0, colour = "black", linewidth = 0.4) +
    geom_segment(aes(xend = lag, yend = 0), colour = "steelblue", linewidth = 0.7) +
    geom_point(colour = "steelblue", size = 1.2) +
    geom_hline(yintercept =  ci, linetype = "dashed", colour = "blue", linewidth = 0.5) +
    geom_hline(yintercept = -ci, linetype = "dashed", colour = "blue", linewidth = 0.5) +
    scale_y_continuous(limits = ylim_range) +
    labs(title = title, x = "Lag", y = "ACF") +
    theme_bw(base_size = 9) +
    theme(plot.title = element_text(size = 9, hjust = 0.5))
}

pacf_ggplot <- function(x, lags = 30, title = "PACF") {
  n    <- length(x)
  vals <- pacf(x, lag.max = lags, plot = FALSE)$acf
  df   <- data.frame(lag = 1:lags, value = vals)
  ci   <- 1.96 / sqrt(n)

  ggplot(df, aes(x = lag, y = value)) +
    geom_hline(yintercept = 0, colour = "black", linewidth = 0.4) +
    geom_segment(aes(xend = lag, yend = 0), colour = "darkorange", linewidth = 0.7) +
    geom_point(colour = "darkorange", size = 1.2) +
    geom_hline(yintercept =  ci, linetype = "dashed", colour = "blue", linewidth = 0.5) +
    geom_hline(yintercept = -ci, linetype = "dashed", colour = "blue", linewidth = 0.5) +
    labs(title = title, x = "Lag", y = "PACF") +
    theme_bw(base_size = 9) +
    theme(plot.title = element_text(size = 9, hjust = 0.5))
}

# =========================================================
# Figure 1a: Deterministic Trend (individual)
# =========================================================
df_det <- data.frame(t = t_index, y = y_det, trend = alpha + beta * t_index)

fig_det <- ggplot(df_det, aes(x = t)) +
  geom_line(aes(y = y, colour = "Series"), linewidth = 0.6) +
  geom_line(aes(y = trend, colour = "Trend"), linetype = "dashed", linewidth = 1.2) +
  geom_hline(yintercept = 0, colour = "gray", linetype = "dotted", linewidth = 0.4) +
  scale_colour_manual(values = c("Series" = "steelblue", "Trend" = "red")) +
  labs(title = "Deterministic Trend: y[t] = alpha + beta*t + u[t], u[t] ~ AR(1)",
       x = "Time", y = "y[t]", colour = NULL) +
  theme_bw(base_size = 11) +
  theme(legend.position = "top")

ggsave(file.path(out_dir, "deterministic_trend.png"),
       fig_det, width = 10, height = 5, dpi = 150)

# =========================================================
# Figure 1b: Random Walk (individual)
# =========================================================
df_rw <- data.frame(t = 1:T, y = y_rw)

fig_rw <- ggplot(df_rw, aes(x = t, y = y)) +
  geom_ribbon(aes(ymin = pmin(y, 0), ymax = pmax(y, 0)), fill = "darkgreen", alpha = 0.25) +
  geom_line(colour = "darkgreen", linewidth = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red",
             linewidth = 1.0, alpha = 0.8) +
  labs(title = "Random Walk: y[t] = y[t-1] + epsilon[t]",
       x = "Time", y = "y[t]") +
  theme_bw(base_size = 11)

ggsave(file.path(out_dir, "random_walk.png"),
       fig_rw, width = 10, height = 5, dpi = 150)

# =========================================================
# Figure 1c: Random Walk with Drift (individual)
# =========================================================
df_rwd <- data.frame(t = t_index, y = y_rwd, drift = delta * t_index)

fig_rwd <- ggplot(df_rwd, aes(x = t)) +
  geom_line(aes(y = y,     colour = "Series"), linewidth = 0.6) +
  geom_line(aes(y = drift, colour = "Drift"),  linetype = "dashed", linewidth = 1.2) +
  scale_colour_manual(values = c("Series" = "purple", "Drift" = "red")) +
  labs(title = "Random Walk with Drift: y[t] = 0.15 + y[t-1] + epsilon[t]",
       x = "Time", y = "y[t]", colour = NULL) +
  theme_bw(base_size = 11) +
  theme(legend.position = "top")

ggsave(file.path(out_dir, "random_walk_drift.png"),
       fig_rwd, width = 10, height = 5, dpi = 150)

# =========================================================
# Figure 2: Combined Time Series
# =========================================================
p_det <- ggplot(df_det, aes(x = t)) +
  geom_line(aes(y = y),     colour = "steelblue", linewidth = 0.6) +
  geom_line(aes(y = trend), colour = "red", linetype = "dashed", linewidth = 1.0) +
  labs(title = "Deterministic Trend: y[t] = 2 + 0.1t + u[t], u[t] = 0.7*u[t-1] + eps[t]",
       x = NULL, y = "y[t]") +
  theme_bw(base_size = 10)

p_rw <- ggplot(df_rw, aes(x = t, y = y)) +
  geom_line(colour = "darkgreen", linewidth = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red", linewidth = 0.8) +
  labs(title = "Random Walk: y[t] = y[t-1] + epsilon[t] (variance grows with t)",
       x = NULL, y = "y[t]") +
  theme_bw(base_size = 10)

p_rwd <- ggplot(df_rwd, aes(x = t)) +
  geom_line(aes(y = y),     colour = "purple", linewidth = 0.6) +
  geom_line(aes(y = drift), colour = "red", linetype = "dashed", linewidth = 1.0) +
  labs(title = "Random Walk with Drift: y[t] = 0.15 + y[t-1] + epsilon[t]",
       x = "Time", y = "y[t]") +
  theme_bw(base_size = 10)

fig2 <- p_det / p_rw / p_rwd
ggsave(file.path(out_dir, "nonstationary_time_series.png"),
       fig2, width = 12, height = 10, dpi = 150)

# =========================================================
# Figure 3: ACF Comparison
# =========================================================
fig3 <- (acf_ggplot(wn,    nlags, "ACF: White Noise (Stationary)") +
          acf_ggplot(y_det, nlags, "ACF: Deterministic Trend (before detrending)")) /
        (acf_ggplot(y_rw,  nlags, "ACF: Random Walk") +
          acf_ggplot(y_rwd, nlags, "ACF: Random Walk with Drift"))

ggsave(file.path(out_dir, "nonstationary_acf_comparison.png"),
       fig3, width = 12, height = 8, dpi = 150)

# =========================================================
# Figure 4: PACF Comparison
# =========================================================
fig4 <- (pacf_ggplot(wn,    nlags, "PACF: White Noise (Stationary)") +
          pacf_ggplot(y_det, nlags, "PACF: Deterministic Trend")) /
        (pacf_ggplot(y_rw,  nlags, "PACF: Random Walk") +
          pacf_ggplot(y_rwd, nlags, "PACF: Random Walk with Drift"))

ggsave(file.path(out_dir, "nonstationary_pacf_comparison.png"),
       fig4, width = 12, height = 8, dpi = 150)

# =========================================================
# Figure 5: ACF Levels vs First Differences
# =========================================================
dy_det <- diff(y_det)
dy_rw  <- diff(y_rw)
dy_rwd <- diff(y_rwd)

fig5 <- (acf_ggplot(y_det,  nlags, "Deterministic Trend (levels)",             c(-0.3, 1.1)) +
          acf_ggplot(y_rw,  nlags, "Random Walk (levels)",                      c(-0.3, 1.1)) +
          acf_ggplot(y_rwd, nlags, "Random Walk with Drift (levels)",           c(-0.3, 1.1))) /
        (acf_ggplot(dy_det, nlags, "Deterministic Trend (first diff.)",         c(-0.5, 1.1)) +
          acf_ggplot(dy_rw,  nlags, "Random Walk (first diff.) = White Noise",  c(-0.5, 1.1)) +
          acf_ggplot(dy_rwd, nlags, "RW with Drift (first diff.) = White Noise",c(-0.5, 1.1))) +
  plot_annotation(
    title = "ACF in Levels vs. First Differences: Diagnosing Nonstationarity",
    theme = theme(plot.title = element_text(size = 13, hjust = 0.5))
  )

ggsave(file.path(out_dir, "acf_levels_vs_differences.png"),
       fig5, width = 14, height = 8, dpi = 150)

# =========================================================
# Figure 6: Variance Growth Comparison
# =========================================================
n_sim <- 100
T_var <- 200

rw_sims  <- matrix(0, n_sim, T_var)
det_sims <- matrix(0, n_sim, T_var)

for (i in 1:n_sim) {
  rw_sims[i, ]  <- generate_random_walk(T_var, sigma)$y
  det_sims[i, ] <- generate_deterministic_trend(T_var, 0, 0, phi_u, sigma)$y
}

rw_var  <- apply(rw_sims,  2, var)
det_var <- apply(det_sims, 2, var)
theo_rw_var  <- sigma^2 * 1:T_var
theo_det_var <- sigma^2 / (1 - phi_u^2)

df_var_rw  <- data.frame(t = 1:T_var, sample = rw_var,  theoretical = theo_rw_var)
df_var_det <- data.frame(t = 1:T_var, sample = det_var)

p_var_rw <- ggplot(df_var_rw, aes(x = t)) +
  geom_line(aes(y = sample,      colour = "Sample variance"), linewidth = 0.9) +
  geom_line(aes(y = theoretical, colour = "Theoretical: t*sigma^2"),
            linetype = "dashed", linewidth = 1.0) +
  scale_colour_manual(values = c("Sample variance" = "darkgreen",
                                  "Theoretical: t*sigma^2" = "red")) +
  labs(title = "Random Walk: Variance Grows Linearly with Time",
       x = "Time", y = "Variance", colour = NULL) +
  theme_bw(base_size = 10) + theme(legend.position = "top")

p_var_det <- ggplot(df_var_det, aes(x = t)) +
  geom_line(aes(y = sample, colour = "Sample variance"), linewidth = 0.9) +
  geom_hline(aes(yintercept = theo_det_var, colour = "Theoretical: sigma^2/(1-phi^2)"),
             linetype = "dashed", linewidth = 1.0) +
  scale_colour_manual(values = c("Sample variance" = "steelblue",
                                  "Theoretical: sigma^2/(1-phi^2)" = "red")) +
  coord_cartesian(ylim = c(0, max(det_var) * 1.5)) +
  labs(title = "Stationary AR(1): Variance Remains Constant",
       x = "Time", y = "Variance", colour = NULL) +
  theme_bw(base_size = 10) + theme(legend.position = "top")

fig6 <- p_var_rw + p_var_det
ggsave(file.path(out_dir, "variance_growth_comparison.png"),
       fig6, width = 12, height = 5, dpi = 150)

# =========================================================
# Figure 7: Multiple Realizations (Fan Chart)
# =========================================================
# Random Walk paths
rw_paths  <- data.frame(t = integer(), y = numeric(), sim = integer())
det_paths <- data.frame(t = integer(), y = numeric(), sim = integer())

for (i in 1:30) {
  y_tmp <- generate_random_walk(T, sigma)$y
  rw_paths <- rbind(rw_paths,
                    data.frame(t = 1:T, y = y_tmp, sim = i))
  y_tmp <- generate_deterministic_trend(T, alpha, beta, phi_u, sigma)$y
  det_paths <- rbind(det_paths,
                     data.frame(t = 1:T, y = y_tmp, sim = i))
}

p_fan_rw <- ggplot(rw_paths, aes(x = t, y = y, group = sim)) +
  geom_line(alpha = 0.35, linewidth = 0.4) +
  geom_hline(yintercept = 0, colour = "red", linetype = "dashed", linewidth = 1.2) +
  labs(title = "Random Walk: Multiple Realizations (Variance Expansion)",
       x = "Time", y = "y[t]") +
  theme_bw(base_size = 10)

trend_line <- data.frame(t = 0:(T-1), trend = alpha + beta * 0:(T-1))
p_fan_det <- ggplot(det_paths, aes(x = t, y = y, group = sim)) +
  geom_line(alpha = 0.35, linewidth = 0.4) +
  geom_line(data = trend_line, aes(x = t, y = trend, group = NULL),
            colour = "red", linetype = "dashed", linewidth = 1.2) +
  labs(title = "Deterministic Trend: Multiple Realizations (Constant Variance)",
       x = "Time", y = "y[t]") +
  theme_bw(base_size = 10)

fig7 <- p_fan_rw + p_fan_det
ggsave(file.path(out_dir, "multiple_realizations.png"),
       fig7, width = 12, height = 5, dpi = 150)

# =========================================================
# Print Summary Statistics
# =========================================================
cat(strrep("=", 70), "\n")
cat("NONSTATIONARY PROCESSES: SUMMARY STATISTICS\n")
cat(strrep("=", 70), "\n")

acf_val <- function(x, k) acf(x, lag.max = k, plot = FALSE)$acf[k + 1]

cat("\n--- Deterministic Trend: y_t = 2 + 0.1t + u_t ---\n")
cat(sprintf("Sample Mean: %.3f\n",     mean(y_det)))
cat(sprintf("Sample Variance: %.3f\n", var(y_det)))
cat(sprintf("ACF(1): %.3f\n",          acf_val(y_det, 1)))
cat(sprintf("ACF(10): %.3f\n",         acf_val(y_det, 10)))

cat("\n--- Random Walk: y_t = y_{t-1} + epsilon_t ---\n")
cat(sprintf("Sample Mean: %.3f\n",     mean(y_rw)))
cat(sprintf("Sample Variance: %.3f\n", var(y_rw)))
cat(sprintf("Theoretical Variance at T=%d: %.3f\n", T, T * sigma^2))
cat(sprintf("ACF(1): %.3f\n",          acf_val(y_rw, 1)))
cat(sprintf("ACF(10): %.3f\n",         acf_val(y_rw, 10)))

cat("\n--- Random Walk with Drift: y_t = 0.15 + y_{t-1} + epsilon_t ---\n")
cat(sprintf("Sample Mean: %.3f\n",     mean(y_rwd)))
cat(sprintf("Sample Variance: %.3f\n", var(y_rwd)))
cat(sprintf("ACF(1): %.3f\n",          acf_val(y_rwd, 1)))
cat(sprintf("ACF(10): %.3f\n",         acf_val(y_rwd, 10)))

cat("\n--- First Differences ---\n")
cat(sprintf("Delta-y (Random Walk)    - Mean: %.3f, Var: %.3f\n", mean(dy_rw), var(dy_rw)))
cat(sprintf("Delta-y (RW with Drift)  - Mean: %.3f, Var: %.3f\n", mean(dy_rwd), var(dy_rwd)))

cat("\n", strrep("=", 70), "\n")
cat("KEY INSIGHT: WHY CORRELOGRAM FAILS FOR UNIT ROOTS\n")
cat(strrep("=", 70), "\n")
cat("
For a random walk, the sample ACF at lag k is approximately:

    rho_k ~ (T - k) / T -> 1  as T -> Inf

This means:
1. ALL autocorrelations appear close to 1, regardless of lag
2. The slow linear decay is an artifact, not a true decay pattern
3. Standard confidence bands (based on 1/sqrt(T)) are INVALID
4. The ACF cannot distinguish between:
   - A near-unit-root AR(1) with phi = 0.99
   - A true unit root with phi = 1.00

This is why we need FORMAL UNIT ROOT TESTS (ADF, PP, KPSS) rather
than visual inspection of the correlogram!
\n")

cat("\nFigures saved:\n")
for (f in c("deterministic_trend.png", "random_walk.png", "random_walk_drift.png",
            "nonstationary_time_series.png", "nonstationary_acf_comparison.png",
            "nonstationary_pacf_comparison.png", "acf_levels_vs_differences.png",
            "variance_growth_comparison.png", "multiple_realizations.png"))
  cat(sprintf("  - %s\n", f))
