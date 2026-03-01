# =========================================================
# ARMA Process Simulation and Visualization
# =========================================================
# This script generates artificial time series for:
#   - White Noise
#   - AR(1): y_t = phi * y_{t-1} + epsilon_t
#   - MA(1): y_t = epsilon_t + theta * epsilon_{t-1}
#   - ARMA(1,1): y_t = phi * y_{t-1} + epsilon_t + theta * epsilon_{t-1}
#
# Author:   Alessia Paccagnini
# Textbook: Macroeconometrics
# =========================================================

library(ggplot2)
library(patchwork)
library(stats)   # acf, pacf built-in

set.seed(42)

# Output directory: same folder as this script (portable)
out_dir <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)),
                    error = function(e) getwd())

# =========================================================
# Parameters
# =========================================================
T     <- 500
sigma <- 1.0
phi   <- 0.7
theta <- 0.5

# =========================================================
# Generation functions
# =========================================================

generate_white_noise <- function(T, sigma = 1.0) {
  rnorm(T, 0, sigma)
}

generate_ar1 <- function(T, phi, sigma = 1.0) {
  if (abs(phi) >= 1)
    stop(sprintf("AR(1) requires |phi| < 1 for stationarity. Got phi = %g", phi))
  epsilon <- rnorm(T, 0, sigma)
  y       <- numeric(T)
  y[1]    <- rnorm(1, 0, sigma / sqrt(1 - phi^2))
  for (t in 2:T)
    y[t] <- phi * y[t-1] + epsilon[t]
  list(y = y, epsilon = epsilon)
}

generate_ma1 <- function(T, theta, sigma = 1.0) {
  epsilon <- rnorm(T + 1, 0, sigma)
  y       <- numeric(T)
  for (t in 1:T)
    y[t] <- epsilon[t + 1] + theta * epsilon[t]
  list(y = y, epsilon = epsilon[2:(T+1)])
}

generate_arma11 <- function(T, phi, theta, sigma = 1.0) {
  if (abs(phi) >= 1)
    stop(sprintf("ARMA(1,1) requires |phi| < 1 for stationarity. Got phi = %g", phi))
  epsilon <- rnorm(T + 1, 0, sigma)
  y       <- numeric(T)
  y[1]    <- epsilon[2] + theta * epsilon[1]
  for (t in 2:T)
    y[t] <- phi * y[t-1] + epsilon[t + 1] + theta * epsilon[t]
  list(y = y, epsilon = epsilon[2:(T+1)])
}

# =========================================================
# Generate all processes
# =========================================================
wn    <- generate_white_noise(T, sigma)
ar1   <- generate_ar1(T, phi, sigma)$y
ma1   <- generate_ma1(T, theta, sigma)$y
arma11 <- generate_arma11(T, phi, theta, sigma)$y

# =========================================================
# Helper: ACF/PACF bar plot via ggplot2
# =========================================================
acf_ggplot <- function(x, lags = 20, title = "ACF", type = "acf", ci = 1.96) {
  n    <- length(x)
  vals <- if (type == "acf") acf(x, lag.max = lags, plot = FALSE)$acf[-1]
          else                pacf(x, lag.max = lags, plot = FALSE)$acf
  df   <- data.frame(lag = 1:lags, value = vals)
  ci_band <- ci / sqrt(n)

  ggplot(df, aes(x = lag, y = value)) +
    geom_hline(yintercept = 0, colour = "black", linewidth = 0.4) +
    geom_segment(aes(xend = lag, yend = 0), colour = "steelblue", linewidth = 0.8) +
    geom_point(colour = "steelblue", size = 1.5) +
    geom_hline(yintercept =  ci_band, linetype = "dashed", colour = "blue", linewidth = 0.5) +
    geom_hline(yintercept = -ci_band, linetype = "dashed", colour = "blue", linewidth = 0.5) +
    scale_x_continuous(breaks = seq(0, lags, by = 5)) +
    ylim(-1, 1) +
    labs(title = title, x = "Lag", y = if (type == "acf") "ACF" else "PACF") +
    theme_bw(base_size = 9) +
    theme(plot.title = element_text(size = 9, hjust = 0.5))
}

# =========================================================
# Figure 1: Time Series Plots
# =========================================================
t_seq <- 1:T
df_ts <- data.frame(
  t      = rep(t_seq, 4),
  y      = c(wn, ar1, ma1, arma11),
  series = factor(rep(c("White Noise", "AR(1)", "MA(1)", "ARMA(1,1)"), each = T),
                  levels = c("White Noise", "AR(1)", "MA(1)", "ARMA(1,1)"))
)

colours <- c("White Noise" = "steelblue", "AR(1)" = "darkgreen",
             "MA(1)"       = "darkorange","ARMA(1,1)" = "purple")

titles_ts <- c(
  "White Noise" = "White Noise: e_t ~ N(0,1)",
  "AR(1)"       = "AR(1): y_t = 0.7*y_{t-1} + e_t",
  "MA(1)"       = "MA(1): y_t = e_t + 0.5*e_{t-1}",
  "ARMA(1,1)"   = "ARMA(1,1): y_t = 0.7*y_{t-1} + e_t + 0.5*e_{t-1}"
)

make_ts_panel <- function(ser) {
  sub_df <- df_ts[df_ts$series == ser, ]
  ggplot(sub_df, aes(x = t, y = y)) +
    geom_line(colour = colours[ser], linewidth = 0.5) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "red",
               linewidth = 0.5, alpha = 0.7) +
    labs(title = titles_ts[ser], x = NULL, y = "y_t") +
    theme_bw(base_size = 10) +
    theme(plot.title = element_text(size = 10))
}

fig1 <- (make_ts_panel("White Noise") /
          make_ts_panel("AR(1)") /
          make_ts_panel("MA(1)") /
          make_ts_panel("ARMA(1,1)")) +
  plot_annotation(title = "ARMA Processes: Time Series") &
  theme(plot.title = element_text(hjust = 0.5))

ggsave(file.path(out_dir, "arma_time_series.png"),
       fig1, width = 12, height = 10, dpi = 150)

# =========================================================
# Figure 2: ACF Comparison
# =========================================================
fig2 <- (acf_ggplot(wn,     20, "ACF: White Noise")    +
          acf_ggplot(ar1,   20, "ACF: AR(1) phi=0.7")) /
        (acf_ggplot(ma1,    20, "ACF: MA(1) theta=0.5") +
          acf_ggplot(arma11,20, "ACF: ARMA(1,1) phi=0.7, theta=0.5"))

ggsave(file.path(out_dir, "arma_acf_comparison.png"),
       fig2, width = 12, height = 8, dpi = 150)

# =========================================================
# Figure 3: PACF Comparison
# =========================================================
fig3 <- (acf_ggplot(wn,     20, "PACF: White Noise",              type = "pacf") +
          acf_ggplot(ar1,   20, "PACF: AR(1) phi=0.7",            type = "pacf")) /
        (acf_ggplot(ma1,    20, "PACF: MA(1) theta=0.5",          type = "pacf") +
          acf_ggplot(arma11,20, "PACF: ARMA(1,1) phi=0.7, theta=0.5", type = "pacf"))

ggsave(file.path(out_dir, "arma_pacf_comparison.png"),
       fig3, width = 12, height = 8, dpi = 150)

# =========================================================
# Print Theoretical vs Sample Statistics
# =========================================================
cat(strrep("=", 70), "\n")
cat("THEORETICAL vs SAMPLE STATISTICS\n")
cat(strrep("=", 70), "\n")

cat("\n--- White Noise ---\n")
cat(sprintf("Theoretical Mean: 0.000 | Sample Mean: %.3f\n", mean(wn)))
cat(sprintf("Theoretical Var:  %.3f | Sample Var:  %.3f\n", sigma^2, var(wn)))

ar1_theo_var  <- sigma^2 / (1 - phi^2)
cat(sprintf("\n--- AR(1): phi = %.1f ---\n", phi))
cat(sprintf("Theoretical Mean: 0.000 | Sample Mean: %.3f\n", mean(ar1)))
cat(sprintf("Theoretical Var:  %.3f | Sample Var:  %.3f\n", ar1_theo_var, var(ar1)))
cat(sprintf("Theoretical ACF(1): %.3f | Sample ACF(1): %.3f\n",
            phi, cor(ar1[-T], ar1[-1])))

ma1_theo_var  <- sigma^2 * (1 + theta^2)
ma1_theo_acf1 <- theta / (1 + theta^2)
cat(sprintf("\n--- MA(1): theta = %.1f ---\n", theta))
cat(sprintf("Theoretical Mean: 0.000 | Sample Mean: %.3f\n", mean(ma1)))
cat(sprintf("Theoretical Var:  %.3f | Sample Var:  %.3f\n", ma1_theo_var, var(ma1)))
cat(sprintf("Theoretical ACF(1): %.3f | Sample ACF(1): %.3f\n",
            ma1_theo_acf1, cor(ma1[-T], ma1[-1])))

arma_theo_var  <- sigma^2 * (1 + theta^2 + 2*phi*theta) / (1 - phi^2)
arma_theo_acf1 <- (phi + theta) * (1 + phi*theta) / (1 + theta^2 + 2*phi*theta)
cat(sprintf("\n--- ARMA(1,1): phi = %.1f, theta = %.1f ---\n", phi, theta))
cat(sprintf("Theoretical Mean: 0.000 | Sample Mean: %.3f\n", mean(arma11)))
cat(sprintf("Theoretical Var:  %.3f | Sample Var:  %.3f\n", arma_theo_var, var(arma11)))
cat(sprintf("Theoretical ACF(1): %.3f | Sample ACF(1): %.3f\n",
            arma_theo_acf1, cor(arma11[-T], arma11[-1])))

cat("\n", strrep("=", 70), "\n")
cat("ACF/PACF IDENTIFICATION PATTERNS\n")
cat(strrep("=", 70), "\n")
cat("
Process     | ACF Pattern                    | PACF Pattern
------------|--------------------------------|----------------------------------
White Noise | No significant autocorrelations| No significant partial autocorr.
AR(p)       | Geometric decay (or damped     | Cuts off after lag p
            | oscillations if complex roots) |
MA(q)       | Cuts off after lag q           | Geometric decay (or damped
            |                                | oscillations)
ARMA(p,q)   | Decays after lag q             | Decays after lag p
\n")

cat("\nFigures saved:\n")
cat("  - arma_time_series.png\n")
cat("  - arma_acf_comparison.png\n")
cat("  - arma_pacf_comparison.png\n")
