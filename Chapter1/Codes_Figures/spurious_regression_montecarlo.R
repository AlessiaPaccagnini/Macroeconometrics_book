# =========================================================
# Spurious Regression: Monte Carlo Simulations
# =========================================================
# Author:   Alessia Paccagnini
# Textbook: Macroeconometrics
#
# Replicates three figures:
#   1. Monte Carlo evidence (rejection rates, mean R2, mean DW)
#   2. Granger-Newbold rule of thumb (R2 vs DW scatter)
#   3. t-statistic distributions (spurious vs valid)
#
# References:
#   Granger & Newbold (1974). Journal of Econometrics, 2(2), 111-120.
#   Phillips (1986). Journal of Econometrics, 33(3), 311-340.
# =========================================================

library(ggplot2)
library(gridExtra)
library(patchwork)   # for combining ggplots side by side

set.seed(42)

# Output directory: same folder as this script (portable)
out_dir <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)),
                    error = function(e) getwd())

# --------------------------------------------------------
# Helper functions
# --------------------------------------------------------
dw_stat <- function(resid) sum(diff(resid)^2) / sum(resid^2)

run_regression <- function(T, spurious = TRUE) {
  if (spurious) {
    x <- cumsum(rnorm(T))
    y <- cumsum(rnorm(T))
  } else {
    x <- rnorm(T)
    y <- rnorm(T)
  }
  fit    <- lm(y ~ x)
  s      <- summary(fit)
  list(
    r2       = s$r.squared,
    t_stat   = coef(s)["x", "t value"],
    dw       = dw_stat(residuals(fit)),
    beta_hat = coef(fit)["x"]
  )
}

# =========================================================
# Figure 1: Monte Carlo Evidence
# =========================================================
cat(strrep("=", 60), "\n")
cat("Figure 1: Monte Carlo Evidence on Spurious Regression\n")
cat(strrep("=", 60), "\n")

sample_sizes  <- c(50, 100, 200, 500)
n_simulations <- 1000
critical_value <- 1.96

fig1_stats <- data.frame(
  T               = sample_sizes,
  rejection_rate  = NA_real_,
  mean_r2         = NA_real_,
  mean_dw         = NA_real_
)

for (i in seq_along(sample_sizes)) {
  T <- sample_sizes[i]
  cat(sprintf("  Simulating T = %d...\n", T))

  sims <- replicate(n_simulations, run_regression(T, spurious = TRUE),
                    simplify = FALSE)

  t_stats <- sapply(sims, `[[`, "t_stat")
  r2s     <- sapply(sims, `[[`, "r2")
  dws     <- sapply(sims, `[[`, "dw")

  fig1_stats$rejection_rate[i] <- mean(abs(t_stats) > critical_value) * 100
  fig1_stats$mean_r2[i]        <- mean(r2s)
  fig1_stats$mean_dw[i]        <- mean(dws)

  cat(sprintf("  T=%d: Rejection rate=%.1f%%, Mean R2=%.3f, Mean DW=%.3f\n",
              T, fig1_stats$rejection_rate[i],
              fig1_stats$mean_r2[i], fig1_stats$mean_dw[i]))
}

fig1_stats$T_label <- factor(as.character(fig1_stats$T),
                              levels = as.character(sample_sizes))

# Panel (a)
pa <- ggplot(fig1_stats, aes(x = T_label, y = rejection_rate)) +
  geom_col(fill = "#E74C6F", colour = "white", width = 0.6) +
  geom_hline(yintercept = 5, linetype = "dashed", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.0f%%", rejection_rate)),
            vjust = -0.5, fontface = "bold", size = 3.8) +
  annotate("text", x = 0.7, y = 6.5, label = "Nominal 5%", size = 3) +
  scale_y_continuous(limits = c(0, 100)) +
  labs(title = "(a) False Rejection Rate at 5% Level",
       x = "Sample Size (T)", y = "Rejection Rate (%)") +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 12))

# Panel (b)
pb <- ggplot(fig1_stats, aes(x = T_label, y = mean_r2)) +
  geom_col(fill = "#6C9BD1", colour = "white", width = 0.6) +
  scale_y_continuous(limits = c(0, 0.5)) +
  labs(title = expression(bold("(b) Mean " * R^2 * " from Spurious Regressions")),
       x = "Sample Size (T)", y = expression("Mean " * R^2)) +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 12))

# Panel (c)
pc <- ggplot(fig1_stats, aes(x = T_label, y = mean_dw)) +
  geom_col(fill = "#2E7D32", colour = "white", width = 0.6) +
  geom_hline(yintercept = 2, linetype = "dashed", colour = "red", linewidth = 0.8) +
  annotate("text", x = 0.7, y = 2.1, label = "DW = 2", colour = "red", size = 3) +
  scale_y_continuous(limits = c(0, 2.5)) +
  labs(title = "(c) Mean DW Statistic (Lower = More Autocorrelation)",
       x = "Sample Size (T)", y = "Mean Durbin-Watson") +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 12))

fig1 <- pa + pb + pc
ggsave(file.path(out_dir, "spurious_regression_montecarlo.png"),
       fig1, width = 16, height = 5, dpi = 300)
ggsave(file.path(out_dir, "spurious_regression_montecarlo.pdf"),
       fig1, width = 16, height = 5)
cat("  -> Saved: spurious_regression_montecarlo.png/pdf\n\n")


# =========================================================
# Figure 2: Granger-Newbold Rule of Thumb
# =========================================================
cat(strrep("=", 60), "\n")
cat("Figure 2: Granger-Newbold Rule of Thumb\n")
cat(strrep("=", 60), "\n")

n_sims_scatter <- 1000
T_scatter      <- 200

cat("  Simulating spurious regressions...\n")
spur <- replicate(n_sims_scatter, run_regression(T_scatter, spurious = TRUE),
                  simplify = FALSE)
spurious_df <- data.frame(
  dw   = sapply(spur, `[[`, "dw"),
  r2   = sapply(spur, `[[`, "r2"),
  type = "Spurious (independent I(1))"
)

cat("  Simulating valid regressions...\n")
valid_list <- lapply(seq_len(n_sims_scatter), function(...) {
  x   <- rnorm(T_scatter)
  y   <- 0.5 * x + rnorm(T_scatter)
  fit <- lm(y ~ x)
  list(r2 = summary(fit)$r.squared, dw = dw_stat(residuals(fit)))
})
valid_df <- data.frame(
  dw   = sapply(valid_list, `[[`, "dw"),
  r2   = sapply(valid_list, `[[`, "r2"),
  type = "Valid (stationary with true relationship)"
)

scatter_df <- rbind(spurious_df, valid_df)

fig2 <- ggplot(scatter_df, aes(x = dw, y = r2, colour = type, shape = type)) +
  # Shaded spurious region
  annotate("ribbon", x = c(0, 1), ymin = c(0, 1), ymax = 1,
           fill = "red", alpha = 0.10) +
  # 45-degree line
  geom_abline(intercept = 0, slope = 1, linetype = "dashed",
              linewidth = 1.2, colour = "black") +
  geom_point(alpha = 0.45, size = 1.8) +
  annotate("text", x = 0.18, y = 0.90,
           label = "Spurious\nRegion\n(R² > DW)",
           colour = "red", fontface = "bold.italic", size = 4.5) +
  scale_colour_manual(values = c("Spurious (independent I(1))"              = "#E74C6F",
                                  "Valid (stationary with true relationship)" = "#5DADE2")) +
  scale_shape_manual(values  = c("Spurious (independent I(1))"              = 16,
                                  "Valid (stationary with true relationship)" = 17)) +
  scale_x_continuous(limits = c(0, 2.5)) +
  scale_y_continuous(limits = c(0, 1.0)) +
  labs(
    title  = expression(bold("Granger-Newbold Rule of Thumb: If " * R^2 * " > DW, Suspect Spurious Regression")),
    x      = "Durbin-Watson Statistic",
    y      = expression(R^2),
    colour = NULL, shape = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold", size = 13))

ggsave(file.path(out_dir, "spurious_regression_rule_of_thumb.png"),
       fig2, width = 10, height = 7, dpi = 300)
ggsave(file.path(out_dir, "spurious_regression_rule_of_thumb.pdf"),
       fig2, width = 10, height = 7)
cat("  -> Saved: spurious_regression_rule_of_thumb.png/pdf\n\n")


# =========================================================
# Figure 3: t-statistic distributions
# =========================================================
cat(strrep("=", 60), "\n")
cat("Figure 3: Distribution of t-statistics\n")
cat(strrep("=", 60), "\n")

n_sims_tstat <- 5000
T_tstat      <- 200

cat("  Simulating spurious regressions...\n")
spurious_t <- replicate(n_sims_tstat,
                         run_regression(T_tstat, spurious = TRUE)$t_stat)

cat("  Simulating valid regressions...\n")
valid_t <- replicate(n_sims_tstat,
                      run_regression(T_tstat, spurious = FALSE)$t_stat)

# Theoretical t-density
t_grid <- seq(-15, 15, length.out = 500)
theory_df <- data.frame(t = t_grid, density = dt(t_grid, df = T_tstat - 2))

make_tstat_panel <- function(t_vals, fill_col, panel_label, subtitle) {
  df <- data.frame(t = t_vals)
  ggplot(df, aes(x = t)) +
    geom_histogram(aes(y = after_stat(density)), bins = 50,
                   fill = fill_col, alpha = 0.6, colour = "white") +
    geom_line(data = theory_df, aes(x = t, y = density),
              linetype = "dashed", linewidth = 0.9, colour = "black") +
    geom_vline(xintercept = c(-1.96, 1.96),
               linetype = "dotted", linewidth = 0.8, colour = "blue") +
    scale_x_continuous(limits = c(-15, 15)) +
    scale_y_continuous(limits = c(0, 0.40)) +
    labs(title    = panel_label,
         subtitle = subtitle,
         x        = "t-statistic",
         y        = "Density") +
    theme_bw(base_size = 11) +
    theme(plot.title    = element_text(face = "bold", size = 12),
          plot.subtitle = element_text(size = 10))
}

p3a <- make_tstat_panel(spurious_t, "#E74C6F",
                         "(a) t-statistics from Spurious Regressions",
                         "Two Independent Random Walks")
p3b <- make_tstat_panel(valid_t,    "#5DADE2",
                         "(b) t-statistics from Valid Regressions",
                         "Two Independent White Noise Series")

fig3 <- p3a + p3b
ggsave(file.path(out_dir, "spurious_regression_tstat_distribution.png"),
       fig3, width = 14, height = 5.5, dpi = 300)
ggsave(file.path(out_dir, "spurious_regression_tstat_distribution.pdf"),
       fig3, width = 14, height = 5.5)
cat("  -> Saved: spurious_regression_tstat_distribution.png/pdf\n\n")

cat(strrep("=", 60), "\n")
cat("All figures generated successfully!\n")
cat(strrep("=", 60), "\n")
