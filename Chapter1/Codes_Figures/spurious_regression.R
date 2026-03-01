# =========================================================
# Spurious Regression: Nine Regressions of Independent Random Walks
# =========================================================
# Author:   Alessia Paccagnini
# Textbook: Macroeconometrics
# =========================================================

library(ggplot2)
library(gridExtra)

set.seed(NULL)  # Random seed: we'll try until we find good results

# Output directory: same folder as this script (portable)
out_dir <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)),
                    error = function(e) getwd())

# --------------------------------------------------------
# Helper: Durbin-Watson statistic
# --------------------------------------------------------
durbin_watson <- function(resid) {
  sum(diff(resid)^2) / sum(resid^2)
}

# --------------------------------------------------------
# Search for 9 pairs with R2 > 0.80 and DW < 0.20
# --------------------------------------------------------
T            <- 300
max_attempts <- 100000
attempt      <- 0
results      <- list()

while (length(results) < 9 && attempt < max_attempts) {
  attempt <- attempt + 1

  x <- cumsum(rnorm(T))
  y <- cumsum(rnorm(T))

  fit   <- lm(y ~ x)
  r2    <- summary(fit)$r.squared
  t_stat <- coef(summary(fit))["x", "t value"]
  dw    <- durbin_watson(residuals(fit))

  if (r2 > 0.80 && dw < 0.20) {
    results[[length(results) + 1]] <- list(
      x      = x,
      y      = y,
      fitted = fitted(fit),
      r2     = r2,
      t      = t_stat,
      dw     = dw
    )
    cat(sprintf("Found pair %d: R2=%.2f, t=%.1f, DW=%.2f\n",
                length(results), r2, t_stat, dw))
  }
}

cat(sprintf("\nTotal attempts: %d\n", attempt))
cat(sprintf("Pairs found: %d\n", length(results)))

# --------------------------------------------------------
# Build one ggplot panel per pair
# --------------------------------------------------------
make_panel <- function(res) {
  df      <- data.frame(x = res$x, y = res$y, fitted = res$fitted)
  df      <- df[order(df$x), ]
  abs_t   <- abs(res$t)
  stars   <- ifelse(abs_t > 3.29, "***",
             ifelse(abs_t > 2.58, "**",
             ifelse(abs_t > 1.96, "*", "")))
  t_sign  <- ifelse(res$t < 0, "-", "")
  ttl     <- sprintf("R2=%.2f, t=%s%.1f%s, DW=%.2f",
                     res$r2, t_sign, abs_t, stars, res$dw)

  ggplot(df, aes(x = x, y = y)) +
    geom_point(alpha = 0.45, size = 1.2,
               colour = "#9B59B6") +
    geom_line(aes(y = fitted), colour = "red", linewidth = 0.9) +
    labs(title = ttl, x = expression(x[t]), y = expression(y[t])) +
    theme_bw(base_size = 9) +
    theme(plot.title = element_text(size = 8, hjust = 0.5))
}

panels <- lapply(results, make_panel)

# --------------------------------------------------------
# Arrange and save
# --------------------------------------------------------
combined <- arrangeGrob(
  grobs = panels,
  ncol  = 3,
  top   = "Nine Regressions of Independent Random Walks\n(All relationships are SPURIOUS)"
)

ggsave(file.path(out_dir, "spurious_regression_multiple.png"),
       combined, width = 14, height = 12, dpi = 300)
ggsave(file.path(out_dir, "spurious_regression_multiple.pdf"),
       combined, width = 14, height = 12)

cat("\nPlots saved!\n")
