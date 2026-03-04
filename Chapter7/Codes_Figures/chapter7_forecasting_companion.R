# ============================================================================
# CHAPTER 7: FORECASTING — Figures 7.1–7.4 (R Companion)
# Macroeconometrics Textbook
# Author: Alessia Paccagnini
# ============================================================================
#
# This script generates four publication-ready figures for Chapter 7:
#   Figure 7.1: bias_variance_tradeoff.pdf
#   Figure 7.2: bias_variance_examples.pdf
#   Figure 7.3: forecast_errors_example.pdf  (needs GDPC1.xlsx)
#   Figure 7.4: giacomini_rossi_fluctuation_test.pdf
#
# Required packages: readxl (for Fig 7.3 only)
# Install: install.packages("readxl")
#
# Run all:  source("chapter7_forecasting_companion.R")
# Or individually: run_figure_7_1(), run_figure_7_2(), etc.
# ============================================================================


# ============================================================================
#  FIGURE 7.1 — The Bias-Variance Tradeoff in Forecasting
# ============================================================================

run_figure_7_1 <- function() {
  cat("==== FIGURE 7.1 — The Bias-Variance Tradeoff ====\n")

  complexity <- seq(0, 10, length.out = 200)
  bias_sq    <- 8 * exp(-0.5 * complexity) + 0.5
  variance   <- 0.1 + 0.15 * complexity^1.5
  irreducible <- rep(2.0, length(complexity))
  total      <- bias_sq + variance + irreducible

  opt_idx <- which.min(total)
  opt_x   <- complexity[opt_idx]
  opt_y   <- total[opt_idx]

  pdf("bias_variance_tradeoff.pdf", width = 10, height = 6)
  par(mar = c(6, 4.5, 3, 1), family = "serif")

  plot(complexity, total, type = "l", lwd = 3, col = "green3",
       ylim = c(0, 12), xaxt = "n",
       xlab = "", ylab = "Mean Squared Error",
       main = "The Bias-Variance Tradeoff in Forecasting",
       cex.main = 1.3, cex.lab = 1.1)
  lines(complexity, bias_sq, lwd = 2.5, col = "blue")
  lines(complexity, variance, lwd = 2.5, col = "red")
  lines(complexity, irreducible, lwd = 1.5, col = "black", lty = 2)

  points(opt_x, opt_y, pch = 19, col = "darkgreen", cex = 2)
  abline(v = opt_x, col = "grey", lty = 3, lwd = 1.5)

  axis(1, at = c(0, 2.5, 5, 7.5, 10),
       labels = c("Random\nWalk", "AR(1)", "AR(4)\nBVAR",
                   "Unrestr.\nVAR", "High-Dim\nModel"),
       cex.axis = 0.85, padj = 0.5)
  mtext("Model Complexity", side = 1, line = 4.5, cex = 1.1, font = 2)

  # Annotations
  text(1.5, 7, "Underfitting\n(High Bias)", cex = 0.95)
  rect(0.3, 6.2, 2.7, 7.8, col = rgb(0.68, 0.85, 0.9, 0.5), border = NA)
  text(1.5, 7, "Underfitting\n(High Bias)", cex = 0.95)

  text(opt_x, opt_y - 1.5, "Optimal\nComplexity", cex = 0.95)
  text(8, 8, "Overfitting\n(High Variance)", cex = 0.95)

  legend("top", ncol = 5,
         legend = c(expression(Bias^2), "Variance", "Irreduc. Error",
                    "Total Error", "Optimal"),
         col = c("blue", "red", "black", "green3", "darkgreen"),
         lwd = c(2.5, 2.5, 1.5, 3, NA), lty = c(1, 1, 2, 1, NA),
         pch = c(NA, NA, NA, NA, 19), pt.cex = 1.5,
         cex = 0.85, bty = "n")
  grid(col = "grey90")
  dev.off()
  cat("Saved: bias_variance_tradeoff.pdf\n\n")
}


# ============================================================================
#  FIGURE 7.2 — Bias-Variance Decomposition in Practice
# ============================================================================

run_figure_7_2 <- function() {
  cat("==== FIGURE 7.2 — Bias-Variance Decomposition ====\n")

  set.seed(42)

  # --- Left panel data ---
  simple  <- rnorm(1000, mean = 1.5, sd = 0.5)
  complex <- rnorm(1000, mean = 0.3, sd = 1.8)
  optimal <- rnorm(1000, mean = 0.5, sd = 0.9)

  # --- Right panel data ---
  models   <- c("RW", "AR(1)", "AR(2)", "AR(4)",
                 "VAR(2)", "VAR(4)", "BVAR", "VAR(8)")
  rmse_val <- c(3.2, 2.9, 2.6, 2.3, 2.4, 2.5, 2.35, 2.8)
  bias_val <- c(2.8, 2.2, 1.5, 0.8, 0.7, 0.5, 0.6, 0.3)
  var_val  <- c(0.4, 0.7, 1.1, 1.5, 1.7, 2.0, 1.75, 2.5)

  pdf("bias_variance_examples.pdf", width = 14, height = 5)
  par(mfrow = c(1, 2), mar = c(5, 4.5, 3, 1), family = "serif")

  # Left panel: histograms
  brk <- seq(-6, 7, by = 0.3)
  h_s <- hist(simple,  breaks = brk, plot = FALSE)
  h_c <- hist(complex, breaks = brk, plot = FALSE)
  h_o <- hist(optimal, breaks = brk, plot = FALSE)

  # Normalise to density
  h_s$density <- h_s$counts / (sum(h_s$counts) * diff(h_s$breaks)[1])
  h_c$density <- h_c$counts / (sum(h_c$counts) * diff(h_c$breaks)[1])
  h_o$density <- h_o$counts / (sum(h_o$counts) * diff(h_o$breaks)[1])

  ymax <- max(h_s$density, h_c$density, h_o$density) * 1.1

  plot(h_s, freq = FALSE, col = rgb(0, 0, 1, 0.4), border = NA,
       xlim = c(-5, 7), ylim = c(0, ymax),
       main = "Distribution of Forecast Errors",
       xlab = "Forecast Error", ylab = "Density",
       cex.main = 1.2, cex.lab = 1.1)
  plot(h_c, freq = FALSE, col = rgb(1, 0, 0, 0.4), border = NA, add = TRUE)
  plot(h_o, freq = FALSE, col = rgb(0, 0.7, 0, 0.4), border = NA, add = TRUE)
  abline(v = 0, lty = 2, lwd = 2)

  legend("topright",
         legend = c(sprintf("RW (Bias=%.2f, SD=%.2f)", mean(simple), sd(simple)),
                    sprintf("VAR(8) (Bias=%.2f, SD=%.2f)", mean(complex), sd(complex)),
                    sprintf("AR(4) (Bias=%.2f, SD=%.2f)", mean(optimal), sd(optimal)),
                    "Target (zero)"),
         fill = c(rgb(0,0,1,0.5), rgb(1,0,0,0.5), rgb(0,0.7,0,0.5), NA),
         border = c(NA, NA, NA, "black"), lty = c(NA, NA, NA, 2),
         lwd = c(NA, NA, NA, 2), cex = 0.75, bty = "n")

  # Right panel: stacked bars + RMSE line
  par(mar = c(6, 4.5, 3, 4.5))
  x <- barplot(rbind(bias_val, var_val), beside = FALSE,
               col = c("steelblue", "coral"), border = NA,
               names.arg = models, las = 2,
               ylab = "Error Components",
               main = "Bias-Variance Decomposition by Model",
               cex.main = 1.2, cex.lab = 1.1, cex.names = 0.9)

  par(new = TRUE)
  plot(x, rmse_val, type = "o", pch = 19, col = "darkgreen", lwd = 2.5,
       axes = FALSE, xlab = "", ylab = "", ylim = c(2, 3.5))
  axis(4, col = "green4", col.axis = "green4")
  mtext("Total RMSE", side = 4, line = 3, col = "green4", cex = 0.9, font = 2)

  opt_i <- which.min(rmse_val)
  points(x[opt_i], rmse_val[opt_i], pch = 8, col = "darkgreen", cex = 2, lwd = 2)

  legend("topleft",
         legend = c(expression(Bias^2), "Variance", "Total RMSE"),
         fill = c("steelblue", "coral", NA),
         border = c(NA, NA, NA),
         col = c(NA, NA, "darkgreen"),
         lwd = c(NA, NA, 2.5), pch = c(NA, NA, 19),
         cex = 0.85, bty = "n")

  dev.off()
  cat("Saved: bias_variance_examples.pdf\n\n")
}


# ============================================================================
#  FIGURE 7.3 — Visualising Forecast Performance
# ============================================================================

run_figure_7_3 <- function(file_path = "GDPC1.xlsx") {
  cat("==== FIGURE 7.3 — Visualising Forecast Performance ====\n")

  # --- Load data ---
  use_real <- FALSE
  tryCatch({
    library(readxl)
    df <- read_excel(file_path, sheet = "Quarterly", col_names = TRUE)
    colnames(df) <- c("date", "gdpc1")
    df$date  <- as.Date(df$date)
    df$gdpc1 <- as.numeric(df$gdpc1)
    df <- df[complete.cases(df), ]
    # Quarterly log growth (%)
    df$growth <- c(NA, 100 * diff(log(df$gdpc1)))
    df <- df[complete.cases(df), ]

    # 1985Q1 to 2023Q4
    df <- df[df$date >= as.Date("1985-01-01") & df$date <= as.Date("2023-12-31"), ]
    cat(sprintf("   Loaded GDPC1: %s to %s (%d obs)\n",
                df$date[1], tail(df$date, 1), nrow(df)))
    use_real <- TRUE
  }, error = function(e) {
    cat(sprintf("   Could not load %s: %s\n", file_path, e$message))
    cat("   Using simulated data.\n")
  })

  if (!use_real) {
    # Simulated fallback
    set.seed(2026)
    dates <- seq(as.Date("1985-01-01"), as.Date("2023-10-01"), by = "quarter")
    T <- length(dates)
    y <- numeric(T)
    y[1] <- 0.8; y[2] <- 0.6
    for (t in 3:T) {
      y[t] <- 0.15 + 0.30*y[t-1] + 0.12*y[t-2] + rnorm(1, sd = 0.5)
    }
    # COVID shock
    covid_i <- which(dates >= as.Date("2020-04-01"))[1]
    y[covid_i] <- -8.2; y[covid_i+1] <- 7.5; y[covid_i+2] <- 1.1
    df <- data.frame(date = dates, growth = y)
  }

  dates <- df$date
  y     <- df$growth
  T     <- length(y)

  # --- Expanding-window AR(2) forecasts ---
  R <- 95
  forecasts <- rep(NA_real_, T)

  for (t in R:(T - 1)) {
    Y <- y[3:(t)]           # dependent: y_3 ... y_t
    X <- cbind(1, y[2:(t-1)], y[1:(t-2)])  # lags
    beta <- tryCatch(
      solve(t(X) %*% X, t(X) %*% Y),
      error = function(e) NULL
    )
    if (!is.null(beta)) {
      forecasts[t + 1] <- beta[1] + beta[2]*y[t] + beta[3]*y[t-1]
    }
  }

  fmask  <- !is.na(forecasts)
  errors <- ifelse(fmask, y - forecasts, NA)
  valid  <- errors[fmask]
  rmse   <- sqrt(mean(valid^2))
  mae    <- mean(abs(valid))

  cat(sprintf("   Forecasts: %s to %s (%d obs)\n",
              dates[which(fmask)[1]], dates[tail(which(fmask), 1)],
              sum(fmask)))
  cat(sprintf("   RMSE = %.2f,  MAE = %.2f\n", rmse, mae))

  # --- Recession dates ---
  rec_starts <- as.Date(c("2007-12-01", "2020-02-01"))
  rec_ends   <- as.Date(c("2009-06-30", "2020-04-30"))

  # --- Plot range: from 2009 ---
  pi <- which(dates >= as.Date("2009-01-01"))

  pdf("forecast_errors_example.pdf", width = 12, height = 9)
  par(mfrow = c(2, 1), family = "serif")

  # Panel (a): Actual vs Forecast
  par(mar = c(3, 4.5, 3, 1))
  plot(dates[pi], y[pi], type = "l", lwd = 1.8, col = "black",
       ylab = "GDP Growth (%)", xlab = "",
       main = "(a) Actual vs. Forecast",
       cex.main = 1.3, cex.lab = 1.1)
  # Recession shading
  for (k in seq_along(rec_starts)) {
    rect(rec_starts[k], par("usr")[3], rec_ends[k], par("usr")[4],
         col = rgb(0.8, 0.8, 0.8, 0.5), border = NA)
  }
  lines(dates[pi], y[pi], lwd = 1.8, col = "black")
  fc_i <- pi[fmask[pi]]
  lines(dates[fc_i], forecasts[fc_i], lwd = 1.8, col = "blue", lty = 2)
  abline(h = 0, col = "black", lwd = 0.5)
  legend("bottomright",
         legend = c("Actual", "Forecast"),
         col = c("black", "blue"), lwd = 1.8, lty = c(1, 2),
         cex = 1, bty = "o", bg = "white", box.col = "grey")
  grid(col = "grey90")

  # Panel (b): Forecast Errors
  par(mar = c(4, 4.5, 3, 1))
  err_pi <- errors[pi]
  fc_dates_pi <- dates[pi]
  ylim_b <- c(-10, 10)

  plot(fc_dates_pi, err_pi, type = "n",
       ylim = ylim_b, ylab = "Forecast Error (%)",
       xlab = "Time", main = "(b) Forecast Errors",
       cex.main = 1.3, cex.lab = 1.1)

  # Recession shading
  for (k in seq_along(rec_starts)) {
    rect(rec_starts[k], ylim_b[1], rec_ends[k], ylim_b[2],
         col = rgb(0.8, 0.8, 0.8, 0.5), border = NA)
  }

  # Bars coloured by sign
  bar_w <- 75  # days
  for (j in seq_along(fc_dates_pi)) {
    if (!is.na(err_pi[j])) {
      col_bar <- ifelse(err_pi[j] >= 0, "steelblue", "coral")
      rect(fc_dates_pi[j] - bar_w/2, 0,
           fc_dates_pi[j] + bar_w/2, err_pi[j],
           col = col_bar, border = NA)
    }
  }

  abline(h = 0, lwd = 0.8)
  abline(h = rmse,  col = "darkred", lty = 2, lwd = 1.5)
  abline(h = -rmse, col = "darkred", lty = 2, lwd = 1.5)
  grid(col = "grey90")

  legend("bottomright",
         legend = c(sprintf("+RMSE = %.2f", rmse),
                    sprintf("-RMSE = -%.2f", rmse)),
         col = "darkred", lty = 2, lwd = 1.5,
         cex = 0.9, bty = "o", bg = "white", box.col = "grey")

  # Text box
  legend("topleft",
         legend = c(sprintf("RMSE = %.2f", rmse),
                    sprintf("MAE = %.2f", mae)),
         bty = "o", bg = "white", box.col = "grey",
         cex = 1, text.font = 1)

  dev.off()
  cat("Saved: forecast_errors_example.pdf\n\n")
}


# ============================================================================
#  FIGURE 7.4 — Giacomini-Rossi Fluctuation Test
# ============================================================================

giacomini_rossi_test <- function(loss_diff, window_size = 40, alpha = 0.05) {
  T <- length(loss_diff)
  n_win <- T - window_size + 1
  t_stats <- numeric(n_win)

  for (i in 1:n_win) {
    window <- loss_diff[i:(i + window_size - 1)]
    m  <- mean(window)
    se <- sd(window) / sqrt(window_size)
    t_stats[i] <- ifelse(se > 0, m / se, 0)
  }

  max_stat <- max(abs(t_stats))
  cv <- switch(as.character(alpha),
               "0.1" = 1.73, "0.05" = 1.95, "0.01" = 2.37, 1.95)
  p_val <- 2 * (1 - pnorm(max_stat))
  periods <- seq(floor(window_size/2),
                 floor(window_size/2) + n_win - 1)

  list(t_stats = t_stats, periods = periods,
       max_stat = max_stat, critical_value = cv, p_value = p_val)
}


run_figure_7_4 <- function() {
  cat("==== FIGURE 7.4 — Giacomini-Rossi Fluctuation Test ====\n")

  set.seed(42)
  T <- 200
  t_vec <- 0:(T - 1)
  trend <- 0.5 * (t_vec - T/2) / T
  loss_diff <- trend + 0.3 * rnorm(T)

  window_size <- 40
  res <- giacomini_rossi_test(loss_diff, window_size = window_size)

  t_stats <- res$t_stats
  periods <- res$periods
  cv      <- res$critical_value

  pdf("giacomini_rossi_fluctuation_test.pdf", width = 12, height = 6)
  par(mar = c(5, 4.5, 4, 1), family = "serif")

  plot(periods, t_stats, type = "l", lwd = 2, col = "blue",
       ylim = range(c(t_stats, cv, -cv)) * 1.1,
       xlab = "Time Period (centered on rolling window)",
       ylab = "Rolling t-statistic",
       main = "Giacomini-Rossi Fluctuation Test: AR(1) vs AR(4)",
       cex.main = 1.3, cex.lab = 1.1)

  abline(h = cv,  col = "red", lty = 2, lwd = 1.5)
  abline(h = -cv, col = "red", lty = 2, lwd = 1.5)
  abline(h = 0,   col = "black", lwd = 0.5)

  # Shade rejection regions
  upper <- t_stats > cv
  lower <- t_stats < -cv
  if (any(upper)) {
    idx <- which(upper)
    for (i in idx) {
      rect(periods[i] - 0.5, cv, periods[i] + 0.5, t_stats[i],
           col = rgb(0, 0.8, 0, 0.2), border = NA)
    }
  }
  if (any(lower)) {
    idx <- which(lower)
    for (i in idx) {
      rect(periods[i] - 0.5, t_stats[i], periods[i] + 0.5, -cv,
           col = rgb(1, 0.65, 0, 0.2), border = NA)
    }
  }

  # Re-draw line on top
  lines(periods, t_stats, lwd = 2, col = "blue")
  grid(col = "grey90")

  legend("bottomright",
         legend = c("Rolling t-statistic",
                    sprintf("95%% CV (+/-%.2f)", cv),
                    "AR(4) better", "AR(1) better"),
         col = c("blue", "red", rgb(0,0.8,0,0.4), rgb(1,0.65,0,0.4)),
         lwd = c(2, 1.5, 8, 8), lty = c(1, 2, 1, 1),
         cex = 0.85, bty = "o", bg = "white")

  # Text box
  legend("topleft",
         legend = c(sprintf("Test stat: %.3f", res$max_stat),
                    sprintf("p-value: %.3f", res$p_value)),
         bty = "o", bg = "wheat", cex = 0.9)

  dev.off()

  cat(sprintf("   Max |t| = %.3f,  CV(5%%) = %.3f,  p = %.3f\n",
              res$max_stat, cv, res$p_value))
  if (res$max_stat > cv) {
    cat("   => REJECT: time-varying forecast performance\n")
  } else {
    cat("   => Do not reject equal predictive ability\n")
  }
  cat("Saved: giacomini_rossi_fluctuation_test.pdf\n\n")
}


# ============================================================================
#  MAIN
# ============================================================================

cat("************************************************************\n")
cat("*  CHAPTER 7: FORECASTING — R Companion (Figures 7.1–7.4)\n")
cat("*  Macroeconometrics Textbook — Alessia Paccagnini\n")
cat("************************************************************\n\n")

run_figure_7_1()
run_figure_7_2()
run_figure_7_3()   # set file_path argument if GDPC1.xlsx is elsewhere
run_figure_7_4()

cat("************************************************************\n")
cat("*  ALL FIGURES COMPLETE\n")
cat("************************************************************\n")
