# =========================================================================
#  Figure 5.1: Common Prior Distributions for Bayesian Estimation
# =========================================================================
#  Author: Alessia Paccagnini
#  Textbook: Macroeconometrics
# =========================================================================

rm(list = ls())

# Color palette
colors <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")

# --- Save to PNG and display in RStudio ---
dev.new(width = 14, height = 4.5)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3.5, 1), oma = c(0, 0, 0, 0))

# =================================================================
# Panel (a): Normal Prior
# =================================================================
x <- seq(-6, 6, length.out = 500)

normal_params <- list(
  list(mu = 0, sigma = 0.5, lab = expression(N(0, 0.5^2) ~ "— Tight")),
  list(mu = 0, sigma = 1.0, lab = expression(N(0, 1^2) ~ "— Moderate")),
  list(mu = 0, sigma = 2.0, lab = expression(N(0, 2^2) ~ "— Diffuse")),
  list(mu = 1, sigma = 1.0, lab = expression(N(1, 1^2) ~ "— Shifted"))
)

plot(NULL, xlim = c(-6, 6), ylim = c(0, 0.85),
     xlab = expression(theta), ylab = "Density",
     main = "(a) Normal Prior\nLocation parameters", font.main = 2)
grid(col = "gray90")
for (i in seq_along(normal_params)) {
  p <- normal_params[[i]]
  lines(x, dnorm(x, p$mu, p$sigma), col = colors[i], lwd = 2.5)
}
legend("topright", legend = sapply(normal_params, function(p) {
  paste0("N(", p$mu, ", ", p$sigma, "²)")
}), col = colors[1:4], lwd = 2.5, cex = 0.85, bg = "white",
  title = NULL)

# =================================================================
# Panel (b): Inverse Gamma Prior
# =================================================================
x_ig <- seq(0.01, 5, length.out = 500)

# scipy invgamma(a=alpha, scale=beta) matches R's dinvgamma
# Using manual formula: beta^alpha / Gamma(alpha) * x^(-alpha-1) * exp(-beta/x)
dinvgamma <- function(x, alpha, beta) {
  (beta^alpha / gamma(alpha)) * x^(-alpha - 1) * exp(-beta / x)
}

ig_params <- list(
  list(alpha = 3, beta = 2, lab = "IG(3, 2) — Moderate"),
  list(alpha = 2, beta = 1, lab = "IG(2, 1) — Diffuse"),
  list(alpha = 1, beta = 1, lab = "IG(1, 1) — Heavy tail"),
  list(alpha = 8, beta = 3, lab = "IG(8, 3) — Informative")
)

plot(NULL, xlim = c(0, 5), ylim = c(0, 4),
     xlab = expression(sigma^2), ylab = "Density",
     main = "(b) Inverse Gamma Prior\nVariance parameters", font.main = 2)
grid(col = "gray90")
for (i in seq_along(ig_params)) {
  p <- ig_params[[i]]
  lines(x_ig, dinvgamma(x_ig, p$alpha, p$beta), col = colors[i], lwd = 2.5)
}
legend("topright", legend = sapply(ig_params, function(p) p$lab),
       col = colors[1:4], lwd = 2.5, cex = 0.85, bg = "white")

# =================================================================
# Panel (c): Beta Prior
# =================================================================
x_beta <- seq(0.001, 0.999, length.out = 500)

beta_params <- list(
  list(a = 1,   b = 1, lab = "Beta(1,1) — Uniform"),
  list(a = 0.5, b = 0.5, lab = "Beta(0.5,0.5) — U-shaped"),
  list(a = 2,   b = 5, lab = "Beta(2,5) — Skewed right"),
  list(a = 5,   b = 2, lab = "Beta(5,2) — Skewed left"),
  list(a = 5,   b = 5, lab = "Beta(5,5) — Symmetric")
)

plot(NULL, xlim = c(0, 1), ylim = c(0, 4),
     xlab = expression(p), ylab = "Density",
     main = "(c) Beta Prior\nProbabilities & proportions", font.main = 2)
grid(col = "gray90")
for (i in seq_along(beta_params)) {
  p <- beta_params[[i]]
  lines(x_beta, dbeta(x_beta, p$a, p$b), col = colors[i], lwd = 2.5)
}
legend("topright", legend = sapply(beta_params, function(p) p$lab),
       col = colors[1:5], lwd = 2.5, cex = 0.85, bg = "white")

# --- Save ---
dev.copy(png, filename = "prior_distributions.png", width = 1400, height = 450, res = 150)
dev.off()
cat("Saved: prior_distributions.png\n")
