# =============================================================================
# Growth-at-Risk (GaR) with the Chicago Fed NFCI
# =============================================================================
# Textbook : Macroeconometrics
# Author   : Alessia Paccagnini
# Chapter  : 12 — Quantile Regression and Growth-at-Risk
# Section  : 12.7 — Empirical Application: US Growth-at-Risk with NFCI
#
# Empirical specification (eq. 12.9):
#   Q_tau(Dy_{t+4} | Omega_t) = alpha(tau) + beta_y(tau)*Dy_t
#                              + beta_pi(tau)*pi_t + beta_f(tau)*NFCI_t
#
# Estimation sample : 1971Q1-2024Q3   (N = 215)
# Forecast horizon  : h = 4 quarters ahead
# Quantiles         : {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95} + OLS
#
# Files needed (place in working directory):
#   GDPC1.xlsx   - Real GDP, quarterly  (sheet: Quarterly)
#   GDPDEF.xlsx  - GDP Deflator, qtrly  (sheet: Quarterly)
#   NFCI.csv     - NFCI weekly, FRED    (cols: observation_date, NFCI)
#
# Script structure - aligned with Chapter 12 exercises:
#   Step 1 - Load data       (Exercise: data collection)
#   Step 2 - Transform       (Exercise: stationarity, summary stats)
#   Step 3 - Quantile reg.   (Exercise: estimate at multiple tau, Table 12.3)
#   Step 4 - Asymmetry       (Exercise: coefficient asymmetry plot, Fig 12.1)
#   Step 5 - Coverage        (Exercise: model evaluation, Table 12.4)
#   Step 6 - Risk assessment (Section 12.7.5 current conditions)
#   Step 7 - Figures         (Fig 12.1-12.4 + extras)
#
# Required packages:
#   readxl, dplyr, tidyr, lubridate, quantreg, ggplot2, patchwork, scales
#
# Install if needed:
#   install.packages(c("readxl","dplyr","tidyr","lubridate",
#                      "quantreg","ggplot2","patchwork","scales"))
# =============================================================================

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(lubridate)
  library(quantreg)
  library(ggplot2)
  library(patchwork)
  library(scales)
})

# ── Working directory ────────────────────────────────────────────────────
# Option A (RStudio): runs automatically — sets wd to the folder of this script
if (interactive() && requireNamespace("rstudioapi", quietly = TRUE)) {
  script_dir <- tryCatch(
    dirname(rstudioapi::getSourceEditorContext()$path),
    error = function(e) NULL
  )
  if (!is.null(script_dir) && nchar(script_dir) > 0) setwd(script_dir)
}
# Option B (manual): uncomment and set your path explicitly
# setwd("C:/Users/yourname/Documents/chapter12")   # Windows
# setwd("/Users/yourname/Documents/chapter12")      # Mac/Linux

# Output directory — figures saved here (same folder as data by default)
OUTPUT_DIR <- getwd()

cat(strrep("=", 70), "\n")
cat("GROWTH-AT-RISK ANALYSIS WITH NFCI\n")
cat("Chapter 12, Section 12.7  |  Macroeconometrics  |  Alessia Paccagnini\n")
cat(strrep("=", 70), "\n\n")

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================
cat(strrep("-", 70), "\n")
cat("Step 1 — Loading data\n")

# Real GDP (quarterly)
gdp <- read_excel("GDPC1.xlsx", sheet = "Quarterly") |>
  mutate(date = as.Date(observation_date)) |>
  select(date, GDPC1)
cat(sprintf("  GDPC1  : %d quarterly obs  (%s -> %s)\n",
            nrow(gdp), min(gdp$date), max(gdp$date)))

# GDP Deflator (quarterly)
defl <- read_excel("GDPDEF.xlsx", sheet = "Quarterly") |>
  mutate(date = as.Date(observation_date)) |>
  select(date, GDPDEF)
cat(sprintf("  GDPDEF : %d quarterly obs\n", nrow(defl)))

# NFCI (weekly -> quarterly average)
if (!file.exists("NFCI.csv")) {
  stop("[ERROR] NFCI.csv not found. Download from https://fred.stlouisfed.org/series/NFCI")
}
nfci_w <- read.csv("NFCI.csv") |>
  mutate(date = as.Date(observation_date))

# Aggregate weekly NFCI to quarterly average (quarter start = first day of quarter)
nfci_q <- nfci_w |>
  mutate(qdate = floor_date(date, unit = "quarter")) |>
  group_by(qdate) |>
  summarise(nfci = mean(NFCI, na.rm = TRUE), .groups = "drop") |>
  rename(date = qdate)

cat(sprintf("  NFCI   : %d weekly -> %d quarterly avg  (%s -> %s)\n",
            nrow(nfci_w), nrow(nfci_q), min(nfci_w$date), max(nfci_w$date)))

# =============================================================================
# STEP 2 — MERGE AND TRANSFORM
# =============================================================================
cat(strrep("-", 70), "\n")
cat("Step 2 — Transformations and sample restriction\n")

H <- 4   # forecast horizon

df <- gdp |>
  inner_join(defl, by = "date") |>
  left_join(nfci_q, by = "date") |>
  arrange(date) |>
  mutate(
    gdp_growth  = c(rep(NA, 4), diff(log(GDPC1),  lag = 4)) * 100,   # Dy_t (eq. 12.9)
    inflation   = c(rep(NA, 4), diff(log(GDPDEF), lag = 4)) * 100,   # pi_t
    gdp_forward = lead(gdp_growth, H)                                  # target t+4
  ) |>
  filter(date >= as.Date("1971-01-01"),
         date <= as.Date("2024-09-30")) |>
  drop_na(gdp_growth, inflation, nfci, gdp_forward)

cat(sprintf("  Sample : 1971Q1 -> 2024Q3   N = %d\n", nrow(df)))

cat("\nDescriptive statistics:\n")
print(summary(df[, c("gdp_growth", "inflation", "nfci")]))

# =============================================================================
# STEP 3 — QUANTILE REGRESSION  (eq. 12.9 / Table 12.3)
# =============================================================================
cat(strrep("-", 70), "\n")
cat("Step 3 — Quantile regression  (eq. 12.9 / Table 12.3)\n")
cat("Note: SEs are asymptotic (Huber sandwich). Block bootstrap recommended\n")
cat("      for overlapping observations (Section 12.2.3).\n\n")

QUANTILES <- c(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
FORMULA   <- gdp_forward ~ gdp_growth + inflation + nfci

# Fit quantile regressions
qr_fits <- lapply(QUANTILES, function(q) {
  rq(FORMULA, data = df, tau = q, method = "br")
})
names(qr_fits) <- as.character(QUANTILES)

# OLS benchmark
ols_fit <- lm(FORMULA, data = df)

# Print Table 12.3 (show tau = 0.05, 0.10, 0.25, 0.50, 0.90 + OLS)
SHOW      <- c(0.05, 0.10, 0.25, 0.50, 0.90)
VARNAMES  <- c("(Intercept)", "gdp_growth", "inflation", "nfci")
VARLABELS <- c("Intercept", "GDP_t", "Inflation_t", "NFCI_t")

stars <- function(tstat) {
  if (abs(tstat) > 2.58) "***"
  else if (abs(tstat) > 1.96) "** "
  else if (abs(tstat) > 1.64) "*  "
  else "   "
}

header <- sprintf("%-12s", "Variable")
for (q in SHOW) header <- paste0(header, sprintf("  tau=%.2f", q))
header <- paste0(header, "     OLS")
cat(header, "\n")
cat(strrep("-", 72), "\n")

for (i in seq_along(VARNAMES)) {
  var   <- VARNAMES[i]
  label <- VARLABELS[i]

  # Coef row
  row_coef <- sprintf("%-12s", label)
  for (q in SHOW) {
    fit  <- qr_fits[[as.character(q)]]
    s    <- summary(fit, se = "nid")
    coef <- coef(fit)[var]
    se   <- s$coefficients[var, "Std. Error"]
    tval <- coef / se
    row_coef <- paste0(row_coef, sprintf("  %5.2f%s", coef, stars(tval)))
  }
  ols_coef <- coef(ols_fit)[var]
  ols_se   <- summary(ols_fit)$coefficients[var, "Std. Error"]
  ols_t    <- ols_coef / ols_se
  row_coef <- paste0(row_coef, sprintf("  %5.2f%s", ols_coef, stars(ols_t)))
  cat(row_coef, "\n")

  # SE row
  row_se <- sprintf("%-12s", "")
  for (q in SHOW) {
    s  <- summary(qr_fits[[as.character(q)]], se = "nid")
    se <- s$coefficients[var, "Std. Error"]
    row_se <- paste0(row_se, sprintf("  (%4.2f)  ", se))
  }
  row_se <- paste0(row_se, sprintf("  (%4.2f)", summary(ols_fit)$coefficients[var, "Std. Error"]))
  cat(row_se, "\n")
}
cat(strrep("-", 72), "\n")
cat("* p<0.10  ** p<0.05  *** p<0.01\n")

# =============================================================================
# STEP 4 — NFCI ASYMMETRY  (Section 12.7.2)
# =============================================================================
cat(strrep("-", 70), "\n")
cat("Step 4 — NFCI asymmetry  (Section 12.7.2)\n\n")
cat("NFCI coefficient across quantiles:\n")

nfci_coef_vec <- sapply(QUANTILES, function(q)
  unname(coef(qr_fits[[as.character(q)]])[["nfci"]]))
nfci_se_vec   <- sapply(QUANTILES, function(q)
  unname(summary(qr_fits[[as.character(q)]], se = "nid")$coefficients["nfci", "Std. Error"]))

for (j in seq_along(QUANTILES)) {
  q    <- QUANTILES[j]
  coef <- nfci_coef_vec[j]
  se   <- nfci_se_vec[j]
  tval <- coef / se
  cat(sprintf("  tau = %.2f:  %7.3f  (SE %.3f)  %s\n", q, coef, se, stars(tval)))
}

b05   <- nfci_coef_vec[QUANTILES == 0.05]
b50   <- nfci_coef_vec[QUANTILES == 0.50]
b90   <- nfci_coef_vec[QUANTILES == 0.90]
ratio <- abs(b05) / abs(b50)

cat(sprintf("\n  beta_f(0.05) = %.3f\n", b05))
cat(sprintf("  beta_f(0.50) = %.3f\n", b50))
cat(sprintf("  beta_f(0.90) = %.3f\n", b90))
cat(sprintf("  Asymmetry    = %.2fx\n", ratio))

# =============================================================================
# FITTED QUANTILES
# =============================================================================
for (q in QUANTILES) {
  col_name    <- sprintf("gar_%02d", round(q * 100))
  df[[col_name]] <- predict(qr_fits[[as.character(q)]], newdata = df)
}

# =============================================================================
# STEP 5 — COVERAGE EVALUATION  (Table 12.4)
# =============================================================================
cat(strrep("-", 70), "\n")
cat("Step 5 — Coverage evaluation  (Table 12.4)\n\n")
cat(sprintf("%-8s %10s %12s %8s   %s\n", "tau", "Nominal%", "Empirical%", "|Diff|", "OK?"))
cat(strrep("-", 48), "\n")

for (q in QUANTILES) {
  col_name <- sprintf("gar_%02d", round(q * 100))
  emp      <- mean(df$gdp_forward < df[[col_name]]) * 100
  diff_val <- abs(emp - q * 100)
  ok       <- if (diff_val < 2.0) "V" else "!"
  cat(sprintf("%-8.2f %10.1f %12.1f %8.1f   %s\n", q, q * 100, emp, diff_val, ok))
}

# =============================================================================
# STEP 6 — CURRENT RISK ASSESSMENT  (Section 12.7.5)
# =============================================================================
cat(strrep("-", 70), "\n")
cat("Step 6 — Current risk assessment  (Section 12.7.5)\n\n")

last      <- tail(df, 1)
q_label   <- sprintf("%dQ%d", year(last$date), quarter(last$date))
gar5_cur  <- last$gar_05
med_cur   <- last$gar_50
nfci_cur  <- last$nfci

# P(GDP < 0): interpolate CDF using Chernozhukov rearrangement
gar_vals  <- as.numeric(last[, sprintf("gar_%02d", round(QUANTILES * 100))])
sort_idx  <- order(gar_vals)
v_sorted  <- gar_vals[sort_idx]
q_sorted  <- QUANTILES[sort_idx]
if (v_sorted[1] < 0 && v_sorted[length(v_sorted)] > 0) {
  prob_neg <- approx(v_sorted, q_sorted, xout = 0)$y
} else if (v_sorted[length(v_sorted)] <= 0) {
  prob_neg <- 1.0
} else {
  prob_neg <- 0.0
}

cat(sprintf("  Last obs  : %s  (NFCI = %.2f)\n", q_label, nfci_cur))
cat(sprintf("  GaR (5%%) : %.2f%%\n", gar5_cur))
cat(sprintf("  Median    : %.2f%%\n", med_cur))
cat(sprintf("  P(GDP<0)  : %.1f%%\n", prob_neg * 100))

# Stress scenario
X_stress     <- data.frame(gdp_growth = last$gdp_growth,
                            inflation  = last$inflation,
                            nfci       = 2.0)
gar5_stress  <- predict(qr_fits[["0.05"]], newdata = X_stress)
cat(sprintf("\n  Stress (NFCI=2.0, as in 2008): GaR(5%%) = %.2f%%\n", gar5_stress))

# =============================================================================
# STEP 7 — FIGURES
# =============================================================================
cat(strrep("-", 70), "\n")
cat("Step 7 — Generating figures\n")

# NBER recession dates
recessions <- data.frame(
  start = as.Date(c("1973-11-01","1980-01-01","1981-07-01","1990-07-01",
                    "2001-03-01","2007-12-01","2020-02-01")),
  end   = as.Date(c("1975-03-01","1980-07-01","1982-11-01","1991-03-01",
                    "2001-11-01","2009-06-01","2020-04-01"))
)

rec_layer <- function() {
  geom_rect(data = recessions,
            aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf),
            fill = "grey70", alpha = 0.2, inherit.aes = FALSE)
}

# ── Figure 12.1: Coefficient asymmetry ────────────────────────────────────
coef_df <- do.call(rbind, lapply(QUANTILES, function(q) {
  fit <- qr_fits[[as.character(q)]]
  s   <- summary(fit, se = "nid")$coefficients
  data.frame(
    tau      = q,
    variable = rownames(s),
    coef     = s[, "Value"],
    se       = s[, "Std. Error"]
  )
})) |>
  filter(variable != "(Intercept)") |>
  mutate(
    variable = recode(variable,
      "gdp_growth" = "Current GDP Growth",
      "inflation"  = "Inflation",
      "nfci"       = "NFCI (Financial Conditions)"
    ),
    ci_lo = coef - 1.96 * se,
    ci_hi = coef + 1.96 * se,
    fill_col = ifelse(coef < 0, "#C00000", "#2E75B6")
  )

p1 <- ggplot(coef_df, aes(x = factor(tau), y = coef)) +
  geom_col(aes(fill = fill_col), alpha = 0.8) +
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.25, colour = "black") +
  geom_hline(yintercept = 0, colour = "black", linewidth = 0.5) +
  scale_fill_identity() +
  facet_wrap(~variable, scales = "free_y", ncol = 2) +
  labs(x = "Quantile (tau)", y = "Coefficient",
       title = "Figure 12.1: Quantile Regression Coefficients",
       subtitle = "Asymmetric Effects on Downside vs Upside Risks") +
  theme_bw(base_size = 11) +
  theme(strip.background = element_rect(fill = "#2E75B6"),
        strip.text = element_text(colour = "white", face = "bold"),
        plot.title = element_text(face = "bold"))

ggsave(file.path(OUTPUT_DIR, "fig1_gar_coefficients.pdf"), p1,
       width = 11, height = 7, device = "pdf")
cat("  Figure 12.1 - coefficient asymmetry\n")

# ── Figure 12.2: Fan chart ─────────────────────────────────────────────────
p2 <- ggplot(df, aes(x = date)) +
  rec_layer() +
  geom_ribbon(aes(ymin = gar_05, ymax = gar_95, fill = "5-95%"),  alpha = 0.12) +
  geom_ribbon(aes(ymin = gar_10, ymax = gar_90, fill = "10-90%"), alpha = 0.22) +
  geom_ribbon(aes(ymin = gar_25, ymax = gar_75, fill = "25-75%"), alpha = 0.38) +
  geom_line(aes(y = gar_50,      colour = "Median"),   linewidth = 1.5) +
  geom_line(aes(y = gdp_forward, colour = "Actual"),   linewidth = 0.7, alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red", alpha = 0.7) +
  scale_fill_manual(values   = c("5-95%"  = "#2E75B6",
                                 "10-90%" = "#2E75B6",
                                 "25-75%" = "#2E75B6"),
                    name = "Interval") +
  scale_colour_manual(values = c("Median" = "#2E75B6", "Actual" = "black"),
                      name = "") +
  coord_cartesian(ylim = c(-10, 12)) +
  labs(x = "Date", y = "GDP Growth (%, YoY)",
       title = sprintf("Figure 12.2: Growth-at-Risk Fan Chart  (%d-Quarter-Ahead)", H),
       subtitle = "1971Q1-2024Q3") +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold"))

ggsave(file.path(OUTPUT_DIR, "fig2_gar_fanchart.pdf"), p2,
       width = 12, height = 6, device = "pdf")
cat("  Figure 12.2 - fan chart\n")

# ── Figure 12.3: Three panels ──────────────────────────────────────────────
pa <- ggplot(df, aes(x = date)) +
  rec_layer() +
  geom_ribbon(data = filter(df, nfci > 0),
              aes(ymin = 0, ymax = nfci), fill = "#C00000", alpha = 0.35) +
  geom_ribbon(data = filter(df, nfci < 0),
              aes(ymin = nfci, ymax = 0), fill = "#70AD47", alpha = 0.25) +
  geom_line(aes(y = nfci), colour = "#2E75B6", linewidth = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  labs(x = NULL, y = "NFCI",
       title = "Panel A: Financial Conditions (NFCI > 0 = Tight)") +
  theme_bw(base_size = 10)

pb <- ggplot(df, aes(x = date)) +
  rec_layer() +
  geom_line(aes(y = inflation), colour = "#C00000", linewidth = 1.2) +
  geom_hline(yintercept = 2, linetype = "dashed", colour = "#70AD47") +
  labs(x = NULL, y = "Inflation (%, YoY)",
       title = "Panel B: Inflation (GDP Deflator)") +
  theme_bw(base_size = 10)

pc <- ggplot(df, aes(x = date)) +
  rec_layer() +
  geom_ribbon(data = filter(df, gar_05 < 0),
              aes(ymin = gar_05, ymax = 0), fill = "#C00000", alpha = 0.25) +
  geom_line(aes(y = gar_05,      colour = "GaR 5th"),  linewidth = 1.5) +
  geom_line(aes(y = gar_50,      colour = "Median"),   linewidth = 1.0, alpha = 0.8) +
  geom_line(aes(y = gdp_forward, colour = "Actual"),   linewidth = 0.6, alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  scale_colour_manual(values = c("GaR 5th" = "#C00000",
                                 "Median"  = "#2E75B6",
                                 "Actual"  = "black"), name = "") +
  labs(x = "Date", y = "GDP Growth (%)",
       title = "Panel C: Growth-at-Risk (5th Pct) vs Median") +
  theme_bw(base_size = 10) +
  theme(legend.position = "bottom")

p3 <- pa / pb / pc +
  plot_annotation(title    = "Figure 12.3: Financial Conditions and Growth-at-Risk",
                  theme    = theme(plot.title = element_text(face = "bold", size = 12)))

ggsave(file.path(OUTPUT_DIR, "fig3_gar_panels.pdf"), p3,
       width = 12, height = 9, device = "pdf")
cat("  Figure 12.3 - three panels\n")

# ── Figure 12.4: Predictive CDF  (Chernozhukov rearrangement) ─────────────
fine_q  <- seq(0.01, 0.99, length.out = 200)
kv_raw  <- as.numeric(last[, sprintf("gar_%02d", round(QUANTILES * 100))])
sort_i  <- order(kv_raw)
kv      <- kv_raw[sort_i]
kq      <- QUANTILES[sort_i]
preds   <- approx(kq, kv, xout = fine_q)$y
gar5_pt <- approx(fine_q, preds, xout = 0.05)$y

cdf_df  <- data.frame(gdp_growth = preds, cum_prob = fine_q)

p4 <- ggplot(cdf_df, aes(x = gdp_growth, y = cum_prob)) +
  geom_ribbon(aes(xmin = min(preds) - 1, xmax = gdp_growth),
              fill = "#2E75B6", alpha = 0.15) +
  geom_line(colour = "#2E75B6", linewidth = 2) +
  geom_vline(xintercept = 0,    linetype = "dashed", colour = "#C00000",
             linewidth = 1, alpha = 0.9) +
  geom_hline(yintercept = 0.05, linetype = "dotted", colour = "orange",  linewidth = 1) +
  geom_hline(yintercept = 0.50, linetype = "dotted", colour = "#70AD47", linewidth = 1) +
  geom_point(aes(x = gar5_pt, y = 0.05), colour = "#C00000", size = 3) +
  annotate("text", x = gar5_pt - 0.3, y = 0.12,
           label = sprintf("GaR(5%%) = %.2f%%", gar5_pt),
           hjust = 1, colour = "#C00000", size = 3.5) +
  annotate("text", x = min(preds), y = 0.92,
           label = sprintf("GaR (5%%): %.2f%%\nP(GDP<0): %.1f%%\nNFCI: %.2f",
                           gar5_pt, prob_neg * 100, nfci_cur),
           hjust = 0, vjust = 1, size = 3.5,
           colour = "black",
           label.padding = unit(0.5, "lines")) +
  labs(x = "GDP Growth (%, YoY)", y = "Cumulative Probability",
       title = sprintf("Figure 12.4: Predictive CDF  (%dQ Ahead)  |  %s  |  NFCI = %.2f",
                       H, q_label, nfci_cur),
       subtitle = "Chernozhukov rearrangement applied (Section 12.3.2)") +
  coord_cartesian(xlim = c(min(preds) - 1, max(preds) + 1), ylim = c(0, 1)) +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(OUTPUT_DIR, "fig4_gar_cdf.pdf"), p4,
       width = 10, height = 6, device = "pdf")
cat("  Figure 12.4 - predictive CDF\n")

# ── Figure 5: NFCI vs GaR scatter ─────────────────────────────────────────
lm_fit <- lm(gar_05 ~ nfci, data = df)
slope  <- round(coef(lm_fit)["nfci"], 2)

p5 <- ggplot(df, aes(x = nfci, y = gar_05, colour = year(date))) +
  geom_point(alpha = 0.6, size = 1.8) +
  geom_smooth(method = "lm", se = FALSE, colour = "#C00000", linewidth = 1.5,
              formula = y ~ x) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
  scale_colour_viridis_c(name = "Year") +
  annotate("text", x = max(df$nfci) * 0.8, y = max(df$gar_05) * 0.9,
           label = sprintf("Slope = %.2f", slope), size = 4) +
  labs(x = "NFCI", y = "GaR 5th Percentile (%)",
       title = "NFCI vs GaR: Tighter Conditions -> Lower Downside Threshold") +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(OUTPUT_DIR, "fig5_gar_scatter.pdf"), p5,
       width = 10, height = 6, device = "pdf")
cat("  Figure 5 - NFCI vs GaR scatter\n")

# ── Figure 6: Early warning ────────────────────────────────────────────────
p6 <- ggplot(df, aes(x = date)) +
  rec_layer() +
  geom_ribbon(aes(ymin = gar_05, ymax = gar_10), fill = "#ED7D31", alpha = 0.25) +
  geom_line(aes(y = gar_05, colour = "GaR 5th"),  linewidth = 1.5) +
  geom_line(aes(y = gar_10, colour = "GaR 10th"), linewidth = 1.2) +
  geom_hline(yintercept =  0, colour = "black", linewidth = 0.5) +
  geom_hline(yintercept = -2, linetype = "dotted", colour = "grey50") +
  scale_colour_manual(values = c("GaR 5th" = "#C00000", "GaR 10th" = "#ED7D31"),
                      name = "") +
  labs(x = "Date", y = "GDP Growth (%)",
       title = "Growth-at-Risk as Early Warning  (Grey = NBER recessions)") +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold"))

ggsave(file.path(OUTPUT_DIR, "fig6_gar_warning.pdf"), p6,
       width = 12, height = 5, device = "pdf")
cat("  Figure 6 - early warning\n")

cat(sprintf("\nAll figures saved to: %s\n", OUTPUT_DIR))

# =============================================================================
# SUMMARY
# =============================================================================
cat(strrep("=", 70), "\n")
cat(sprintf("
Growth-at-Risk with NFCI  |  Chapter 12, Section 12.7
======================================================
Sample       : 1971Q1-2024Q3   N = %d
Horizon      : h = %d quarters

Key finding - NFCI asymmetry (Table 12.3):
  beta_f(0.05) = %.3f
  beta_f(0.50) = %.3f
  beta_f(0.90) = %.3f
  Asymmetry    = %.2fx

Current assessment (Section 12.7.5 / %s):
  NFCI = %.2f  |  GaR(5%%) = %.2f%%  |  Median = %.2f%%
  P(GDP<0) = %.1f%%

Output: %s
  fig1-fig6 individual PDFs
",
nrow(df), H,
b05, b50, b90, ratio,
q_label, nfci_cur, gar5_cur, med_cur, prob_neg * 100,
OUTPUT_DIR))
