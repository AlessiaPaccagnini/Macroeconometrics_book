# =============================================================================
# figure_10_3.R
# Replication code for Figure 10.3
# US Macroeconomic Data with Regime Indicators (1960-2019)
#
# Macroeconometrics — Paccagnini (2026), Chapter 10
#
# Data file required (same folder): macro_data_1960_2019.csv
# Install packages once:
#   install.packages(c("ggplot2","dplyr","tidyr","zoo","patchwork","scales"))
# =============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(zoo)
library(patchwork)
library(scales)

# ── 0. Load data ──────────────────────────────────────────────────────────────
df <- read.csv("macro_data_1960_2019.csv", stringsAsFactors = FALSE)
df$date <- as.Date(df$date)
df$gdp_growth <- as.numeric(df$gdp_growth)
df$inflation  <- as.numeric(df$inflation)
T <- nrow(df)

# ── 1. Regime indicators ──────────────────────────────────────────────────────
# Rolling 8-quarter SD of GDP growth (zoo::rollapply)
df$roll_std <- rollapply(df$gdp_growth, width = 8, FUN = sd,
                         fill = NA, align = "right", partial = TRUE)
p75         <- quantile(df$roll_std, 0.75, na.rm = TRUE)
df$high_vol <- as.integer(df$roll_std > p75)
df$high_vol[is.na(df$high_vol)] <- 0L
df$stress   <- pmax(df$nber_rec, df$high_vol)

gm_date <- as.Date("1984-01-01")
ffr_med <- median(df$ffr, na.rm = TRUE)

# ── 2. Colours ────────────────────────────────────────────────────────────────
C_REC  <- "#C0392B"; C_HVOL <- "#2471A3"; C_STR <- "#7D3C98"
C_GM   <- "#1E8449"; C_GDP  <- "#1A1A2E"; C_INF <- "#B7600A"; C_FFR <- "#154360"

# ── 3. Build recession shading data.frame ─────────────────────────────────────
make_shading <- function(dates, indicator) {
  out <- data.frame(xmin=as.Date(character()), xmax=as.Date(character()))
  in_b <- FALSE
  for (i in seq_along(dates)) {
    if (indicator[i] == 1 && !in_b) { s <- dates[i]; in_b <- TRUE }
    else if (indicator[i] == 0 && in_b) {
      out <- rbind(out, data.frame(xmin=s, xmax=dates[i])); in_b <- FALSE
    }
  }
  if (in_b) out <- rbind(out, data.frame(xmin=s, xmax=tail(dates,1)))
  out
}

rec_shading  <- make_shading(df$date, df$nber_rec)
# Add ymin/ymax for each panel
make_rect <- function(shd, ymin, ymax) {
  if (nrow(shd)==0) return(NULL)
  shd$ymin <- ymin; shd$ymax <- ymax; shd
}

# ── 4. Helper: common theme ───────────────────────────────────────────────────
my_theme <- function() {
  theme_minimal(base_size = 9) +
    theme(panel.grid.minor  = element_blank(),
          panel.grid.major  = element_line(colour="grey92"),
          axis.title.x      = element_blank(),
          axis.text.x       = element_blank(),
          axis.ticks.x      = element_blank(),
          plot.margin       = margin(2,8,2,8))
}

# ── 5. Panel 1: GDP Growth ────────────────────────────────────────────────────
p1 <- ggplot(df, aes(x=date)) +
  geom_rect(data=make_rect(rec_shading,-11,11),
            aes(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax),
            fill=C_REC,alpha=0.18,inherit.aes=FALSE) +
  geom_hline(yintercept=0, linetype="dashed", colour="black", linewidth=0.4, alpha=0.4) +
  geom_line(aes(y=gdp_growth), colour=C_GDP, linewidth=0.8) +
  geom_vline(xintercept=gm_date, colour=C_GM, linetype="dashed", linewidth=1.2, alpha=0.85) +
  annotate("rect",xmin=as.Date("1959-01-01"),xmax=as.Date("1962-01-01"),
           ymin=9,ymax=11,fill=C_REC,alpha=0.18) +
  annotate("text",x=as.Date("1963-01-01"),y=10,label="NBER Recession",
           hjust=0,size=2.8,colour=C_REC) +
  scale_y_continuous(name="GDP Growth (%)") +
  coord_cartesian(ylim=c(-11,11)) +
  labs(title="US Macroeconomic Data with Regime Indicators (1960\u20132019)") +
  my_theme() +
  theme(plot.title=element_text(size=11,face="bold",hjust=0))

# ── 6. Panel 2: Inflation ─────────────────────────────────────────────────────
p2 <- ggplot(df, aes(x=date)) +
  geom_rect(data=make_rect(rec_shading,-0.5,10),
            aes(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax),
            fill=C_REC,alpha=0.18,inherit.aes=FALSE) +
  geom_line(aes(y=inflation), colour=C_INF, linewidth=0.8) +
  geom_vline(xintercept=gm_date, colour=C_GM, linetype="dashed", linewidth=1.2, alpha=0.85) +
  scale_y_continuous(name="Inflation (%)") +
  coord_cartesian(ylim=c(-0.5,10)) +
  my_theme()

# ── 7. Panel 3: Fed Funds Rate ────────────────────────────────────────────────
p3 <- ggplot(df, aes(x=date)) +
  geom_rect(data=make_rect(rec_shading,-0.5,20.5),
            aes(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax),
            fill=C_REC,alpha=0.18,inherit.aes=FALSE) +
  geom_line(aes(y=ffr), colour=C_FFR, linewidth=0.8) +
  geom_hline(yintercept=ffr_med, colour="grey50", linetype="dotted",
             linewidth=0.9, alpha=0.85) +
  geom_vline(xintercept=gm_date, colour=C_GM, linetype="dashed",
             linewidth=1.2, alpha=0.85) +
  annotate("text",x=as.Date("1985-01-01"),y=19,
           label=paste0("Median (",round(ffr_med,1),"%)"),
           hjust=0,size=2.5,colour="grey40") +
  annotate("text",x=as.Date("1984-07-01"),y=0.8,
           label="Great Moderation",hjust=0,size=2.5,colour=C_GM,angle=90) +
  scale_y_continuous(name="Interest Rate (%)") +
  coord_cartesian(ylim=c(-0.5,20.5)) +
  my_theme()

# ── 8. Panel 4: Regime indicators ────────────────────────────────────────────
make_regime_df <- function(dates, ind, label, y) {
  out <- data.frame(xmin=as.Date(character()), xmax=as.Date(character()),
                    label=character(), y=numeric())
  in_b <- FALSE
  for (i in seq_along(dates)) {
    if (ind[i]==1 && !in_b) { s <- dates[i]; in_b <- TRUE }
    else if (ind[i]==0 && in_b) {
      out <- rbind(out, data.frame(xmin=s,xmax=dates[i],label=label,y=y)); in_b<-FALSE
    }
  }
  if (in_b) out <- rbind(out, data.frame(xmin=s,xmax=tail(dates,1),label=label,y=y))
  out
}

H <- 0.18
reg_df <- rbind(
  make_regime_df(df$date, df$nber_rec,  "NBER Recession", 0.67),
  make_regime_df(df$date, df$high_vol,  "High Uncertainty", 0.37),
  make_regime_df(df$date, df$stress,    "Combined Stress",  0.07)
)
reg_df$label <- factor(reg_df$label,
  levels=c("NBER Recession","High Uncertainty","Combined Stress"))

colors4 <- c("NBER Recession"=C_REC, "High Uncertainty"=C_HVOL,
             "Combined Stress"=C_STR)

p4 <- ggplot() +
  geom_rect(data=reg_df,
            aes(xmin=xmin,xmax=xmax,ymin=y-H/2,ymax=y+H/2,fill=label),
            alpha=0.75,inherit.aes=FALSE) +
  geom_vline(xintercept=gm_date,colour=C_GM,linetype="dashed",
             linewidth=1.2,alpha=0.85) +
  scale_fill_manual(values=colors4, name="Regime") +
  scale_y_continuous(breaks=c(0.07,0.37,0.67),
                     labels=c("Stress","High Vol","Recession")) +
  scale_x_date(date_breaks="10 years", date_labels="%Y") +
  labs(x="Date") +
  coord_cartesian(ylim=c(-0.05,0.85)) +
  theme_minimal(base_size=9) +
  theme(panel.grid.minor=element_blank(),
        panel.grid.major.y=element_blank(),
        panel.grid.major.x=element_line(colour="grey92"),
        legend.position="bottom",
        legend.title=element_blank(),
        legend.text=element_text(size=7.5),
        plot.margin=margin(2,8,2,8),
        axis.title.y=element_blank())

# ── 9. Combine with patchwork ─────────────────────────────────────────────────
final <- p1 / p2 / p3 / p4 +
  plot_layout(heights=c(1.8,1.8,1.8,1.1))

ggsave("Figure_10_3.pdf", plot=final, width=13, height=11, units="in")
ggsave("Figure_10_3.png", plot=final, width=13, height=11, units="in", dpi=300)
cat("Saved Figure_10_3.pdf and Figure_10_3.png\n")
