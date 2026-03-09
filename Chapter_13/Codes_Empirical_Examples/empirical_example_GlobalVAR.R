# =============================================================================
# GVAR: US MONETARY POLICY SPILLOVERS
# Replication of Section 13.7.2 — Macroeconometrics Textbook
# Author:   Alessia Paccagnini
# Textbook: Macroeconometrics (De Gruyter)
# =============================================================================
#
# Book specification (eq. 13.74):
#   y_it = alpha_i + Phi_i y_{i,t-1} + Lam0_i y*_it
#                  + Lam1_i y*_{i,t-1} + eps_it
#   USA: closed VAR(2) + oil (dominant unit)
#   y_it = (Dy_it, pi_it, rs_it)'
#
# Steps:
#   1. Simulate GVAR data (T=162, N=8)
#   2. Construct trade-weighted foreign variables
#   3. Estimate VARX*(1,1) per country
#   4. Weak exogeneity tests  (Table 13.7)
#   5. Stack global VAR  (eq. 13.71)
#   6. Stability check + Figure 13.2
#   7. GIRF to US rate shock + Figure 13.3
#   8. FEVD + spillover network + Figure 13.4
# =============================================================================

set.seed(42)

# ── Dimensions ───────────────────────────────────────────────────────────────
COUNTRIES <- c("USA","Euro_Area","UK","Japan","China","Canada","Korea","Brazil")
N  <- length(COUNTRIES)
K  <- 3L    # gdp, inflation, interest rate
TT <- 162L
CIDX <- setNames(seq_along(COUNTRIES), COUNTRIES)  # 1-based

# ── Trade weight matrix ───────────────────────────────────────────────────────
W_raw <- matrix(c(
  0.00, 0.18, 0.10, 0.12, 0.14, 0.22, 0.13, 0.11,
  0.26, 0.00, 0.18, 0.10, 0.14, 0.08, 0.12, 0.12,
  0.22, 0.36, 0.00, 0.08, 0.10, 0.07, 0.08, 0.09,
  0.28, 0.14, 0.07, 0.00, 0.21, 0.08, 0.13, 0.09,
  0.20, 0.18, 0.08, 0.18, 0.00, 0.08, 0.18, 0.10,
  0.55, 0.09, 0.07, 0.07, 0.08, 0.00, 0.07, 0.07,
  0.25, 0.14, 0.08, 0.20, 0.18, 0.07, 0.00, 0.08,
  0.22, 0.20, 0.10, 0.08, 0.14, 0.08, 0.08, 0.00
), N, N, byrow=TRUE)
W <- W_raw / rowSums(W_raw)

# ── DGP parameters ───────────────────────────────────────────────────────────
A1_US <- matrix(c( 0.50,  0.00, -0.28,
                   0.09,  0.55,  0.06,
                   0.22,  0.12,  0.80), 3, 3, byrow=TRUE)
A2_US <- matrix(c( 0.09,  0.00, -0.12,
                   0.02,  0.07,  0.00,
                   0.06,  0.04,  0.08), 3, 3, byrow=TRUE)
GAMMA_OIL_US <- c(-0.06, 0.08, 0.02)

# Phi, Lambda0, Lambda1 for non-US countries
PHI  <- list(
  Euro_Area = diag(c(0.50, 0.62, 0.82)),
  UK        = diag(c(0.48, 0.58, 0.80)),
  Japan     = diag(c(0.44, 0.52, 0.85)),
  China     = diag(c(0.60, 0.65, 0.84)),
  Canada    = diag(c(0.52, 0.58, 0.81)),
  Korea     = diag(c(0.46, 0.54, 0.78)),
  Brazil    = diag(c(0.40, 0.68, 0.74))
)
LAM0 <- list(
  Euro_Area = matrix(c(-0.10, 0.04,-0.08, 0.04, 0.08, 0.04, 0.06, 0.04, 0.14),3,3,byrow=TRUE),
  UK        = matrix(c(-0.12, 0.04,-0.10, 0.04, 0.09, 0.04, 0.07, 0.04, 0.16),3,3,byrow=TRUE),
  Japan     = matrix(c(-0.05, 0.02,-0.04, 0.02, 0.04, 0.02, 0.04, 0.02, 0.10),3,3,byrow=TRUE),
  China     = matrix(c(-0.04, 0.02,-0.03, 0.02, 0.05, 0.02, 0.03, 0.02, 0.08),3,3,byrow=TRUE),
  Canada    = matrix(c(-0.18, 0.06,-0.14, 0.06, 0.12, 0.06, 0.09, 0.06, 0.20),3,3,byrow=TRUE),
  Korea     = matrix(c(-0.07, 0.03,-0.06, 0.03, 0.07, 0.03, 0.05, 0.03, 0.12),3,3,byrow=TRUE),
  Brazil    = matrix(c(-0.06, 0.04,-0.05, 0.04, 0.09, 0.03, 0.05, 0.04, 0.10),3,3,byrow=TRUE)
)
LAM1 <- lapply(LAM0, function(x) 0.35 * x)

SIG <- list(
  USA=c(0.010,0.005,0.006), Euro_Area=c(0.009,0.004,0.005),
  UK=c(0.010,0.005,0.006),  Japan=c(0.009,0.003,0.004),
  China=c(0.013,0.005,0.006),Canada=c(0.010,0.004,0.005),
  Korea=c(0.012,0.005,0.008),Brazil=c(0.018,0.008,0.012)
)
ALPHA <- list(
  USA=c(0.006,0.004,0.010), Euro_Area=c(0.004,0.003,0.008),
  UK=c(0.005,0.004,0.009),  Japan=c(0.003,0.001,0.005),
  China=c(0.013,0.004,0.008),Canada=c(0.005,0.004,0.010),
  Korea=c(0.008,0.004,0.060),Brazil=c(0.004,0.006,0.055)
)

# =============================================================================
# STEP 1 — SIMULATE DATA
# =============================================================================
simulate_gvar <- function(seed = 42) {
  set.seed(seed)
  oil <- numeric(TT)
  for (t in 2:TT) oil[t] <- 0.75*oil[t-1] + rnorm(1,0,0.045)
  
  Y <- lapply(setNames(COUNTRIES,COUNTRIES), function(c) matrix(0, TT, K,
             dimnames=list(NULL, c("gdp","inf","rs"))))
  for (c in COUNTRIES) {
    Y[[c]][1,] <- ALPHA[[c]] + SIG[[c]] * rnorm(K) * 0.5
    Y[[c]][2,] <- ALPHA[[c]] + SIG[[c]] * rnorm(K) * 0.5
  }
  
  for (t in 3:TT) {
    # USA — dominant unit
    eps_us <- SIG[["USA"]] * rnorm(K)
    Y[["USA"]][t,] <- ALPHA[["USA"]] +
                      A1_US %*% Y[["USA"]][t-1,] +
                      A2_US %*% Y[["USA"]][t-2,] +
                      GAMMA_OIL_US * oil[t] + eps_us
    
    # Non-US
    for (i in 2:N) {
      c   <- COUNTRIES[i]
      y_star   <- numeric(K)
      y_star_l <- numeric(K)
      for (j in seq_along(COUNTRIES)) {
        if (j != i) {
          c2  <- COUNTRIES[j]
          wij <- W[i, j]
          if (j == 1) {  # USA: contemporaneous
            y_star   <- y_star   + wij * Y[[c2]][t,]
          } else {
            y_star   <- y_star   + wij * Y[[c2]][t-1,]
          }
          y_star_l <- y_star_l + wij * Y[[c2]][t-1,]
        }
      }
      eps_c <- SIG[[c]] * rnorm(K)
      Y[[c]][t,] <- ALPHA[[c]] +
                    PHI[[c]]  %*% Y[[c]][t-1,] +
                    LAM0[[c]] %*% y_star +
                    LAM1[[c]] %*% y_star_l + eps_c
    }
  }
  Y
}

# =============================================================================
# STEP 2 — FOREIGN VARIABLES
# =============================================================================
make_foreign <- function(Y_data) {
  lapply(setNames(COUNTRIES, COUNTRIES), function(c) {
    i  <- CIDX[c]
    Ys <- matrix(0, TT, K)
    for (j in seq_along(COUNTRIES)) {
      if (j != i) Ys <- Ys + W[i,j] * Y_data[[COUNTRIES[j]]]
    }
    Ys
  })
}

# =============================================================================
# STEP 3 — ESTIMATE VARX* COUNTRY MODELS
# =============================================================================
estimate_varx <- function(Y, Y_star, p = 1, q = 1) {
  Tc <- nrow(Y);  k <- ncol(Y);  ks <- ncol(Y_star)
  ml <- max(p, q);  Te <- Tc - ml
  X  <- cbind(1, Y[ml:(Tc-1),], Y_star[(ml+1):Tc,], Y_star[ml:(Tc-1),])
  Yd <- Y[(ml+1):Tc,]
  B  <- solve(crossprod(X), crossprod(X, Yd))
  U  <- Yd - X %*% B
  Sig <- crossprod(U) / Te
  ss_res <- colSums(U^2)
  ss_tot <- colSums(sweep(Yd,2,colMeans(Yd))^2) + 1e-12
  r2 <- pmax(0, 1 - ss_res/ss_tot)
  n1<-1; n2<-n1+k; n3<-n2+ks; n4<-n3+ks
  list(B=B, Phi=t(B[(n1+1):n2,,drop=FALSE]),
       Lam0=t(B[(n2+1):n3,,drop=FALSE]),
       Lam1=t(B[(n3+1):n4,,drop=FALSE]),
       Sigma=Sig, U=U, r2=r2, r2_avg=mean(r2))
}

# =============================================================================
# STEP 4 — WEAK EXOGENEITY TESTS
# =============================================================================
test_weak_exog <- function(Y_data, foreign, models) {
  crit <- 3.07
  cat("\n", strrep("=",52), "\n")
  cat("Table 13.7  Weak Exogeneity Tests: F-Statistics\n")
  cat(strrep("=",52), "\n")
  cat(sprintf("%-14s %8s %8s %8s\n","Country","Dy*","pi*","rs*"))
  cat(strrep("-",52), "\n")
  
  for (i in 2:N) {
    c     <- COUNTRIES[i]
    U_hat <- models[[c]]$U
    Ys    <- foreign[[c]]
    Te    <- nrow(U_hat)
    # Align dYs and ec
    Ys_sub <- Ys[(nrow(Ys)-Te):nrow(Ys),]
    dYs    <- diff(Ys_sub, 1)      # length Te
    dYs_   <- dYs[-1,, drop=FALSE] # length Te-1
    ec     <- U_hat[-nrow(U_hat),, drop=FALSE]  # length Te-1
    n_obs  <- nrow(dYs_)
    
    f_stats <- numeric(K)
    for (v in seq_len(K)) {
      y     <- dYs_[,v]
      lag_y <- c(0, y[-length(y)])
      X_u   <- cbind(1, lag_y, ec[seq_len(n_obs), 1])
      B_u   <- solve(crossprod(X_u), crossprod(X_u, y))
      ss_u  <- sum((y - X_u %*% B_u)^2)
      X_r   <- X_u[,-ncol(X_u)]
      B_r   <- solve(crossprod(X_r), crossprod(X_r, y))
      ss_r  <- sum((y - X_r %*% B_r)^2)
      df2   <- max(n_obs - ncol(X_u), 1)
      f_stats[v] <- ((ss_r-ss_u)/1) / (ss_u/df2)
    }
    flags <- ifelse(f_stats > crit, "*", " ")
    cat(sprintf("%-14s %7.2f%s %7.2f%s %7.2f%s\n",
                gsub("_"," ",c),
                f_stats[1],flags[1], f_stats[2],flags[2], f_stats[3],flags[3]))
  }
  cat(strrep("-",52), "\n")
  cat(sprintf("5%% critical value ~%.2f  (* exceeds)\n", crit))
  cat(strrep("=",52), "\n")
}

# =============================================================================
# STEP 5 — STACK GLOBAL VAR
# =============================================================================
stack_global <- function(models) {
  Kg <- N * K
  G0 <- diag(Kg)
  G1 <- matrix(0, Kg, Kg)
  
  for (i in seq_len(N)) {
    c    <- COUNTRIES[i]
    rs   <- (i-1)*K + 1;  re <- i*K
    Phi  <- models[[c]]$Phi
    G1[rs:re, rs:re] <- Phi
    if (c == "USA") next       # dominant unit: no foreign variables
    Lam0 <- models[[c]]$Lam0
    Lam1 <- models[[c]]$Lam1
    for (j in seq_len(N)) {
      if (j != i) {
        cs <- (j-1)*K+1;  ce <- j*K
        wij <- W[i,j]
        G0[rs:re, cs:ce] <- G0[rs:re, cs:ce] - Lam0 * wij
        G1[rs:re, cs:ce] <- G1[rs:re, cs:ce] + Lam1 * wij
      }
    }
  }
  G0_inv  <- solve(G0)
  F_mat   <- G0_inv %*% G1
  Sig_blk <- matrix(0, Kg, Kg)
  for (i in seq_len(N)) {
    rs <- (i-1)*K+1; re <- i*K
    Sig_blk[rs:re,rs:re] <- models[[COUNTRIES[i]]]$Sigma
  }
  Sigma_e <- G0_inv %*% Sig_blk %*% t(G0_inv)
  list(G0=G0, G1=G1, G0_inv=G0_inv, F=F_mat, Sigma_e=Sigma_e)
}

# =============================================================================
# STEP 6 — GIRF
# =============================================================================
compute_girf <- function(F_mat, Sigma_e, shock_cid, shock_vid, H = 20) {
  Kg  <- nrow(F_mat)
  j   <- (shock_cid-1)*K + shock_vid
  e_j <- numeric(Kg); e_j[j] <- 1
  b   <- Sigma_e %*% e_j / sqrt(max(Sigma_e[j,j], 1e-12))
  irf_raw <- matrix(0, H, Kg)
  Cs  <- diag(Kg)
  irf_raw[1,] <- as.numeric(Cs %*% b)
  for (h in 2:H) {
    Cs <- Cs %*% F_mat
    irf_raw[h,] <- as.numeric(Cs %*% b)
  }
  setNames(lapply(seq_len(N), function(i) {
    irf_raw[, ((i-1)*K+1):(i*K), drop=FALSE]
  }), COUNTRIES)
}

bootstrap_girf <- function(F_mat, Sigma_e, models, G0_inv,
                            shock_cid, shock_vid, H=20, n_boot=500, seed=10) {
  set.seed(seed)
  Kg    <- nrow(F_mat)
  boots <- lapply(setNames(COUNTRIES,COUNTRIES),
                  function(c) array(0, c(n_boot, H, K)))
  
  for (b in seq_len(n_boot)) {
    Sig_b <- matrix(0, Kg, Kg)
    for (i in seq_len(N)) {
      c   <- COUNTRIES[i]
      U_b <- models[[c]]$U
      idx <- sample.int(nrow(U_b), nrow(U_b), replace=TRUE)
      Ub  <- U_b[idx,]
      rs  <- (i-1)*K+1; re <- i*K
      Sig_b[rs:re,rs:re] <- crossprod(Ub) / nrow(Ub)
    }
    Sigma_eb <- G0_inv %*% Sig_b %*% t(G0_inv)
    F_b      <- F_mat + 0.02*sd(F_mat)*matrix(rnorm(Kg^2), Kg, Kg)
    tryCatch({
      irf_b <- compute_girf(F_b, Sigma_eb, shock_cid, shock_vid, H)
      for (i in seq_len(N)) {
        c <- COUNTRIES[i]
        boots[[c]][b,,] <- irf_b[[c]]
      }
    }, error=function(e) NULL)
  }
  
  irf_lo <- lapply(setNames(COUNTRIES,COUNTRIES), function(c)
                   apply(boots[[c]], c(2,3), quantile, 0.05))
  irf_hi <- lapply(setNames(COUNTRIES,COUNTRIES), function(c)
                   apply(boots[[c]], c(2,3), quantile, 0.95))
  list(lo=irf_lo, hi=irf_hi)
}

# =============================================================================
# STEP 7 — FEVD
# =============================================================================
compute_gfevd_gdp <- function(F_mat, Sigma_e, H_fevd = 8) {
  Kg    <- nrow(F_mat)
  Cs_all <- array(0, c(Kg,Kg,H_fevd))
  Cs_all[,,1] <- diag(Kg)
  for (h in 2:H_fevd) Cs_all[,,h] <- Cs_all[,,h-1] %*% F_mat
  
  fevd <- matrix(0, N, N)
  for (i in seq_len(N)) {
    v   <- (i-1)*K + 1
    e_v <- numeric(Kg); e_v[v] <- 1
    fev_tot <- sum(sapply(seq_len(H_fevd), function(h) {
      Cs <- Cs_all[,,h]
      as.numeric(t(e_v) %*% Cs %*% Sigma_e %*% t(Cs) %*% e_v)
    }))
    fev_tot <- max(fev_tot, 1e-12)
    for (j in seq_len(N)) {
      num <- 0
      for (s in seq_len(K)) {
        js   <- (j-1)*K + s
        e_js <- numeric(Kg); e_js[js] <- 1
        sig_js <- max(Sigma_e[js,js], 1e-12)
        num <- num + sum(sapply(seq_len(H_fevd), function(h) {
          Cs <- Cs_all[,,h]
          (as.numeric(t(e_v) %*% Cs %*% Sigma_e %*% e_js))^2 / sig_js
        }))
      }
      fevd[i,j] <- num / fev_tot
    }
  }
  fevd / rowSums(fevd)
}

# =============================================================================
# FIGURES
# =============================================================================
plot_fig13_2 <- function(F_mat, path=NULL) {
  eigs_F  <- eigen(F_mat, only.values=TRUE)$values
  max_mod <- max(Mod(eigs_F))
  if (!is.null(path)) pdf(path, width=6, height=6)
  theta_c <- seq(0, 2*pi, length.out=300)
  plot(cos(theta_c), sin(theta_c), type="l", lwd=1,
       xlab="Real", ylab="Imaginary", asp=1,
       xlim=c(-1.3,1.3), ylim=c(-1.3,1.3),
       main=sprintf("Eigenvalues of the GVAR Companion Matrix\nMax |lambda| = %.3f",
                    max_mod),
       col="black"); grid(col="grey90")
  points(Re(eigs_F), Im(eigs_F), pch=16, col="#2E75B6", cex=0.9)
  abline(h=0, v=0, col="grey60", lwd=0.4)
  legend("topright", bty="n", cex=0.9,
         legend=c("Unit circle","Eigenvalues"),
         col=c("black","#2E75B6"), lty=c(1,NA), pch=c(NA,16))
  if (!is.null(path)) { dev.off(); cat(sprintf("  Saved -> %s\n", path)) }
  max_mod
}

plot_fig13_3 <- function(irf, irf_lo, irf_hi, H=20, path=NULL) {
  if (!is.null(path)) pdf(path, width=14, height=7)
  par(mfrow=c(2,4), mar=c(3,3,2,1), oma=c(0,0,3,0))
  h_ax <- 0:(H-1)
  col_ <- "#2E75B6"
  for (i in seq_len(N)) {
    c   <- COUNTRIES[i]
    gdp <- irf[[c]][,1]
    lo_ <- irf_lo[[c]][,1]
    hi_ <- irf_hi[[c]][,1]
    ylim_ <- range(c(lo_,hi_,gdp)) + c(-1,1)*0.1*diff(range(c(lo_,hi_,gdp)))
    plot(h_ax, gdp, type="n", xlab="Quarters", ylab="GDP growth (pp)",
         main=gsub("_"," ",c), ylim=ylim_, cex.main=0.9, las=1); grid(col="grey90")
    polygon(c(h_ax,rev(h_ax)), c(lo_,rev(hi_)), col=adjustcolor(col_,0.20), border=NA)
    lines(h_ax, gdp, col=col_, lwd=2.0)
    abline(h=0, lwd=0.6)
    th <- which.min(gdp)
    text(th-1+1, gdp[th], sprintf("%.3f", gdp[th]),
         col="#C00000", cex=0.75, adj=c(0,1))
  }
  mtext(paste("Figure 13.3: GVAR — GDP Response to US Monetary Policy Tightening",
              "\nGIRF to 1 s.d. increase in rs_US (~50 bp)  |  90% bootstrap CI",
              sep=""), outer=TRUE, font=2, cex=0.9)
  if (!is.null(path)) { dev.off(); cat(sprintf("  Saved -> %s\n", path)) }
}

plot_fig13_4 <- function(fevd, path=NULL) {
  short <- c("USA","EUR","GBR","JPN","CHN","CAN","KOR","BRA")
  long_ <- c("United States","Euro Area","United Kingdom","Japan",
             "China","Canada","Korea","Brazil")
  pct   <- round(fevd * 100)
  
  if (!is.null(path)) pdf(path, width=14, height=5.5)
  par(mfrow=c(1,2), mar=c(4,6,3,1), oma=c(0,0,2,0))
  
  # Heatmap (using image)
  image(1:N, 1:N, t(pct[N:1,]),  # flip for conventional row/col display
        col=colorRampPalette(c("white","#084594"))(100),
        xaxt="n", yaxt="n",
        xlab="Shock source (country j)", ylab="",
        main="FEVD (h=8)")
  axis(1, at=1:N, labels=short, las=2, cex.axis=0.8)
  axis(2, at=1:N, labels=rev(long_), las=1, cex.axis=0.7)
  for (i in seq_len(N)) for (j in seq_len(N)) {
    col_ <- if (pct[N+1-i,j] > 55) "white" else "black"
    text(j, i, pct[N+1-i,j], cex=0.7, col=col_)
  }
  
  # Spillover network
  plot(0, 0, type="n", xlim=c(-1.4,1.4), ylim=c(-1.4,1.4),
       asp=1, xlab="", ylab="", main="Spillover Network\n(edges > 3%)", axes=FALSE)
  theta_n <- seq(0, 2*pi, length.out=N+1)[1:N]
  px <- 0.85*cos(theta_n)
  py <- 0.85*sin(theta_n)
  out_conn <- rowSums(fevd) - diag(fevd)
  
  for (i in seq_len(N)) for (j in seq_len(N)) {
    if (i!=j && fevd[i,j]>0.03) {
      lw_ <- 1 + 6*fevd[i,j]
      arrows(px[i], py[i], px[j], py[j], length=0.08,
             col=adjustcolor("#2E75B6",0.55), lwd=lw_)
    }
  }
  for (i in seq_len(N)) {
    r_  <- 0.06 + 0.14*out_conn[i]
    th_ <- seq(0,2*pi,length.out=50)
    polygon(px[i]+r_*cos(th_), py[i]+r_*sin(th_),
            col="#2E75B6", border=NA)
    text(px[i], py[i], short[i], col="white", cex=0.7, font=2)
  }
  
  mtext("Figure 13.4: GVAR — FEVD and Spillover Network (h=8 quarters)",
        outer=TRUE, font=2, cex=0.95)
  if (!is.null(path)) { dev.off(); cat(sprintf("  Saved -> %s\n", path)) }
}

# =============================================================================
# MAIN
# =============================================================================
cat(strrep("=",70), "\n")
cat("GVAR: US MONETARY POLICY SPILLOVERS\n")
cat("Section 13.7.2 | Macroeconometrics | Alessia Paccagnini\n")
cat(strrep("=",70), "\n")

cat("\n[1] Simulating GVAR data (T=162, N=8) ...\n")
Y_data <- simulate_gvar(seed=42)
cat(sprintf("    wCAN,USA = %.2f  (book: 0.55)\n", W[CIDX["Canada"], CIDX["USA"]]))

cat("\n[2] Constructing foreign variables ...\n")
foreign <- make_foreign(Y_data)

estimate_var_usa <- function(Y, p = 2) {
  # Pure VAR(p) for USA — no foreign variables (dominant unit assumption)
  Tc <- nrow(Y);  k <- ncol(Y);  Te <- Tc - p
  X  <- cbind(1, do.call(cbind, lapply(seq_len(p), function(lag)
               Y[(p-lag+1):(Tc-lag), ])))
  Yd <- Y[(p+1):Tc, ]
  B  <- solve(crossprod(X), crossprod(X, Yd))
  U  <- Yd - X %*% B
  Sig <- crossprod(U) / Te
  ss_res <- colSums(U^2)
  ss_tot <- colSums(sweep(Yd, 2, colMeans(Yd))^2) + 1e-12
  r2 <- pmax(0, 1 - ss_res / ss_tot)
  list(B    = B,
       Phi  = t(B[2:(k+1), , drop=FALSE]),   # k x k  (lag-1 block)
       Lam0 = matrix(0, k, k),                # no foreign contemporaneous
       Lam1 = matrix(0, k, k),                # no foreign lagged
       Sigma = Sig, U = U, r2 = r2, r2_avg = mean(r2))
}

cat("\n[3] Estimating VARX*(1,1) country models ...\n")
models <- lapply(setNames(COUNTRIES, COUNTRIES), function(c) {
  Y <- Y_data[[c]]
  m <- if (c == "USA") estimate_var_usa(Y, p=2L)
       else            estimate_varx(Y, foreign[[c]], p=1L, q=1L)
  cat(sprintf("  %-12s: avg R2=%.3f\n", c, m$r2_avg))
  m
})

cat("\n[4] Weak exogeneity tests ...\n")
test_weak_exog(Y_data, foreign, models)

cat("\n[5] Stacking global VAR ...\n")
gvar <- stack_global(models)
Kg   <- N * K
cat(sprintf("    Global system: %dx%d  (%d countries x %d variables)\n", Kg, Kg, N, K))

cat("\n[6] Stability check + Figure 13.2 ...\n")
max_eig <- plot_fig13_2(gvar$F, path="empirical_example_GlobalVAR_eigenvalues.pdf")
cat(sprintf("    Max |lambda| = %.3f  (book: 0.973)\n", max_eig))

cat("\n[7] GIRF to US rate shock + Figure 13.3 ...\n")
H       <- 20L
shock_cid <- CIDX["USA"];  shock_vid <- 3L  # rs = variable 3
irf     <- compute_girf(gvar$F, gvar$Sigma_e, shock_cid, shock_vid, H)
cat("    Bootstrap CIs (500 reps) ...\n")
boot_gi <- bootstrap_girf(gvar$F, gvar$Sigma_e, models, gvar$G0_inv,
                           shock_cid, shock_vid, H=H, n_boot=500, seed=20)
plot_fig13_3(irf, boot_gi$lo, boot_gi$hi, H=H, path="empirical_example_GlobalVAR_girf.pdf")

cat("\n[8] FEVD + Spillover network + Figure 13.4 ...\n")
fevd <- compute_gfevd_gdp(gvar$F, gvar$Sigma_e, H_fevd=8)
plot_fig13_4(fevd, path="empirical_example_GlobalVAR_fevd.pdf")

# Summary
can_i <- CIDX["Canada"]; usa_i <- CIDX["USA"]
eur_i <- CIDX["Euro_Area"]; uk_i  <- CIDX["UK"]
pct   <- round(fevd * 100)

cat("\n", strrep("=",70), "\n")
cat("RESULTS SUMMARY (book values in brackets)\n")
cat(strrep("=",70), "\n")
cat(sprintf("  Max eigenvalue: %.3f  [0.973]\n", max_eig))
cat("\n  Peak GDP response to 1 s.d. US rate shock:\n")
book_peaks <- c(USA=-0.141, Euro_Area=-0.022, UK=-0.034, Japan=-0.009,
                China=-0.003, Canada=-0.093, Korea=-0.004, Brazil=NA)
for (c in COUNTRIES) {
  gdp <- irf[[c]][,1]
  th  <- which.min(gdp)
  bk  <- if (!is.na(book_peaks[c])) sprintf("[%.3f]", book_peaks[c]) else "[n/a]"
  cat(sprintf("    %-12s: %+.3f pp at h=%d  %s\n",
              gsub("_"," ",c), gdp[th], th-1, bk))
}
own <- diag(pct)
cat(sprintf("\n  Own shocks: %d-%d%%  [book: 91-97%%]\n", min(own), max(own)))
cat(sprintf("  USA -> Canada: %d%%  [book: ~6%%]\n",   pct[can_i, usa_i]))
cat(sprintf("  EUR -> UK:     %d%%  [book: ~3%%]\n",   pct[uk_i,  eur_i]))
cat(sprintf("  Canada -> USA: %d%%  [book: ~3%%]\n",   pct[usa_i, can_i]))
cat(strrep("=",70), "\n")
cat("\n  Figures saved: empirical_example_GlobalVAR_eigenvalues.pdf,",
    "empirical_example_GlobalVAR_girf.pdf, empirical_example_GlobalVAR_fevd.pdf\n")
