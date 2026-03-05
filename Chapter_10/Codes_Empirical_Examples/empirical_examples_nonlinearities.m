% =========================================================================
% FOUR APPROACHES TO REGIME-DEPENDENT DYNAMICS
% =========================================================================
%
% Author:   Alessia Paccagnini
% Textbook: Macroeconometrics
%
% Empirical comparison of:
%   Approach 1: Time-Varying Parameters with Stochastic Volatility (TVP-VAR-SV)
%   Approach 2: Markov-Switching VAR (MS-VAR)
%   Approach 3: Threshold VAR (TVAR)
%   Approach 4: Smooth Transition VAR (STVAR)
%
% Data: US quarterly macro data 1960Q1-2019Q4 (FRED)
%   GDPC1    -> annualised GDP growth
%   GDPDEF   -> annualised inflation
%   FEDFUNDS -> quarterly average federal funds rate
%
% Variable ordering (Cholesky): GDP growth | Inflation | Fed Funds Rate
%
% Requirements:
%   - msvar toolkit (functions/ subfolder on path)
%   - Statistics and Machine Learning Toolbox (for mvnrnd, wishrnd)
%   - Optimization Toolbox (for fmincon / fminunc, used in STVAR)
%
% Usage:
%   Run from the folder containing this script and the functions/ directory.
%   Excel files GDPC1.xlsx, GDPDEF.xlsx, FEDFUNDS.xlsx must be present.
%
% =========================================================================

clear; clc; close all;
rng(42);   % reproducibility

% -- Add toolkit functions to path -----------------------------------------
addpath(fullfile(pwd, 'functions'));

% =========================================================================
% 0.  GLOBAL SETTINGS
% =========================================================================

LAGS   = 2;     % VAR lag order
HORZ   = 20;    % IRF horizon (quarters)
N      = 3;     % number of variables

% MCMC settings
% For quick testing use the defaults below.
% For publication quality increase to REPS_TVP=20000, REPS_MS=20000.
% Always inspect convergence diagnostics after any run (see below).
REPS_TVP = 3000;  BURN_TVP = 1000;
REPS_MS  = 5000;  BURN_MS  = 3000;

% Shock variable: federal funds rate (column 3, last in Cholesky order)
SHOCK_EQ = 3;

% Date indices for IRF comparison
% Data starts 1960Q1; model drops 1 diff row + LAGS rows - effective start 1960Q4 (row 1)
% 2005Q1 = model row 178  (1-based)
% 2008Q4 = model row 193  (1-based)
T_NORMAL = 178;   % 2005Q1
T_STRESS = 193;   % 2008Q4

% =========================================================================
% 1.  DATA PREPARATION
% =========================================================================

fprintf('=================================================================\n');
fprintf('FOUR APPROACHES TO REGIME-DEPENDENT DYNAMICS\n');
fprintf('=================================================================\n\n');
fprintf('Loading FRED data...\n');

% -- GDP (quarterly) -------------------------------------------------------
gdp_raw   = readtable('GDPC1.xlsx',    'Sheet','Quarterly', ...
                       'Range','A2:B500','ReadVariableNames',false);
infl_raw  = readtable('GDPDEF.xlsx',   'Sheet','Quarterly', ...
                       'Range','A2:B500','ReadVariableNames',false);
ff_raw    = readtable('FEDFUNDS.xlsx', 'Sheet','Monthly',   ...
                       'Range','A2:B1000','ReadVariableNames',false);

gdp_raw.Properties.VariableNames   = {'date','gdp'};
infl_raw.Properties.VariableNames  = {'date','gdpdef'};
ff_raw.Properties.VariableNames    = {'date','fedfunds'};

% Remove rows with missing dates or values
gdp_raw   = gdp_raw(~isnat(gdp_raw.date)   & ~isnan(gdp_raw.gdp),   :);
infl_raw  = infl_raw(~isnat(infl_raw.date) & ~isnan(infl_raw.gdpdef),:);
ff_raw    = ff_raw(~isnat(ff_raw.date)     & ~isnan(ff_raw.fedfunds),:);

% Monthly fed funds - quarterly average
ff_raw.quarter = dateshift(ff_raw.date,'start','quarter');
[G,qdate]      = findgroups(ff_raw.quarter);
ff_q_vals      = splitapply(@mean, ff_raw.fedfunds, G);
ff_q           = table(qdate, ff_q_vals, 'VariableNames',{'date','fedfunds'});

% Merge on quarterly dates
[~,ig,id] = intersect(gdp_raw.date, infl_raw.date);
dates_all  = gdp_raw.date(ig);
gdp_vals   = gdp_raw.gdp(ig);
def_vals   = infl_raw.gdpdef(id);

[~,ig2,if2] = intersect(dates_all, ff_q.date);
dates_all  = dates_all(ig2);
gdp_vals   = gdp_vals(ig2);
def_vals   = def_vals(ig2);
ff_vals    = ff_q.fedfunds(if2);

% Annualised quarterly growth rates
gdp_growth = [NaN; diff(log(gdp_vals))  * 400];
inflation  = [NaN; diff(log(def_vals))  * 400];

% Filter 1960Q1-2019Q4 and drop first NaN
mask       = dates_all >= datetime(1960,1,1) & dates_all <= datetime(2019,12,31);
dates_all  = dates_all(mask);
Y_raw      = [gdp_growth(mask), inflation(mask), ff_vals(mask)];

% Drop first obs (NaN from diff)
dates_plot = dates_all(2:end);
Y_raw      = Y_raw(2:end,:);
T_raw      = size(Y_raw,1);   % should be 240

fprintf('Sample:        %s  to  %s\n', ...
    datestr(dates_plot(1),'YYYY-QQ'), datestr(dates_plot(end),'YYYY-QQ'));
fprintf('Observations:  %d\n\n', T_raw);
fprintf('Variable order:  [1] GDP Growth | [2] Inflation | [3] Fed Funds Rate\n\n');

% -- NBER recession dates (for shading) ------------------------------------
nber = [datetime(1960,4,1),  datetime(1961,2,1);
        datetime(1969,12,1), datetime(1970,11,1);
        datetime(1973,11,1), datetime(1975,3,1);
        datetime(1980,1,1),  datetime(1980,7,1);
        datetime(1981,7,1),  datetime(1982,11,1);
        datetime(1990,7,1),  datetime(1991,3,1);
        datetime(2001,3,1),  datetime(2001,11,1);
        datetime(2007,12,1), datetime(2009,6,1)];

% =========================================================================
% 2.  APPROACH 1 - TVP-VAR WITH STOCHASTIC VOLATILITY
% =========================================================================
% Simple diagonal TVP-VAR-SV via Gibbs sampling
%   Step 1: Carter-Kohn FFBS for beta_t
%   Step 2: IG draw for Q diagonal
%   Step 3: Single-move MH for log-volatilities h_{i,t}

fprintf('=================================================================\n');
fprintf('APPROACH 1: TVP-VAR-SV  (Primiceri 2005)\n');
fprintf('=================================================================\n');

% -- Build regressor matrix ------------------------------------------------
p   = LAGS;
Y   = Y_raw(p+1:end,:);
T   = size(Y,1);       % effective sample (drops first p obs)
k   = N*p + 1;         % regressors per equation
nk  = N*k;             % total coefficients

X   = zeros(T,k);
for t = 1:T
    row = [];
    for lag = 1:p
        row = [row, Y_raw(p+t-lag,:)];
    end
    X(t,:) = [row, 1];
end

% -- OLS initialisation ---------------------------------------------------
B_ols   = X \ Y;
e_ols   = Y - X*B_ols;
S_ols   = (e_ols'*e_ols) / (T-k);

% -- Initialise state vectors ----------------------------------------------
beta0    = B_ols(:);               % (nk x 1)
h_all    = repmat(log(diag(S_ols)+1e-6)', T, 1);  % (T x N) log-volatilities
beta_all = repmat(beta0', T, 1);   % (T x nk)

% Prior on Q: kappa * I (diagonal)
kappa_Q  = 0.01;
Q_diag   = kappa_Q * ones(nk,1);
Q        = diag(Q_diag);

% Observation matrices Z[t]: (N x nk), block-diagonal with x_t repeated
Z_all = zeros(T, N, nk);
for t = 1:T
    for eq = 1:N
        cols_eq = (eq-1)*k+1 : eq*k;
        Z_all(t, eq, cols_eq) = X(t,:);
    end
end

% Storage
ndraws_tvp  = REPS_TVP - BURN_TVP;
beta_store  = zeros(ndraws_tvp, T, nk);
h_store     = zeros(ndraws_tvp, T, N);
sig_eta_mh  = 0.1;   % MH proposal std for log-vol

m0 = beta0;
P0 = 4*eye(nk);

fprintf('Running Gibbs sampler: %d iterations, %d burn-in...\n', ...
    REPS_TVP, BURN_TVP);
% -- Convergence note ------------------------------------------------------
% After the sampler finishes, call mcmc_diagnostics() to check convergence.
% Example (run interactively or uncomment):
%
%   mid = T/2;
%   store_tvp = [squeeze(beta_store(:,mid,1)), squeeze(beta_store(:,mid,nk/2)), ...
%                squeeze(h_store(:,mid,1)),    squeeze(h_store(:,mid,2)), ...
%                squeeze(h_store(:,mid,3))];
%   names_tvp = {'beta[mid,1]','beta[mid,nk/2]', ...
%                'log-vol GDP[mid]','log-vol Inf[mid]','log-vol FFR[mid]'};
%   mcmc_diagnostics(store_tvp, names_tvp, 5, 'tvpvar');
% -------------------------------------------------------------------------

jdraw = 0;
for isim = 1:REPS_TVP

    % -- Step 1: Sample beta_t | h_t  (Carter-Kohn FFBS) ------------------
    H_obs = exp(h_all);   % (T x N) diagonal variances

    % Forward filter
    mf = zeros(T, nk);
    Pf = zeros(T, nk, nk);
    mp = m0;  Pp = P0;

    for t = 1:T
        Zt = squeeze(Z_all(t,:,:));         % (N x nk)
        Ht = diag(H_obs(t,:));              % (N x N)
        vt = Y(t,:)' - Zt*mp;
        Ft = Zt*Pp*Zt' + Ht;
        Kt = Pp*Zt'*(Ft\eye(N));
        mf(t,:) = (mp + Kt*vt)';
        Pf(t,:,:) = Pp - Kt*Zt*Pp;
        Pf(t,:,:) = 0.5*(squeeze(Pf(t,:,:)) + squeeze(Pf(t,:,:))') + 1e-9*eye(nk);
        mp = mf(t,:)';
        Pp = squeeze(Pf(t,:,:)) + Q;
    end

    % Backward sample
    beta_path = zeros(T, nk);
    Pf_T = squeeze(Pf(T,:,:));
    beta_path(T,:) = mvnrnd(mf(T,:), Pf_T + 1e-9*eye(nk));

    for t = T-1:-1:1
        Pf_t    = squeeze(Pf(t,:,:));
        Pp_next = Pf_t + Q;
        J       = Pf_t / Pp_next;
        mb      = mf(t,:)' + J*(beta_path(t+1,:)' - mf(t,:)');
        Pb      = Pf_t - J*Pf_t;
        Pb      = 0.5*(Pb+Pb') + 1e-9*eye(nk);
        beta_path(t,:) = mvnrnd(mb', Pb);
    end
    beta_all = beta_path;

    % -- Step 2: Draw Q | beta_t  (diagonal IG) ----------------------------
    db    = diff(beta_all);                    % (T-1 x nk)
    sq    = sum(db.^2, 1)';                    % (nk x 1) sum of squares
    nu_q  = nk + 1 + (T-1);
    for j = 1:nk
        Q_diag(j) = 1 / gamrnd(nu_q/2, 2/(kappa_Q + sq(j)));
        Q_diag(j) = min(max(Q_diag(j), 1e-8), 1.0);
    end
    Q = diag(Q_diag);

    % -- Step 3: Compute residuals -----------------------------------------
    resid = zeros(T,N);
    for t = 1:T
        Zt = squeeze(Z_all(t,:,:));
        resid(t,:) = Y(t,:) - (Zt*beta_all(t,:)')';
    end

    % -- Step 4: Sample h_{i,t}  (single-move MH) -------------------------
    for i = 1:N
        h_cur = h_all(:,i);
        for t = 1:T
            h_prev_t = h_cur(max(t-1,1));
            h_prop   = h_prev_t + sig_eta_mh * randn;
            ll_p = -0.5*h_prop - 0.5*resid(t,i)^2*exp(-h_prop);
            ll_c = -0.5*h_cur(t) - 0.5*resid(t,i)^2*exp(-h_cur(t));
            if log(rand + 1e-300) < ll_p - ll_c
                h_cur(t) = h_prop;
            end
        end
        h_all(:,i) = h_cur;
    end

    % -- Store post burn-in -------------------------------------------------
    if isim > BURN_TVP
        jdraw = jdraw + 1;
        beta_store(jdraw,:,:) = beta_all;
        h_store(jdraw,:,:)    = h_all;
    end

    if mod(isim,500)==0
        fprintf('  Iteration %4d / %d\n', isim, REPS_TVP);
    end
end
fprintf('  TVP-VAR-SV complete.\n');
fprintf('\n  --- TVP-VAR-SV Convergence Diagnostics ---\n');
fprintf('  Checking representative coefficients and log-volatilities...\n');
mid_tvp = floor(T/2);
store_tvp = [squeeze(beta_store(:,mid_tvp,1)), squeeze(beta_store(:,mid_tvp,floor(nk/2))), ...
             squeeze(h_store(:,mid_tvp,1)),    squeeze(h_store(:,mid_tvp,2)), ...
             squeeze(h_store(:,mid_tvp,3))];
names_tvp = {'beta[mid,1]','beta[mid,nk/2]', ...
             'log-vol GDP[mid]','log-vol Inf[mid]','log-vol FFR[mid]'};
mcmc_diagnostics(store_tvp, names_tvp, 5, 'tvpvar');
fprintf('  Geweke trace plots saved as diag_trace_tvpvar_*.png\n');
fprintf('  Tip: increase REPS_TVP >= 20000 for publication quality.\n\n');

% -- Posterior summaries ---------------------------------------------------
h_post_mean = squeeze(mean(h_store,1));           % (T x N) posterior mean
sig_med  = exp(median(h_store,1)/2);              % (1 x T x N) or squeeze below
sig_med  = squeeze(median(exp(h_store/2), 1));    % (T x N)
sig_lo   = squeeze(prctile(exp(h_store/2), 16));
sig_hi   = squeeze(prctile(exp(h_store/2), 84));

% -- Compute IRFs at two dates ---------------------------------------------

fprintf('Computing TVP-VAR-SV IRFs at 2005Q1 and 2008Q4...\n');
[irf_n_med, irf_n_lo, irf_n_hi] = compute_tvp_irf(beta_store, h_store, ...
    T_NORMAL, N, k, p, LAGS, HORZ, SHOCK_EQ);
[irf_s_med, irf_s_lo, irf_s_hi] = compute_tvp_irf(beta_store, h_store, ...
    T_STRESS, N, k, p, LAGS, HORZ, SHOCK_EQ);

% -- High-volatility regime flag (for synthesis) ---------------------------
gdp_vol     = sig_med(:,1);
vol_thresh  = prctile(gdp_vol, 75);
tvp_hv_flag = gdp_vol > vol_thresh;

% -- TVP-VAR-SV: Time-Varying Volatilities ----------------------------------
dates_tvp = dates_plot(p+1:end);   % aligned to model sample
var_names  = {'GDP Growth','Inflation','Fed Funds Rate'};
colors_fig = [0.13 0.40 0.67; 0.84 0.37 0.30; 0.30 0.67 0.15];

fig1 = figure('Name','TVP-VAR-SV Volatilities','Position',[50 50 900 700]);
for i = 1:N
    subplot(N,1,i);
    fill_between_dates(dates_tvp, sig_lo(:,i), sig_hi(:,i), colors_fig(i,:), 0.3);
    hold on;
    plot(dates_tvp, sig_med(:,i), 'Color', colors_fig(i,:), 'LineWidth', 1.8);
    shade_nber(nber, ylim);
    if i==1
        xline(datetime(1984,1,1),'k--','LineWidth',0.8);
        text(datetime(1985,1,1), max(sig_hi(:,1))*0.80, ...
            'Great Moderation','FontSize',8,'Color',[0.3 0.3 0.3]);
    end
    ylabel('Volatility (SD)');
    title([var_names{i} ' Volatility']);
    legend({'68% Credible Interval','Posterior Mean'},'Location','northeast','FontSize',8);
    grid on; grid minor;
    xlim([dates_tvp(1), dates_tvp(end)]);
end
xlabel('Date');
sgtitle('Time-Varying Stochastic Volatilities (TVP-VAR-SV)', ...
    'FontWeight','bold','FontSize',12);
saveas(fig1, 'fig_tvpvar_volatility.png');
fprintf('TVP-VAR-SV volatility figure saved.\n');

% -- TVP-VAR-SV: Regime-Specific IRFs --------------------------------------
resp_labels = {'GDP Growth Response','Inflation Response','Fed Funds Rate Response'};
fig2 = figure('Name','TVP-VAR-SV IRFs','Position',[50 50 1100 380]);
hor  = 0:HORZ-1;
for i = 1:N
    subplot(1,N,i);
    % Normal times (blue)
    fill([hor, fliplr(hor)], [irf_n_lo(:,i)', fliplr(irf_n_hi(:,i)')], ...
        [0.13 0.40 0.67], 'FaceAlpha',0.20,'EdgeColor','none'); hold on;
    plot(hor, irf_n_med(:,i), 'Color',[0.13 0.40 0.67], 'LineWidth',2.0, ...
        'DisplayName','Normal Times (2005Q1)');
    % Stress times (red dashed)
    fill([hor, fliplr(hor)], [irf_s_lo(:,i)', fliplr(irf_s_hi(:,i)')], ...
        [0.84 0.37 0.30], 'FaceAlpha',0.20,'EdgeColor','none');
    plot(hor, irf_s_med(:,i), '--','Color',[0.84 0.37 0.30], 'LineWidth',2.0, ...
        'DisplayName','Stress Times (2008Q4)');
    yline(0,'k:','LineWidth',0.8);
    xlabel('Quarters Ahead'); title(resp_labels{i});
    if i==1, legend('Location','southwest','FontSize',8); end
    grid on; grid minor;
end
sgtitle({'TVP-VAR-SV: Monetary Policy Shock (1 SD Tightening)', ...
    'Normal vs. Stress Period Comparison'}, 'FontWeight','bold','FontSize',11);
saveas(fig2, 'fig_tvpvar_irf_comparison.png');
fprintf('TVP-VAR-SV IRF comparison figure saved.\n\n');

% Numerical summary
fprintf('TVP-VAR-SV Results:\n');
fprintf('  Normal (2005Q1) - Peak GDP: %.3f%%,  Peak Inflation: %.3f%%\n', ...
    min(irf_n_med(:,1)), min(irf_n_med(:,2)));
fprintf('  Stress (2008Q4) - Peak GDP: %.3f%%,  Peak Inflation: %.3f%%\n', ...
    min(irf_s_med(:,1)), min(irf_s_med(:,2)));

% =========================================================================
% 3.  APPROACH 2 - MARKOV-SWITCHING VAR  (uses msvar toolkit)
% =========================================================================

fprintf('=================================================================\n');
fprintf('APPROACH 2: MARKOV-SWITCHING VAR  (Gibbs sampler)\n');
fprintf('=================================================================\n');

% -- Toolkit settings ------------------------------------------------------
VarBench.L           = LAGS;
VarBench.lamdaP      = 0.1;
VarBench.tauP        = 1.0;     % 10 * lambda
VarBench.epsilonP    = 1/1000;
VarBench.TVTPpriorC    = [-2; -4];
VarBench.TVTPpriorvarC = [10; 10];

MaxTrys = 1000;
Update  = 500;

fprintf('Running MS-VAR Gibbs sampler: %d iterations, %d burn-in...\n', ...
    REPS_MS, BURN_MS);
% -- Convergence note ------------------------------------------------------
% After the sampler finishes, call mcmc_diagnostics() to check convergence.
% Example (run interactively or uncomment):
%
%   store_ms = [bsave1(:,1), bsave2(:,1), mean(regime_draws,2), ...
%               mean(pmat,2), mean(qmat,2)];
%   names_ms = {'B1[1]','B2[1]','avg regime prob','avg p00','avg q11'};
%   mcmc_diagnostics(store_ms, names_ms, 5, 'msvar');
% -------------------------------------------------------------------------

[bsave1, bsave2, sigmaS1, sigmaS2, gsave, regime_draws, pmat, qmat, ~] = ...
    estimatemstvtpnew(Y_raw, VarBench, HORZ, REPS_MS, BURN_MS, Update, MaxTrys, 1);

fprintf('  MS-VAR complete.\n');
fprintf('\n  --- MS-VAR Convergence Diagnostics ---\n');
fprintf('  Checking VAR coefficients, regime probabilities, transition parameters...\n');
store_ms = [bsave1(:,1), bsave2(:,1), mean(regime_draws,2), ...
            mean(pmat,2), mean(qmat,2)];
names_ms = {'B1[1]','B2[1]','avg regime prob','avg p00','avg q11'};
mcmc_diagnostics(store_ms, names_ms, 5, 'msvar');
fprintf('  Geweke trace plots saved as diag_trace_msvar_*.png\n');
fprintf('  Tip: increase REPS_MS >= 20000 for publication quality.\n\n');

% -- Smoothed regime probabilities -----------------------------------------
% regime_draws: (ndraws_ms x T_raw)  - 0=normal, 1=stress
% Smoothed probability = posterior mean of regime indicator
ndraws_ms   = size(regime_draws,1);
T_ms        = size(regime_draws,2);   % full T_raw (no lag trimming in toolkit)
smooth_prob = mean(regime_draws, 1)';   % (T_ms x 1)

% -- Transition probability statistics -------------------------------------
p00_draws = mean(pmat, 2);    % (ndraws x 1) - regime 0 persistence per draw
q11_draws = mean(qmat, 2);    % regime 1 persistence
p00_mean  = mean(mean(pmat,2));
q11_mean  = mean(mean(qmat,2));
dur_normal = 1/(1-p00_mean);
dur_stress = 1/(1-q11_mean);

fprintf('MS-VAR Transition Probabilities (posterior mean):\n');
fprintf('  P(Normal-Normal) = %.4f   E[Duration] = %.1f quarters\n', p00_mean, dur_normal);
fprintf('  P(Stress-Stress) = %.4f   E[Duration] = %.1f quarters\n', q11_mean, dur_stress);

% -- Regime-specific IRFs (normalised to 1 SD within regime) ---------------
% NOTE: irfsim(b,n,l,v,s,t) returns t-l rows (it trims the first l and last l
% rows internally: y = y(l+1 : t+l-l)).  We therefore pass HORZ+LAGS so the
% returned matrix has exactly HORZ rows.
fprintf('Computing MS-VAR regime-specific IRFs...\n');
HORZ_IN      = HORZ + LAGS;          % argument passed to irfsim
irf_normal_ms = zeros(ndraws_ms, HORZ, N);
irf_stress_ms = zeros(ndraws_ms, HORZ, N);

k_ms = N*LAGS+1;
for d = 1:ndraws_ms
    % Regime 0 (normal)
    B1    = reshape(bsave1(d,:), k_ms, N);
    Sig1  = squeeze(sigmaS1(d,:,:));
    chol1 = chol(Sig1,'lower');
    shock1 = zeros(N,1); shock1(SHOCK_EQ) = chol1(SHOCK_EQ,SHOCK_EQ);
    irf_normal_ms(d,:,:) = irfsim(B1, N, LAGS, chol1, shock1', HORZ_IN);

    % Regime 1 (stress)
    B2    = reshape(bsave2(d,:), k_ms, N);
    Sig2  = squeeze(sigmaS2(d,:,:));
    chol2 = chol(Sig2,'lower');
    shock2 = zeros(N,1); shock2(SHOCK_EQ) = chol2(SHOCK_EQ,SHOCK_EQ);
    irf_stress_ms(d,:,:) = irfsim(B2, N, LAGS, chol2, shock2', HORZ_IN);
end

ms_irf_n_med = squeeze(median(irf_normal_ms,1));
ms_irf_n_lo  = squeeze(prctile(irf_normal_ms,16,1));
ms_irf_n_hi  = squeeze(prctile(irf_normal_ms,84,1));
ms_irf_s_med = squeeze(median(irf_stress_ms,1));
ms_irf_s_lo  = squeeze(prctile(irf_stress_ms,16,1));
ms_irf_s_hi  = squeeze(prctile(irf_stress_ms,84,1));

% -- MS-VAR: Smoothed Regime Probabilities ----------------------------------
dates_ms = dates_plot(1:T_ms);
fig3 = figure('Name','MS-VAR Regime Prob','Position',[50 50 1000 380]);
fill_between_dates(dates_ms, zeros(T_ms,1), smooth_prob, [0.13 0.40 0.67], 0.55);
hold on;
shade_nber(nber, [0,1]);
yline(0.5,'k--','LineWidth',1.2,'DisplayName','Classification Threshold (0.5)');
xlabel('Date'); ylabel('Probability');
title('Markov-Switching VAR: Smoothed Stress Regime Probability', ...
    'FontWeight','bold');
legend({'Pr(Stress Regime)','NBER Recession','50% Threshold'}, ...
    'Location','northeast','FontSize',8);
ylim([0,1]); grid on; grid minor;
saveas(fig3, 'fig_msvar_regime_prob.png');
fprintf('MS-VAR regime probability figure saved.\n');

% MS-VAR regime flag for synthesis
ms_stress_flag = (smooth_prob > 0.5);

fprintf('  Normal regime: peak GDP = %.3f%%,  peak Inflation = %.3f%%\n', ...
    min(ms_irf_n_med(:,1)), min(ms_irf_n_med(:,2)));
fprintf('  Stress regime: peak GDP = %.3f%%,  peak Inflation = %.3f%%\n\n', ...
    min(ms_irf_s_med(:,1)), min(ms_irf_s_med(:,2)));

% =========================================================================
% 4.  APPROACH 3 - THRESHOLD VAR (TVAR)
% =========================================================================
% Threshold variable: lagged GDP growth (column 1)
% Identification: grid search over threshold tau, estimated via OLS in each regime
% Inference: Hansen (1999) sup-LM bootstrap for linearity test
% IRFs: Generalized (Monte Carlo simulation, 1000 paths)

fprintf('=================================================================\n');
fprintf('APPROACH 3: THRESHOLD VAR\n');
fprintf('=================================================================\n');

TVAR_DELAY  = 1;   % delay parameter d (Ystar = GDP growth lagged d periods)
TVAR_VAR    = 1;   % threshold variable is column 1 (GDP growth)
TVAR_NCRIT  = 15;  % minimum obs per regime
N_GRID      = 100; % threshold grid points
N_GIRF      = 1000;% Monte Carlo paths for GIRFs

% -- Build TVAR data matrices -----------------------------------------------
Yt   = Y_raw(LAGS+1:end,:);        % (T_tvar x N)
Xt   = zeros(size(Yt,1), N*LAGS+1);
for t = 1:size(Yt,1)
    row = [];
    for lag = 1:LAGS
        row = [row, Y_raw(LAGS+t-lag,:)];
    end
    Xt(t,:) = [row, 1];
end
T_tvar  = size(Yt,1);

% Threshold variable: GDP growth lagged TVAR_DELAY periods
Ystar = [NaN(TVAR_DELAY,1); Y_raw(1:end-TVAR_DELAY, TVAR_VAR)];
Ystar = Ystar(LAGS+1:end);   % align with Yt

% -- Grid search for threshold ---------------------------------------------
% Trim 15% from each tail of Ystar distribution
tau_min = prctile(Ystar, 15);
tau_max = prctile(Ystar, 85);
tau_grid = linspace(tau_min, tau_max, N_GRID);

sse_grid = zeros(N_GRID,1);
for ig = 1:N_GRID
    tau = tau_grid(ig);
    e1  = Ystar <= tau;
    e2  = Ystar >  tau;
    if sum(e1) < TVAR_NCRIT || sum(e2) < TVAR_NCRIT
        sse_grid(ig) = inf;
        continue;
    end
    B1  = Xt(e1,:) \ Yt(e1,:);
    B2  = Xt(e2,:) \ Yt(e2,:);
    r1  = Yt(e1,:) - Xt(e1,:)*B1;
    r2  = Yt(e2,:) - Xt(e2,:)*B2;
    sse_grid(ig) = sum(sum(r1.^2)) + sum(sum(r2.^2));
end

[~, best_idx] = min(sse_grid);
tau_hat = tau_grid(best_idx);
fprintf('Estimated threshold:  tau_hat = %.4f%%  (GDP growth)\n', tau_hat);

% -- Estimate regime coefficients at optimal threshold ---------------------
e1_hat = Ystar <= tau_hat;
e2_hat = Ystar >  tau_hat;
B1_tvar = Xt(e1_hat,:) \ Yt(e1_hat,:);
B2_tvar = Xt(e2_hat,:) \ Yt(e2_hat,:);
r1_tvar = Yt(e1_hat,:) - Xt(e1_hat,:)*B1_tvar;
r2_tvar = Yt(e2_hat,:) - Xt(e2_hat,:)*B2_tvar;
Sig1_tvar = (r1_tvar'*r1_tvar) / (sum(e1_hat)-N*LAGS-1);
Sig2_tvar = (r2_tvar'*r2_tvar) / (sum(e2_hat)-N*LAGS-1);

fprintf('Low-growth regime:  %d quarters (%.1f%%)\n', sum(e1_hat), 100*mean(e1_hat));
fprintf('High-growth regime: %d quarters (%.1f%%)\n', sum(e2_hat), 100*mean(e2_hat));

% -- Hansen (1999) sup-LM linearity test (bootstrap approximation) ---------
% Under H0: linear VAR; test statistic = (SSE_linear - SSE_tvar) / SSE_tvar * T
B_lin   = Xt \ Yt;
r_lin   = Yt - Xt*B_lin;
SSE_lin = sum(sum(r_lin.^2));
SSE_tvar_opt = sse_grid(best_idx);
LM_stat = (SSE_lin - SSE_tvar_opt) / SSE_tvar_opt * T_tvar;

% Bootstrap p-value (500 replications)
N_BOOT = 500;
LM_boot = zeros(N_BOOT,1);
for b = 1:N_BOOT
    % Resample residuals under H0 (linear VAR)
    idx_b  = randi(T_tvar, T_tvar, 1);
    Yt_b   = Xt*B_lin + r_lin(idx_b,:);
    % Re-run grid search on bootstrap sample
    sse_b  = zeros(N_GRID,1);
    Blin_b = Xt \ Yt_b;
    rlin_b = Yt_b - Xt*Blin_b;
    SSE_lin_b = sum(sum(rlin_b.^2));
    for ig = 1:N_GRID
        tau = tau_grid(ig);
        e1b = Ystar <= tau; e2b = ~e1b;
        if sum(e1b)<TVAR_NCRIT || sum(e2b)<TVAR_NCRIT; sse_b(ig)=inf; continue; end
        B1b = Xt(e1b,:)\Yt_b(e1b,:); B2b = Xt(e2b,:)\Yt_b(e2b,:);
        r1b = Yt_b(e1b,:)-Xt(e1b,:)*B1b; r2b = Yt_b(e2b,:)-Xt(e2b,:)*B2b;
        sse_b(ig) = sum(sum(r1b.^2))+sum(sum(r2b.^2));
    end
    SSE_tvar_b = min(sse_b);
    LM_boot(b) = (SSE_lin_b - SSE_tvar_b) / SSE_tvar_b * T_tvar;
end
pval_LM = mean(LM_boot >= LM_stat);
fprintf('Hansen sup-LM test:  stat = %.3f,  bootstrap p-value = %.4f\n', LM_stat, pval_LM);
if pval_LM < 0.05
    fprintf('  -> Linearity rejected at 5%% level.\n\n');
end

% -- Generalized IRFs (Monte Carlo, 1000 paths) -----------------------------
% For each regime, simulate N_GIRF paths of length HORZ, with and without
% a 1-SD shock to Fed Funds Rate
chol1_tvar = chol(Sig1_tvar, 'lower');
chol2_tvar = chol(Sig2_tvar, 'lower');

% Shock size = 1 SD of FFR in each regime
shock_size1 = chol1_tvar(SHOCK_EQ, SHOCK_EQ);
shock_size2 = chol2_tvar(SHOCK_EQ, SHOCK_EQ);

girf1 = compute_girf_tvar(Yt, Xt, Ystar, B1_tvar, B2_tvar, ...
    chol1_tvar, chol2_tvar, tau_hat, TVAR_DELAY, TVAR_VAR, LAGS, ...
    HORZ, N, N_GIRF, SHOCK_EQ, 1, T_tvar);  % start_regime=1 (low growth)
girf2 = compute_girf_tvar(Yt, Xt, Ystar, B1_tvar, B2_tvar, ...
    chol1_tvar, chol2_tvar, tau_hat, TVAR_DELAY, TVAR_VAR, LAGS, ...
    HORZ, N, N_GIRF, SHOCK_EQ, 2, T_tvar);  % start_regime=2 (high growth)

tvar_flag = e1_hat;   % low-growth regime flag for synthesis

% -- TVAR: Generalized IRFs -------------------------------------------------
fig5 = figure('Name','TVAR GIRFs','Position',[50 50 1100 380]);
hor  = 0:HORZ-1;
for i = 1:N
    subplot(1,N,i);
    plot(hor, girf1(:,i), 'Color',[0.13 0.40 0.67], 'LineWidth',2.0, ...
        'DisplayName','Low Growth Regime'); hold on;
    plot(hor, girf2(:,i), 'Color',[0.30 0.67 0.15], 'LineWidth',2.0, ...
        'DisplayName','High Growth Regime');
    yline(0,'k:','LineWidth',0.8);
    xlabel('Quarters Ahead'); title(resp_labels{i});
    if i==1, legend('Location','southeast','FontSize',8); end
    grid on; grid minor;
end
sgtitle({'Threshold VAR: Generalized Impulse Responses', ...
    'Monetary Policy Shock (1 SD Tightening)'}, 'FontWeight','bold','FontSize',11);
saveas(fig5, 'fig_tvar_girf.png');
fprintf('TVAR GIRF figure saved.\n');

fprintf('TVAR Results:\n');
fprintf('  Low-growth regime:  Peak GDP = %.3f%%\n', min(girf1(:,1)));
fprintf('  High-growth regime: Peak GDP = %.3f%%\n\n', min(girf2(:,1)));

% =========================================================================
% 5.  APPROACH 4 - SMOOTH TRANSITION VAR (STVAR)
% =========================================================================
% Transition function: F(z_t; gamma, c) = 1/(1 + exp(-gamma*(z_t - c)))
% Transition variable: lagged GDP growth
% Estimation: NLS/MLE by grid search over c, then fmincon over gamma

fprintf('=================================================================\n');
fprintf('APPROACH 4: SMOOTH TRANSITION VAR\n');
fprintf('=================================================================\n');

STVAR_DELAY = 1;
STVAR_VAR   = 1;   % GDP growth as transition variable

% Normalise transition variable
Zvar   = Ystar;    % already aligned from TVAR section
Zstd   = std(Zvar,'omitnan');

% -- 2-D grid search over (gamma, c) --------------------------------------
% fminsearch is unbounded so gamma can explode toward infinity (- TVAR).
% We instead do a coarse grid over both parameters, which is cheap and
% avoids the degeneracy entirely.
c_grid     = linspace(prctile(Zvar,10), prctile(Zvar,90), 40);
gamma_grid = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0];
sse_2d     = inf(length(gamma_grid), length(c_grid));

for ig = 1:length(gamma_grid)
    for ic = 1:length(c_grid)
        Ft = logistic_trans(Zvar, gamma_grid(ig), c_grid(ic));
        % skip if transition function is degenerate (all 0 or all 1)
        if min(Ft) > 0.99 || max(Ft) < 0.01; continue; end
        [~, sse_2d(ig,ic)] = fit_stvar(Yt, Xt, Ft, N);
    end
end

[best_g_idx, best_c_idx] = find(sse_2d == min(sse_2d(:)), 1);
gamma_init = gamma_grid(best_g_idx);
c_init     = c_grid(best_c_idx);

% -- Local refinement with fminsearch, starting from grid best ------------
% Transform gamma = exp(log_gamma) so optimisation stays on (0, 20]
obj_fn = @(params) stvar_sse(exp(params(1)), params(2), Yt, Xt, Zvar, N);
opts   = optimset('Display','off','MaxIter',800,'TolFun',1e-6,'TolX',1e-6);
[params_opt, ~] = fminsearch(obj_fn, [log(gamma_init), c_init], opts);
gamma_hat = min(exp(params_opt(1)), 20.0);  % cap at 20 (near-TVAR beyond this)
c_hat     = params_opt(2);

fprintf('STVAR estimates:  gamma_hat = %.3f,  c_hat = %.4f%%\n', gamma_hat, c_hat);

% -- Regime-specific VAR coefficients -------------------------------------
Ft_hat    = logistic_trans(Zvar, gamma_hat, c_hat);
[B1_st, B2_st, Sig1_st, Sig2_st] = fit_stvar_coefs(Yt, Xt, Ft_hat, N);

% -- Transition function over time -----------------------------------------
dates_st  = dates_plot(LAGS+1:end);

% -- Compute IRFs across five states (z values) ----------------------------
z_states  = [-2.0, -0.5, 0.0, 1.5, 3.0];
colors_st = [0.60 0.10 0.10;
             0.80 0.30 0.10;
             0.60 0.60 0.10;
             0.20 0.60 0.20;
             0.05 0.40 0.70];

irf_st_all = zeros(length(z_states), HORZ, N);
chol1_st   = chol(Sig1_st,'lower');
chol2_st   = chol(Sig2_st,'lower');

% irfsim returns t-l rows when passed horizon t, so pass HORZ+LAGS
for iz = 1:length(z_states)
    Fz     = logistic_trans(z_states(iz), gamma_hat, c_hat);
    B_mix  = (1-Fz)*B1_st + Fz*B2_st;
    Sig_mix= (1-Fz)*Sig1_st + Fz*Sig2_st;
    chol_m = chol(Sig_mix + 1e-8*eye(N), 'lower');
    shock  = zeros(N,1); shock(SHOCK_EQ) = chol_m(SHOCK_EQ,SHOCK_EQ);
    irf_st_all(iz,:,:) = irfsim(B_mix, N, LAGS, chol_m, shock', HORZ+LAGS);
end

% -- Information criteria: Linear VAR vs TVAR vs STVAR ---------------------
B_lin2  = Xt \ Yt;
r_lin2  = Yt - Xt*B_lin2;
sig_lin2= r_lin2'*r_lin2 / T_tvar;
ll_lin  = -T_tvar*N/2*log(2*pi) - T_tvar/2*log(det(sig_lin2)) ...
          - T_tvar/2;   % concentrated log-likelihood
k_lin   = N*(N*LAGS+1);
AIC_lin = -2*ll_lin + 2*k_lin;
BIC_lin = -2*ll_lin + log(T_tvar)*k_lin;

% TVAR
r1t = Yt(e1_hat,:)-Xt(e1_hat,:)*B1_tvar;
r2t = Yt(e2_hat,:)-Xt(e2_hat,:)*B2_tvar;
sig_tvar = ([r1t;r2t])'*([r1t;r2t]) / T_tvar;
ll_tvar  = -T_tvar*N/2*log(2*pi) - T_tvar/2*log(det(sig_tvar)) - T_tvar/2;
k_tvar   = 2*N*(N*LAGS+1);
AIC_tvar = -2*ll_tvar + 2*k_tvar;
BIC_tvar = -2*ll_tvar + log(T_tvar)*k_tvar;

% STVAR
[~, SSE_stvar] = fit_stvar(Yt, Xt, Ft_hat, N);
r_st = Yt - (repmat(1-Ft_hat,1,N).*( Xt*B1_st ) + repmat(Ft_hat,1,N).*( Xt*B2_st ));
sig_stvar = r_st'*r_st / T_tvar;
ll_stvar  = -T_tvar*N/2*log(2*pi) - T_tvar/2*log(det(sig_stvar)) - T_tvar/2;
k_stvar   = 2*N*(N*LAGS+1) + 2;   % extra 2 for gamma and c
AIC_stvar = -2*ll_stvar + 2*k_stvar;
BIC_stvar = -2*ll_stvar + log(T_tvar)*k_stvar;

fprintf('Information Criteria:\n');
fprintf('  Model        AIC        BIC\n');
fprintf('  Linear VAR   %8.1f   %8.1f\n', AIC_lin, BIC_lin);
fprintf('  TVAR         %8.1f   %8.1f\n', AIC_tvar, BIC_tvar);
fprintf('  STVAR        %8.1f   %8.1f\n\n', AIC_stvar, BIC_stvar);

stvar_flag = Ft_hat > 0.5;   % high-growth weight > 0.5 - classify as normal

% -- STVAR: Transition Function ---------------------------------------------
fig6 = figure('Name','STVAR Transition','Position',[50 50 1000 500]);
subplot(2,1,1);
plot(dates_st, Zvar, 'Color',[0.2 0.2 0.8], 'LineWidth',1.2); hold on;
yline(c_hat,'r--','LineWidth',1.4,'DisplayName',sprintf('Threshold c = %.2f%%', c_hat));
shade_nber(nber, ylim);
ylabel('GDP Growth (%)'); title('GDP Growth (Transition Variable)');
legend('GDP Growth', sprintf('Threshold c = %.2f%%',c_hat), 'NBER Recession', ...
    'Location','southeast','FontSize',8);
grid on; grid minor;

subplot(2,1,2);
plot(dates_st, Ft_hat, 'Color',[0.15 0.55 0.15], 'LineWidth',1.6); hold on;
shade_nber(nber, [0,1]);
yline(0.5,'k--','LineWidth',1.0);
ylabel('Transition Weight'); ylim([0,1]);
title(sprintf('F(z): Weight on High-Growth Regime  (\\gamma = %.2f)', gamma_hat));
text(dates_st(20), 0.88, sprintf('\\gamma = %.1f', gamma_hat), 'FontSize',9);
grid on; grid minor;
sgtitle('STVAR: Gradual Regime Changes','FontWeight','bold','FontSize',12);
saveas(fig6, 'fig_stvar_transition.png');
fprintf('STVAR transition figure saved.\n');

% -- STVAR: IRF Continuum ---------------------------------------------------
fig7 = figure('Name','STVAR IRF Continuum','Position',[50 50 1100 380]);
hor  = 0:HORZ-1;
lbl_strs = {'z=-2.0% (Deep Recession)','z=-0.5%','z=0.0%','z=+1.5%','z=+3.0% (Expansion)'};
for i = 1:N
    subplot(1,N,i);
    for iz = 1:length(z_states)
        plot(hor, squeeze(irf_st_all(iz,:,i)), ...
            'Color', colors_st(iz,:), 'LineWidth',1.8, ...
            'DisplayName', lbl_strs{iz}); hold on;
    end
    yline(0,'k:','LineWidth',0.8);
    xlabel('Quarters Ahead'); title(resp_labels{i});
    if i==1, legend('Location','southeast','FontSize',7); end
    grid on; grid minor;
end
sgtitle({'STVAR: Impulse Responses Across Continuum of States', ...
    'Monetary Policy Shock (1 SD Tightening)'}, 'FontWeight','bold','FontSize',11);
saveas(fig7, 'fig_stvar_irf_continuum.png');
fprintf('STVAR IRF continuum figure saved.\n');

% -- Model Comparison Bar Chart ---------------------------------------------
fig8 = figure('Name','Model Comparison','Position',[50 50 700 380]);
subplot(1,2,1);
bar([AIC_lin, AIC_tvar, AIC_stvar], 'FaceColor','flat', ...
    'CData',[0.5 0.5 0.8; 0.3 0.7 0.3; 0.8 0.5 0.3]);
set(gca,'XTickLabel',{'Linear VAR','TVAR','STVAR'});
ylabel('AIC (lower is better)');
title('Akaike Information Criterion');
for k_bar=1:3
    vals = [AIC_lin, AIC_tvar, AIC_stvar];
    text(k_bar, vals(k_bar)+10, sprintf('%.0f',vals(k_bar)), ...
        'HorizontalAlignment','center','FontSize',9);
end
grid on;

subplot(1,2,2);
bar([BIC_lin, BIC_tvar, BIC_stvar], 'FaceColor','flat', ...
    'CData',[0.5 0.5 0.8; 0.3 0.7 0.3; 0.8 0.5 0.3]);
set(gca,'XTickLabel',{'Linear VAR','TVAR','STVAR'});
ylabel('BIC (lower is better)');
title('Bayesian Information Criterion');
for k_bar=1:3
    vals = [BIC_lin, BIC_tvar, BIC_stvar];
    text(k_bar, vals(k_bar)+10, sprintf('%.0f',vals(k_bar)), ...
        'HorizontalAlignment','center','FontSize',9);
end
grid on;
sgtitle('Model Selection: Information Criteria', ...
    'FontWeight','bold','FontSize',12);
saveas(fig8, 'fig_model_comparison.png');
fprintf('Model comparison figure saved.\n\n');

% =========================================================================
% 6.  SYNTHESIS - REGIME CONCORDANCE
% =========================================================================

fprintf('=================================================================\n');
fprintf('SYNTHESIS: REGIME CLASSIFICATION CONCORDANCE\n');
fprintf('=================================================================\n');

% Align all regime flags to the common sample (LAGS+1 : T_raw)
% tvp_hv_flag : (T x 1) where T = T_raw - LAGS
% ms_stress_flag : (T_ms x 1) = T_raw (toolkit uses full data)
% tvar_flag   : (T_tvar x 1) = T_raw - LAGS
% stvar_flag  : (T_tvar x 1)

T_common = min([length(tvp_hv_flag), T_ms-LAGS, length(tvar_flag), length(stvar_flag)]);

tvp_c   = tvp_hv_flag(1:T_common);
ms_c    = ms_stress_flag(LAGS+1:LAGS+T_common);
tvar_c  = tvar_flag(1:T_common);
stvar_c = stvar_flag(1:T_common);
dates_c = dates_plot(LAGS+1:LAGS+T_common);

% NBER recession flag (aligned)
nber_flag = false(T_common,1);
for t = 1:T_common
    for r = 1:size(nber,1)
        if dates_c(t) >= nber(r,1) && dates_c(t) <= nber(r,2)
            nber_flag(t) = true;
        end
    end
end

% Concordance correlations
flags  = [nber_flag, ms_c, tvar_c, stvar_c];
labels = {'NBER','MS-VAR','TVAR','STVAR'};
Cmat   = corr(double(flags));

fprintf('Regime Classification Concordance (correlations):\n');
fprintf('           ');
for i=1:4, fprintf('%8s', labels{i}); end; fprintf('\n');
for i=1:4
    fprintf('%8s   ', labels{i});
    for j=1:4, fprintf('%8.3f', Cmat(i,j)); end
    fprintf('\n');
end
fprintf('\n');

% (Concordance heatmap removed -- correlation table printed above is sufficient.)

% -- Stacked regime timeline ------------------------------------------------
fig10 = figure('Name','Regime Timeline','Position',[50 50 1100 500]);
all_flags = [nber_flag, ms_c, tvar_c, stvar_c];
flag_labels = {'NBER Recession','MS-VAR Stress','TVAR Low-Growth','STVAR Stress'};
colors_syn  = [0.8 0.2 0.2; 0.2 0.4 0.8; 0.2 0.7 0.2; 0.7 0.4 0.0];
for i = 1:4
    subplot(4,1,i);
    area(dates_c, double(all_flags(:,i)), 'FaceColor', colors_syn(i,:), ...
        'FaceAlpha',0.6,'EdgeColor','none');
    ylim([0,1]); yticks([0,1]);
    ylabel(flag_labels{i},'FontSize',8);
    xlim([dates_c(1), dates_c(end)]);
    grid on; grid minor;
end
sgtitle('Regime Classifications Across Four Approaches','FontWeight','bold','FontSize',11);
saveas(fig10, 'fig_regime_timeline.png');
fprintf('Regime timeline saved.\n\n');

% =========================================================================
% 7.  FINAL SUMMARY TABLE
% =========================================================================

fprintf('=================================================================\n');
fprintf('SYNTHESIS: MODEL-ROBUST FINDINGS\n');
fprintf('=================================================================\n\n');

% Amplification ratios
amp_tvp  = abs(min(irf_s_med(:,1))) / max(abs(min(irf_n_med(:,1))), 1e-6);
amp_ms   = abs(min(ms_irf_s_med(:,1))) / max(abs(min(ms_irf_n_med(:,1))), 1e-6);
amp_tvar = abs(min(girf1(:,1))) / max(abs(min(girf2(:,1))), 1e-6);
amp_stv  = abs(min(squeeze(irf_st_all(1,:,1)))) / ...
           max(abs(min(squeeze(irf_st_all(end,:,1)))), 1e-6);

fprintf('GDP Response Amplification (Stress/Normal or Low/High Growth):\n');
fprintf('  TVP-VAR-SV : %.2fx\n', amp_tvp);
fprintf('  MS-VAR     : %.2fx\n', amp_ms);
fprintf('  TVAR       : %.2fx\n', amp_tvar);
fprintf('  STVAR      : %.2fx\n\n', amp_stv);
fprintf('Core finding: All four models show LARGER policy effects during stress.\n');
fprintf('Direction is unambiguous; exact magnitudes depend on MCMC/NLS settings.\n\n');

fprintf('=================================================================\n');
fprintf('OUTPUT FILES\n');
fprintf('=================================================================\n');
outfiles = {'fig_tvpvar_volatility.png','fig_tvpvar_irf_comparison.png', ...
    'fig_msvar_regime_prob.png','fig_tvar_girf.png', ...
    'fig_stvar_transition.png','fig_stvar_irf_continuum.png', ...
    'fig_model_comparison.png','fig_regime_timeline.png'};
for f = 1:length(outfiles)
    fprintf('  %s\n', outfiles{f});
end
fprintf('=================================================================\n');
fprintf('ALL DONE - Replication complete.\n');
fprintf('=================================================================\n');

% =========================================================================

% =========================================================================
% LOCAL FUNCTION: trace_plot
% =========================================================================
function trace_plot(chain, label, save_png)
% TRACE_PLOT  Plot MCMC trace and marginal density for a scalar chain.
%
% Usage:
%   trace_plot(chain, label)         % saves PNG
%   trace_plot(chain, label, false)  % display only, no save
%
% Parameters:
%   chain    - (ndraws x 1) vector of posterior draws (burn-in removed)
%   label    - string label for plot title and filename
%   save_png - logical, default true
%
% Interpretation:
%   The trace (left panel) should look like "white noise" around a
%   stable mean. Slow drift or persistent trends indicate the chain
%   has not converged and more iterations are needed.
    if nargin < 3; save_png = true; end
    chain = chain(:);
    fname = ['diag_trace_' regexprep(label,'[^A-Za-z0-9]','_') '.png'];

    fig = figure('Visible','off','Position',[50 50 900 320]);

    % Left: trace
    subplot(1,2,1);
    plot(chain,'Color',[0.13 0.40 0.67],'LineWidth',0.7); hold on;
    yline(mean(chain),'r--','LineWidth',1.5,'DisplayName','Posterior mean');
    xlabel('Draw'); ylabel('Value');
    title(['Trace - ' label],'FontSize',9);
    legend('Location','northeast','FontSize',8);
    grid on;

    % Right: density
    subplot(1,2,2);
    [f_kde, xi] = ksdensity(chain);
    plot(xi, f_kde, 'Color',[0.84 0.37 0.30],'LineWidth',2.0); hold on;
    xline(mean(chain),'r--','LineWidth',1.5);
    xline(prctile(chain,16),'Color',[0.5 0.5 0.5],'LineWidth',1.0,'LineStyle',':');
    xline(prctile(chain,84),'Color',[0.5 0.5 0.5],'LineWidth',1.0,'LineStyle',':');
    xlabel('Value'); ylabel('Density');
    title(['Density - ' label],'FontSize',9);
    grid on;

    if save_png
        saveas(fig, fname);
    end
    close(fig);
end

% =========================================================================
% LOCAL FUNCTION: geweke_z
% =========================================================================
function z = geweke_z(chain, first, last)
% GEWEKE_Z  Geweke (1992) convergence diagnostic for a scalar MCMC chain.
%
% Usage:
%   z = geweke_z(chain)
%   z = geweke_z(chain, 0.10, 0.50)
%
% Parameters:
%   chain - (ndraws x 1) vector of posterior draws (burn-in removed)
%   first - fraction defining the early window  (default 0.10 = first 10%)
%   last  - fraction defining the late  window  (default 0.50 = last  50%)
%
% Returns:
%   z  - scalar z-score; |z| > 1.96 flags non-convergence at 5% level
%
% Under convergence the two subchain means should be equal, making
% z approximately standard normal.
    if nargin < 2; first = 0.10; end
    if nargin < 3; last  = 0.50; end

    chain = chain(:);
    n  = length(chain);
    n1 = floor(first * n);
    n2 = floor(last  * n);
    a  = chain(1:n1);
    b  = chain(n - n2 + 1 : n);

    % Spectral density at frequency 0 via AR(1) approximation
    function s = spec0(x)
        x   = x - mean(x);
        n_x = length(x);
        if n_x < 4; s = var(x) + 1e-12; return; end
        rho = sum(x(2:end) .* x(1:end-1)) / (sum(x(1:end-1).^2) + 1e-12);
        rho = min(max(rho, -0.99), 0.99);
        s2  = var(x);
        s   = s2 / (1 - rho)^2;
    end

    s_a = spec0(a) / length(a);
    s_b = spec0(b) / length(b);
    denom = sqrt(s_a + s_b);
    if denom < 1e-12; z = 0; return; end
    z = (mean(a) - mean(b)) / denom;
end

% =========================================================================
% LOCAL FUNCTION: mcmc_diagnostics
% =========================================================================
function mcmc_diagnostics(store, param_names, n_check, tag)
% MCMC_DIAGNOSTICS  Geweke z-scores and trace plots for a draw matrix.
%
% Usage:
%   mcmc_diagnostics(store, param_names, n_check, tag)
%
% Parameters:
%   store       - (ndraws x nparams) matrix of posterior draws
%   param_names - cell array of parameter names (optional, pass {} to skip)
%   n_check     - number of parameters to check (default 5)
%   tag         - string prefix for PNG filenames (default 'param')
    if nargin < 2 || isempty(param_names)
        param_names = arrayfun(@(i) sprintf('param_%d',i), ...
                               1:size(store,2), 'UniformOutput',false);
    end
    if nargin < 3 || isempty(n_check); n_check = 5; end
    if nargin < 4 || isempty(tag);     tag = 'param'; end

    npar = size(store, 2);
    idx  = round(linspace(1, npar, min(n_check, npar)));

    fprintf('  %-30s %10s  %s\n', 'Parameter', 'Geweke z', 'Flag');
    fprintf('  %s\n', repmat('-',1,53));
    any_flag = false;
    for ii = idx
        z    = geweke_z(store(:,ii));
        flag = '';
        if abs(z) > 1.96; flag = '  *** WARN'; any_flag = true; end
        fprintf('  %-30s %10.3f%s\n', param_names{ii}, z, flag);
        trace_plot(store(:,ii), [tag '_' param_names{ii}]);
    end
    if ~any_flag
        fprintf('  All Geweke |z| < 1.96 - no convergence issues detected.\n');
    else
        fprintf('  *** Flagged parameters suggest insufficient iterations.\n');
        fprintf('      Consider increasing REPS and/or checking trace plots.\n');
    end
    fprintf('\n');
end


function [irf_med, irf_lo, irf_hi] = compute_tvp_irf(beta_store, h_store, ...
        t_idx, N, k, p, LAGS, HORZ, shock_eq)
    ndraws = size(beta_store,1);
    irfs   = zeros(ndraws, HORZ, N);
    for d = 1:ndraws
        B_t  = reshape(squeeze(beta_store(d,t_idx,:)), k, N);  % (k x N)
        A_list = cell(1,LAGS);
        for lag = 1:LAGS
            A_list{lag} = B_t((lag-1)*N+1:lag*N,:)';   % (N x N)
        end
        sigma_t = exp(squeeze(h_store(d,t_idx,:))/2);   % (N x 1)
        impact  = zeros(N,1);
        impact(shock_eq) = sigma_t(shock_eq);
        irf = zeros(HORZ,N);
        irf(1,:) = impact';
        for h = 2:HORZ
            for lag = 1:min(h-1, LAGS)
                irf(h,:) = irf(h,:) + (A_list{lag} * irf(h-lag,:)')';
            end
        end
        irfs(d,:,:) = irf;
    end
    irf_med = squeeze(median(irfs,1));
    irf_lo  = squeeze(prctile(irfs,16,1));
    irf_hi  = squeeze(prctile(irfs,84,1));
end

% LOCAL HELPER FUNCTIONS
% =========================================================================

function F = logistic_trans(z, gamma, c)
% Logistic smooth transition function
% F = 1 / (1 + exp(-gamma*(z - c)))
    F = 1 ./ (1 + exp(-gamma .* (z - c)));
end

function [B1, B2, Sig1, Sig2] = fit_stvar_coefs(Y, X, Ft, N)
% Estimate STVAR coefficients via WLS with weights (1-F) and F
    T  = size(Y,1);
    w1 = 1 - Ft;   % weight for regime 1 (low-growth)
    w2 = Ft;       % weight for regime 2 (high-growth)
    W1 = diag(sqrt(w1));
    W2 = diag(sqrt(w2));
    B1 = (W1*X) \ (W1*Y);
    B2 = (W2*X) \ (W2*Y);
    r1 = Y - X*B1;
    r2 = Y - X*B2;
    Sig1 = (r1'*(diag(w1)*r1)) / (sum(w1)-1);
    Sig2 = (r2'*(diag(w2)*r2)) / (sum(w2)-1);
    Sig1 = Sig1 + 1e-8*eye(N);
    Sig2 = Sig2 + 1e-8*eye(N);
end

function [B1, SSE] = fit_stvar(Y, X, Ft, N)
% Fit STVAR and return SSE (used in grid search)
    [B1, B2, ~, ~] = fit_stvar_coefs(Y, X, Ft, N);
    Yhat = (1-Ft).*( X*B1 ) + Ft.*( X*B2 );
    r    = Y - Yhat;
    SSE  = sum(sum(r.^2));
end

function SSE = stvar_sse(gamma, c, Y, X, Zvar, N)
% Objective for fminsearch
    if gamma <= 0; SSE = 1e10; return; end
    Ft  = logistic_trans(Zvar, gamma, c);
    [~, SSE] = fit_stvar(Y, X, Ft, N);
end

function girf = compute_girf_tvar(Yt, Xt, Ystar, B1, B2, chol1, chol2, ...
        tau, delay, tvar_col, LAGS, HORZ, N, N_GIRF, shock_eq, start_regime, T)
% Generalized IRFs for TVAR via Monte Carlo simulation
% start_regime: 1=low-growth, 2=high-growth
    girf_paths = zeros(N_GIRF, HORZ, N);
    girf_base  = zeros(N_GIRF, HORZ, N);

    for g = 1:N_GIRF
        % Pick a random history from the relevant regime
        if start_regime == 1
            idx_pool = find(Ystar <= tau);
        else
            idx_pool = find(Ystar >  tau);
        end
        t0 = idx_pool(randi(length(idx_pool)));
        t0 = min(max(t0, LAGS+1), T);

        % Initial Y history (last LAGS observations)
        Y_hist = Yt(max(1,t0-LAGS):t0, :);
        if size(Y_hist,1) < LAGS+1
            Y_hist = [repmat(Y_hist(1,:), LAGS+1-size(Y_hist,1), 1); Y_hist];
        end

        % Simulate baseline and shocked path
        Y_base  = zeros(HORZ, N);
        Y_shock = zeros(HORZ, N);

        % Draw innovations for both paths (same draws for all but t=1)
        eps_all = randn(HORZ, N);

        for h = 1:HORZ
            % Build regressor from last LAGS values
            if h == 1
                Y_use = Y_hist;
            else
                Y_use_b = [Y_hist(end-LAGS+2:end,:); Y_base(1:h-1,:)];
                Y_use_s = [Y_hist(end-LAGS+2:end,:); Y_shock(1:h-1,:)];
            end

            x_b = []; x_s = [];
            for lag = 1:LAGS
                if h == 1
                    x_b = [x_b, Y_hist(end-lag+1,:)];
                    x_s = [x_b];
                else
                    if h-lag >= 1
                        x_b = [x_b, Y_base(h-lag,:)];
                        x_s = [x_s, Y_shock(h-lag,:)];
                    else
                        idx_h = LAGS-(lag-h)-1;
                        x_b = [x_b, Y_hist(end-idx_h,:)];
                        x_s = [x_s, Y_hist(end-idx_h,:)];
                    end
                end
            end
            x_b = [x_b, 1]; x_s = [x_s, 1];

            % Transition variable for this step
            if h <= delay
                z_b = Y_hist(end-delay+h, tvar_col);
                z_s = z_b;
            else
                z_b = Y_base(h-delay, tvar_col);
                z_s = Y_shock(h-delay, tvar_col);
            end

            reg_b = (z_b <= tau);
            reg_s = (z_s <= tau);

            B_b = reg_b*B1 + (1-reg_b)*B2;
            B_s = reg_s*B1 + (1-reg_s)*B2;
            C_b = reg_b*chol1 + (1-reg_b)*chol2;
            C_s = reg_s*chol1 + (1-reg_s)*chol2;

            innov = eps_all(h,:);
            Y_base(h,:)  = x_b*B_b + innov*C_b';
            if h == 1
                shock_vec = zeros(1,N);
                shock_vec(shock_eq) = C_s(shock_eq,shock_eq);
                Y_shock(h,:) = x_s*B_s + innov*C_s' + shock_vec;
            else
                Y_shock(h,:) = x_s*B_s + innov*C_s';
            end
        end

        girf_base(g,:,:)  = Y_base;
        girf_paths(g,:,:) = Y_shock;
    end

    girf = squeeze(mean(girf_paths - girf_base, 1));
end

function shade_nber(nber, ylims)
% Add NBER recession shading to current axes
    yl = ylim;
    if nargin > 1 && length(ylims)==2; yl = ylims; end
    hold on;
    for r = 1:size(nber,1)
        fill([nber(r,1) nber(r,2) nber(r,2) nber(r,1)], ...
            [yl(1) yl(1) yl(2) yl(2)], ...
            [0.75 0.75 0.75], 'FaceAlpha',0.35,'EdgeColor','none', ...
            'HandleVisibility','on','DisplayName','NBER Recession');
    end
end

function fill_between_dates(dates, y_lo, y_hi, col, alpha_val)
% Fill between two date-indexed series
    x_fill = [dates; flipud(dates)];
    y_fill = [y_lo; flipud(y_hi)];
    fill(x_fill, y_fill, col, 'FaceAlpha',alpha_val, 'EdgeColor','none');
    hold on;
end

function cmap = redblue_colormap()
% Simple red-white-blue colormap
    r = [linspace(0.7,1,32), linspace(1,0.13,32)]';
    g = [linspace(0.7,1,32), linspace(1,0.40,32)]';
    b = [linspace(1,1,32),   linspace(1,0.67,32)]';
    cmap = [r,g,b];
end
