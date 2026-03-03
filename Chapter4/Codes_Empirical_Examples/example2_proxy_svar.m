%% ========================================================================
%  EXAMPLE 2: PROXY SVAR (EXTERNAL INSTRUMENTS)
%  Complete Analysis with IRFs and Comparison to Cholesky
%  ========================================================================
%  Data: U.S. FRED (GDPC1, GDPDEF, FEDFUNDS)
%  Instrument: Romer-Romer (2004) shocks, updated by Wieland-Yang (2020)
%  Sample: 1970:Q1 - 2007:Q4
%
%  Author: Alessia Paccagnini
%  Textbook: Macroeconometrics
%  ========================================================================
clear; clc; close all;

fprintf(repmat('=',1,70)); fprintf('\n');
fprintf('EXAMPLE 2: PROXY SVAR (EXTERNAL INSTRUMENTS)\n');
fprintf('Using Romer-Romer (2004) Monetary Policy Shocks\n');
fprintf(repmat('=',1,70)); fprintf('\n\n');

%% ========================================================================
%  SECTION 1: LOAD AND PREPARE DATA
%  ========================================================================
fprintf('[1/7] Loading FRED data...\n');

% --- Load GDP ---
gdp_raw  = readtable('GDPC1.xlsx', 'Sheet', 'Quarterly');
gdp_date = datetime(gdp_raw.observation_date);
gdp_val  = gdp_raw.GDPC1;

% --- Load GDP Deflator ---
def_raw  = readtable('GDPDEF.xlsx', 'Sheet', 'Quarterly');
def_date = datetime(def_raw.observation_date);
def_val  = def_raw.GDPDEF;

% --- Load Federal Funds Rate (monthly -> quarterly average) ---
ff_raw   = readtable('FEDFUNDS.xlsx', 'Sheet', 'Monthly');
ff_date  = datetime(ff_raw.observation_date);
ff_val   = ff_raw.FEDFUNDS;

% Convert monthly to quarterly (average)
ff_tt    = timetable(ff_date, ff_val);
ff_q     = retime(ff_tt, 'quarterly', 'mean');
ff_qdate = ff_q.ff_date;
ff_qval  = ff_q.ff_val;

% --- Compute annualized growth rates ---
gdp_growth = 400 * log(gdp_val(2:end) ./ gdp_val(1:end-1));
gdp_dates  = gdp_date(2:end);

inflation  = 400 * log(def_val(2:end) ./ def_val(1:end-1));
inf_dates  = def_date(2:end);

fprintf('   Done.\n');

%% ========================================================================
%  SECTION 2: LOAD ROMER-ROMER SHOCKS
%  ========================================================================
fprintf('\n[2/7] Loading Romer-Romer monetary policy shocks...\n');

rr_raw  = readtable('RR_monetary_shock_quarterly.xlsx');
rr_date = datetime(rr_raw.date);
rr_val  = rr_raw.resid_romer;

fprintf('   Source: Romer & Romer (2004), updated by Wieland & Yang (2020)\n');
fprintf('   Observations: %d\n', length(rr_val));

%% ========================================================================
%  SECTION 3: MERGE AND ALIGN DATA
%  ========================================================================
fprintf('\n[3/7] Merging datasets...\n');

% Build a common timetable
tt_gdp = timetable(gdp_dates, gdp_growth, 'VariableNames', {'gdp_growth'});
tt_inf = timetable(inf_dates, inflation, 'VariableNames', {'inflation'});
tt_ff  = timetable(ff_qdate, ff_qval, 'VariableNames', {'fedfunds'});
tt_rr  = timetable(rr_date, rr_val, 'VariableNames', {'rr_shock'});

tt = synchronize(tt_gdp, tt_inf, tt_ff, tt_rr, 'intersection');

% Restrict to 1970:Q1 - 2007:Q4
idx_sample = tt.gdp_dates >= datetime(1970,1,1) & tt.gdp_dates <= datetime(2007,12,31);
tt = tt(idx_sample, :);

% Remove rows with any NaN
valid = ~any(ismissing(tt), 2);
tt = tt(valid, :);

dates = tt.gdp_dates;
T_full = height(tt);

fprintf('   Final sample: %s to %s\n', datestr(dates(1)), datestr(dates(end)));
fprintf('   Observations: %d\n', T_full);

% Extract data matrix
Y = [tt.gdp_growth, tt.inflation, tt.fedfunds];
z_full = tt.rr_shock;

var_names  = {'GDP Growth', 'Inflation', 'Federal Funds Rate'};
[~, K] = size(Y);

%% ========================================================================
%  SECTION 4: VAR ESTIMATION
%  ========================================================================
fprintf('\n[4/7] Estimating VAR(4)...\n');

p = 4;  % number of lags
H = 40; % IRF horizon

[T_eff, B_hat, u_hat, Sigma_u, X_var] = estimate_var(Y, p);

fprintf('   VAR(%d) estimated\n', p);
fprintf('   Effective sample: %d observations\n', T_eff);

%% ========================================================================
%  SECTION 5: STRUCTURAL IDENTIFICATION
%  ========================================================================
fprintf('\n[5/7] Structural Identification...\n');

% --- A) Cholesky ---
P_chol = chol(Sigma_u, 'lower');

fprintf('\n   A) CHOLESKY IDENTIFICATION\n');
fprintf('   --------------------------------------------------\n');
fprintf('              eps_GDP    eps_pi    eps_MP\n');
row_labels = {'GDP      ', 'Inflation', 'FedFunds '};
for i = 1:K
    fprintf('   %s  %9.4f  %9.4f  %9.4f\n', row_labels{i}, P_chol(i,:));
end

% --- B) Proxy SVAR ---
fprintf('\n   B) PROXY SVAR IDENTIFICATION\n');
fprintf('   --------------------------------------------------\n');
fprintf('   Instrument: Romer-Romer (2004) narrative shocks\n');

% Align instrument with VAR residuals
z = z_full(p+1:end);

[b_proxy, se_proxy, n_valid, F_stat, R2] = proxy_svar_identification(u_hat, z);

fprintf('\n   First-stage diagnostics:\n');
fprintf('   F-statistic: %.1f\n', F_stat);
fprintf('   R-squared: %.4f\n', R2);
fprintf('   Observations: %d\n', n_valid);

if F_stat > 10
    fprintf('   Strong instrument (F > 10)\n');
else
    fprintf('   WARNING: Weak instrument (F < 10)\n');
end

fprintf('\n   Structural impact multipliers:\n');
fprintf('   %-20s %12s %12s %10s\n', 'Variable', 'Estimate', 'Std. Error', 't-stat');
fprintf('   %s\n', repmat('-',1,54));
for i = 1:K
    if se_proxy(i) > 0
        t_stat = b_proxy(i) / se_proxy(i);
    else
        t_stat = NaN;
    end
    fprintf('   %-20s %12.4f %12.4f %10.2f\n', var_names{i}, b_proxy(i), se_proxy(i), t_stat);
end

%% ========================================================================
%  SECTION 6: COMPUTE IRFs
%  ========================================================================
fprintf('\n\n[6/7] Computing IRFs...\n');

% --- Cholesky IRFs ---
IRF_chol = compute_irfs_cholesky(B_hat, P_chol, p, K, H);
irf_chol_mp = squeeze(IRF_chol(:, :, 3));  % MP shock = 3rd column

% --- Proxy SVAR IRFs ---
irf_proxy = compute_irfs_proxy(B_hat, b_proxy, p, K, H);

% --- Wild Bootstrap CIs ---
fprintf('   Computing bootstrap confidence intervals...\n');

rng(42);  % reproducibility

fprintf('   Proxy SVAR: wild bootstrap (Mammen weights, 2000 reps, 90%% CI)...\n');
[irf_med_proxy, irf_lo_proxy, irf_hi_proxy] = ...
    bootstrap_irfs_proxy(Y, z_full, p, H, 2000, 0.90);

fprintf('   Cholesky: residual bootstrap (500 reps, 90%% CI)...\n');
[irf_med_chol, irf_lo_chol, irf_hi_chol] = ...
    bootstrap_irfs_cholesky(Y, p, H, 500, 0.90);

fprintf('   Done.\n');

%% ========================================================================
%  SECTION 7: GENERATE FIGURES
%  ========================================================================
fprintf('\n\n[7/7] Generating figures...\n');

col_proxy = [46 134 171]/255;   % #2E86AB
col_chol  = [233 79 55]/255;    % #E94F37
horizons  = 0:H;

% --- FIGURE 1: Proxy SVAR IRFs ---
figure('Position', [100 100 1200 350]);
for i = 1:K
    subplot(1,3,i);
    fill([horizons, fliplr(horizons)], ...
         [irf_lo_proxy(:,i)', fliplr(irf_hi_proxy(:,i)')], ...
         col_proxy, 'FaceAlpha', 0.25, 'EdgeColor', 'none'); hold on;
    plot(horizons, irf_med_proxy(:,i), 'Color', col_proxy, 'LineWidth', 2.5);
    yline(0, 'k-', 'LineWidth', 0.8);
    xlabel('Quarters after shock');
    ylabel('Percentage points');
    title(['Response of ', var_names{i}]);
    xlim([0 H]);
    grid on; box on;
end
sgtitle({'Impulse Responses to Monetary Policy Shock', ...
         '(Proxy SVAR, Romer-Romer Instrument, 90% Wild Bootstrap CI, U.S. 1970-2007)'}, ...
         'FontSize', 13, 'FontWeight', 'bold');
exportgraphics(gcf, 'figure1_proxy_irf.png', 'Resolution', 150);
fprintf('   Saved: figure1_proxy_irf.png\n');

% --- FIGURE 2: Proxy vs Cholesky Comparison ---
figure('Position', [100 100 1200 350]);
for i = 1:K
    subplot(1,3,i);
    fill([horizons, fliplr(horizons)], ...
         [irf_lo_proxy(:,i)', fliplr(irf_hi_proxy(:,i)')], ...
         col_proxy, 'FaceAlpha', 0.20, 'EdgeColor', 'none'); hold on;
    h1 = plot(horizons, irf_med_proxy(:,i), 'Color', col_proxy, 'LineWidth', 2.5);
    h2 = plot(horizons, irf_med_chol(:,i),  'Color', col_chol,  'LineWidth', 2.5, 'LineStyle', '--');
    yline(0, 'k-', 'LineWidth', 0.8);
    xlabel('Quarters after shock');
    ylabel('Percentage points');
    title(['Response of ', var_names{i}]);
    xlim([0 H]);
    legend([h1 h2], {'Proxy SVAR (R-R)', 'Cholesky'}, 'Location', 'northeast');
    grid on; box on;
end
sgtitle({'Monetary Policy Shock: Proxy SVAR vs Cholesky', ...
         '(90% Bootstrap CI, U.S. Data 1970-2007)'}, ...
         'FontSize', 13, 'FontWeight', 'bold');
exportgraphics(gcf, 'figure2_proxy_vs_cholesky.png', 'Resolution', 150);
fprintf('   Saved: figure2_proxy_vs_cholesky.png\n');

fprintf('\n   All figures saved!\n');

%% ========================================================================
%  SUMMARY
%  ========================================================================
fprintf('\n'); fprintf(repmat('=',1,70)); fprintf('\n');
fprintf('SUMMARY RESULTS\n');
fprintf(repmat('=',1,70)); fprintf('\n');

fprintf('\n--- First-Stage Diagnostics ---\n');
fprintf('F-statistic: %.1f\n', F_stat);
fprintf('R-squared: %.4f\n', R2);

fprintf('\n--- Impact Multipliers Comparison ---\n');
fprintf('%-25s %12s %12s\n', 'Variable', 'Cholesky', 'Proxy SVAR');
fprintf('%s\n', repmat('-',1,50));
for i = 1:K
    fprintf('%-25s %12.4f %12.4f\n', var_names{i}, irf_chol_mp(1,i), irf_proxy(1,i));
end

fprintf('\n--- Comparison with Textbook Table 4.5 ---\n');
fprintf('%-25s %10s %11s %10s %11s\n', 'Variable', 'Book Chol', 'Book Proxy', 'Code Chol', 'Code Proxy');
fprintf('%s\n', repmat('-',1,68));
book_chol  = [0.00, 0.00, 0.90];
book_proxy = [0.78, 0.19, 1.00];
for i = 1:K
    fprintf('%-25s %10.2f %11.2f %10.4f %11.4f\n', ...
        var_names{i}, book_chol(i), book_proxy(i), irf_chol_mp(1,i), irf_proxy(1,i));
end

fprintf('\n'); fprintf(repmat('=',1,70)); fprintf('\n');
fprintf('DONE\n');
fprintf(repmat('=',1,70)); fprintf('\n');


%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function [T_eff, B_hat, u, Sigma_u, X] = estimate_var(Y, p)
% ESTIMATE_VAR  Estimate a VAR(p) by OLS
%   [T_eff, B_hat, u, Sigma_u, X] = estimate_var(Y, p)
%
%   Y      : T x K data matrix
%   p      : number of lags
%   B_hat  : (1+K*p) x K coefficient matrix [constant; lags]
%   u      : T_eff x K residual matrix
%   Sigma_u: K x K covariance matrix (dof-adjusted)
%   X      : T_eff x (1+K*p) regressor matrix

    [T, K] = size(Y);
    T_eff = T - p;

    % Dependent variable
    Y_dep = Y(p+1:end, :);

    % Regressors: constant + lags
    X = ones(T_eff, 1);
    for lag = 1:p
        X = [X, Y(p+1-lag:T-lag, :)]; %#ok<AGROW>
    end

    % OLS
    B_hat = (X' * X) \ (X' * Y_dep);
    u = Y_dep - X * B_hat;
    Sigma_u = (u' * u) / (T_eff - K*p - 1);
end


function [b, se, n, F_stat, R2] = proxy_svar_identification(u, z)
% PROXY_SVAR_IDENTIFICATION  Identify structural impact using external instrument
%   b_k = Cov(u_k, z) / Cov(u_mp, z),  normalized so b_mp = 1
%
%   u : T_eff x K  reduced-form residuals
%   z : T_eff x 1  external instrument (may contain NaN)

    K = size(u, 2);
    min_len = min(size(u,1), length(z));
    u = u(1:min_len, :);
    z = z(1:min_len);

    % Remove NaN observations
    valid = ~isnan(z);
    u = u(valid, :);
    z = z(valid);
    n = length(z);

    % Covariances
    z_dm = z - mean(z);
    cov_uz = zeros(K, 1);
    for k = 1:K
        u_dm = u(:,k) - mean(u(:,k));
        cov_uz(k) = (u_dm' * z_dm) / (n - 1);
    end
    cov_mp_z = cov_uz(K);   % last variable = policy equation

    % Impact coefficients (normalized: b_mp = 1)
    b = cov_uz / cov_mp_z;

    % First-stage regression: u_mp = alpha + beta*z + eta
    X_fs = [ones(n,1), z];
    beta_fs = X_fs \ u(:,K);
    resid_fs = u(:,K) - X_fs * beta_fs;

    TSS = sum((u(:,K) - mean(u(:,K))).^2);
    RSS = sum(resid_fs.^2);
    R2 = 1 - RSS / TSS;
    F_stat = (R2 / 1) / ((1 - R2) / (n - 2));

    % Bootstrap standard errors for b
    n_boot = 500;
    b_boot = NaN(n_boot, K);
    for ib = 1:n_boot
        idx = randi(n, n, 1);
        z_b = z(idx);
        u_b = u(idx, :);
        z_dm_b = z_b - mean(z_b);
        cov_b = zeros(K,1);
        for k = 1:K
            cov_b(k) = ((u_b(:,k) - mean(u_b(:,k)))' * z_dm_b) / (n - 1);
        end
        if abs(cov_b(K)) > 1e-10
            b_boot(ib, :) = cov_b / cov_b(K);
        end
    end
    se = nanstd(b_boot, 0, 1)';
end


function Phi = compute_ma_coefficients(B_hat, p, K, H)
% COMPUTE_MA_COEFFICIENTS  MA representation Phi_h from VAR coefficients
%   B_hat: (1+K*p) x K   (first row = constant)
%   Returns Phi: (H+1) x K x K

    % Extract companion-form lag matrices B_1, ..., B_p  (each K x K)
    B = zeros(K, K, p);
    for j = 1:p
        rows = (1 + (j-1)*K + 1) : (1 + j*K);   % skip constant
        B(:,:,j) = B_hat(rows, :)';               % transpose to K x K
    end

    Phi = zeros(H+1, K, K);
    Phi(1,:,:) = eye(K);  % Phi_0 = I

    for h = 1:H
        tmp = zeros(K);
        for j = 1:min(h, p)
            tmp = tmp + B(:,:,j) * squeeze(Phi(h-j+1, :, :));
        end
        Phi(h+1, :, :) = tmp;
    end
end


function IRF = compute_irfs_cholesky(B_hat, P, p, K, H)
% COMPUTE_IRFS_CHOLESKY  IRFs using Cholesky identification
%   Returns IRF: (H+1) x K x K

    Phi = compute_ma_coefficients(B_hat, p, K, H);
    IRF = zeros(H+1, K, K);
    for h = 0:H
        IRF(h+1, :, :) = squeeze(Phi(h+1, :, :)) * P;
    end
end


function IRF = compute_irfs_proxy(B_hat, b, p, K, H)
% COMPUTE_IRFS_PROXY  IRFs for Proxy SVAR (single identified shock)
%   b: K x 1 structural impact vector
%   Returns IRF: (H+1) x K

    Phi = compute_ma_coefficients(B_hat, p, K, H);
    IRF = zeros(H+1, K);
    for h = 0:H
        IRF(h+1, :) = squeeze(Phi(h+1, :, :)) * b;
    end
end


function [irf_median, irf_lower, irf_upper] = ...
    bootstrap_irfs_proxy(Y, z_full, p, H, n_boot, ci)
% BOOTSTRAP_IRFS_PROXY  Wild bootstrap for Proxy SVAR IRFs
%   Uses Mammen (1993) two-point weights.
%   Same weights for residuals and instrument (preserves covariance).
%   Returns bootstrap median + percentile bands.

    [T, K] = size(Y);

    % Baseline VAR
    [T_eff, B_base, u_base, ~, X_base] = estimate_var(Y, p);

    % Align instrument
    z_aligned = z_full(p+1:end);
    min_len = min(size(u_base,1), length(z_aligned));
    z_trim = z_aligned(1:min_len);

    % Mammen probabilities
    p_mammen = (sqrt(5) + 1) / (2*sqrt(5));
    w_lo = -(sqrt(5) - 1) / 2;
    w_hi =  (sqrt(5) + 1) / 2;

    irfs_boot = NaN(n_boot, H+1, K);
    n_fail = 0;

    for ib = 1:n_boot
        try
            % Mammen weights
            w = w_hi * ones(T_eff, 1);
            w(rand(T_eff,1) < p_mammen) = w_lo;

            % Wild bootstrap residuals
            u_star = u_base .* w;

            % Reconstruct data
            Y_star = zeros(T, K);
            Y_star(1:p, :) = Y(1:p, :);
            for t = 1:T_eff
                Y_star(p+t, :) = X_base(t,:) * B_base + u_star(t,:);
            end

            % Re-estimate VAR
            [~, B_b, u_b, ~, ~] = estimate_var(Y_star, p);

            % Wild bootstrap instrument (same weights)
            w_z = w(1:min_len);
            z_star = z_trim .* w_z;

            % Re-identify
            [b_est, ~, ~, ~, ~] = proxy_svar_identification(u_b, z_star);

            % Bootstrap IRFs
            irfs_boot(ib, :, :) = compute_irfs_proxy(B_b, b_est, p, K, H);
        catch
            n_fail = n_fail + 1;
        end
    end

    fprintf('   Wild bootstrap: %d replications, %d failures\n', n_boot, n_fail);

    alpha = (1 - ci) / 2;
    irf_median = squeeze(nanmedian(irfs_boot, 1));
    irf_lower  = squeeze(prctile_nan(irfs_boot, alpha*100, 1));
    irf_upper  = squeeze(prctile_nan(irfs_boot, (1-alpha)*100, 1));
end


function [irf_median, irf_lower, irf_upper] = ...
    bootstrap_irfs_cholesky(Y, p, H, n_boot, ci)
% BOOTSTRAP_IRFS_CHOLESKY  Residual bootstrap for Cholesky IRFs
%   Returns bootstrap median + percentile bands for the MP shock.

    [T, K] = size(Y);
    [T_eff, B_orig, u_orig, ~, X_orig] = estimate_var(Y, p);

    irfs_boot = NaN(n_boot, H+1, K);

    for ib = 1:n_boot
        try
            idx = randi(T_eff, T_eff, 1);
            u_boot = u_orig(idx, :);

            Y_boot = zeros(T, K);
            Y_boot(1:p, :) = Y(1:p, :);
            Y_boot(p+1:end, :) = X_orig * B_orig + u_boot;

            [~, B_b, ~, Sigma_b, ~] = estimate_var(Y_boot, p);
            P_b = chol(Sigma_b, 'lower');
            IRF_b = compute_irfs_cholesky(B_b, P_b, p, K, H);
            irfs_boot(ib, :, :) = IRF_b(:, :, 3);  % MP shock
        catch
            % leave as NaN
        end
    end

    alpha = (1 - ci) / 2;
    irf_median = squeeze(nanmedian(irfs_boot, 1));
    irf_lower  = squeeze(prctile_nan(irfs_boot, alpha*100, 1));
    irf_upper  = squeeze(prctile_nan(irfs_boot, (1-alpha)*100, 1));
end


function q = prctile_nan(X, pct, dim)
% PRCTILE_NAN  Percentile ignoring NaN (along dimension dim)
%   Workaround for older MATLAB versions where prctile doesn't skip NaN.
    sz = size(X);
    if dim == 1
        q = zeros(1, sz(2), sz(3));
        for j = 1:sz(2)
            for k = 1:sz(3)
                vals = X(:, j, k);
                vals = vals(~isnan(vals));
                if ~isempty(vals)
                    q(1, j, k) = prctile(vals, pct);
                else
                    q(1, j, k) = NaN;
                end
            end
        end
        q = squeeze(q);
    end
end
