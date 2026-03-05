%% Replicating Bernanke, Boivin & Eliasz (2005) FAVAR Analysis
% =========================================================================
% Textbook: Macroeconometrics
% Author:   Alessia Paccagnini
%
% This script:
%   1. Loads FRED-MD dataset (McCracken & Ng, 2016)
%   2. Applies FRED-MD transformation codes for stationarity
%   3. Extracts 5 factors via PCA (excluding FFR)
%   4. Estimates 3-variable VAR (IP, CPI, FFR)
%   5. Estimates FAVAR (5 factors + FFR) — two-step approach
%   6. Compares IRFs across subsamples (Pre-Volcker / Great Moderation / Full)
%   7. Rolling out-of-sample forecasting: FAVAR vs VAR vs Random Walk
%   8. Clark-West (2007) test for nested models
%   9. Giacomini-Rossi (2010) fluctuation test for forecast stability
%
% References:
%   Bernanke, Boivin & Eliasz (2005), QJE
%   McCracken & Ng (2016), JBES
%   Clark & West (2007), JoE
%   Giacomini & Rossi (2010), ReStud
%
% Data: 2025-12-MD.csv  (FRED-MD monthly)
% Sample: 1962-01-01 to 2007-12-01 (pre-crisis, consistent with textbook)
% =========================================================================

clear; clc; close all;

%% Global Settings
n_factors     = 5;
p             = 12;        % VAR lag order (monthly → 1 year)
horizon       = 48;        % IRF horizon (months)
n_boot        = 500;       % Bootstrap replications
alpha_ci      = 0.10;      % CI significance level
normalize_shk = 0.25;      % Normalize to 25bp shock
initial_win   = 120;       % Initial window for rolling forecasts (10 yrs)

% Subsamples (consistent with textbook Section 8.13.5 / Figure 8.1)
SUBSAMPLES = {
    '1962-01-01', '1984-12-01', 'Pre-Volcker Era (1962-1984)';
    '1985-01-01', '2007-12-01', 'Great Moderation (1985-2007)';
    '1962-01-01', '2007-12-01', 'Full Sample (1962-2007)'
};

fprintf('%s\n', repmat('=',1,70));
fprintf('BBE (2005) FAVAR Replication with FRED-MD Data\n');
fprintf('%s\n\n', repmat('=',1,70));

%% =========================================================================
%  SECTION 1: LOAD DATA
%% =========================================================================
fprintf('1. Loading FRED-MD data...\n');

DATA_FILE = '2025-12-MD.csv';
opts = detectImportOptions(DATA_FILE);
opts.VariableNamingRule = 'preserve';
% Fix ambiguous date format warning (MM/dd/uuuu vs dd/MM/uuuu)
opts = setvaropts(opts, opts.VariableNames{1}, 'InputFormat', 'MM/dd/uuuu');
raw = readtable(DATA_FILE, opts);

var_names  = raw.Properties.VariableNames(2:end);
tcodes     = table2array(raw(1, 2:end));
% Extract dates (row 1 = tcode row, so skip it; rows 2:end = actual data)
% After setvaropts, the column may be datetime or char depending on MATLAB version
date_col = raw{2:end, 1};
if isa(date_col, 'datetime')
    dates = date_col;
elseif iscell(date_col)
    dates = datetime(date_col, 'InputFormat', 'MM/dd/uuuu');
else
    dates = datetime(string(date_col), 'InputFormat', 'MM/dd/uuuu');
end
data_raw   = table2array(raw(2:end, 2:end));

fprintf('   Raw data : %d obs, %d variables\n', size(data_raw,1), size(data_raw,2));
fprintf('   Range    : %s — %s\n', datestr(dates(1),'yyyy-mm-dd'), ...
        datestr(dates(end),'yyyy-mm-dd'));

%% =========================================================================
%  SECTION 2: TRANSFORM DATA
%% =========================================================================
fprintf('\n2. Applying transformations...\n');

[T_raw, N_raw] = size(data_raw);
data_trans = nan(T_raw, N_raw);
for j = 1:N_raw
    data_trans(:, j) = transform_series(data_raw(:, j), tcodes(j));
end

%% =========================================================================
%  SECTION 3: SUBSAMPLE IRF ANALYSIS
%% =========================================================================
fprintf('\n3. Subsample IRF analysis...\n');

results_irf = struct();

for s = 1:size(SUBSAMPLES, 1)
    sd    = datetime(SUBSAMPLES{s,1});
    ed    = datetime(SUBSAMPLES{s,2});
    sname = SUBSAMPLES{s,3};

    fprintf('\n%s\n  %s\n  %s  →  %s\n%s\n', ...
        repmat('=',1,60), sname, SUBSAMPLES{s,1}, SUBSAMPLES{s,2}, repmat('=',1,60));

    % Subset dates and remove columns with > 10% missing
    idx  = dates >= sd & dates <= ed;
    Ds   = data_trans(idx, :);
    miss = sum(isnan(Ds)) / sum(idx);
    keep = miss < 0.10;
    Ds   = Ds(:, keep);
    vn   = var_names(keep);

    % Remove rows with any NaN (explicit — avoids silent sample shrinkage)
    complete = ~any(isnan(Ds), 2);
    Ds = Ds(complete, :);
    T  = size(Ds, 1);

    % Find key variable indices
    idx_IP  = find(strcmp(vn, 'INDPRO'));
    idx_CPI = find(strcmp(vn, 'CPIAUCSL'));
    idx_FFR = find(strcmp(vn, 'FEDFUNDS'));

    % ---- Factor extraction (exclude FFR) ----
    fac_cols  = setdiff(1:size(Ds,2), idx_FFR);
    Xf        = Ds(:, fac_cols);
    vn_fac    = vn(fac_cols);
    mu_f      = mean(Xf);
    sig_f     = std(Xf);
    % Drop zero-variance columns (would cause Inf/NaN after standardisation)
    nonzero   = sig_f > 0;
    Xf        = Xf(:, nonzero);
    vn_fac    = vn_fac(nonzero);
    mu_f      = mu_f(nonzero);
    sig_f     = sig_f(nonzero);
    Xs        = (Xf - mu_f) ./ sig_f;
    % Safety check: replace any remaining NaN/Inf with 0
    Xs(~isfinite(Xs)) = 0;
    [U, S, V] = svds(Xs, n_factors);
    F_hat     = U * S;         % T × n_factors
    loadings  = V;             % n_vars_factor × n_factors
    var_exp   = diag(S).^2 / sum(sum(Xs.^2));

    fprintf('   Variables : %d   |   T = %d\n', size(Xf,2), T);
    fprintf('   Var. exp. : %.1f%%\n', sum(var_exp)*100);

    % VAR and FAVAR data matrices
    Y_var   = [Ds(:,idx_IP), Ds(:,idx_CPI), Ds(:,idx_FFR)];
    Y_favar = [F_hat, Ds(:,idx_FFR)];

    % ---- VAR IRFs with bootstrap ----
    [irf_var, irf_lo, irf_hi] = bootstrap_var_irf(Y_var, p, horizon, ...
        n_boot, 3, alpha_ci, normalize_shk);

    % ---- FAVAR IRFs ----
    K_fav   = n_factors + 1;
    [B_fav, Sig_fav, ~] = estimate_var(Y_favar, p);
    irf_fav_raw = compute_irf(B_fav, Sig_fav, K_fav, p, horizon, ...
                               K_fav, normalize_shk);

    % Recover IP and CPI from factor space (eq. 8.29)
    idx_IP_f  = find(strcmp(vn_fac, 'INDPRO'));
    idx_CPI_f = find(strcmp(vn_fac, 'CPIAUCSL'));

    irf_fav_IP  = irf_fav_raw(:, 1:n_factors) * loadings(idx_IP_f,  :)' ...
                  * sig_f(idx_IP_f);
    irf_fav_CPI = irf_fav_raw(:, 1:n_factors) * loadings(idx_CPI_f, :)' ...
                  * sig_f(idx_CPI_f);
    irf_fav_FFR = irf_fav_raw(:, end);

    % Store
    results_irf(s).name         = sname;
    results_irf(s).T            = T;
    results_irf(s).irf_var      = irf_var;
    results_irf(s).irf_lo       = irf_lo;
    results_irf(s).irf_hi       = irf_hi;
    results_irf(s).irf_fav_IP   = irf_fav_IP;
    results_irf(s).irf_fav_CPI  = irf_fav_CPI;
    results_irf(s).irf_fav_FFR  = irf_fav_FFR;
    results_irf(s).Y_var        = Y_var;
    results_irf(s).F_hat        = F_hat;
    results_irf(s).Y_favar      = Y_favar;

    var_cpi_pos   = sum(irf_var(1:12, 2) > 0);
    favar_cpi_pos = sum(irf_fav_CPI(1:12) > 0);
    results_irf(s).var_cpi_pos   = var_cpi_pos;
    results_irf(s).favar_cpi_pos = favar_cpi_pos;
    fprintf('   Price puzzle — VAR: %d/12 pos  FAVAR: %d/12 pos\n', ...
            var_cpi_pos, favar_cpi_pos);
end

%% =========================================================================
%  SECTION 4: PLOT FIGURE 8.1 — IRF COMPARISON
%% =========================================================================
fprintf('\n4. Plotting IRF comparison (Figure 8.1)...\n');

months = 0:(horizon-1);
col_blue = [0.122 0.467 0.706];
col_red  = [0.839 0.153 0.157];

figure('Position', [100, 100, 1400, 800]);

for col = 1:3
    res = results_irf(col);

    % --- CPI row ---
    subplot(2, 3, col);
    hold on;
    fill([months, fliplr(months)], ...
         [res.irf_lo(:,2)'*100, fliplr(res.irf_hi(:,2)'*100)], ...
         col_blue, 'FaceAlpha', 0.20, 'EdgeColor', 'none');
    plot(months, res.irf_var(:,2)*100, '-',  'Color', col_blue, 'LineWidth', 2, ...
         'DisplayName', 'VAR');
    plot(months, res.irf_fav_CPI*100,  '--', 'Color', col_red,  'LineWidth', 2, ...
         'DisplayName', 'FAVAR');
    yline(0, 'k-', 'LineWidth', 0.7);
    % Shade first 12 months (price puzzle region)
    yl = ylim;
    patch([0 12 12 0], [yl(1) yl(1) yl(2) yl(2)], 'r', ...
          'FaceAlpha', 0.08, 'EdgeColor', 'none');
    text(1, yl(2)*0.88, sprintf('VAR: %d/12 pos.', res.var_cpi_pos), ...
         'Color', col_blue, 'FontSize', 9);
    title(res.name, 'FontWeight', 'bold');
    if col == 1; ylabel('CPI Response (%)'); legend('Location','northeast'); end
    xlim([0, horizon-1]); grid on; hold off;

    % --- IP row ---
    subplot(2, 3, col + 3);
    hold on;
    fill([months, fliplr(months)], ...
         [res.irf_lo(:,1)'*100, fliplr(res.irf_hi(:,1)'*100)], ...
         col_blue, 'FaceAlpha', 0.20, 'EdgeColor', 'none');
    plot(months, res.irf_var(:,1)*100, '-',  'Color', col_blue, 'LineWidth', 2);
    plot(months, res.irf_fav_IP*100,   '--', 'Color', col_red,  'LineWidth', 2);
    yline(0, 'k-', 'LineWidth', 0.7);
    if col == 1; ylabel('IP Response (%)'); end
    xlabel('Months');
    xlim([0, horizon-1]); grid on; hold off;
end

sgtitle('VAR vs FAVAR Across Monetary Policy Regimes — Response to 25bp FFR Shock', ...
        'FontSize', 13, 'FontWeight', 'bold');
saveas(gcf, 'bbe_favar_irf.pdf');
saveas(gcf, 'bbe_favar_irf.png');
close;
fprintf('   Saved: bbe_favar_irf.pdf/.png\n');

%% =========================================================================
%  SECTION 5: PRICE PUZZLE SUMMARY
%% =========================================================================
fprintf('\n%s\nPRICE PUZZLE SUMMARY\n%s\n', repmat('=',1,65), repmat('=',1,65));
fprintf('%-35s %5s %9s %10s\n', 'Sample', 'T', 'VAR pos', 'FAVAR pos');
fprintf('%s\n', repmat('-',1,65));
for s = 1:3
    fprintf('%-35s %5d %6d/12 %7d/12\n', ...
            results_irf(s).name, results_irf(s).T, ...
            results_irf(s).var_cpi_pos, results_irf(s).favar_cpi_pos);
end

%% =========================================================================
%  SECTION 6: ROLLING FORECAST COMPARISON
%% =========================================================================
fprintf('\n5. Rolling forecasting on Full Sample (h=1, initial_win=%d)...\n', ...
        initial_win);

full    = results_irf(3);    % Full Sample
Y_var_f = full.Y_var;
F_hat_f = full.F_hat;
Y_fav_f = full.Y_favar;

[fc_rw, fc_var, fc_favar, actuals] = rolling_forecasts_fn( ...
    Y_var_f, Y_fav_f, p, 1, initial_win);

fprintf('   Forecast origins: %d\n', length(actuals));

%% =========================================================================
%  SECTION 7: FORECAST EVALUATION TESTS
%% =========================================================================
fprintf('\n6. Computing forecast metrics and test statistics...\n');

% Point accuracy
e_rw    = actuals - fc_rw;
e_var   = actuals - fc_var;
e_favar = actuals - fc_favar;

rmse_rw    = sqrt(mean(e_rw.^2));
rmse_var   = sqrt(mean(e_var.^2));
rmse_favar = sqrt(mean(e_favar.^2));
mae_rw     = mean(abs(e_rw));
mae_var    = mean(abs(e_var));
mae_favar  = mean(abs(e_favar));

% Clark-West tests
[cw_var_rw,    p_cw_vr]  = clark_west_test(actuals, fc_rw,  fc_var);
[cw_favar_rw,  p_cw_fr]  = clark_west_test(actuals, fc_rw,  fc_favar);
[cw_favar_var, p_cw_fv]  = clark_west_test(actuals, fc_var, fc_favar);

% Giacomini-Rossi fluctuation tests
[gr_var_rw,    cv_gr, gr_s_vr,  ti_vr]  = giacomini_rossi_test(actuals, fc_rw,  fc_var);
[gr_favar_rw,  cv_gr, gr_s_fr,  ti_fr]  = giacomini_rossi_test(actuals, fc_rw,  fc_favar);
[gr_favar_var, cv_gr, gr_s_fv,  ti_fv]  = giacomini_rossi_test(actuals, fc_var, fc_favar);

% Print table
fprintf('\n%s\n', repmat('=',1,65));
fprintf('OUT-OF-SAMPLE FORECAST EVALUATION  (target: FFR, h=1)\n');
fprintf('%s\n', repmat('=',1,65));
fprintf('%-10s %10s %10s\n', 'Model', 'RMSE', 'MAE');
fprintf('%s\n', repmat('-',1,32));
fprintf('%-10s %10.4f %10.4f\n', 'RW',    rmse_rw,    mae_rw);
fprintf('%-10s %10.4f %10.4f\n', 'VAR',   rmse_var,   mae_var);
fprintf('%-10s %10.4f %10.4f\n', 'FAVAR', rmse_favar, mae_favar);

fprintf('\n--- Clark-West (2007) [H1: model > benchmark, one-sided] ---\n');
fprintf('  %-20s %10s %10s %5s\n', 'Comparison','CW stat','p-value','sig');
fprintf('  %s\n', repmat('-',1,50));
print_cw_row('VAR vs RW',    cw_var_rw,    p_cw_vr);
print_cw_row('FAVAR vs RW',  cw_favar_rw,  p_cw_fr);
print_cw_row('FAVAR vs VAR', cw_favar_var, p_cw_fv);

fprintf('\n--- Giacomini-Rossi (2010) Fluctuation Test ---\n');
fprintf('  %-20s %10s %10s %8s\n', 'Comparison','GR stat','CV (10%)','stable?');
fprintf('  %s\n', repmat('-',1,52));
print_gr_row('VAR vs RW',    gr_var_rw,    cv_gr);
print_gr_row('FAVAR vs RW',  gr_favar_rw,  cv_gr);
print_gr_row('FAVAR vs VAR', gr_favar_var, cv_gr);
fprintf('%s\n', repmat('=',1,65));

%% =========================================================================
%  SECTION 8: FORECAST EVALUATION FIGURE
%% =========================================================================
fprintf('\n7. Saving forecast evaluation figure...\n');

figure('Position', [100, 100, 1400, 600]);

% ---- Left: RMSE bars ----
subplot(1, 2, 1);
bar_vals = [rmse_rw; rmse_var; rmse_favar];
bar_h = bar(1:3, bar_vals, 0.6, 'FaceColor', 'flat');
bar_h.CData = [col_blue; 0.173 0.627 0.173; col_red];
hold on;
for k = 1:3
    text(k, bar_vals(k) + max(bar_vals)*0.015, ...
         sprintf('%.4f', bar_vals(k)), 'HorizontalAlignment', 'center', ...
         'FontSize', 10);
end
% CW annotations
y_top = max(bar_vals) * 1.12;
text(1.5, y_top, sprintf('CW: %.2f%s', cw_var_rw,    stars(p_cw_vr)),  'FontSize', 8, 'Color', [0.4 0.4 0.4], 'HorizontalAlignment', 'center');
text(2.5, y_top, sprintf('CW: %.2f%s', cw_favar_rw,  stars(p_cw_fr)),  'FontSize', 8, 'Color', [0.4 0.4 0.4], 'HorizontalAlignment', 'center');
text(2.0, y_top*1.06, sprintf('CW: %.2f%s', cw_favar_var, stars(p_cw_fv)), 'FontSize', 8, 'Color', [0.4 0.4 0.4], 'HorizontalAlignment', 'center');
hold off;
set(gca, 'XTickLabel', {'RW','VAR','FAVAR'});
ylabel('RMSE (FFR, percentage points)');
title('Forecast Accuracy', 'FontWeight', 'bold');
ylim([0, max(bar_vals)*1.25]); grid on;

% ---- Right: GR fluctuation paths ----
subplot(1, 2, 2);
hold on;
plot(ti_vr, gr_s_vr, '-',  'Color', col_blue, 'LineWidth', 1.8, ...
     'DisplayName', 'VAR vs RW');
plot(ti_fr, gr_s_fr, '--', 'Color', col_red,  'LineWidth', 1.8, ...
     'DisplayName', 'FAVAR vs RW');
plot(ti_fv, gr_s_fv, ':',  'Color', [0.173 0.627 0.173], 'LineWidth', 1.8, ...
     'DisplayName', 'FAVAR vs VAR');
yline( cv_gr, 'k--', 'LineWidth', 1.5);
yline(-cv_gr, 'k--', 'LineWidth', 1.5);
yline(0, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.7);
hold off;
xlabel('Rolling window end (observation)');
ylabel('GR statistic');
title(sprintf('Giacomini-Rossi (2010) Fluctuation Test\nDashed lines: ±%.2f CV (10%%)', cv_gr), ...
      'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;

sgtitle('Out-of-Sample Forecast Comparison: FFR | RW vs VAR vs FAVAR', ...
        'FontSize', 13, 'FontWeight', 'bold');
saveas(gcf, 'bbe_favar_forecast_eval.pdf');
saveas(gcf, 'bbe_favar_forecast_eval.png');
close;
fprintf('   Saved: bbe_favar_forecast_eval.pdf/.png\n');

%% =========================================================================
%  SUMMARY
%% =========================================================================
fprintf('\n%s\nDone. Files generated:\n', repmat('=',1,70));
fprintf('  bbe_favar_irf.pdf/.png           — Figure 8.1 (IRF comparison)\n');
fprintf('  bbe_favar_forecast_eval.pdf/.png — Forecast evaluation figure\n');
fprintf('%s\n', repmat('=',1,70));


%% =========================================================================
%%  HELPER FUNCTIONS
%% =========================================================================

function y = transform_series(x, tcode)
% FRED-MD transformation codes (1-7).
    small = 1e-6;
    n = length(x);
    y = nan(n, 1);
    switch tcode
        case 1,  y = x;
        case 2,  y(2:end) = diff(x);
        case 3,  y(3:end) = diff(x, 2);
        case 4,  y = log(max(x, small));
        case 5,  y(2:end) = diff(log(max(x, small)));
        case 6,  y(3:end) = diff(log(max(x, small)), 2);
        case 7
            pct = x(2:end) ./ x(1:end-1) - 1;
            y(3:end) = diff(pct);
        otherwise, y = x;
    end
end

function [B, Sigma, resid] = estimate_var(Y, p)
% Estimate VAR(p) by OLS.  B is K x (Kp+1), intercept in column 1.
    [T, K] = size(Y);
    X = ones(T-p, 1);
    for i = 1:p
        X = [X, Y(p+1-i:T-i, :)]; %#ok<AGROW>
    end
    Ydep  = Y(p+1:end, :);
    B     = (X'*X \ (X'*Ydep))';    % K x (Kp+1)
    resid = Ydep - X*B';
    Sigma = (resid'*resid) / (T-p - K*p - 1);
end

function C = var_companion(B, K, p)
% Companion matrix.  B is K x (Kp+1), drop intercept column first.
    C = zeros(K*p, K*p);
    C(1:K, :) = B(:, 2:end);
    if p > 1
        C(K+1:end, 1:K*(p-1)) = eye(K*(p-1));
    end
end

function irf = compute_irf(B, Sigma, K, p, horizon, shock_var, normalize_to)
% Cholesky IRF, normalized to normalize_to if supplied.
    P     = chol(Sigma, 'lower');
    Cmat  = var_companion(B, K, p);
    e     = zeros(K, 1); e(shock_var) = 1;
    imp   = P * e;
    if ~isempty(normalize_to)
        imp = imp * (normalize_to / imp(shock_var));
    end
    irf   = zeros(horizon, K);
    state = zeros(K*p, 1);
    state(1:K) = imp;
    for h = 1:horizon
        irf(h, :) = state(1:K)';
        state = Cmat * state;
    end
end

function [irf_pt, irf_lo, irf_hi] = bootstrap_var_irf(Y, p, horizon, ...
        n_boot, shock_var, alpha, normalize_to)
% Residual-bootstrap CIs for Cholesky IRFs.
% Bug fix: explicit lag stacking avoids flatten/reverse ordering errors.
    [T, K] = size(Y);
    [B, Sigma, resid] = estimate_var(Y, p);
    irf_pt = compute_irf(B, Sigma, K, p, horizon, shock_var, normalize_to);
    irf_store = zeros(n_boot, horizon, K);

    for b = 1:n_boot
        idx_b  = randi(size(resid,1), size(resid,1), 1);
        resid_b = resid(idx_b, :);
        Yb = zeros(T, K); Yb(1:p, :) = Y(1:p, :);
        for t = p+1:T
            % Explicit lag stacking
            Ylag = reshape(flipud(Yb(t-p:t-1, :))', [], 1);
            Yb(t, :) = B(:,1)' + (B(:,2:end) * Ylag)' + resid_b(t-p, :);
        end
        try
            [Bb, Sb, ~] = estimate_var(Yb, p);
            irf_store(b,:,:) = compute_irf(Bb, Sb, K, p, horizon, ...
                                            shock_var, normalize_to);
        catch
            irf_store(b,:,:) = irf_pt;
        end
    end
    irf_lo = squeeze(prctile(irf_store, alpha/2*100,  1));
    irf_hi = squeeze(prctile(irf_store, (1-alpha/2)*100, 1));
end

function [fc_rw, fc_var, fc_favar, actuals] = rolling_forecasts_fn( ...
        Y_var, Y_favar, p, h, init_win)
% Rolling h-step-ahead forecasts: RW, VAR, FAVAR.
    T = size(Y_var, 1);
    K_var = size(Y_var, 2);
    K_fav = size(Y_favar, 2);
    fc_rw = []; fc_var = []; fc_favar = []; actuals = [];

    for t = init_win:(T-h)
        actual = Y_var(t+h, 3);
        actuals = [actuals; actual]; %#ok<AGROW>

        % Random Walk
        fc_rw = [fc_rw; Y_var(t, 3)]; %#ok<AGROW>

        % VAR
        try
            [Bv, ~, ~] = estimate_var(Y_var(1:t,:), p);
            st = zeros(K_var*p, 1); st(1:K_var) = Y_var(t,:)';
            fc_h = Y_var(t, :)';
            for s_ = 1:h
                fc_h = Bv(:,1) + Bv(:,2:end)*st;
                st   = [fc_h; st(1:K_var*(p-1))];
            end
            fc_var = [fc_var; fc_h(3)]; %#ok<AGROW>
        catch
            fc_var = [fc_var; Y_var(t,3)]; %#ok<AGROW>
        end

        % FAVAR
        try
            [Bf, ~, ~] = estimate_var(Y_favar(1:t,:), p);
            st = zeros(K_fav*p, 1); st(1:K_fav) = Y_favar(t,:)';
            fc_h = Y_favar(t, :)';
            for s_ = 1:h
                fc_h = Bf(:,1) + Bf(:,2:end)*st;
                st   = [fc_h; st(1:K_fav*(p-1))];
            end
            fc_favar = [fc_favar; fc_h(end)]; %#ok<AGROW>
        catch
            fc_favar = [fc_favar; Y_var(t,3)]; %#ok<AGROW>
        end
    end
end

function [cw_stat, pval] = clark_west_test(actual, fc_bench, fc_model)
% Clark & West (2007) MSPE-adjusted test.
% H0: benchmark forecasts as well as larger model (nested).
% Reference: Clark & West (2007), Journal of Econometrics 138, 291-311.
    e1   = actual - fc_bench;
    e2   = actual - fc_model;
    ct   = e1.^2 - (e2.^2 - (fc_bench - fc_model).^2);
    cbar = mean(ct);
    se   = std(ct) / sqrt(length(ct));
    cw_stat = cbar / se;
    pval    = 1 - normcdf(cw_stat);   % one-sided
end

function [gr_stat, cv, gr_series, time_idx] = giacomini_rossi_test( ...
        actual, fc1, fc2, window, alpha)
% Giacomini & Rossi (2010) fluctuation test for forecast stability.
% Reference: Giacomini & Rossi (2010), Review of Economic Studies 77, 530-561.
    if nargin < 4 || isempty(window)
        window = max(floor(length(actual)*0.2), 10);
    end
    if nargin < 5; alpha = 0.10; end

    e1 = actual - fc1; e2 = actual - fc2;
    d  = e1.^2 - e2.^2;
    n  = length(d);
    d_dm = d - mean(d);
    bw   = window - 1;
    lrv  = var(d);
    for k = 1:bw
        w   = 1 - k/(bw+1);
        lrv = lrv + 2*w*mean(d_dm(k+1:end) .* d_dm(1:end-k));
    end
    lrv = max(lrv, 1e-12);

    rolling = arrayfun(@(t) mean(d(t-window+1:t)), window:n);
    gr_series = sqrt(window) * rolling / sqrt(lrv);
    gr_stat   = max(abs(gr_series));
    time_idx  = window:n;

    cv_map = containers.Map({0.10, 0.05, 0.01}, {2.49, 2.80, 3.40});
    cv = cv_map(alpha);
end

function s = stars(pval)
    if pval < 0.01;      s = '***';
    elseif pval < 0.05;  s = '**';
    elseif pval < 0.10;  s = '*';
    else;                s = '';
    end
end

function print_cw_row(label, stat, pval)
    fprintf('  %-20s %10.3f %10.3f %5s\n', label, stat, pval, stars(pval));
end

function print_gr_row(label, gr_stat, cv)
    stable = 'NO'; if gr_stat < cv; stable = 'YES'; end
    fprintf('  %-20s %10.3f %10.3f %8s\n', label, gr_stat, cv, stable);
end
