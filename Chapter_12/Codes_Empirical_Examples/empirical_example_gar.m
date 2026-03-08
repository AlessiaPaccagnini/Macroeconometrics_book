% =============================================================================
% Growth-at-Risk (GaR) with the Chicago Fed NFCI
% =============================================================================
% Textbook : Macroeconometrics
% Author   : Alessia Paccagnini
% Chapter  : 12 — Quantile Regression and Growth-at-Risk
% Section  : 12.7 — Empirical Application: US Growth-at-Risk with NFCI
%
% Empirical specification (eq. 12.9):
%   Q_tau(Dy_{t+4} | Omega_t) = alpha(tau) + beta_y(tau)*Dy_t
%                              + beta_pi(tau)*pi_t + beta_f(tau)*NFCI_t
%
% Estimation sample : 1971Q1-2024Q3   (N = 215)
% Forecast horizon  : h = 4 quarters ahead
% Quantiles         : [0.05 0.10 0.25 0.50 0.75 0.90 0.95] + OLS
%
% Files needed (place in working directory):
%   GDPC1.xlsx   - Real GDP, quarterly  (sheet: Quarterly)
%   GDPDEF.xlsx  - GDP Deflator, qtrly  (sheet: Quarterly)
%   NFCI.csv     - NFCI weekly, FRED    (cols: observation_date, NFCI)
%
% Script structure - aligned with Chapter 12 exercises:
%   Step 1 - Load data       (Exercise: data collection)
%   Step 2 - Transform       (Exercise: stationarity, summary stats)
%   Step 3 - Quantile reg.   (Exercise: estimate at multiple tau, Table 12.3)
%   Step 4 - Asymmetry       (Exercise: coefficient asymmetry plot, Fig 12.1)
%   Step 5 - Coverage        (Exercise: model evaluation, Table 12.4)
%   Step 6 - Risk assessment (Section 12.7.5 current conditions)
%   Step 7 - Figures         (Fig 12.1-12.4 + extras)
%
% Requirements: Statistics and Machine Learning Toolbox (quantile regression
%   is implemented here from scratch via linear programming using the
%   Optimization Toolbox, so no additional toolbox is strictly required
%   for Steps 1-6; only base MATLAB + Optimization Toolbox).
%
% Note: FEDFUNDS is NOT part of the Chapter 12 GaR model (eq. 12.9).
% =============================================================================

clear; clc; close all;

OUTPUT_DIR = './';   % change if needed

sep = repmat('=', 1, 70);
fprintf('%s\n', sep);
fprintf('GROWTH-AT-RISK ANALYSIS WITH NFCI\n');
fprintf('Chapter 12, Section 12.7  |  Macroeconometrics  |  Alessia Paccagnini\n');
fprintf('%s\n\n', sep);

% =============================================================================
% STEP 1 — LOAD DATA
% =============================================================================
fprintf('%s\n', repmat('-', 1, 70));
fprintf('Step 1 — Loading data\n');

% ── Real GDP (quarterly) ──────────────────────────────────────────────────
opts_gdp       = detectImportOptions('GDPC1.xlsx', 'Sheet', 'Quarterly');
opts_gdp       = setvartype(opts_gdp, 'observation_date', 'datetime');
gdp_tbl        = readtable('GDPC1.xlsx', opts_gdp);
gdp_dates      = dateshift(gdp_tbl.observation_date, 'start', 'quarter');
gdp_vals       = gdp_tbl.GDPC1;
fprintf('  GDPC1  : %d quarterly obs  (%s -> %s)\n', numel(gdp_vals), ...
        datestr(min(gdp_dates),'yyyy-mm-dd'), datestr(max(gdp_dates),'yyyy-mm-dd'));

% ── GDP Deflator (quarterly) ──────────────────────────────────────────────
opts_def       = detectImportOptions('GDPDEF.xlsx', 'Sheet', 'Quarterly');
opts_def       = setvartype(opts_def, 'observation_date', 'datetime');
def_tbl        = readtable('GDPDEF.xlsx', opts_def);
def_dates      = dateshift(def_tbl.observation_date, 'start', 'quarter');
def_vals       = def_tbl.GDPDEF;
fprintf('  GDPDEF : %d quarterly obs\n', numel(def_vals));

% ── NFCI weekly -> quarterly average ─────────────────────────────────────
if ~isfile('NFCI.csv')
    error('[ERROR] NFCI.csv not found. Download from https://fred.stlouisfed.org/series/NFCI');
end
opts_nfci      = detectImportOptions('NFCI.csv');
opts_nfci      = setvartype(opts_nfci, 'observation_date', 'datetime');
nfci_tbl       = readtable('NFCI.csv', opts_nfci);
nfci_w_dates   = nfci_tbl.observation_date;
nfci_w_vals    = nfci_tbl.NFCI;

% Aggregate to quarterly average (quarter start dates)
nfci_q_starts  = unique(dateshift(nfci_w_dates, 'start', 'quarter'));
nfci_q_vals    = arrayfun(@(qd) ...
    mean(nfci_w_vals(dateshift(nfci_w_dates,'start','quarter') == qd), 'omitnan'), ...
    nfci_q_starts);

fprintf('  NFCI   : %d weekly -> %d quarterly avg  (%s -> %s)\n', ...
        numel(nfci_w_vals), numel(nfci_q_vals), ...
        datestr(min(nfci_w_dates),'yyyy-mm-dd'), datestr(max(nfci_w_dates),'yyyy-mm-dd'));

% =============================================================================
% STEP 2 — MERGE AND TRANSFORM
% =============================================================================
fprintf('%s\n', repmat('-', 1, 70));
fprintf('Step 2 — Transformations and sample restriction\n');

H = 4;   % forecast horizon

% Align all series to a common quarterly date grid
all_dates   = intersect(gdp_dates, def_dates);
[~, ig]     = ismember(all_dates, gdp_dates);
[~, id]     = ismember(all_dates, def_dates);
gdp_a       = gdp_vals(ig);
def_a       = def_vals(id);

% Match NFCI (some quarters may be missing at edges)
[~, in_]    = ismember(all_dates, nfci_q_starts);
nfci_a      = nan(size(all_dates));
ok          = in_ > 0;
nfci_a(ok)  = nfci_q_vals(in_(ok));

% Year-on-year log growth x100  (eq. 12.9)
n           = numel(gdp_a);
gdp_growth  = [nan(4,1); (log(gdp_a(5:end)) - log(gdp_a(1:end-4))) * 100];
inflation   = [nan(4,1); (log(def_a(5:end)) - log(def_a(1:end-4))) * 100];

% h=4 forward target
gdp_forward = [gdp_growth(H+1:end); nan(H,1)];

% Sample restriction: 1971Q1-2024Q3
d_start     = datetime(1971,1,1);
d_end       = datetime(2024,9,30);
in_sample   = all_dates >= d_start & all_dates <= d_end;

dates       = all_dates(in_sample);
gdp_growth  = gdp_growth(in_sample);
inflation   = inflation(in_sample);
nfci        = nfci_a(in_sample);
gdp_forward = gdp_forward(in_sample);

% Drop rows with any NaN in model variables
valid       = ~isnan(gdp_growth) & ~isnan(inflation) & ~isnan(nfci) & ~isnan(gdp_forward);
dates       = dates(valid);
gdp_growth  = gdp_growth(valid);
inflation   = inflation(valid);
nfci        = nfci(valid);
gdp_forward = gdp_forward(valid);
N           = numel(dates);

fprintf('  Sample : 1971Q1 -> 2024Q3   N = %d\n', N);

fprintf('\nDescriptive statistics:\n');
vars_mat = [gdp_growth, inflation, nfci];
vnames   = {'gdp_growth', 'inflation', 'nfci'};
fprintf('%-14s %8s %8s %8s %8s %8s\n', 'Variable', 'N', 'Mean', 'Std', 'Min', 'Max');
for k = 1:3
    v = vars_mat(:,k);
    fprintf('%-14s %8d %8.2f %8.2f %8.2f %8.2f\n', ...
            vnames{k}, sum(~isnan(v)), mean(v,'omitnan'), std(v,'omitnan'), ...
            min(v,[],'omitnan'), max(v,[],'omitnan'));
end

% =============================================================================
% QUANTILE REGRESSION — local helper
% =============================================================================
% Solves the quantile regression problem via linear programming:
%   min_{b, u, v} tau*1'u + (1-tau)*1'v
%   s.t.  y - Xb = u - v,   u>=0, v>=0
%
% This requires the Optimization Toolbox (linprog).
%
function beta = quantreg_lp(X, y, tau)
    [n, k] = size(X);
    % Variables: [beta (k); u (n); v (n)]
    f   = [zeros(k,1); tau*ones(n,1); (1-tau)*ones(n,1)];
    Aeq = [X, eye(n), -eye(n)];
    beq = y;
    lb  = [-inf(k,1); zeros(2*n,1)];
    ub  = [];
    opts = optimoptions('linprog','Display','none', ...
                        'Algorithm','dual-simplex');
    sol  = linprog(f, [], [], Aeq, beq, lb, ub, opts);
    beta = sol(1:k);
end

% Asymptotic SE via Hendricks-Koenker sandwich (Powell kernel density)
function se = quantreg_se(X, y, beta, tau)
    n      = size(X, 1);
    resid  = y - X * beta;
    h      = max(tau*(1-tau), 0.05);   % bandwidth (simple rule)
    bw     = norminv(min(tau + h, 0.999)) - norminv(max(tau - h, 0.001));
    bw     = bw * std(resid) * n^(-1/5);
    in_bnd = abs(resid) < bw;
    fhat   = sum(in_bnd) / (2 * n * bw);
    fhat   = max(fhat, 1e-6);
    V      = inv(X' * X) * (X' * diag(double(in_bnd)) * X) * inv(X' * X);
    V      = tau*(1-tau) / fhat^2 * V;
    se     = sqrt(diag(V));
end

% =============================================================================
% STEP 3 — QUANTILE REGRESSION  (eq. 12.9 / Table 12.3)
% =============================================================================
fprintf('%s\n', repmat('-', 1, 70));
fprintf('Step 3 — Quantile regression  (eq. 12.9 / Table 12.3)\n');
fprintf('Note: SEs use Powell kernel sandwich. Block bootstrap recommended\n');
fprintf('      for overlapping observations (Section 12.2.3).\n\n');

% Design matrix: [1  gdp_growth  inflation  nfci]
X_mat     = [ones(N,1), gdp_growth, inflation, nfci];
y_vec     = gdp_forward;
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95];
nq        = numel(QUANTILES);
nvar      = 4;   % intercept + 3 regressors

% Storage
betas     = nan(nvar, nq);
ses       = nan(nvar, nq);

for j = 1:nq
    betas(:,j) = quantreg_lp(X_mat, y_vec, QUANTILES(j));
    ses(:,j)   = quantreg_se(X_mat, y_vec, betas(:,j), QUANTILES(j));
end

% OLS
beta_ols  = (X_mat' * X_mat) \ (X_mat' * y_vec);
resid_ols = y_vec - X_mat * beta_ols;
s2        = sum(resid_ols.^2) / (N - nvar);
se_ols    = sqrt(diag(s2 * inv(X_mat' * X_mat)));

% Print table (show tau = 0.05, 0.10, 0.25, 0.50, 0.90 + OLS)
SHOW_IDX  = [1, 2, 3, 4, 6];   % indices in QUANTILES
VARLABELS = {'Intercept', 'GDP_t', 'Inflation_t', 'NFCI_t'};

function s = stars(tstat)
    if abs(tstat) > 2.58, s = '***';
    elseif abs(tstat) > 1.96, s = '** ';
    elseif abs(tstat) > 1.64, s = '*  ';
    else, s = '   ';
    end
end

hdr = sprintf('%-12s', 'Variable');
for idx = SHOW_IDX
    hdr = [hdr, sprintf('  tau=%.2f', QUANTILES(idx))]; %#ok<AGROW>
end
hdr = [hdr, sprintf('     OLS')];
fprintf('%s\n%s\n', hdr, repmat('-', 1, 72));

for i = 1:nvar
    % Coef row
    row = sprintf('%-12s', VARLABELS{i});
    for idx = SHOW_IDX
        tval = betas(i,idx) / ses(i,idx);
        row  = [row, sprintf('  %5.2f%s', betas(i,idx), stars(tval))]; %#ok<AGROW>
    end
    tval_ols = beta_ols(i) / se_ols(i);
    row = [row, sprintf('  %5.2f%s', beta_ols(i), stars(tval_ols))];
    fprintf('%s\n', row);
    % SE row
    row = sprintf('%-12s', '');
    for idx = SHOW_IDX
        row = [row, sprintf('  (%4.2f)  ', ses(i,idx))]; %#ok<AGROW>
    end
    row = [row, sprintf('  (%4.2f)', se_ols(i))];
    fprintf('%s\n', row);
end
fprintf('%s\n', repmat('-', 1, 72));
fprintf('* p<0.10  ** p<0.05  *** p<0.01\n');

% =============================================================================
% STEP 4 — NFCI ASYMMETRY  (Section 12.7.2)
% =============================================================================
fprintf('%s\n', repmat('-', 1, 70));
fprintf('Step 4 — NFCI asymmetry  (Section 12.7.2)\n\n');
fprintf('NFCI coefficient across quantiles:\n');

nfci_row = 4;   % row index in betas
for j = 1:nq
    tval = betas(nfci_row,j) / ses(nfci_row,j);
    fprintf('  tau = %.2f:  %7.3f  (SE %.3f)  %s\n', ...
            QUANTILES(j), betas(nfci_row,j), ses(nfci_row,j), stars(tval));
end

b05   = betas(nfci_row, QUANTILES == 0.05);
b50   = betas(nfci_row, QUANTILES == 0.50);
b90   = betas(nfci_row, QUANTILES == 0.90);
ratio = abs(b05) / abs(b50);

fprintf('\n  beta_f(0.05) = %.3f\n', b05);
fprintf('  beta_f(0.50) = %.3f\n', b50);
fprintf('  beta_f(0.90) = %.3f\n', b90);
fprintf('  Asymmetry    = %.2fx\n', ratio);

% =============================================================================
% FITTED QUANTILES
% =============================================================================
fitted = X_mat * betas;   % N x nq matrix: columns = tau order

% =============================================================================
% STEP 5 — COVERAGE EVALUATION  (Table 12.4)
% =============================================================================
fprintf('%s\n', repmat('-', 1, 70));
fprintf('Step 5 — Coverage evaluation  (Table 12.4)\n\n');
fprintf('%-8s %10s %12s %8s   %s\n', 'tau', 'Nominal%', 'Empirical%', '|Diff|', 'OK?');
fprintf('%s\n', repmat('-', 1, 48));

for j = 1:nq
    emp      = mean(gdp_forward < fitted(:,j)) * 100;
    diff_val = abs(emp - QUANTILES(j)*100);
    ok       = 'V'; if diff_val >= 2.0, ok = '!'; end
    fprintf('%-8.2f %10.1f %12.1f %8.1f   %s\n', ...
            QUANTILES(j), QUANTILES(j)*100, emp, diff_val, ok);
end

% =============================================================================
% STEP 6 — CURRENT RISK ASSESSMENT  (Section 12.7.5)
% =============================================================================
fprintf('%s\n', repmat('-', 1, 70));
fprintf('Step 6 — Current risk assessment  (Section 12.7.5)\n\n');

last_date   = dates(end);
q_yr        = year(last_date);
q_qtr       = quarter(last_date);
q_label     = sprintf('%dQ%d', q_yr, q_qtr);
last_fitted = fitted(end, :);   % 1 x nq
gar5_cur    = last_fitted(QUANTILES == 0.05);
med_cur     = last_fitted(QUANTILES == 0.50);
nfci_cur    = nfci(end);

% P(GDP<0): interpolate CDF with Chernozhukov rearrangement
[v_sorted, si] = sort(last_fitted);
q_sorted       = QUANTILES(si);
if v_sorted(1) < 0 && v_sorted(end) > 0
    prob_neg = interp1(v_sorted, q_sorted, 0, 'linear');
elseif v_sorted(end) <= 0
    prob_neg = 1.0;
else
    prob_neg = 0.0;
end

fprintf('  Last obs  : %s  (NFCI = %.2f)\n', q_label, nfci_cur);
fprintf('  GaR (5%%) : %.2f%%\n', gar5_cur);
fprintf('  Median    : %.2f%%\n', med_cur);
fprintf('  P(GDP<0)  : %.1f%%\n', prob_neg * 100);

% Stress scenario
last_obs      = [gdp_growth(end), inflation(end)];
X_stress      = [1, last_obs(1), last_obs(2), 2.0];
gar5_stress   = X_stress * betas(:, QUANTILES == 0.05);
fprintf('\n  Stress (NFCI=2.0, as in 2008): GaR(5%%) = %.2f%%\n', gar5_stress);

% =============================================================================
% STEP 7 — FIGURES
% =============================================================================
fprintf('%s\n', repmat('-', 1, 70));
fprintf('Step 7 — Generating figures\n');

% NBER recession dates
rec_start = [datetime(1973,11,1), datetime(1980,1,1), datetime(1981,7,1), ...
             datetime(1990,7,1),  datetime(2001,3,1), datetime(2007,12,1), ...
             datetime(2020,2,1)];
rec_end   = [datetime(1975,3,1),  datetime(1980,7,1), datetime(1982,11,1), ...
             datetime(1991,3,1),  datetime(2001,11,1),datetime(2009,6,1), ...
             datetime(2020,4,1)];

function add_recessions(ax, rec_start, rec_end)
    yl = ylim(ax);
    hold(ax, 'on');
    for r = 1:numel(rec_start)
        patch(ax, ...
            [rec_start(r), rec_end(r), rec_end(r), rec_start(r)], ...
            [yl(1), yl(1), yl(2), yl(2)], ...
            [0.7 0.7 0.7], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
end

t_num = datenum(dates);   % for compatibility with some plot functions

% ── Figure 12.1: Coefficient asymmetry ────────────────────────────────────
fig1 = figure('Position', [100 100 1100 750]);
var_labels_plot = {'Current GDP Growth', 'Inflation', 'NFCI (Financial Conditions)'};
for k = 1:3
    subplot(2, 2, k);
    row_k  = k + 1;   % rows 2,3,4 in betas (skip intercept)
    coefs_k = betas(row_k, :);
    ses_k   = ses(row_k, :);
    bar_colors = repmat([0.18 0.46 0.71], nq, 1);   % default blue
    bar_colors(coefs_k < 0, :) = repmat([0.75 0 0], sum(coefs_k<0), 1);
    b = bar(1:nq, coefs_k, 'FaceColor', 'flat');
    for bj = 1:nq
        b.CData(bj,:) = bar_colors(bj,:);
    end
    hold on;
    errorbar(1:nq, coefs_k, 1.96*ses_k, 'k.', 'LineWidth', 1.2);
    yline(0, 'k-', 'LineWidth', 0.8);
    yline(beta_ols(row_k), '--', 'Color', [0 0.5 0], 'LineWidth', 1.5, ...
          'DisplayName', 'OLS');
    xticks(1:nq);
    xticklabels(arrayfun(@(q) sprintf('%.0f%%', q*100), QUANTILES, 'UniformOutput', false));
    xlabel('Quantile (\tau)'); ylabel('Coefficient');
    title(var_labels_plot{k}, 'FontWeight', 'bold');
    grid on; box on;
    if k == 1, legend('show', 'Location', 'best'); end
end
subplot(2, 2, 4); axis off;
txt = sprintf('KEY FINDING: NFCI Asymmetry\n\nbeta_f(tau):\n  tau=0.05: %.3f\n  tau=0.50: %.3f\n  tau=0.90: %.3f\n\n%.2fx larger at 5th\nthan median\n\nFinancial conditions predict\nthe SHAPE of the distribution,\nnot just its mean.\n\nAdrian et al. (2019)', ...
              b05, b50, b90, ratio);
text(0.05, 0.95, txt, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
     'FontName', 'Courier', 'FontSize', 9, ...
     'BackgroundColor', [1 1 0.8], 'EdgeColor', [0.6 0.6 0.6]);
sgtitle('Figure 12.1: Quantile Regression Coefficients', 'FontWeight', 'bold', 'FontSize', 12);
exportgraphics(fig1, fullfile(OUTPUT_DIR, 'fig1_gar_coefficients.pdf'), 'Resolution', 300);
fprintf('  Figure 12.1 - coefficient asymmetry\n');

% ── Figure 12.2: Fan chart ─────────────────────────────────────────────────
idx_05 = find(QUANTILES == 0.05);  idx_10 = find(QUANTILES == 0.10);
idx_25 = find(QUANTILES == 0.25);  idx_50 = find(QUANTILES == 0.50);
idx_75 = find(QUANTILES == 0.75);  idx_90 = find(QUANTILES == 0.90);
idx_95 = find(QUANTILES == 0.95);

fig2 = figure('Position', [100 100 1100 500]);
ax2  = axes(fig2);
add_recessions(ax2, rec_start, rec_end);
hold on;
fill([dates; flipud(dates)], [fitted(:,idx_05); flipud(fitted(:,idx_95))], ...
     [0.18 0.46 0.71], 'FaceAlpha', 0.12, 'EdgeColor', 'none', 'DisplayName', '5-95%');
fill([dates; flipud(dates)], [fitted(:,idx_10); flipud(fitted(:,idx_90))], ...
     [0.18 0.46 0.71], 'FaceAlpha', 0.22, 'EdgeColor', 'none', 'DisplayName', '10-90%');
fill([dates; flipud(dates)], [fitted(:,idx_25); flipud(fitted(:,idx_75))], ...
     [0.18 0.46 0.71], 'FaceAlpha', 0.38, 'EdgeColor', 'none', 'DisplayName', '25-75%');
plot(dates, fitted(:,idx_50), '-', 'Color', [0.18 0.46 0.71], 'LineWidth', 2, 'DisplayName', 'Median');
p_act = plot(dates, gdp_forward, '-', 'Color', [0 0 0], 'LineWidth', 0.8, 'DisplayName', 'Actual');
p_act.Color(4) = 0.7;   % set alpha via Color 4th element (R2014b+)
yline(0, '--', 'Color', [1 0 0 0.7], 'LineWidth', 0.8);
xlabel('Date'); ylabel('GDP Growth (%, YoY)');
title(sprintf('Figure 12.2: Growth-at-Risk Fan Chart  (%d-Quarter-Ahead)  1971Q1-2024Q3', H), ...
      'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 9);
ylim([-10 12]); grid on; box on;
exportgraphics(fig2, fullfile(OUTPUT_DIR, 'fig2_gar_fanchart.pdf'), 'Resolution', 300);
fprintf('  Figure 12.2 - fan chart\n');

% ── Figure 12.3: Three panels ──────────────────────────────────────────────
fig3 = figure('Position', [100 100 1100 850]);

ax3a = subplot(3, 1, 1);
tight_pos = abs(nfci) .* (nfci > 0);
loose_pos = abs(nfci) .* (nfci < 0);
fill([dates; flipud(dates)], [zeros(N,1); flipud(tight_pos)],  [0.75 0 0],    'FaceAlpha', 0.35, 'EdgeColor', 'none'); hold on;
fill([dates; flipud(dates)], [-loose_pos; zeros(N,1)],         [0.44 0.68 0.28], 'FaceAlpha', 0.25, 'EdgeColor', 'none');
plot(dates, nfci, '-', 'Color', [0.18 0.46 0.71], 'LineWidth', 1.5);
yline(0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
ylabel('NFCI'); title('Panel A: Financial Conditions (NFCI > 0 = Tight)', 'FontWeight', 'bold');
add_recessions(ax3a, rec_start, rec_end); grid on; box on;

ax3b = subplot(3, 1, 2);
plot(dates, inflation, '-', 'Color', [0.75 0 0], 'LineWidth', 1.5); hold on;
yline(2, '--', 'Color', [0.44 0.68 0.28], 'LineWidth', 1, 'DisplayName', '2% target');
ylabel('Inflation (%, YoY)'); title('Panel B: Inflation (GDP Deflator)', 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);
add_recessions(ax3b, rec_start, rec_end); grid on; box on;

ax3c = subplot(3, 1, 3);
gar05_neg = fitted(:, idx_05);
gar05_neg(gar05_neg >= 0) = 0;
fill([dates; flipud(dates)], [gar05_neg; zeros(N,1)], [0.75 0 0], 'FaceAlpha', 0.25, 'EdgeColor', 'none'); hold on;
plot(dates, fitted(:,idx_05), '-', 'Color', [0.75 0 0],    'LineWidth', 2,   'DisplayName', 'GaR 5th');
p_med = plot(dates, fitted(:,idx_50), '-', 'Color', [0.18 0.46 0.71], 'LineWidth', 1.5, 'DisplayName', 'Median');
p_med.Color(4) = 0.8;
p_act2 = plot(dates, gdp_forward, 'k-', 'LineWidth', 0.8, 'DisplayName', 'Actual');
p_act2.Color(4) = 0.5;
yline(0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
xlabel('Date'); ylabel('GDP Growth (%)'); title('Panel C: Growth-at-Risk (5th Pct) vs Median', 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 9);
add_recessions(ax3c, rec_start, rec_end); grid on; box on;

sgtitle('Figure 12.3: Financial Conditions and Growth-at-Risk', 'FontWeight', 'bold', 'FontSize', 12);
exportgraphics(fig3, fullfile(OUTPUT_DIR, 'fig3_gar_panels.pdf'), 'Resolution', 300);
fprintf('  Figure 12.3 - three panels\n');

% ── Figure 12.4: Predictive CDF  (Chernozhukov rearrangement) ─────────────
fine_q   = linspace(0.01, 0.99, 200)';
kv_raw   = last_fitted;
[kv, si] = sort(kv_raw);
kq       = QUANTILES(si);
preds    = interp1(kq, kv, fine_q, 'linear', 'extrap');
gar5_pt  = interp1(fine_q, preds, 0.05);

fig4 = figure('Position', [100 100 900 600]);
ax4  = axes(fig4);
fill([min(preds)-1; preds; min(preds)-1], [fine_q(1); fine_q; fine_q(end)], ...
     [0.18 0.46 0.71], 'FaceAlpha', 0.15, 'EdgeColor', 'none'); hold on;
plot(preds, fine_q, '-', 'Color', [0.18 0.46 0.71], 'LineWidth', 2.5);
xline(0,    '--', 'Color', [0.75 0 0],    'LineWidth', 1.5, 'DisplayName', 'Zero growth');
yline(0.05, ':',  'Color', [1 0.6 0],     'LineWidth', 1.5, 'DisplayName', '5th pct (GaR)');
yline(0.50, ':',  'Color', [0.44 0.68 0.28], 'LineWidth', 1.5, 'DisplayName', 'Median');
plot(gar5_pt, 0.05, 'o', 'Color', [0.75 0 0], 'MarkerFaceColor', [0.75 0 0], 'MarkerSize', 8);
text(gar5_pt - 0.3, 0.12, sprintf('GaR(5%%) = %.2f%%', gar5_pt), ...
     'Color', [0.75 0 0], 'FontSize', 11, 'HorizontalAlignment', 'right');
annotation_txt = sprintf('GaR (5%%): %.2f%%\nP(GDP<0): %.1f%%\nNFCI: %.2f', ...
                          gar5_pt, prob_neg*100, nfci_cur);
text(0.02, 0.95, annotation_txt, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
     'FontSize', 10, 'BackgroundColor', [1 1 0.8], 'EdgeColor', [0.6 0.6 0.6]);
xlabel('GDP Growth (%, YoY)'); ylabel('Cumulative Probability');
title(sprintf('Figure 12.4: Predictive CDF  (%dQ Ahead)  |  %s  |  NFCI = %.2f', ...
              H, q_label, nfci_cur), 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 9);
xlim([min(preds)-1, max(preds)+1]); ylim([0 1]);
grid on; box on;
exportgraphics(fig4, fullfile(OUTPUT_DIR, 'fig4_gar_cdf.pdf'), 'Resolution', 300);
fprintf('  Figure 12.4 - predictive CDF\n');

% ── Figure 5: NFCI vs GaR scatter ─────────────────────────────────────────
fig5      = figure('Position', [100 100 900 600]);
yr_num    = year(dates);
scatter(nfci, fitted(:,idx_05), 25, yr_num, 'filled', 'MarkerFaceAlpha', 0.6); hold on;
colormap(viridis_approx(256));  % see helper below; falls back to parula
cb = colorbar; cb.Label.String = 'Year';
p_fit = polyfit(nfci, fitted(:,idx_05), 1);
x_fit = linspace(min(nfci), max(nfci), 100);
plot(x_fit, polyval(p_fit, x_fit), '-', 'Color', [0.75 0 0], 'LineWidth', 2, ...
     'DisplayName', sprintf('Slope = %.2f', p_fit(1)));
yline(0, '--', 'Color', [0.5 0.5 0.5]); xline(0, '--', 'Color', [0.5 0.5 0.5]);
xlabel('NFCI'); ylabel('GaR 5th Percentile (%)');
title('NFCI vs GaR: Tighter Conditions -> Lower Downside Threshold', 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);
grid on; box on;
exportgraphics(fig5, fullfile(OUTPUT_DIR, 'fig5_gar_scatter.pdf'), 'Resolution', 300);
fprintf('  Figure 5 - NFCI vs GaR scatter\n');

% ── Figure 6: Early warning ────────────────────────────────────────────────
fig6 = figure('Position', [100 100 1100 450]);
ax6  = axes(fig6);
add_recessions(ax6, rec_start, rec_end);
hold on;
fill([dates; flipud(dates)], [fitted(:,idx_05); flipud(fitted(:,idx_10))], ...
     [0.93 0.49 0.19], 'FaceAlpha', 0.25, 'EdgeColor', 'none');
plot(dates, fitted(:,idx_05), '-', 'Color', [0.75 0 0],    'LineWidth', 2,   'DisplayName', 'GaR 5th pct');
plot(dates, fitted(:,idx_10), '-', 'Color', [0.93 0.49 0.19], 'LineWidth', 1.5, 'DisplayName', 'GaR 10th pct');
yline(0,  'k-', 'LineWidth', 0.5);
yline(-2, ':',  'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'DisplayName', '-2% reference');
xlabel('Date'); ylabel('GDP Growth (%)');
title('Growth-at-Risk as Early Warning  (Grey = NBER recessions)', 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 9);
grid on; box on;
exportgraphics(fig6, fullfile(OUTPUT_DIR, 'fig6_gar_warning.pdf'), 'Resolution', 300);
fprintf('  Figure 6 - early warning\n');

fprintf('\nAll figures saved to: %s\n', OUTPUT_DIR);

% =============================================================================
% SUMMARY
% =============================================================================
fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('\nGrowth-at-Risk with NFCI  |  Chapter 12, Section 12.7\n');
fprintf('======================================================\n');
fprintf('Sample       : 1971Q1-2024Q3   N = %d\n', N);
fprintf('Horizon      : h = %d quarters\n\n', H);
fprintf('Key finding - NFCI asymmetry (Table 12.3):\n');
fprintf('  beta_f(0.05) = %.3f\n', b05);
fprintf('  beta_f(0.50) = %.3f\n', b50);
fprintf('  beta_f(0.90) = %.3f\n', b90);
fprintf('  Asymmetry    = %.2fx\n\n', ratio);
fprintf('Current assessment (Section 12.7.5 / %s):\n', q_label);
fprintf('  NFCI = %.2f  |  GaR(5%%) = %.2f%%  |  Median = %.2f%%\n', ...
        nfci_cur, gar5_cur, med_cur);
fprintf('  P(GDP<0) = %.1f%%\n\n', prob_neg * 100);
fprintf('Output: %s\n', OUTPUT_DIR);
fprintf('  fig1-fig6 individual PDFs\n\n');

% =============================================================================
% HELPER: approximate viridis colormap (MATLAB < R2022a fallback)
% =============================================================================
function cmap = viridis_approx(n)
    try
        cmap = viridis(n);
    catch
        cmap = parula(n);   % fallback if viridis not available
    end
end
