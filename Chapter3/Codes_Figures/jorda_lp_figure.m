%% =========================================================================
%  Jordà (2005) Local Projections Example with Real U.S. Data — MATLAB
%  =========================================================================
%  Comparing LP vs VAR impulse responses for monetary policy analysis
%  Using FRED data: GDP (GDPC1), GDP Deflator (GDPDEF), Federal Funds Rate
%
%  Instructions:
%    1. Place GDPC1.xlsx, GDPDEF.xlsx, and FEDFUNDS.xlsx in the current
%       working directory (or adjust the file paths below).
%    2. Run this script: >> jorda_lp_real_data
%
%  Requires: Econometrics Toolbox (for hpfilter, varm), 
%            or use the manual implementations provided below.
% Author: Alessia Paccagnini
% Textbook: Macroeconometrics
%  =========================================================================

clear; clc; close all;

%% =========================================================================
%  1. Load Data
%  =========================================================================
fprintf('Loading FRED data...\n');

gdp_raw      = readtable('GDPC1.xlsx',    'Sheet', 'Quarterly');
deflator_raw = readtable('GDPDEF.xlsx',   'Sheet', 'Quarterly');
fedfunds_raw = readtable('FEDFUNDS.xlsx', 'Sheet', 'Monthly');

fprintf('  GDP:       %d obs\n', height(gdp_raw));
fprintf('  Deflator:  %d obs\n', height(deflator_raw));
fprintf('  Fed Funds: %d obs\n', height(fedfunds_raw));

%% =========================================================================
%  2. Process and Merge
%  =========================================================================

% Rename columns
gdp_raw.Properties.VariableNames      = {'date', 'gdp'};
deflator_raw.Properties.VariableNames = {'date', 'deflator'};
fedfunds_raw.Properties.VariableNames = {'date', 'fedfunds'};

% Convert to datetime
gdp_raw.date      = datetime(gdp_raw.date);
deflator_raw.date = datetime(deflator_raw.date);
fedfunds_raw.date = datetime(fedfunds_raw.date);

% Fed funds: monthly → quarterly average
ff_year  = year(fedfunds_raw.date);
ff_month = month(fedfunds_raw.date);
ff_qtr   = ceil(ff_month / 3);
fedfunds_raw.qtr_id = ff_year * 10 + ff_qtr;

[qtr_ids, ~, idx] = unique(fedfunds_raw.qtr_id);
ff_q_vals = accumarray(idx, fedfunds_raw.fedfunds, [], @mean);
ff_q_year = floor(qtr_ids / 10);
ff_q_qtr  = qtr_ids - ff_q_year * 10;
ff_q_month = (ff_q_qtr - 1) * 3 + 1;
ff_q_date  = datetime(ff_q_year, ff_q_month, 1);
fedfunds_q = table(ff_q_date, ff_q_vals, 'VariableNames', {'date', 'fedfunds'});

% Align GDP/deflator to quarter-start
gdp_m = month(gdp_raw.date);
gdp_raw.date = datetime(year(gdp_raw.date), (ceil(gdp_m/3)-1)*3+1, 1);
def_m = month(deflator_raw.date);
deflator_raw.date = datetime(year(deflator_raw.date), (ceil(def_m/3)-1)*3+1, 1);

% Merge via innerjoin
df = innerjoin(gdp_raw, deflator_raw, 'Keys', 'date');
df = innerjoin(df, fedfunds_q, 'Keys', 'date');
df = sortrows(df, 'date');

fprintf('\nMerged data: %d observations\n', height(df));
fprintf('Date range:  %s to %s\n', datestr(df.date(1), 'yyyy-mm'), ...
        datestr(df.date(end), 'yyyy-mm'));

%% =========================================================================
%  3. Construct Variables
%  =========================================================================

% Log GDP
log_gdp = 100 * log(df.gdp);

% HP filter (lambda = 1600)
% If you have the Econometrics Toolbox, use:
%   [trend, cycle] = hpfilter(log_gdp, 'Smoothing', 1600);
% Otherwise, manual HP filter:
T_hp = length(log_gdp);
A = zeros(T_hp, T_hp);
for i = 1:T_hp
    A(i,i) = 1;
end
F = zeros(T_hp-2, T_hp);
for i = 1:(T_hp-2)
    F(i, i)   =  1;
    F(i, i+1) = -2;
    F(i, i+2) =  1;
end
lambda_hp = 1600;
trend = (A + lambda_hp * (F' * F)) \ log_gdp;
output_gap = log_gdp - trend;

% Inflation: 4-quarter change in log GDP deflator
log_deflator = 100 * log(df.deflator);
inflation = [NaN(4,1); log_deflator(5:end) - log_deflator(1:end-4)];

% Federal funds rate
fed_funds = df.fedfunds;

% Combine
dates = df.date;
all_data = table(dates, output_gap, inflation, fed_funds);

% Sample: 1960Q1 – 2007Q4
sample_mask = dates >= datetime(1960,1,1) & dates <= datetime(2007,12,31);
sample_data = all_data(sample_mask, :);
sample_data = sample_data(~any(ismissing(sample_data{:, 2:end}), 2), :);

T_s = height(sample_data);
s_start = sample_data.dates(1);
s_end   = sample_data.dates(end);
start_q = sprintf('%dQ%d', year(s_start), ceil(month(s_start)/3));
end_q   = sprintf('%dQ%d', year(s_end),   ceil(month(s_end)/3));

fprintf('\nEstimation sample: %s – %s  (T = %d)\n', start_q, end_q, T_s);

Y = [sample_data.output_gap, sample_data.inflation, sample_data.fed_funds];
var_names = {'output\_gap', 'inflation', 'fed\_funds'};

fprintf('\nDescriptive statistics:\n');
fprintf('  %15s  %10s  %10s  %10s\n', '', 'output_gap', 'inflation', 'fed_funds');
fprintf('  %15s  %10.2f  %10.2f  %10.2f\n', 'Mean', mean(Y));
fprintf('  %15s  %10.2f  %10.2f  %10.2f\n', 'Std', std(Y));
fprintf('  %15s  %10.2f  %10.2f  %10.2f\n', 'Min', min(Y));
fprintf('  %15s  %10.2f  %10.2f  %10.2f\n', 'Max', max(Y));

%% =========================================================================
%  4. Estimate VAR Models
%  =========================================================================
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('Estimating VAR models...\n');
fprintf('%s\n', repmat('=', 1, 60));

H = 20;  % horizon
K = 3;   % number of variables

% --- Estimate VAR by OLS --------------------------------------------------
function [A, Sigma, resid] = estimate_var(Y, p)
    [T, K] = size(Y);
    % Build lagged matrix
    X = ones(T - p, 1);  % constant
    for lag = 1:p
        X = [X, Y(p+1-lag:T-lag, :)];
    end
    Yf = Y(p+1:T, :);
    B = (X' * X) \ (X' * Yf);     % OLS coefficients
    resid = Yf - X * B;
    Sigma = (resid' * resid) / (T - p - K*p - 1);
    A = B;  % constant in row 1, then lags
end

[A4, Sigma4, resid4] = estimate_var(Y, 4);
[A1, Sigma1, resid1] = estimate_var(Y, 1);

fprintf('  VAR(4) estimated\n');
fprintf('  VAR(1) estimated (misspecified benchmark)\n');

% --- Cholesky IRFs ---------------------------------------------------------
function irfs = compute_cholesky_irf(A, Sigma, p, K, H)
    % A: coefficient matrix [const; A1; A2; ... Ap]  size (1+K*p) x K
    % Returns irfs: (H+1) x K x K  — irfs(h+1, i, j) = response of i to shock j at horizon h
    
    P = chol(Sigma, 'lower');  % Cholesky factor
    
    % Companion form
    A_comp = zeros(K * p, K * p);
    for lag = 1:p
        rows_A = (lag - 1) * K + (1:K);
        A_comp(1:K, rows_A) = A(1 + (lag-1)*K + 1 : 1 + lag*K, :)';
    end
    if p > 1
        A_comp(K+1:K*p, 1:K*(p-1)) = eye(K*(p-1));
    end
    
    irfs = zeros(H + 1, K, K);
    irfs(1, :, :) = P;
    
    Phi = eye(K * p);
    for h = 1:H
        Phi = A_comp * Phi;
        irfs(h + 1, :, :) = Phi(1:K, 1:K) * P;
    end
end

irf4 = compute_cholesky_irf(A4, Sigma4, 4, K, H);
irf1 = compute_cholesky_irf(A1, Sigma1, 1, K, H);

% --- Bootstrap confidence bands for VAR(4) --------------------------------
fprintf('  Bootstrapping VAR(4) confidence bands...\n');
n_boot = 500;
irf_boot = zeros(H + 1, K, K, n_boot);
[T_est, ~] = size(resid4);

rng(42);
for b = 1:n_boot
    % Resample residuals
    boot_idx = randi(T_est, T_est, 1);
    resid_b = resid4(boot_idx, :);
    
    % Simulate data from VAR(4)
    Y_b = zeros(T_est + 4, K);
    Y_b(1:4, :) = Y(1:4, :);  % initial values
    for t = 5:(T_est + 4)
        Xt = [1];
        for lag = 1:4
            Xt = [Xt, Y_b(t - lag, :)];
        end
        Y_b(t, :) = Xt * A4 + resid_b(t - 4, :);
    end
    
    % Re-estimate and compute IRFs
    [A_b, Sigma_b, ~] = estimate_var(Y_b, 4);
    irf_boot(:, :, :, b) = compute_cholesky_irf(A_b, Sigma_b, 4, K, H);
end

irf4_lo = quantile(irf_boot, 0.025, 4);
irf4_hi = quantile(irf_boot, 0.975, 4);

%% =========================================================================
%  5. Estimate Local Projections
%  =========================================================================
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('Estimating Local Projections...\n');
fprintf('%s\n', repmat('=', 1, 60));

function [irfs, ses, ci_lo, ci_hi] = estimate_lp(Y, shock_col, resp_col, H, n_lags)
    % Jordà (2005) LP with Newey–West standard errors
    %
    % Y         : T x K data matrix
    % shock_col : column index of shock variable
    % resp_col  : column index of response variable
    % H         : maximum horizon
    % n_lags    : number of control lags
    
    [T, K] = size(Y);
    irfs = zeros(H + 1, 1);
    ses  = zeros(H + 1, 1);
    
    for h = 0:H
        % Dependent variable: y_{t+h}
        y = [Y(1+h:T, resp_col); NaN(h, 1)];
        
        % Controls: contemporaneous shock + lags of all variables + constant
        X = Y(:, shock_col);  % contemporaneous shock
        for v = 1:K
            for lag = 1:n_lags
                lagged = [NaN(lag, 1); Y(1:T-lag, v)];
                X = [X, lagged];
            end
        end
        X = [ones(T, 1), X];  % constant first
        
        % Drop incomplete rows
        ok = ~any(isnan([y, X]), 2);
        y_c = y(ok);
        X_c = X(ok, :);
        
        % OLS
        beta = (X_c' * X_c) \ (X_c' * y_c);
        e = y_c - X_c * beta;
        n = length(y_c);
        k = size(X_c, 2);
        
        % Newey–West HAC covariance
        nw_lags = max(h + 1, 4);
        S = zeros(k, k);
        for j = 0:nw_lags
            Gj = zeros(k, k);
            for t = (j+1):n
                Gj = Gj + (X_c(t,:)' * e(t)) * (X_c(t-j,:)' * e(t-j))';
            end
            Gj = Gj / n;
            if j == 0
                S = S + Gj;
            else
                w = 1 - j / (nw_lags + 1);  % Bartlett kernel
                S = S + w * (Gj + Gj');
            end
        end
        V = (n / (n - k)) * ((X_c' * X_c) \ S) / (X_c' * X_c) * n;
        
        % Coefficient on shock (column 2: after constant)
        irfs(h + 1) = beta(2);
        ses(h + 1)  = sqrt(V(2, 2));
    end
    
    ci_lo = irfs - 1.96 * ses;
    ci_hi = irfs + 1.96 * ses;
end

n_lags_lp = 4;

fprintf('  Output gap response...\n');
[lp_y, ~, lp_lo_y, lp_hi_y]    = estimate_lp(Y, 3, 1, H, n_lags_lp);

fprintf('  Inflation response...\n');
[lp_pi, ~, lp_lo_pi, lp_hi_pi] = estimate_lp(Y, 3, 2, H, n_lags_lp);

fprintf('  Fed funds own response...\n');
[lp_ff, ~, lp_lo_ff, lp_hi_ff] = estimate_lp(Y, 3, 3, H, n_lags_lp);

fprintf('  Done!\n');

%% =========================================================================
%  6. Publication-Quality Figure
%  =========================================================================
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('Creating figure...\n');
fprintf('%s\n', repmat('=', 1, 60));

% Cholesky indices: shock = fed_funds (col 3)
shock_j = 3;
resp_y  = 1;  % output_gap
resp_pi = 2;  % inflation

% VAR(4) IRFs
v4_y   = squeeze(irf4(:, resp_y,  shock_j));
v4_pi  = squeeze(irf4(:, resp_pi, shock_j));
v4_lo_y  = squeeze(irf4_lo(:, resp_y,  shock_j));
v4_hi_y  = squeeze(irf4_hi(:, resp_y,  shock_j));
v4_lo_pi = squeeze(irf4_lo(:, resp_pi, shock_j));
v4_hi_pi = squeeze(irf4_hi(:, resp_pi, shock_j));

% VAR(1) IRFs
v1_y  = squeeze(irf1(:, resp_y,  shock_j));
v1_pi = squeeze(irf1(:, resp_pi, shock_j));

horizons = 0:H;

% Colours
c_var4 = [0.130 0.400 0.675];   % blue
c_lp   = [0.698 0.094 0.169];   % red
c_var1 = [0.302 0.686 0.290];   % green

fig = figure('Position', [100 100 1200 450], 'Color', 'w');

% --- Panel (a): Output Gap ------------------------------------------------
subplot(1, 2, 1); hold on; box on;

% VAR(4) band
fill([horizons, fliplr(horizons)], [v4_lo_y', fliplr(v4_hi_y')], ...
     c_var4, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
% LP band
fill([horizons, fliplr(horizons)], [lp_lo_y', fliplr(lp_hi_y')], ...
     c_lp, 'FaceAlpha', 0.15, 'EdgeColor', 'none');

h1 = plot(horizons, v4_y,  '-',  'Color', c_var4, 'LineWidth', 2.5);
h2 = plot(horizons, lp_y,  '--', 'Color', c_lp,   'LineWidth', 2.5);
h3 = plot(horizons, v1_y,  ':',  'Color', c_var1,  'LineWidth', 2);
yline(0, 'k-', 'LineWidth', 1);

xlabel('Quarters after shock', 'FontSize', 13);
ylabel('Percent', 'FontSize', 13);
title('(a) Response of Output Gap to Fed Funds Shock', ...
      'FontSize', 13, 'FontWeight', 'bold');
legend([h1 h2 h3], {'VAR(4)', 'Local Projections', 'VAR(1) — misspecified'}, ...
       'Location', 'southeast', 'FontSize', 10);
set(gca, 'XTick', 0:4:20, 'FontSize', 11);
xlim([0 20]);
ym = max(abs([v4_lo_y; v4_hi_y; lp_lo_y; lp_hi_y; v1_y])) * 1.15;
ylim([-ym ym]);

% --- Panel (b): Inflation --------------------------------------------------
subplot(1, 2, 2); hold on; box on;

fill([horizons, fliplr(horizons)], [v4_lo_pi', fliplr(v4_hi_pi')], ...
     c_var4, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
fill([horizons, fliplr(horizons)], [lp_lo_pi', fliplr(lp_hi_pi')], ...
     c_lp, 'FaceAlpha', 0.15, 'EdgeColor', 'none');

h1 = plot(horizons, v4_pi,  '-',  'Color', c_var4, 'LineWidth', 2.5);
h2 = plot(horizons, lp_pi,  '--', 'Color', c_lp,   'LineWidth', 2.5);
h3 = plot(horizons, v1_pi,  ':',  'Color', c_var1,  'LineWidth', 2);
yline(0, 'k-', 'LineWidth', 1);

xlabel('Quarters after shock', 'FontSize', 13);
ylabel('Percent', 'FontSize', 13);
title('(b) Response of Inflation to Fed Funds Shock', ...
      'FontSize', 13, 'FontWeight', 'bold');
legend([h1 h2 h3], {'VAR(4)', 'Local Projections', 'VAR(1) — misspecified'}, ...
       'Location', 'northeast', 'FontSize', 10);
set(gca, 'XTick', 0:4:20, 'FontSize', 11);
xlim([0 20]);
ym_pi = max(abs([v4_lo_pi; v4_hi_pi; lp_lo_pi; lp_hi_pi; v1_pi])) * 1.15;
ylim([-ym_pi ym_pi]);

% Main title
sgtitle(sprintf('Impulse Responses to a Monetary Policy Shock: LP vs VAR\nU.S. Quarterly Data, %s–%s', ...
        start_q, end_q), 'FontSize', 14, 'FontWeight', 'bold');

% Save
print(fig, 'jorda_lp_example_real_data', '-dpng', '-r200');
fprintf('Figure saved as jorda_lp_example_real_data.png\n');

%% =========================================================================
%  7. Summary
%  =========================================================================
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('SUMMARY: LP vs VAR Comparison (Real U.S. Data)\n');
fprintf('%s\n', repmat('=', 1, 60));
fprintf('\nSample: %s – %s  (T = %d)\n', start_q, end_q, T_s);
fprintf('Variables: Output Gap, Inflation, Federal Funds Rate\n');
fprintf('Identification: Cholesky (output → inflation → fed funds)\n');

fprintf('\n--- Output Gap Response (h = 8) ---\n');
fprintf('  VAR(4): %7.4f  [%7.4f, %7.4f]\n', v4_y(9), v4_lo_y(9), v4_hi_y(9));
fprintf('  LP:     %7.4f  [%7.4f, %7.4f]\n', lp_y(9), lp_lo_y(9), lp_hi_y(9));
fprintf('  VAR(1): %7.4f  (misspecified)\n', v1_y(9));

fprintf('\n--- Inflation Response (h = 8) ---\n');
fprintf('  VAR(4): %7.4f  [%7.4f, %7.4f]\n', v4_pi(9), v4_lo_pi(9), v4_hi_pi(9));
fprintf('  LP:     %7.4f  [%7.4f, %7.4f]\n', lp_pi(9), lp_lo_pi(9), lp_hi_pi(9));
fprintf('  VAR(1): %7.4f  (misspecified)\n', v1_pi(9));

fprintf('\n%s\n', repmat('=', 1, 60));
