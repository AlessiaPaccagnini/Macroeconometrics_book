% =============================================================================
% Author:   Alessia Paccagnini
% Textbook: Macroeconometrics
% MIDAS (Mixed Data Sampling) Estimation in MATLAB
% =============================================================================
% Translation of midas_estimation.py for the Macroeconometrics Textbook
% Chapter 11: Mixed-Frequency Data, Section 11.9
%
% Key References:
%   Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). "The MIDAS Touch"
%   Ghysels, E., & Marcellino, M. (2018). "Applied Economic Forecasting
%     Using Time Series Methods"
%
% Requirements: MATLAB R2019b+ (uses readtable, datetime, fmincon)
%   Toolboxes: Optimization Toolbox (for fmincon)
%
% Usage:
%   Run this script directly from the MATLAB command window or editor.
%   Data files must be in the working directory (or update paths below).
% =============================================================================

clear; clc; close all;

fprintf('\n%s\n', repmat('=',1,70));
fprintf('MIDAS ESTIMATION: Nowcasting GDP Growth with Fed Funds Rate\n');
fprintf('%s\n', repmat('=',1,70));

% =============================================================================
% 1. DATA LOADING AND PREPARATION
% =============================================================================

fprintf('\n>>> Loading and preparing data...\n');

% Load Excel files
fedfunds_tbl = readtable('FEDFUNDS.xlsx', 'Sheet', 'Monthly');
gdp_tbl      = readtable('GDPC1.xlsx',    'Sheet', 'Quarterly');
gdpdef_tbl   = readtable('GDPDEF.xlsx',   'Sheet', 'Quarterly');

% Rename columns for clarity
fedfunds_tbl.Properties.VariableNames = {'date','fedfunds'};
gdp_tbl.Properties.VariableNames      = {'date','gdp'};
gdpdef_tbl.Properties.VariableNames   = {'date','gdpdef'};

% Convert dates to datetime if needed
if ~isdatetime(fedfunds_tbl.date)
    fedfunds_tbl.date = datetime(fedfunds_tbl.date);
end
if ~isdatetime(gdp_tbl.date)
    gdp_tbl.date = datetime(gdp_tbl.date);
end
if ~isdatetime(gdpdef_tbl.date)
    gdpdef_tbl.date = datetime(gdpdef_tbl.date);
end

% Sort by date
fedfunds_tbl = sortrows(fedfunds_tbl, 'date');
gdp_tbl      = sortrows(gdp_tbl,      'date');
gdpdef_tbl   = sortrows(gdpdef_tbl,   'date');

% Annualised quarterly GDP growth: 400 * log(gdp_t / gdp_{t-1})
n_gdp = height(gdp_tbl);
gdp_growth = NaN(n_gdp, 1);
for t = 2:n_gdp
    gdp_growth(t) = 400 * log(gdp_tbl.gdp(t) / gdp_tbl.gdp(t-1));
end
gdp_tbl.gdp_growth = gdp_growth;

% Annualised inflation
n_def = height(gdpdef_tbl);
inflation = NaN(n_def, 1);
for t = 2:n_def
    inflation(t) = 400 * log(gdpdef_tbl.gdpdef(t) / gdpdef_tbl.gdpdef(t-1));
end
gdpdef_tbl.inflation = inflation;

% Merge quarterly tables on date
[~, ia, ib] = intersect(gdp_tbl.date, gdpdef_tbl.date);
quarterly.date       = gdp_tbl.date(ia);
quarterly.gdp_growth = gdp_tbl.gdp_growth(ia);
quarterly.inflation  = gdpdef_tbl.inflation(ib);

fprintf('Monthly Fed Funds:  %d observations\n', height(fedfunds_tbl));
fprintf('  Range: %s to %s\n', ...
    datestr(fedfunds_tbl.date(1),'yyyy-mm'), ...
    datestr(fedfunds_tbl.date(end),'yyyy-mm'));
fprintf('Quarterly GDP:      %d observations\n', height(gdp_tbl));

% =============================================================================
% ALIGN_MIXED_FREQUENCY
% =============================================================================

fprintf('\n>>> Aligning mixed-frequency data...\n');

m = 3;   % months per quarter
K = 12;  % number of monthly lags (= 4 quarters)

[y, X_hf, dates_aligned] = align_mixed_frequency( ...
    quarterly, fedfunds_tbl, m, K);

fprintf('Aligned sample:     %d quarterly observations\n', length(y));
fprintf('HF matrix:          %d x %d\n', size(X_hf,1), size(X_hf,2));

% =============================================================================
% 2. ESTIMATE MIDAS MODELS
% =============================================================================

fprintf('\n>>> Estimating MIDAS with Beta polynomial weights...\n');
midas_beta = fit_midas(y, X_hf, 'beta', [1.0, 3.0]);
print_midas_summary(midas_beta, 'BETA');

fprintf('\n>>> Estimating MIDAS with Exponential Almon weights...\n');
midas_almon = fit_midas(y, X_hf, 'exp_almon', [-0.05, -0.01]);
print_midas_summary(midas_almon, 'EXP. ALMON');

fprintf('\n>>> Estimating Unrestricted MIDAS (U-MIDAS)...\n');
umidas = fit_umidas(y, X_hf);
print_umidas_summary(umidas);

% =============================================================================
% 3. MODEL COMPARISON
% =============================================================================

fprintf('\n%s\n', repmat('=',1,70));
fprintf('MODEL COMPARISON\n');
fprintf('%s\n', repmat('=',1,70));
fprintf('%-25s %10s %10s %12s %12s\n','Model','R2','Adj.R2','AIC','BIC');
fprintf('%s\n', repmat('-',1,70));
fprintf('%-25s %10.4f %10.4f %12.2f %12.2f\n', ...
    'MIDAS (Beta)',       midas_beta.r2,  midas_beta.adj_r2,  midas_beta.aic,  midas_beta.bic);
fprintf('%-25s %10.4f %10.4f %12.2f %12.2f\n', ...
    'MIDAS (Exp. Almon)', midas_almon.r2, midas_almon.adj_r2, midas_almon.aic, midas_almon.bic);
fprintf('%-25s %10.4f %10.4f %12.2f %12.2f\n', ...
    'U-MIDAS',           umidas.r2,      umidas.adj_r2,      umidas.aic,      umidas.bic);
fprintf('%s\n', repmat('=',1,70));

% =============================================================================
% 4. VISUALISATIONS
% =============================================================================

fprintf('\n>>> Generating visualisations...\n');

% Plot: Compare weight functions
fig1 = compare_weight_functions(12);
saveas(fig1, 'midas_weight_comparison.png');
fprintf('  Saved: midas_weight_comparison.png\n');

% Plot: Estimated Beta weights
% midas_beta.K = m*K_lags (total HF lags = 36); use actual length of weights vector
K_w  = midas_beta.K;   % total number of HF lags used in estimation (m*K = 36)
lags = (1:K_w)';
fig2 = figure('Visible','off');
bar(lags, midas_beta.weights, 'FaceColor',[0.27 0.51 0.71],'FaceAlpha',0.7,'EdgeColor','k');
xlabel('Lag (months)','FontSize',12);
ylabel('Weight','FontSize',12);
title('Estimated Beta Polynomial Weights','FontSize',14);
grid on; box on;
saveas(fig2, 'midas_beta_weights.png');
fprintf('  Saved: midas_beta_weights.png\n');

% Plot: Fitted vs Actual
fig3 = figure('Visible','off','Position',[100 100 900 600]);
subplot(2,1,1);
plot(dates_aligned, y, 'b-', 'LineWidth', 1.5); hold on;
plot(dates_aligned, midas_beta.fitted, 'r--', 'LineWidth', 1.5);
yline(0,'k-','LineWidth',0.5);
legend('Actual GDP Growth','MIDAS (Beta) Fitted','Location','northeast');
xlabel('Date'); ylabel('GDP Growth (%)');
title('MIDAS (Beta): GDP Growth - Fitted vs Actual','FontSize',13);
grid on;
subplot(2,1,2);
scatter(midas_beta.fitted, y, 30, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
mn = min([y; midas_beta.fitted]); mx = max([y; midas_beta.fitted]);
plot([mn mx],[mn mx],'r--','LineWidth',1.5);
xlabel('Fitted Values'); ylabel('Actual Values');
title('Fitted vs Actual'); grid on;
saveas(fig3, 'midas_fitted_vs_actual.png');
fprintf('  Saved: midas_fitted_vs_actual.png\n');

% Plot: Parametric vs U-MIDAS weights
umidas_coefs = umidas.params(2:end);   % length = K_w = 36
umidas_norm  = abs(umidas_coefs) / sum(abs(umidas_coefs));
fig4 = figure('Visible','off');
bar(lags, umidas_norm, 'FaceColor',[0.7 0.7 0.7], 'FaceAlpha',0.6); hold on;
plot(lags, midas_beta.weights,  'ro-', 'LineWidth',1.5,'MarkerSize',6);
plot(lags, midas_almon.weights, 'gs--','LineWidth',1.5,'MarkerSize',6);
legend('U-MIDAS (norm. |beta|)','Beta weights','Exp. Almon weights');
xlabel('Lag (months)','FontSize',12);
ylabel('Weight','FontSize',12);
title('Comparison: Parametric vs Unrestricted MIDAS Weights','FontSize',13);
grid on;
saveas(fig4, 'midas_weights_comparison.png');
fprintf('  Saved: midas_weights_comparison.png\n');

fprintf('\n>>> MIDAS estimation complete!\n');
fprintf('%s\n', repmat('=',1,70));


% =============================================================================
% LOCAL FUNCTIONS
% =============================================================================

function [y, X_hf, dates_out] = align_mixed_frequency(quarterly, monthly_tbl, m, K)
%ALIGN_MIXED_FREQUENCY  Align quarterly and monthly data for MIDAS.
%
%  quarterly  - struct with fields: date (datetime), gdp_growth (double)
%  monthly_tbl - table with columns: date (datetime), fedfunds (double)
%  m           - months per quarter (3)
%  K           - number of lags
%
%  Returns:
%    y         - (T,1) quarterly GDP growth
%    X_hf      - (T, m*K) high-frequency matrix
%    dates_out - (T,1) datetime of each observation

n_q    = length(quarterly.date);
y_out  = [];
X_rows = {};
d_out  = datetime.empty(0,1);

% Pre-compute quarter label for each monthly row
monthly_quarter = dateshift(monthly_tbl.date, 'start', 'quarter');

for idx = K+1 : n_q
    q_date = quarterly.date(idx);
    y_val  = quarterly.gdp_growth(idx);
    if isnan(y_val), continue; end

    hf_vals = [];
    valid   = true;

    for lag = 0 : K-1
        lag_q_date = quarterly.date(idx - lag);
        mask       = (monthly_quarter == lag_q_date);
        q_monthly  = monthly_tbl.fedfunds(mask);
        q_monthly  = q_monthly(~isnan(q_monthly));

        if length(q_monthly) < m
            valid = false; break
        end
        % Take last m, most-recent first
        vals    = q_monthly(end-m+1:end);
        hf_vals = [hf_vals; flipud(vals)];  %#ok<AGROW>
    end

    if valid && length(hf_vals) == m*K
        y_out  = [y_out; y_val];          %#ok<AGROW>
        X_rows{end+1} = hf_vals';         %#ok<AGROW>
        d_out  = [d_out; q_date];         %#ok<AGROW>
    end
end

y       = y_out;
X_hf    = cell2mat(X_rows');
dates_out = d_out;
end


function w = exponential_almon_weights(K, theta1, theta2)
%EXPONENTIAL_ALMON_WEIGHTS  Normalised exp-Almon weights.
k = (1:K)';
w = exp(theta1*k + theta2*k.^2);
w = w / sum(w);
end


function w = beta_weights(K, theta1, theta2)
%BETA_WEIGHTS  Normalised Beta polynomial weights.
eps_val = 1e-6;
theta1  = max(theta1, eps_val);
theta2  = max(theta2, eps_val);
k       = (1:K)';
x       = k / (K + 1);
w       = x.^(theta1-1) .* (1-x).^(theta2-1);
w       = max(w, eps_val);
w       = w / sum(w);
end


function result = fit_midas(y, X_hf, weight_type, theta_init)
%FIT_MIDAS  Fit parametric MIDAS regression via nonlinear least squares.
%
%  weight_type: 'beta' or 'exp_almon'
%  theta_init:  1x2 initial values for theta

T = length(y);
K = size(X_hf, 2);

if nargin < 4 || isempty(theta_init)
    theta_init = [1.0 1.0];
end

% OLS initialisation with equal weights
X_eq  = mean(X_hf, 2);
b_ols = [ones(T,1) X_eq] \ y;

params0 = [b_ols(:); theta_init(:)];

% Objective function
    function ssr = objective(params)
        alpha = params(1); beta = params(2);
        theta = params(3:4);
        if strcmp(weight_type,'beta')
            w = beta_weights(K, theta(1), theta(2));
        else
            w = exponential_almon_weights(K, theta(1), theta(2));
        end
        yhat = alpha + beta * (X_hf * w);
        ssr  = sum((y - yhat).^2);
    end

% Bounds
if strcmp(weight_type,'beta')
    lb = [-Inf -Inf 0.01 0.01];
    ub = [ Inf  Inf 10.0 10.0];
else
    lb = [-Inf -Inf -5 -5];
    ub = [ Inf  Inf  5  5];
end

opts = optimoptions('fmincon', 'Display','off', ...
    'MaxIterations', 2000, 'OptimalityTolerance', 1e-9, ...
    'StepTolerance', 1e-10, 'Algorithm', 'interior-point');

[params_opt, ssr_opt] = fmincon(@objective, params0, [], [], [], [], lb, ub, [], opts);

alpha_est = params_opt(1);
beta_est  = params_opt(2);
theta_est = params_opt(3:4);

if strcmp(weight_type,'beta')
    w = beta_weights(K, theta_est(1), theta_est(2));
else
    w = exponential_almon_weights(K, theta_est(1), theta_est(2));
end

fitted    = alpha_est + beta_est * (X_hf * w);
residuals = y - fitted;

% Numerical Hessian for standard errors
n_par = 4;
H     = zeros(n_par, n_par);
eps_h = 1e-5;
for ii = 1:n_par
    for jj = 1:n_par
        pp = params_opt; pm = params_opt; mp = params_opt; mm = params_opt;
        pp(ii) = pp(ii)+eps_h; pp(jj) = pp(jj)+eps_h;
        pm(ii) = pm(ii)+eps_h; pm(jj) = pm(jj)-eps_h;
        mp(ii) = mp(ii)-eps_h; mp(jj) = mp(jj)+eps_h;
        mm(ii) = mm(ii)-eps_h; mm(jj) = mm(jj)-eps_h;
        H(ii,jj) = (objective(pp)-objective(pm)-objective(mp)+objective(mm)) / (4*eps_h^2);
    end
end

sigma2 = sum(residuals.^2) / (T - n_par);
try
    vcov = 2 * sigma2 * inv(H); %#ok<MINV>
    se   = sqrt(abs(diag(vcov)));
catch
    se   = NaN(n_par,1);
end

ss_tot  = sum((y - mean(y)).^2);
ss_res  = sum(residuals.^2);
r2      = 1 - ss_res / ss_tot;
adj_r2  = 1 - (1-r2) * (T-1) / (T-n_par);
aic_val = T * log(ss_res/T) + 2*n_par;
bic_val = T * log(ss_res/T) + n_par*log(T);

result.params      = params_opt;
result.weights     = w;
result.fitted      = fitted;
result.residuals   = residuals;
result.se          = se;
result.r2          = r2;
result.adj_r2      = adj_r2;
result.sigma       = sqrt(sigma2);
result.aic         = aic_val;
result.bic         = bic_val;
result.ssr         = ssr_opt;
result.T           = T;
result.K           = K;
result.weight_type = weight_type;
end


function result = fit_umidas(y, X_hf)
%FIT_UMIDAS  Unrestricted MIDAS via OLS.
T = length(y);
K = size(X_hf, 2);
X = [ones(T,1) X_hf];

params    = X \ y;
fitted    = X * params;
residuals = y - fitted;

sigma2 = sum(residuals.^2) / (T - K - 1);
vcov   = sigma2 * inv(X'*X); %#ok<MINV>
se     = sqrt(diag(vcov));

ss_tot = sum((y - mean(y)).^2);
ss_res = sum(residuals.^2);
r2     = 1 - ss_res / ss_tot;
adj_r2 = 1 - (1-r2) * (T-1) / (T-K-1);
aic_v  = T * log(ss_res/T) + 2*(K+1);
bic_v  = T * log(ss_res/T) + (K+1)*log(T);

result.params    = params;
result.fitted    = fitted;
result.residuals = residuals;
result.se        = se;
result.r2        = r2;
result.adj_r2    = adj_r2;
result.sigma     = sqrt(sigma2);
result.aic       = aic_v;
result.bic       = bic_v;
result.T         = T;
result.K         = K;
end


function print_midas_summary(m, label)
%PRINT_MIDAS_SUMMARY  Display estimation results.
fprintf('\n%s\n', repmat('=',1,70));
fprintf('MIDAS REGRESSION RESULTS — %s\n', label);
fprintf('%s\n', repmat('=',1,70));
fprintf('Observations: %d   HF lags: %d\n', m.T, m.K);
fprintf('%s\n', repmat('-',1,70));
fprintf('%-15s %12s %12s %12s\n','Parameter','Estimate','Std.Err','t-stat');
fprintf('%s\n', repmat('-',1,70));
nms = {'alpha','beta','theta1','theta2'};
for i = 1:4
    fprintf('%-15s %12.4f %12.4f %12.2f\n', ...
        nms{i}, m.params(i), m.se(i), m.params(i)/m.se(i));
end
fprintf('%s\n', repmat('-',1,70));
fprintf('  R-squared:      %.4f\n', m.r2);
fprintf('  Adj. R-squared: %.4f\n', m.adj_r2);
fprintf('  Sigma:          %.4f\n', m.sigma);
fprintf('  AIC:            %.2f\n',  m.aic);
fprintf('  BIC:            %.2f\n',  m.bic);
fprintf('%s\n', repmat('=',1,70));
end


function print_umidas_summary(m)
%PRINT_UMIDAS_SUMMARY  Display U-MIDAS results.
fprintf('\n%s\n', repmat('=',1,70));
fprintf('UNRESTRICTED MIDAS (U-MIDAS) REGRESSION RESULTS\n');
fprintf('%s\n', repmat('=',1,70));
fprintf('Observations: %d   HF lags: %d\n', m.T, m.K);
fprintf('%s\n', repmat('-',1,70));
fprintf('%-15s %12s %12s %12s\n','Parameter','Estimate','Std.Err','t-stat');
fprintf('%s\n', repmat('-',1,70));
show = [1, 2:min(6,length(m.params))];
if length(m.params) > 10
    show = [show, length(m.params)-4:length(m.params)];
end
show = unique(show);
prev = 0;
for i = show
    if i > 6 && i ~= prev+1, fprintf('  ...\n'); end
    nm = 'alpha'; if i > 1, nm = sprintf('beta_lag%d',i-1); end
    fprintf('%-15s %12.4f %12.4f %12.2f\n', nm, m.params(i), m.se(i), m.params(i)/m.se(i));
    prev = i;
end
fprintf('%s\n', repmat('-',1,70));
fprintf('  R-squared:      %.4f\n', m.r2);
fprintf('  Adj. R-squared: %.4f\n', m.adj_r2);
fprintf('  Sigma:          %.4f\n', m.sigma);
fprintf('  AIC:            %.2f\n',  m.aic);
fprintf('  BIC:            %.2f\n',  m.bic);
fprintf('%s\n', repmat('=',1,70));
end


function fig = compare_weight_functions(K)
%COMPARE_WEIGHT_FUNCTIONS  Plot different MIDAS weighting schemes.
lags = 1:K;

fig = figure('Visible','off','Position',[50 50 1200 800]);

% Beta weights
subplot(2,2,1);
configs = {1,1,'Uniform'; 1,5,'Declining'; 2,2,'Hump'; 5,1,'Increasing'};
for c = 1:size(configs,1)
    w = beta_weights(K, configs{c,1}, configs{c,2});
    plot(lags, w, 'o-','LineWidth',1.2,'DisplayName', ...
        sprintf('%s (%d,%d)',configs{c,3},configs{c,1},configs{c,2}));
    hold on;
end
legend('Location','northeast','FontSize',8); grid on;
xlabel('Lag'); ylabel('Weight');
title('Beta Polynomial Weights');

% Exp Almon weights
subplot(2,2,2);
configs_a = {0,0,'Uniform'; -0.1,0,'Declining'; 0.1,-0.02,'Hump'; 0.1,0,'Increasing'};
for c = 1:size(configs_a,1)
    w = exponential_almon_weights(K, configs_a{c,1}, configs_a{c,2});
    plot(lags, w, 'o-','LineWidth',1.2,'DisplayName', ...
        sprintf('%s',configs_a{c,3}));
    hold on;
end
legend('Location','northeast','FontSize',8); grid on;
xlabel('Lag'); ylabel('Weight');
title('Exponential Almon Weights');

% Step function
subplot(2,2,3);
for n_steps = [2,3,4]
    step_size = K / n_steps;
    w = zeros(K,1);
    for s = 1:n_steps
        st = round((s-1)*step_size)+1;
        en = round(s*step_size);
        w(st:en) = 1/n_steps;
    end
    stairs(lags, w, 'LineWidth',2,'DisplayName',sprintf('%d steps',n_steps));
    hold on;
end
legend; grid on;
xlabel('Lag'); ylabel('Weight');
title('Step Function Weights');

% PDL basis
subplot(2,2,4);
k_vec = (1:K)';
plot(lags, ones(K,1),          'o-','LineWidth',1.2,'DisplayName','Constant');
plot(lags, k_vec/K,            's-','LineWidth',1.2,'DisplayName','Linear (scaled)');
plot(lags, (k_vec.^2)/K^2,     '^-','LineWidth',1.2,'DisplayName','Quadratic (scaled)');
legend; grid on;
xlabel('Lag'); ylabel('Basis Value');
title('PDL Polynomial Basis');

sgtitle('Comparison of MIDAS Weighting Schemes','FontSize',14);
end
