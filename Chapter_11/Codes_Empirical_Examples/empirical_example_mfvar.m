% =============================================================================
% Author:   Alessia Paccagnini
% Textbook: Macroeconometrics
% Mixed-Frequency VAR (MF-VAR) Estimation in MATLAB
% =============================================================================
% Translation of mfvar_estimation.py for the Macroeconometrics Textbook
% Chapter 11: Mixed-Frequency Data, Section 11.9
%
% Following Ghysels (2016), Journal of Econometrics
% "Macroeconomics and the Reality of Mixed Frequency Data"
%
% Key References:
%   Ghysels, E. (2016). JoE.
%   Ghysels, E., Hill, J., & Motegi, K. (2016). Granger causality tests.
%   Foroni, C. & Marcellino, M. (2014). Mixed frequency structural VARs.
%
% Requirements: MATLAB R2019b+
%   Toolboxes: Statistics and Machine Learning (for chi2cdf)
%
% Usage:
%   Run this script directly from the MATLAB command window or editor.
%   Data files must be in the working directory (or update paths below).
% =============================================================================

clear; clc; close all;

fprintf('\n%s\n', repmat('=',1,75));
fprintf('MF-VAR ESTIMATION: Following Ghysels (2016, JoE)\n');
fprintf('%s\n', repmat('=',1,75));

% =============================================================================
% 1. DATA LOADING
% =============================================================================

fprintf('\n>>> Loading data...\n');

fedfunds_tbl = readtable('FEDFUNDS.xlsx', 'Sheet', 'Monthly');
gdp_tbl      = readtable('GDPC1.xlsx',    'Sheet', 'Quarterly');
gdpdef_tbl   = readtable('GDPDEF.xlsx',   'Sheet', 'Quarterly');

fedfunds_tbl.Properties.VariableNames = {'date','fedfunds'};
gdp_tbl.Properties.VariableNames      = {'date','gdp'};
gdpdef_tbl.Properties.VariableNames   = {'date','gdpdef'};

if ~isdatetime(fedfunds_tbl.date), fedfunds_tbl.date = datetime(fedfunds_tbl.date); end
if ~isdatetime(gdp_tbl.date),      gdp_tbl.date      = datetime(gdp_tbl.date);      end
if ~isdatetime(gdpdef_tbl.date),   gdpdef_tbl.date   = datetime(gdpdef_tbl.date);   end

fedfunds_tbl = sortrows(fedfunds_tbl, 'date');
gdp_tbl      = sortrows(gdp_tbl,      'date');

fprintf('Monthly Fed Funds:  %d observations\n', height(fedfunds_tbl));
fprintf('Quarterly GDP:      %d observations\n', height(gdp_tbl));

% =============================================================================
% 2. CREATE STACKED MF-VAR DATA
% =============================================================================

fprintf('\n>>> Creating stacked MF-VAR data...\n');

[Z, var_names] = create_stacked_mfvar_data(fedfunds_tbl, gdp_tbl, ...
    'fedfunds', 'gdp', 'growth', 3);

fprintf('Stacked data:       %d rows x %d variables\n', size(Z,1), size(Z,2));
fprintf('Variables: %s\n', strjoin(var_names, ', '));
fprintf('First 5 observations:\n');
disp(array2table(Z(1:5,:), 'VariableNames', var_names));

% =============================================================================
% 3. ESTIMATE MF-VAR(1)
% =============================================================================

fprintf('\n>>> Estimating MF-VAR(1)...\n');
mfvar1 = fit_mfvar(Z, var_names, 1);
print_mfvar_summary(mfvar1);

% =============================================================================
% 4. GRANGER CAUSALITY TESTS
% =============================================================================

fprintf('\n>>> Granger Causality Tests:\n%s\n', repmat('-',1,75));

% Monthly FFR => quarterly GDP
gc_ffr_gdp = granger_causality_test(mfvar1, ...
    {'fedfunds_m1','fedfunds_m2','fedfunds_m3'}, {'gdp'});
fprintf('\nTest: Monthly Fed Funds => GDP Growth\n');
fprintf('  Wald statistic: %.4f\n', gc_ffr_gdp.Wald);
fprintf('  p-value:        %.4f\n', gc_ffr_gdp.pval);
fprintf('  Conclusion:     %s at 5%% level\n', ...
    conditional_str(gc_ffr_gdp.pval < 0.05, 'Reject H0', 'Fail to reject H0'));

% Quarterly GDP => monthly FFR
gc_gdp_ffr = granger_causality_test(mfvar1, ...
    {'gdp'}, {'fedfunds_m1','fedfunds_m2','fedfunds_m3'});
fprintf('\nTest: GDP Growth => Monthly Fed Funds\n');
fprintf('  Wald statistic: %.4f\n', gc_gdp_ffr.Wald);
fprintf('  p-value:        %.4f\n', gc_gdp_ffr.pval);
fprintf('  Conclusion:     %s at 5%% level\n', ...
    conditional_str(gc_gdp_ffr.pval < 0.05, 'Reject H0', 'Fail to reject H0'));

% =============================================================================
% 5. VISUALISATIONS — IRF (no CI, quick)
% =============================================================================

fprintf('\n>>> Generating visualisations...\n');

fig1 = plot_irf_simple(mfvar1, 'fedfunds_m1', 16);
saveas(fig1, 'mfvar_irf_ffr_m1.png');
fprintf('  Saved: mfvar_irf_ffr_m1.png\n');

fig2 = plot_irf_simple(mfvar1, 'gdp', 16);
saveas(fig2, 'mfvar_irf_gdp.png');
fprintf('  Saved: mfvar_irf_gdp.png\n');

fig3 = plot_fevd(mfvar1, 16);
saveas(fig3, 'mfvar_fevd.png');
fprintf('  Saved: mfvar_fevd.png\n');

% =============================================================================
% 6. IRFs WITH BOOTSTRAP CI
% =============================================================================

fprintf('\n>>> Computing IRFs with 90%% bootstrap CI (500 reps)...\n');
ci_ffr_m1 = bootstrap_irf_ci(mfvar1, 'fedfunds_m1', 16, 500, 0.90, 42);

fig4 = plot_irf_with_ci(ci_ffr_m1, mfvar1.var_names, 'fedfunds_m1', 16, 0.90);
saveas(fig4, 'mfvar_irf_ffr_m1_ci.png');
fprintf('  Saved: mfvar_irf_ffr_m1_ci.png\n');

% Compare GDP response to shocks in different months
fprintf('\n>>> Comparing GDP response to shocks at different months...\n');
fig5 = figure('Visible','off','Position',[50 50 1400 500]);
shock_list = {'fedfunds_m1','fedfunds_m2','fedfunds_m3'};
gdp_idx    = find(strcmp(mfvar1.var_names,'gdp'));
colours    = {'b','g','r'};
for si = 1:3
    ci_s = bootstrap_irf_ci(mfvar1, shock_list{si}, 16, 500, 0.90, 120+si);
    subplot(1,3,si);
    h_vec = (0:16)';
    fill([h_vec; flipud(h_vec)], ...
         [ci_s.lower(:,gdp_idx); flipud(ci_s.upper(:,gdp_idx))], ...
         colours{si}, 'FaceAlpha',0.25,'EdgeColor','none'); hold on;
    plot(h_vec, ci_s.irf(:,gdp_idx), [colours{si},'-'], 'LineWidth', 1.8);
    yline(0,'k-','LineWidth',0.5);
    title(sprintf('Shock to %s', shock_list{si}));
    xlabel('Quarters'); ylabel('GDP Response');
    grid on;
end
sgtitle('Response of GDP to Different Interest Rate Shocks (90% Bootstrap CI)','FontSize',12);
saveas(fig5, 'mfvar_gdp_response_by_month.png');
fprintf('  Saved: mfvar_gdp_response_by_month.png\n');

% =============================================================================
% 7. IRF SIGNIFICANCE SUMMARY
% =============================================================================

fprintf('\n>>> IRF Significance Summary (90%% CI):\n%s\n', repmat('-',1,75));
for si = 1:3
    sv  = shock_list{si};
    ci_s = bootstrap_irf_ci(mfvar1, sv, 8, 300, 0.90, 42);
    fprintf('\n  Shock to %s:\n', sv);
    for h = [0,1,2,4,8]
        pt  = ci_s.irf(h+1,   gdp_idx);
        lo  = ci_s.lower(h+1, gdp_idx);
        hi  = ci_s.upper(h+1, gdp_idx);
        sig = conditional_str(lo>0 || hi<0, '*', '');
        fprintf('    h=%d: %7.3f [%7.3f, %7.3f] %s\n', h, pt, lo, hi, sig);
    end
end

% =============================================================================
% 8. MF-VAR vs LF-VAR COMPARISON
% =============================================================================

fprintf('\n>>> Comparing MF-VAR with traditional LF-VAR...\n');

% Quarterly average of monthly Fed Funds
monthly_quarter = dateshift(fedfunds_tbl.date, 'start', 'quarter');
q_dates_ffr     = unique(monthly_quarter);
ffr_q_vals      = arrayfun(@(q) mean(fedfunds_tbl.fedfunds(monthly_quarter == q)), q_dates_ffr);

% GDP growth series
n_gdp = height(gdp_tbl);
gdp_growth = NaN(n_gdp,1);
for t = 2:n_gdp
    gdp_growth(t) = 400 * log(gdp_tbl.gdp(t) / gdp_tbl.gdp(t-1));
end
gdp_growth_dates = gdp_tbl.date;

% Merge on common dates
[common_dates, ia, ib] = intersect(q_dates_ffr, gdp_growth_dates);
ffr_q_matched  = ffr_q_vals(ia);
gdp_g_matched  = gdp_growth(ib);
valid_idx       = ~isnan(gdp_g_matched);
lf_data         = [ffr_q_matched(valid_idx), gdp_g_matched(valid_idx)];
lf_var_names    = {'fedfunds','gdp_growth'};

lfvar = fit_mfvar(lf_data, lf_var_names, 1);

fprintf('\nLF-VAR vs MF-VAR coefficient comparison:\n');
fprintf('  A1[FFR->FFR]: LF=%.4f  vs  MF(m1->m1)=%.4f\n', ...
    lfvar.A{1}(1,1), mfvar1.A{1}(1,1));
fprintf('  A1[FFR->GDP]: LF=%.4f  vs  MF(m1->GDP)=%.4f\n', ...
    lfvar.A{1}(2,1), mfvar1.A{1}(4,1));
fprintf('  A1[GDP->FFR]: LF=%.4f  vs  MF(GDP->m1)=%.4f\n', ...
    lfvar.A{1}(1,2), mfvar1.A{1}(1,4));
fprintf('  A1[GDP->GDP]: LF=%.4f  vs  MF(GDP->GDP)=%.4f\n', ...
    lfvar.A{1}(2,2), mfvar1.A{1}(4,4));

% Comparison plot
irf_mf = compute_irf(mfvar1, 'fedfunds_m1', 16);
irf_lf = compute_irf(lfvar,  'fedfunds',    16);
h_vec  = (0:16)';

fig6 = figure('Visible','off','Position',[50 50 1200 500]);
subplot(1,2,1);
plot(h_vec, irf_mf(:,4), 'b-',  'LineWidth',1.8, 'DisplayName','MF-VAR'); hold on;
plot(h_vec, irf_lf(:,2), 'r--', 'LineWidth',1.8, 'DisplayName','LF-VAR');
yline(0,'k-','LineWidth',0.5);
legend; grid on;
xlabel('Quarters'); ylabel('Response');
title('Response of GDP Growth');

subplot(1,2,2);
plot(h_vec, irf_mf(:,1), 'b-',  'LineWidth',1.8, 'DisplayName','MF-VAR'); hold on;
plot(h_vec, irf_lf(:,1), 'r--', 'LineWidth',1.8, 'DisplayName','LF-VAR');
yline(0,'k-','LineWidth',0.5);
legend; grid on;
xlabel('Quarters'); ylabel('Response');
title('Response of Interest Rate');

sgtitle('MF-VAR vs LF-VAR: Impulse Response Comparison','FontSize',13);
saveas(fig6, 'mfvar_vs_lfvar_comparison.png');
fprintf('  Saved: mfvar_vs_lfvar_comparison.png\n');

% =============================================================================
% 9. MF-VAR(2) LAG SELECTION
% =============================================================================

fprintf('\n>>> Estimating MF-VAR(2) for comparison...\n');
mfvar2 = fit_mfvar(Z, var_names, 2);

ic1 = compute_ic(mfvar1);
ic2 = compute_ic(mfvar2);

fprintf('\nLag Selection:\n%s\n', repmat('-',1,60));
fprintf('%-12s %12s %12s %12s\n','Model','Log-Lik','AIC','BIC');
fprintf('%s\n', repmat('-',1,60));
fprintf('%-12s %12.2f %12.2f %12.2f\n','MF-VAR(1)', ic1.ll, ic1.aic, ic1.bic);
fprintf('%-12s %12.2f %12.2f %12.2f\n','MF-VAR(2)', ic2.ll, ic2.aic, ic2.bic);
fprintf('BIC prefers: %s\n', conditional_str(ic1.bic < ic2.bic,'MF-VAR(1)','MF-VAR(2)'));

fprintf('\n>>> MF-VAR estimation complete!\n');
fprintf('%s\n', repmat('=',1,75));


% =============================================================================
% LOCAL FUNCTIONS
% =============================================================================

function [Z_mat, var_names] = create_stacked_mfvar_data( ...
    monthly_tbl, quarterly_tbl, monthly_var, quarterly_var, transform, m)
%CREATE_STACKED_MFVAR_DATA  Build the Ghysels (2016) stacked MF-VAR dataset.

quarterly_tbl = sortrows(quarterly_tbl, 'date');
monthly_tbl   = sortrows(monthly_tbl,   'date');

n_q        = height(quarterly_tbl);
gdp_vals   = quarterly_tbl.(quarterly_var);
gdp_dates  = quarterly_tbl.date;

% Compute transformed quarterly variable
y_q = NaN(n_q,1);
if strcmp(transform,'growth')
    for t = 2:n_q
        y_q(t) = 400 * log(gdp_vals(t) / gdp_vals(t-1));
    end
else
    y_q = gdp_vals;
end

% Quarter label for monthly data
monthly_quarter = dateshift(monthly_tbl.date, 'start', 'quarter');

rows    = {};
for i = 1:n_q
    q_date = gdp_dates(i);
    y_val  = y_q(i);
    if isnan(y_val), continue; end

    mask      = (monthly_quarter == q_date);
    q_monthly = monthly_tbl.(monthly_var)(mask);
    q_monthly = sort(q_monthly);   % chronological (already sorted)

    if length(q_monthly) < m, continue; end

    row = [q_monthly(1), q_monthly(2), q_monthly(3), y_val];
    rows{end+1} = row; %#ok<AGROW>
end

Z_mat     = cell2mat(rows');
var_names = {[monthly_var '_m1'], [monthly_var '_m2'], ...
             [monthly_var '_m3'], quarterly_var};
end


function [Y_mat, X_mat] = create_var_matrices(data_mat, p)
%CREATE_VAR_MATRICES  Build Y and X matrices for VAR estimation.
T_full = size(data_mat,1);
k      = size(data_mat,2);

Y_mat = data_mat(p+1:end, :);

X_mat = ones(T_full-p, 1);
for lag = 1:p
    X_mat = [X_mat, data_mat(p-lag+1:T_full-lag, :)]; %#ok<AGROW>
end
end


function model = fit_mfvar(Z, var_names, p)
%FIT_MFVAR  Estimate MF-VAR by OLS.
%
%  Z         - (T_full x k) numeric matrix OR table/struct with var_names cols
%  var_names - cell array of variable name strings
%  p         - lag order

if isstruct(Z) || istable(Z)
    if istable(Z)
        data_mat = table2array(Z(:, var_names));
    else
        data_mat = zeros(length(Z.(var_names{1})), length(var_names));
        for vi = 1:length(var_names)
            data_mat(:,vi) = Z.(var_names{vi});
        end
    end
else
    data_mat = Z;
end

k = size(data_mat,2);
[Y_mat, X_mat] = create_var_matrices(data_mat, p);
T_eff = size(Y_mat,1);

XtX_inv = inv(X_mat' * X_mat); %#ok<MINV>
B        = XtX_inv * (X_mat' * Y_mat);   % (1+k*p) x k

c_vec = B(1,:)';                          % k x 1
A_cell = cell(p,1);
for lag = 1:p
    idx = (2 + (lag-1)*k) : (1 + lag*k);
    A_cell{lag} = B(idx,:)';              % k x k
end

fitted    = X_mat * B;
residuals = Y_mat - fitted;
Sigma     = (residuals' * residuals) / T_eff;

% Standard errors
se_c = zeros(k,1);
se_A = cell(p,1);
for lag = 1:p, se_A{lag} = zeros(k,k); end

for eq = 1:k
    var_b = Sigma(eq,eq) * diag(XtX_inv);
    se_b  = sqrt(var_b);
    se_c(eq) = se_b(1);
    for lag = 1:p
        idx = (2 + (lag-1)*k) : (1 + lag*k);
        se_A{lag}(eq,:) = se_b(idx)';
    end
end

model.A         = A_cell;
model.c         = c_vec;
model.Sigma     = Sigma;
model.residuals = residuals;
model.fitted    = fitted;
model.se_c      = se_c;
model.se_A      = se_A;
model.T         = T_eff;
model.k         = k;
model.p         = p;
model.var_names = var_names;
model.data_mat  = data_mat;
end


function Psi = compute_ma_coefs(model, periods)
%COMPUTE_MA_COEFS  MA-representation Psi_0, ..., Psi_periods (cell array).
k   = model.k;
Psi = cell(periods+1,1);
Psi{1} = eye(k);
for h = 1:periods
    Psi_h = zeros(k,k);
    for j = 1:min(h, model.p)
        Psi_h = Psi_h + Psi{h-j+1} * model.A{j};
    end
    Psi{h+1} = Psi_h;
end
end


function irf_mat = compute_irf(model, shock_var, periods, orthogonalized, shock_size)
%COMPUTE_IRF  Compute impulse response functions.

if nargin < 4, orthogonalized = true; end
if nargin < 5, shock_size     = 1.0;  end

if ischar(shock_var)
    shock_idx = find(strcmp(model.var_names, shock_var));
else
    shock_idx = shock_var;
end

k   = model.k;
Psi = compute_ma_coefs(model, periods);

if orthogonalized
    P = chol(model.Sigma, 'lower');   % lower triangular
else
    P = eye(k);
end

impulse = P(:, shock_idx) * shock_size;

irf_mat = zeros(periods+1, k);
for h = 0:periods
    irf_mat(h+1,:) = (Psi{h+1} * impulse)';
end
end


function ci = bootstrap_irf_ci(model, shock_var, periods, n_boot, ci_level, seed)
%BOOTSTRAP_IRF_CI  Bootstrap confidence intervals for IRFs.

if nargin < 6, seed = 42; end
rng(seed);

if ischar(shock_var)
    shock_idx = find(strcmp(model.var_names, shock_var));
else
    shock_idx = shock_var;
end

k      = model.k;
p      = model.p;
T_eff  = model.T;

irf_pt   = compute_irf(model, shock_idx, periods);
boot_irfs = zeros(n_boot, periods+1, k);

for b = 1:n_boot
    idx_boot   = randi(T_eff, T_eff, 1);
    boot_resid = model.residuals(idx_boot,:);

    Z_orig  = model.data_mat;
    Z_new   = zeros(T_eff, k);
    Z_hist  = Z_orig(1:p,:);

    for t = 1:T_eff
        z_t = model.c';
        for lag = 1:p
            z_t = z_t + (model.A{lag} * Z_hist(end-lag+1,:)')';
        end
        z_t      = z_t + boot_resid(t,:);
        Z_new(t,:) = z_t;
        Z_hist   = [Z_hist; z_t];
    end

    Z_full = [Z_orig(1:p,:); Z_new];
    try
        m_b = fit_mfvar(Z_full, model.var_names, p);
        boot_irfs(b,:,:) = compute_irf(m_b, shock_idx, periods);
    catch
        boot_irfs(b,:,:) = irf_pt;
    end
end

alpha_level = 1 - ci_level;
lower = squeeze(quantile(boot_irfs, alpha_level/2,     1));
upper = squeeze(quantile(boot_irfs, 1-alpha_level/2,   1));

ci.irf    = irf_pt;
ci.lower  = lower;
ci.upper  = upper;
ci.n_boot = n_boot;
ci.ci_level = ci_level;
end


function fevd_arr = compute_fevd(model, periods)
%COMPUTE_FEVD  Forecast Error Variance Decomposition.
k   = model.k;
P   = chol(model.Sigma,'lower');
Psi = compute_ma_coefs(model, periods);
Theta = cellfun(@(psi) psi*P, Psi, 'UniformOutput', false);

fevd_arr = zeros(periods+1, k, k);
for h = 0:periods
    mse = zeros(k,k);
    for s = 0:h
        mse = mse + Theta{s+1} * Theta{s+1}';
    end
    mse_diag = diag(mse);
    for j = 1:k
        contrib = zeros(k,1);
        for s = 0:h
            contrib = contrib + Theta{s+1}(:,j).^2;
        end
        fevd_arr(h+1,:,j) = contrib ./ mse_diag;
    end
end
end


function gc = granger_causality_test(model, cause_vars, effect_vars)
%GRANGER_CAUSALITY_TEST  Wald test for Granger non-causality.

if ischar(cause_vars),  cause_vars  = {cause_vars};  end
if ischar(effect_vars), effect_vars = {effect_vars}; end

cause_idx  = cellfun(@(v) find(strcmp(model.var_names,v)), cause_vars);
effect_idx = cellfun(@(v) find(strcmp(model.var_names,v)), effect_vars);

k              = model.k;
p              = model.p;
n_restrictions = length(cause_idx) * length(effect_idx) * p;

[Y_mat, X_mat] = create_var_matrices(model.data_mat, p);
T_eff = size(Y_mat,1);

ssr_u = sum(sum(model.residuals(:,effect_idx).^2));

% Restricted X (drop cause columns from each lag block)
keep = 1;  % constant
for lag = 1:p
    for v = 1:k
        if ~ismember(v, cause_idx)
            keep(end+1) = 1 + (lag-1)*k + v; %#ok<AGROW>
        end
    end
end
X_r   = X_mat(:,keep);
B_r   = X_r \ Y_mat(:,effect_idx);
resid_r = Y_mat(:,effect_idx) - X_r*B_r;
ssr_r = sum(sum(resid_r.^2));

df1   = n_restrictions;
df2   = T_eff*length(effect_idx) - size(X_mat,2)*length(effect_idx);
F_val = ((ssr_r - ssr_u)/df1) / (ssr_u/df2);
W_val = F_val * df1;
p_val = 1 - chi2cdf(W_val, df1);

gc.F     = F_val;
gc.Wald  = W_val;
gc.pval  = p_val;
gc.df    = [df1, df2];
end


function print_mfvar_summary(model)
%PRINT_MFVAR_SUMMARY  Display MF-VAR estimation results.
fprintf('\n%s\n', repmat('=',1,75));
fprintf('MIXED-FREQUENCY VAR ESTIMATION RESULTS\n');
fprintf('Following Ghysels (2016, Journal of Econometrics)\n');
fprintf('%s\n', repmat('=',1,75));
fprintf('Sample size (T):      %d\n', model.T);
fprintf('Variables (k):        %d\n', model.k);
fprintf('Lags (p):             %d\n', model.p);
fprintf('Parameters per eq.:   %d\n', 1+model.k*model.p);
fprintf('Total parameters:     %d\n', model.k*(1+model.k*model.p));

fprintf('\nA1 matrix (first lag):\n');
nms = model.var_names;
k   = model.k;
A1  = model.A{1};
SE1 = model.se_A{1};

hdr = sprintf('%-16s','');
for j = 1:k, hdr = [hdr, sprintf('%14s', nms{j}(1:min(end,13)))]; end %#ok<AGROW>
fprintf('%s\n%s\n', hdr, repmat('-',1,length(hdr)));

for i = 1:k
    row_str = sprintf('%-16s', nms{i}(1:min(end,15)));
    for j = 1:k
        coef  = A1(i,j);
        se_v  = SE1(i,j);
        t_val = abs(coef/se_v);
        if     t_val > 2.576
            stars = '***';
        elseif t_val > 1.960
            stars = '** ';
        elseif t_val > 1.645
            stars = '*  ';
        else
            stars = '   ';
        end
        row_str = [row_str, sprintf('%10.4f%s', coef, stars)]; %#ok<AGROW>
    end
    fprintf('%s\n', row_str);
end
fprintf('Note: *** p<0.01, ** p<0.05, * p<0.10\n');

fprintf('\nIntercepts:\n%s\n', repmat('-',1,50));
for i = 1:k
    fprintf('  %-20s: %8.4f (SE: %.4f)\n', nms{i}, model.c(i), model.se_c(i));
end

ic = compute_ic(model);
fprintf('\n  Log-likelihood: %12.2f\n', ic.ll);
fprintf('  AIC:            %12.2f\n',  ic.aic);
fprintf('  BIC:            %12.2f\n',  ic.bic);
fprintf('%s\n', repmat('=',1,75));
end


function ic = compute_ic(model)
%COMPUTE_IC  Compute log-likelihood, AIC, BIC.
k     = model.k;
T_eff = model.T;
det_S = det(model.Sigma);
ll    = -0.5*T_eff*(k*log(2*pi) + log(det_S) + k);
n_par = k*(1+k*model.p);
ic.ll  = ll;
ic.aic = -2*ll + 2*n_par;
ic.bic = -2*ll + n_par*log(T_eff);
end


function fig = plot_irf_simple(model, shock_var, periods)
%PLOT_IRF_SIMPLE  Plot IRFs without confidence intervals.
irf_mat = compute_irf(model, shock_var, periods);
k       = model.k;
h_vec   = (0:periods)';
nms     = model.var_names;
if ischar(shock_var), shock_name = shock_var; else, shock_name = nms{shock_var}; end

n_col = 2;
n_row = ceil(k/2);
fig = figure('Visible','off','Position',[50 50 1200 800]);
for i = 1:k
    subplot(n_row, n_col, i);
    area(h_vec, irf_mat(:,i), 'FaceColor',[0.27 0.51 0.71], ...
        'FaceAlpha',0.2,'EdgeColor','none'); hold on;
    plot(h_vec, irf_mat(:,i), 'b-', 'LineWidth',1.8);
    yline(0,'k-','LineWidth',0.5);
    title(sprintf('Response of %s', nms{i}));
    xlabel('Quarters'); ylabel('Response');
    grid on;
end
sgtitle(sprintf('Impulse Responses to %s Shock (1 Std Dev)', shock_name),'FontSize',13);
end


function fig = plot_irf_with_ci(ci_res, var_names, shock_name, periods, ci_level)
%PLOT_IRF_WITH_CI  Plot IRFs with bootstrap confidence intervals.
k      = size(ci_res.irf,2);
h_vec  = (0:periods)';
ci_pct = round(ci_level*100);

n_col = 2; n_row = ceil(k/2);
fig = figure('Visible','off','Position',[50 50 1200 800]);

for i = 1:k
    subplot(n_row, n_col, i);
    fill([h_vec; flipud(h_vec)], ...
         [ci_res.lower(:,i); flipud(ci_res.upper(:,i))], ...
         [0.27 0.51 0.71], 'FaceAlpha',0.25,'EdgeColor','none'); hold on;
    plot(h_vec, ci_res.irf(:,i), 'b-','LineWidth',1.8);
    yline(0,'k-','LineWidth',0.5);
    title(sprintf('Response of %s', var_names{i}));
    xlabel('Quarters'); ylabel('Response');
    grid on;
end
sgtitle(sprintf('Impulse Responses to %s (%d%% Bootstrap CI)', shock_name, ci_pct),'FontSize',12);
end


function fig = plot_fevd(model, periods)
%PLOT_FEVD  Plot Forecast Error Variance Decomposition.
fevd_arr = compute_fevd(model, periods);
k        = model.k;
h_vec    = (0:periods)';
nms      = model.var_names;
colours  = lines(k);

n_col = 2; n_row = ceil(k/2);
fig = figure('Visible','off','Position',[50 50 1200 800]);

for i = 1:k
    subplot(n_row, n_col, i);
    bottom = zeros(periods+1,1);
    for j = 1:k
        share = fevd_arr(:,i,j);
        fill([h_vec; flipud(h_vec)], ...
             [bottom; flipud(bottom+share)], ...
             colours(j,:), 'FaceAlpha',0.8,'EdgeColor','none', ...
             'DisplayName', nms{j}); hold on;
        bottom = bottom + share;
    end
    ylim([0,1]); grid on;
    if i==1, legend('Location','northeast','FontSize',8); end
    title(sprintf('FEVD of %s',nms{i}));
    xlabel('Quarters'); ylabel('Share');
end
sgtitle('Forecast Error Variance Decomposition','FontSize',13);
end


function out = conditional_str(cond, str_true, str_false)
%CONDITIONAL_STR  Ternary-like string selector.
if cond, out = str_true; else, out = str_false; end
end
