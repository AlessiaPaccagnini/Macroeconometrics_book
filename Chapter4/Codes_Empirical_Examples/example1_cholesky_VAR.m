%==========================================================================
% EXAMPLE 1: CHOLESKY IDENTIFICATION - 4-VARIABLE VAR
% Author: Alessia Paccagnini
% Textbook: Macroeconometrics
% 
% Variables: GDP Growth, Oil Price Growth, Inflation, Federal Funds Rate
% Sample: 1971:Q1 - 2007:Q4
% Identification: Cholesky (Recursive)
%
% Produces 4 figures:
%   1. IRFs_MP_shock_matlab.png
%   2. Cumulative_responses_matlab.png
%   3. FEVD_matlab.png
%   4. Historical_decomposition_matlab.png
%==========================================================================

clear all; close all; clc;

%% ========================================================================
% SETUP
%==========================================================================
fprintf('========================================\n');
fprintf('4-VARIABLE CHOLESKY IDENTIFICATION\n');
fprintf('========================================\n\n');

%% ========================================================================
% LOAD DATA
%==========================================================================
fprintf('[1/7] Loading data...\n');

% Read Excel file as table
opts = detectImportOptions('2026-01-QD.xlsx');
data_raw = readtable('2026-01-QD.xlsx', opts);

fprintf('   File loaded: %d rows, %d columns\n', height(data_raw), width(data_raw));

% Skip first 2 rows (factor loadings and transformations)
data = data_raw(3:end, :);

% Extract date column
if iscell(data.sasdate)
    dates = datetime(data.sasdate);
elseif isdatetime(data.sasdate)
    dates = data.sasdate;
else
    dates = datetime(data.sasdate, 'ConvertFrom', 'excel');
end

% Function to safely extract numeric data from table
extract_numeric = @(tbl, varname) double(tbl.(varname));

% Extract variables
gdp = extract_numeric(data, 'GDPC1');
deflator = extract_numeric(data, 'GDPCTPI');
fedfunds = extract_numeric(data, 'FEDFUNDS');
oil = extract_numeric(data, 'OILPRICEx');

fprintf('   GDP: %d non-NaN values (out of %d)\n', sum(~isnan(gdp)), length(gdp));
fprintf('   Oil: %d non-NaN values (out of %d)\n', sum(~isnan(oil)), length(oil));
fprintf('   Deflator: %d non-NaN values (out of %d)\n', sum(~isnan(deflator)), length(deflator));
fprintf('   FedFunds: %d non-NaN values (out of %d)\n', sum(~isnan(fedfunds)), length(fedfunds));

% If still all NaN, try alternative extraction
if sum(~isnan(gdp)) == 0
    fprintf('   Trying alternative extraction method...\n');
    if iscell(data.GDPC1)
        gdp = cellfun(@(x) str2double(x), data.GDPC1);
        deflator = cellfun(@(x) str2double(x), data.GDPCTPI);
        fedfunds = cellfun(@(x) str2double(x), data.FEDFUNDS);
        oil = cellfun(@(x) str2double(x), data.OILPRICEx);
    else
        var_indices = [find(strcmp(data.Properties.VariableNames, 'GDPC1')), ...
                      find(strcmp(data.Properties.VariableNames, 'GDPCTPI')), ...
                      find(strcmp(data.Properties.VariableNames, 'FEDFUNDS')), ...
                      find(strcmp(data.Properties.VariableNames, 'OILPRICEx'))];
        data_matrix = table2array(data(:, var_indices));
        gdp = data_matrix(:, 1);
        deflator = data_matrix(:, 2);
        fedfunds = data_matrix(:, 3);
        oil = data_matrix(:, 4);
    end
    fprintf('   After alternative extraction:\n');
    fprintf('   GDP: %d non-NaN values\n', sum(~isnan(gdp)));
    fprintf('   Oil: %d non-NaN values\n', sum(~isnan(oil)));
end

% Compute growth rates
gdp_growth = 100 * diff(log(gdp));
oil_growth = 100 * diff(log(oil));
inflation = 100 * diff(log(deflator));
fedfunds_level = fedfunds(2:end);
dates_growth = dates(2:end);

% Remove any remaining NaN rows
valid_idx = ~isnan(gdp_growth) & ~isnan(oil_growth) & ...
            ~isnan(inflation) & ~isnan(fedfunds_level);

Y_full = [gdp_growth(valid_idx), oil_growth(valid_idx), ...
          inflation(valid_idx), fedfunds_level(valid_idx)];
dates_full = dates_growth(valid_idx);

fprintf('   After cleaning: %d valid observations\n', size(Y_full, 1));

% Filter to sample period 1971:Q1 - 2007:Q4
start_date = datetime(1971, 1, 1);
end_date = datetime(2007, 12, 31);

idx = (dates_full >= start_date) & (dates_full <= end_date);
Y = Y_full(idx, :);
dates_sample = dates_full(idx);

[T, K] = size(Y);

if T == 0
    error('No data in sample period 1971-2007! Check date filtering.');
end

fprintf('   Sample: %d observations\n', T);
fprintf('   Period: %s to %s\n\n', datestr(dates_sample(1)), datestr(dates_sample(end)));

%% ========================================================================
% VAR ESTIMATION
%==========================================================================
fprintf('[2/7] Estimating VAR(4)...\n');

p = 4;  % Lag order
H = 20; % IRF horizon

% Create lagged matrix
X = ones(T-p, 1);  % Constant
for lag = 1:p
    X = [X, Y(p-lag+1:T-lag, :)];
end
Y_est = Y(p+1:T, :);

% OLS estimation
B_hat = (X' * X) \ (X' * Y_est);
U = Y_est - X * B_hat;

% Residual covariance matrix
T_eff = T - p;
Sigma_u = (U' * U) / (T_eff - K*p - 1);

fprintf('   Effective sample: %d observations\n\n', T_eff);

%% ========================================================================
% CHOLESKY IDENTIFICATION
%==========================================================================
fprintf('[3/7] Cholesky identification...\n');

P = chol(Sigma_u, 'lower');
mp_shock = P(4,4);

fprintf('   MP shock: %.2f pp\n\n', mp_shock);

%% ========================================================================
% IMPULSE RESPONSE FUNCTIONS
%==========================================================================
fprintf('[4/7] Computing impulse responses...\n');

% Companion form
F = zeros(K*p, K*p);
F(1:K, :) = B_hat(2:end, :)';
if p > 1
    F(K+1:end, 1:K*(p-1)) = eye(K*(p-1));
end

J = [eye(K), zeros(K, K*(p-1))];

% Compute IRFs
IRF = zeros(H+1, K, K);
IRF(1, :, :) = P;

Fh = eye(K*p);
for h = 1:H
    Fh = Fh * F;
    IRF(h+1, :, :) = J * Fh * J' * P;
end

% Extract MP shock responses
irf_mp = squeeze(IRF(:, :, 4));

fprintf('   IRFs computed\n\n');

%% ========================================================================
% BOOTSTRAP CONFIDENCE INTERVALS
%==========================================================================
fprintf('[5/7] Bootstrap confidence intervals (300 reps)...\n');

B_sim = 300;
IRF_boot = zeros(B_sim, H+1, K, K);
U_centered = U - mean(U);

rng(42);

for b = 1:B_sim
    if mod(b, 100) == 0
        fprintf('   Progress: %d/%d\n', b, B_sim);
    end
    
    % Resample residuals
    idx_boot = randi(size(U_centered, 1), T_eff, 1);
    U_star = U_centered(idx_boot, :);
    
    % Generate bootstrap data
    Y_star = zeros(T, K);
    Y_star(1:p, :) = Y(1:p, :);
    
    for t = p+1:T
        lags = [];
        for lag = 1:p
            lags = [lags, Y_star(t-lag, :)];
        end
        X_t = [1, lags];
        Y_star(t, :) = X_t * B_hat + U_star(t-p, :);
    end
    
    % Re-estimate VAR
    Y_boot = Y_star(p+1:end, :);
    X_boot = ones(T-p, 1);
    for lag = 1:p
        X_boot = [X_boot, Y_star(p-lag+1:T-lag, :)];
    end
    
    B_boot = (X_boot' * X_boot) \ (X_boot' * Y_boot);
    U_boot = Y_boot - X_boot * B_boot;
    Sigma_boot = (U_boot' * U_boot) / (T_eff - K*p - 1);
    
    try
        P_boot = chol(Sigma_boot, 'lower');
        
        F_boot = zeros(K*p, K*p);
        F_boot(1:K, :) = B_boot(2:end, :)';
        if p > 1
            F_boot(K+1:end, 1:K*(p-1)) = eye(K*(p-1));
        end
        
        IRF_b = zeros(H+1, K, K);
        IRF_b(1, :, :) = P_boot;
        Fh_b = eye(K*p);
        for h = 1:H
            Fh_b = Fh_b * F_boot;
            IRF_b(h+1, :, :) = J * Fh_b * J' * P_boot;
        end
        
        IRF_boot(b, :, :, :) = IRF_b;
    catch
        IRF_boot(b, :, :, :) = IRF;
    end
end

% Bias correction
bias = squeeze(mean(IRF_boot, 1)) - IRF;
IRF_centered = IRF_boot - reshape(bias, [1, H+1, K, K]);

% Confidence intervals
CI_68 = prctile(IRF_centered, [16, 84], 1);
CI_90 = prctile(IRF_centered, [5, 95], 1);

fprintf('   Bootstrap complete!\n\n');

%% ========================================================================
% FORECAST ERROR VARIANCE DECOMPOSITION
%==========================================================================
fprintf('[6/7] Computing FEVD...\n');

FEVD = zeros(H+1, K, K);
for h = 0:H
    mse = zeros(K, K);
    for j = 0:h
        mse = mse + squeeze(IRF(j+1, :, :)) * squeeze(IRF(j+1, :, :))';
    end
    
    for i = 1:K
        for j = 1:K
            shock_contrib = 0;
            for s = 0:h
                shock_contrib = shock_contrib + IRF(s+1, i, j)^2;
            end
            if mse(i,i) > 0
                FEVD(h+1, i, j) = shock_contrib / mse(i, i);
            end
        end
    end
end

fprintf('   FEVD computed\n\n');

%% ========================================================================
% HISTORICAL DECOMPOSITION
%==========================================================================
fprintf('[7/7] Computing historical decomposition...\n');

% Structural shocks
eps = U / P';

% Historical decomposition
HD = zeros(T_eff, K, K);
for t = 1:T_eff
    for shock = 1:K
        contrib = zeros(K, 1);
        for s = 1:t
            Fs = F^(t-s);
            impact = J * Fs * J' * P(:, shock);
            contrib = contrib + impact * eps(s, shock);
        end
        HD(t, :, shock) = contrib';
    end
end

fprintf('   Historical decomposition computed\n\n');

%% ========================================================================
% PLOTTING - Matching Python Style
%==========================================================================
fprintf('Creating figures...\n\n');

% Colors matching Python COLORS dict (RGB normalized)
colors = [0.180, 0.525, 0.671;   % #2E86AB - GDP (blue)
          0.024, 0.655, 0.490;   % #06A77D - Oil (green)
          0.635, 0.231, 0.447;   % #A23B72 - Inflation (purple)
          0.945, 0.561, 0.004];  % #F18F01 - Fed Funds (orange)

var_names = {'GDP Growth', 'Oil Price Growth', 'Inflation', 'Federal Funds'};
shock_names = {'GDP Shock', 'Oil Shock', 'Inflation Shock', 'MP Shock'};
h_vec = 0:H;

%% FIGURE 1: Impulse Responses to Monetary Policy Shock
figure('Position', [50, 50, 1600, 400], 'Color', 'w');

for i = 1:K
    subplot(1, 4, i);
    hold on; box on;
    
    % 90% CI (lighter shade)
    ci90_lo = squeeze(CI_90(1, :, i, 4));
    ci90_hi = squeeze(CI_90(2, :, i, 4));
    fill([h_vec, fliplr(h_vec)], [ci90_lo, fliplr(ci90_hi)], ...
         colors(i,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    
    % 68% CI (darker shade)
    ci68_lo = squeeze(CI_68(1, :, i, 4));
    ci68_hi = squeeze(CI_68(2, :, i, 4));
    fill([h_vec, fliplr(h_vec)], [ci68_lo, fliplr(ci68_hi)], ...
         colors(i,:), 'FaceAlpha', 0.30, 'EdgeColor', 'none');
    
    % IRF line
    plot(h_vec, irf_mp(:, i), 'Color', colors(i,:), 'LineWidth', 2.5);
    
    % Zero line
    plot(h_vec, zeros(size(h_vec)), 'k--', 'LineWidth', 0.8);
    
    xlabel('Quarters');
    title(var_names{i}, 'FontWeight', 'bold');
    set(gca, 'XGrid', 'on', 'YGrid', 'on', 'GridAlpha', 0.3);
    if i == 1
        ylabel('Percentage Points');
    end
    if i == 4
        legend('90%', '68%', 'IRF', 'Location', 'best', 'FontSize', 8);
    end
end

sgtitle('Impulse Responses to Monetary Policy Shock', ...
    'FontSize', 13, 'FontWeight', 'bold');

print(gcf, 'IRFs_MP_shock_matlab', '-dpng', '-r150');
fprintf('  Saved: IRFs_MP_shock_matlab.png\n');

%% FIGURE 2: Cumulative Responses (GDP Level + Price Level)
figure('Position', [50, 50, 1200, 450], 'Color', 'w');

irf_cum = cumsum(irf_mp);
CI_68_cum = cumsum(CI_68, 2);  % cumsum along the horizon dimension

% GDP Level
subplot(1, 2, 1);
hold on; box on;
ci68_cum_lo = squeeze(CI_68_cum(1, :, 1, 4));
ci68_cum_hi = squeeze(CI_68_cum(2, :, 1, 4));
fill([h_vec, fliplr(h_vec)], [ci68_cum_lo, fliplr(ci68_cum_hi)], ...
     colors(1,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(h_vec, irf_cum(:, 1), 'Color', colors(1,:), 'LineWidth', 2.5);
plot(h_vec, zeros(size(h_vec)), 'k--', 'LineWidth', 0.8);
xlabel('Quarters');
ylabel('Percentage Points');
title('GDP Level Response', 'FontWeight', 'bold');
set(gca, 'XGrid', 'on', 'YGrid', 'on', 'GridAlpha', 0.3);

% Price Level
subplot(1, 2, 2);
hold on; box on;
ci68_cum_lo = squeeze(CI_68_cum(1, :, 3, 4));
ci68_cum_hi = squeeze(CI_68_cum(2, :, 3, 4));
fill([h_vec, fliplr(h_vec)], [ci68_cum_lo, fliplr(ci68_cum_hi)], ...
     colors(3,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(h_vec, irf_cum(:, 3), 'Color', colors(3,:), 'LineWidth', 2.5);
plot(h_vec, zeros(size(h_vec)), 'k--', 'LineWidth', 0.8);
xlabel('Quarters');
title('Price Level Response', 'FontWeight', 'bold');
set(gca, 'XGrid', 'on', 'YGrid', 'on', 'GridAlpha', 0.3);

sgtitle('Cumulative Responses: GDP and Price Level Effects', ...
    'FontSize', 13, 'FontWeight', 'bold');

print(gcf, 'Cumulative_responses_matlab', '-dpng', '-r150');
fprintf('  Saved: Cumulative_responses_matlab.png\n');

%% FIGURE 3: Forecast Error Variance Decomposition (stacked area)
figure('Position', [50, 50, 1400, 1000], 'Color', 'w');

for i = 1:K
    subplot(2, 2, i);
    hold on; box on;
    
    % Build stacked areas
    bottom = zeros(1, H+1);
    patch_handles = gobjects(K, 1);
    for j = 1:K
        top = bottom + squeeze(FEVD(:, i, j))' * 100;
        patch_handles(j) = fill([h_vec, fliplr(h_vec)], ...
             [bottom, fliplr(top)], ...
             colors(j,:), 'FaceAlpha', 0.8, 'EdgeColor', 'none');
        bottom = top;
    end
    
    xlabel('Quarters');
    ylabel('Percent');
    title(var_names{i}, 'FontWeight', 'bold');
    ylim([0 100]);
    set(gca, 'YGrid', 'on', 'GridAlpha', 0.3);
    legend(patch_handles, shock_names, 'Location', 'best', 'FontSize', 9);
end

sgtitle('Forecast Error Variance Decomposition', ...
    'FontSize', 14, 'FontWeight', 'bold');

print(gcf, 'FEVD_matlab', '-dpng', '-r150');
fprintf('  Saved: FEVD_matlab.png\n');

%% FIGURE 4: Historical Decomposition (stacked positive/negative)
figure('Position', [50, 50, 1400, 1000], 'Color', 'w');

dates_hd = dates_sample(p+1:end);

for i = 1:K
    subplot(2, 2, i);
    hold on; box on;
    
    % Stacked bar-style: positive contributions stacked up, negative down
    bottom_pos = zeros(T_eff, 1);
    bottom_neg = zeros(T_eff, 1);
    patch_handles = gobjects(K, 1);
    
    for j = 1:K
        values = squeeze(HD(:, i, j));
        pos_vals = max(values, 0);
        neg_vals = min(values, 0);
        
        % Positive stack
        top_pos = bottom_pos + pos_vals;
        patch_handles(j) = fill([dates_hd; flipud(dates_hd)], ...
             [bottom_pos; flipud(top_pos)], ...
             colors(j,:), 'FaceAlpha', 0.7, 'EdgeColor', 'none');
        bottom_pos = top_pos;
        
        % Negative stack
        top_neg = bottom_neg + neg_vals;
        fill([dates_hd; flipud(dates_hd)], ...
             [bottom_neg; flipud(top_neg)], ...
             colors(j,:), 'FaceAlpha', 0.7, 'EdgeColor', 'none');
        bottom_neg = top_neg;
    end
    
    % Actual (demeaned) series
    actual_demeaned = Y(p+1:end, i) - mean(Y(p+1:end, i));
    plot(dates_hd, actual_demeaned, 'k-', 'LineWidth', 1.5);
    
    % Zero line
    plot([dates_hd(1), dates_hd(end)], [0 0], 'k-', 'LineWidth', 0.8, ...
        'Color', [0 0 0 0.3]);
    
    ylabel('Percentage Points');
    title(var_names{i}, 'FontWeight', 'bold');
    set(gca, 'YGrid', 'on', 'GridAlpha', 0.3);
    legend([patch_handles; plot(NaN, NaN, 'k-', 'LineWidth', 1.5)], ...
           [shock_names, {'Actual'}], 'Location', 'best', 'FontSize', 8);
    xtickangle(45);
end

sgtitle('Historical Decomposition', 'FontSize', 14, 'FontWeight', 'bold');

print(gcf, 'Historical_decomposition_matlab', '-dpng', '-r150');
fprintf('  Saved: Historical_decomposition_matlab.png\n');

%% ========================================================================
% SUMMARY
%==========================================================================
fprintf('\n========================================\n');
fprintf('  ANALYSIS COMPLETE!\n');
fprintf('========================================\n\n');

fprintf('Key Results:\n');
fprintf('   MP shock:        %.2f pp\n', mp_shock);
fprintf('   GDP peak:        %.2f at Q%d\n', min(irf_mp(:,1)), find(irf_mp(:,1)==min(irf_mp(:,1)))-1);
fprintf('   Oil peak:        %.2f at Q%d\n', min(irf_mp(:,2)), find(irf_mp(:,2)==min(irf_mp(:,2)))-1);
fprintf('   Price puzzle:    %d quarters\n', sum(irf_mp(1:6,3) > 0));

fprintf('\nFEVD at Horizon 20 (5 years):\n');
fprintf('   %-15s %7s %7s %7s %7s\n', 'Variable', 'GDP', 'Oil', 'Infl', 'MP');
fprintf('   %s\n', repmat('-', 1, 50));
for i = 1:K
    fevd_h20 = squeeze(FEVD(21, i, :)) * 100;
    fprintf('   %-15s %6.1f%% %6.1f%% %6.1f%% %6.1f%%\n', var_names{i}, fevd_h20);
end

fprintf('\nFigures saved:\n');
fprintf('   1. IRFs_MP_shock_matlab.png\n');
fprintf('   2. Cumulative_responses_matlab.png\n');
fprintf('   3. FEVD_matlab.png\n');
fprintf('   4. Historical_decomposition_matlab.png\n');
fprintf('========================================\n');
