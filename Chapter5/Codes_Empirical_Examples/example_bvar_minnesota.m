%% ========================================================================
%  EXAMPLE: BAYESIAN VAR WITH MINNESOTA PRIOR
%  ========================================================================
%  Data: U.S. FRED (GDPC1, GDPDEF, FEDFUNDS)
%  Sample: 1970:Q1 - 2007:Q4
%
%  This example demonstrates:
%  1. Minnesota prior implementation via dummy observations
%  2. Posterior simulation from Normal-Inverse Wishart
%  3. Bayesian IRFs with credible intervals
%  4. Comparison with frequentist (OLS) estimates
%  5. Prior sensitivity analysis
%
%  Author: Alessia Paccagnini
%  Textbook: Macroeconometrics
%  ========================================================================

clear; clc; close all;

fprintf('======================================================================\n');
fprintf('BAYESIAN VAR WITH MINNESOTA PRIOR\n');
fprintf('Comparison with Frequentist Estimation\n');
fprintf('======================================================================\n');

%% ========================================================================
%  SECTION 1: LOAD DATA
%  ========================================================================
fprintf('\n[1/7] Loading FRED data...\n');

% Load GDP (Real GDP, Quarterly)
gdp_data = readtable('GDPC1.xlsx', 'Sheet', 'Quarterly');
gdp_dates = datetime(gdp_data{:,1});
gdp_values = gdp_data{:,2};

% Load GDP Deflator
deflator_data = readtable('GDPDEF.xlsx', 'Sheet', 'Quarterly');
deflator_dates = datetime(deflator_data{:,1});
deflator_values = deflator_data{:,2};

% Load Federal Funds Rate (Monthly -> Quarterly)
fedfunds_data = readtable('FEDFUNDS.xlsx', 'Sheet', 'Monthly');
fedfunds_dates = datetime(fedfunds_data{:,1});
fedfunds_values = fedfunds_data{:,2};

% Convert Fed Funds to quarterly (average)
fedfunds_q_dates = dateshift(fedfunds_dates, 'end', 'quarter');
[unique_quarters, ~, idx] = unique(fedfunds_q_dates);
fedfunds_q = accumarray(idx, fedfunds_values, [], @mean);

% Compute growth rates (annualized)
gdp_growth = 400 * diff(log(gdp_values));
gdp_growth_dates = gdp_dates(2:end);

inflation = 400 * diff(log(deflator_values));
inflation_dates = deflator_dates(2:end);

% Align dates (find common sample)
gdp_growth_dates = dateshift(gdp_growth_dates, 'end', 'quarter');
inflation_dates = dateshift(inflation_dates, 'end', 'quarter');

% Create aligned dataset
start_date = datetime('1970-01-01');
end_date = datetime('2007-12-31');

% Find indices for the sample period
gdp_idx = (gdp_growth_dates >= start_date) & (gdp_growth_dates <= end_date);
infl_idx = (inflation_dates >= start_date) & (inflation_dates <= end_date);

% Get the dates for our sample
sample_dates = gdp_growth_dates(gdp_idx);
n_obs = length(sample_dates);

% Align Fed Funds with the sample dates
fedfunds_aligned = nan(n_obs, 1);
for i = 1:n_obs
    match_idx = find(unique_quarters == sample_dates(i));
    if ~isempty(match_idx)
        fedfunds_aligned(i) = fedfunds_q(match_idx);
    end
end

% Create data matrix
Y = [gdp_growth(gdp_idx), inflation(infl_idx), fedfunds_aligned];

% Remove any NaN rows
valid_idx = ~any(isnan(Y), 2);
Y = Y(valid_idx, :);
sample_dates = sample_dates(valid_idx);

fprintf('   ✓ Data loaded: %d observations\n', size(Y,1));
fprintf('   Sample: %s to %s\n', datestr(sample_dates(1), 'yyyy-QQ'), ...
    datestr(sample_dates(end), 'yyyy-QQ'));

%% ========================================================================
%  SECTION 2: OLS VAR ESTIMATION (BENCHMARK)
%  ========================================================================
fprintf('\n[2/7] Estimating frequentist VAR(4) by OLS...\n');

var_names = {'gdp_growth', 'inflation', 'fedfunds'};
p = 4;  % Lag order
[T, K] = size(Y);
T_eff = T - p;

% Construct data matrices
Y_dep = Y(p+1:end, :);
X_ols = ones(T_eff, 1);  % Constant
for lag = 1:p
    X_ols = [X_ols, Y(p+1-lag:T-lag, :)];
end

% OLS estimation
B_ols = (X_ols' * X_ols) \ (X_ols' * Y_dep);
u_ols = Y_dep - X_ols * B_ols;
Sigma_ols = (u_ols' * u_ols) / (T_eff - K*p - 1);

fprintf('   ✓ OLS estimation complete\n');
fprintf('   Variables: %d, Lags: %d, Observations: %d\n', K, p, T_eff);

%% ========================================================================
%  SECTION 3: MINNESOTA PRIOR SPECIFICATION
%  ========================================================================
fprintf('\n[3/7] Setting up Minnesota prior...\n');

% Hyperparameters
lambda1 = 0.2;   % Overall tightness (smaller = more shrinkage)
lambda2 = 0.5;   % Cross-variable shrinkage
lambda3 = 1.0;   % Lag decay

fprintf('   Minnesota prior hyperparameters:\n');
fprintf('   λ₁ (overall tightness) = %.2f\n', lambda1);
fprintf('   λ₂ (cross-variable)    = %.2f\n', lambda2);
fprintf('   λ₃ (lag decay)         = %.2f\n', lambda3);

% Function to create Minnesota prior dummies
[Y_d, X_d, sigma_scale] = create_minnesota_dummies(Y, p, lambda1, lambda2, lambda3);

fprintf('   ✓ Created %d dummy observations\n', size(Y_d, 1));

%% ========================================================================
%  SECTION 4: BAYESIAN ESTIMATION
%  ========================================================================
fprintf('\n[4/7] Bayesian VAR estimation...\n');

% Stack actual data with dummy observations
Y_star = [Y_d; Y_dep];
X_star = [X_d; X_ols];

% Posterior parameters (conjugate Normal-Inverse Wishart)
XtX_star = X_star' * X_star;
XtY_star = X_star' * Y_star;

B_bvar = XtX_star \ XtY_star;

% Posterior scale matrix for Inverse Wishart
resid_star = Y_star - X_star * B_bvar;
S_bvar = resid_star' * resid_star;

% Posterior degrees of freedom
nu_bvar = size(Y_star, 1) - size(X_star, 2);

% Posterior precision for coefficients
V_bvar_inv = XtX_star;

% Posterior mean of Sigma
Sigma_bvar = S_bvar / (nu_bvar - K - 1);

fprintf('   ✓ BVAR estimation complete\n');
fprintf('   Posterior degrees of freedom: %d\n', nu_bvar);

% Compare coefficient estimates
fprintf('\n   Comparison: First lag coefficients (own effects)\n');
fprintf('   --------------------------------------------------\n');
fprintf('   %-15s %12s %12s %12s\n', 'Variable', 'OLS', 'BVAR', 'Shrinkage');
fprintf('   --------------------------------------------------\n');
for i = 1:K
    ols_coef = B_ols(1 + i, i);
    bvar_coef = B_bvar(1 + i, i);
    if abs(ols_coef) > 0.01
        shrink = (1 - bvar_coef / ols_coef) * 100;
    else
        shrink = 0;
    end
    fprintf('   %-15s %12.4f %12.4f %11.1f%%\n', var_names{i}, ols_coef, bvar_coef, shrink);
end

%% ========================================================================
%  SECTION 5: POSTERIOR SIMULATION
%  ========================================================================
fprintf('\n[5/7] Drawing from posterior distribution...\n');

n_draws = 2000;
n_burn = 500;
total_draws = n_draws + n_burn;

fprintf('   Drawing %d samples (discarding %d burn-in)...\n', total_draws, n_burn);

% Inverse of posterior precision
V_bvar = inv(V_bvar_inv);
n_coefs = size(B_bvar, 1);

% Pre-allocate storage
B_draws = zeros(n_coefs, K, total_draws);
Sigma_draws = zeros(K, K, total_draws);

% Draw from posterior
for d = 1:total_draws
    % Draw Sigma from Inverse Wishart
    Sigma_draw = iwishrnd(S_bvar, nu_bvar);
    Sigma_draws(:,:,d) = Sigma_draw;
    
    % Draw B | Sigma from matrix normal
    L_sigma = chol(Sigma_draw, 'lower');
    L_V = chol(V_bvar, 'lower');
    
    Z = randn(n_coefs, K);
    B_draw = B_bvar + L_V * Z * L_sigma';
    B_draws(:,:,d) = B_draw;
end

% Discard burn-in
B_draws = B_draws(:,:,n_burn+1:end);
Sigma_draws = Sigma_draws(:,:,n_burn+1:end);

fprintf('   ✓ %d posterior draws retained\n', n_draws);

%% ========================================================================
%  SECTION 6: COMPUTE IRFs FROM POSTERIOR
%  ========================================================================
fprintf('\n[6/7] Computing Bayesian IRFs with credible intervals...\n');

H = 40;  % Horizon

% Pre-allocate IRF storage
IRF_draws = zeros(H+1, K, K, n_draws);

fprintf('   Computing IRFs for each posterior draw...\n');

for d = 1:n_draws
    IRF_draws(:,:,:,d) = compute_irf_cholesky(B_draws(:,:,d), Sigma_draws(:,:,d), p, K, H);
end

% Compute posterior median and credible intervals
IRF_median = median(IRF_draws, 4);
IRF_lower_68 = prctile(IRF_draws, 16, 4);
IRF_upper_68 = prctile(IRF_draws, 84, 4);
IRF_lower_90 = prctile(IRF_draws, 5, 4);
IRF_upper_90 = prctile(IRF_draws, 95, 4);

fprintf('   ✓ IRF computation complete\n');

% Compute OLS IRFs for comparison
IRF_ols = compute_irf_cholesky(B_ols, Sigma_ols, p, K, H);

%% ========================================================================
%  SECTION 7: GENERATE FIGURES
%  ========================================================================
fprintf('\n[7/7] Generating figures...\n');

var_labels = {'GDP Growth', 'Inflation', 'Federal Funds Rate'};
shock_idx = 3;  % Monetary policy shock (third variable)
horizons = 0:H;

colors_var = [0.18 0.53 0.67;    % GDP (blue)
              0.64 0.23 0.45;    % Inflation (purple)
              0.95 0.56 0.00];   % Rate (orange)
color_bayes = [0.90 0.22 0.27];  % Red
color_ols = [0.27 0.48 0.62];    % Blue-gray

% --- FIGURE 1: Bayesian IRFs to MP Shock with Credible Intervals ---
fprintf('   Figure 1: Bayesian IRFs to MP shock...\n');

figure('Position', [100, 100, 1400, 400]);

for i = 1:K
    subplot(1, 3, i);
    
    % 90% credible interval
    fill([horizons, fliplr(horizons)], ...
         [IRF_lower_90(:,i,shock_idx)', fliplr(IRF_upper_90(:,i,shock_idx)')], ...
         colors_var(i,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    hold on;
    
    % 68% credible interval
    fill([horizons, fliplr(horizons)], ...
         [IRF_lower_68(:,i,shock_idx)', fliplr(IRF_upper_68(:,i,shock_idx)')], ...
         colors_var(i,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    
    % Posterior median
    plot(horizons, IRF_median(:,i,shock_idx), 'Color', colors_var(i,:), ...
         'LineWidth', 2.5);
    
    % Zero line
    yline(0, 'k-', 'LineWidth', 0.8);
    
    xlabel('Quarters after shock');
    ylabel('Percentage points');
    title(sprintf('Response of %s', var_labels{i}));
    xlim([0, H]);
    legend({'90% CI', '68% CI', 'Posterior median'}, 'Location', 'northeast', ...
           'FontSize', 8);
    grid on;
    box off;
end

sgtitle(sprintf('Bayesian VAR: Impulse Responses to Monetary Policy Shock\n(Minnesota Prior, λ₁=%.2f, U.S. Data 1970-2007)', lambda1), ...
        'FontSize', 13, 'FontWeight', 'bold');

saveas(gcf, 'figure1_bvar_irf.png');

% --- FIGURE 2: Comparison BVAR vs OLS ---
fprintf('   Figure 2: BVAR vs OLS comparison...\n');

figure('Position', [100, 100, 1400, 400]);

for i = 1:K
    subplot(1, 3, i);
    
    % BVAR credible interval
    fill([horizons, fliplr(horizons)], ...
         [IRF_lower_68(:,i,shock_idx)', fliplr(IRF_upper_68(:,i,shock_idx)')], ...
         color_bayes, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    hold on;
    
    % BVAR median
    plot(horizons, IRF_median(:,i,shock_idx), 'Color', color_bayes, ...
         'LineWidth', 2.5, 'DisplayName', 'BVAR median');
    
    % OLS
    plot(horizons, IRF_ols(:,i,shock_idx), 'Color', color_ols, ...
         'LineWidth', 2.5, 'LineStyle', '--', 'DisplayName', 'OLS');
    
    yline(0, 'k-', 'LineWidth', 0.8);
    
    xlabel('Quarters after shock');
    ylabel('Percentage points');
    title(sprintf('Response of %s', var_labels{i}));
    xlim([0, H]);
    legend({'BVAR 68% CI', 'BVAR median', 'OLS'}, 'Location', 'northeast', ...
           'FontSize', 8);
    grid on;
    box off;
end

sgtitle('Comparison: Bayesian VAR (Minnesota Prior) vs OLS\n(U.S. Data 1970-2007, Cholesky Identification)', ...
        'FontSize', 13, 'FontWeight', 'bold');

saveas(gcf, 'figure2_bvar_vs_ols.png');

% --- FIGURE 3: Prior Sensitivity Analysis ---
fprintf('   Figure 3: Prior sensitivity analysis...\n');

lambda_values = [0.05, 0.1, 0.2, 0.5, 1.0];
n_lambda = length(lambda_values);
IRF_sensitivity = zeros(H+1, K, K, n_lambda);

for j = 1:n_lambda
    [Y_d_temp, X_d_temp, ~] = create_minnesota_dummies(Y, p, lambda_values(j), lambda2, lambda3);
    Y_star_temp = [Y_d_temp; Y_dep];
    X_star_temp = [X_d_temp; X_ols];
    
    B_temp = (X_star_temp' * X_star_temp) \ (X_star_temp' * Y_star_temp);
    resid_temp = Y_star_temp - X_star_temp * B_temp;
    S_temp = resid_temp' * resid_temp;
    nu_temp = size(Y_star_temp, 1) - size(X_star_temp, 2);
    Sigma_temp = S_temp / (nu_temp - K - 1);
    
    IRF_sensitivity(:,:,:,j) = compute_irf_cholesky(B_temp, Sigma_temp, p, K, H);
end

figure('Position', [100, 100, 1400, 400]);
cmap = parula(n_lambda);

for i = 1:K
    subplot(1, 3, i);
    hold on;
    
    for j = 1:n_lambda
        plot(horizons, IRF_sensitivity(:,i,shock_idx,j), 'Color', cmap(j,:), ...
             'LineWidth', 2, 'DisplayName', sprintf('λ₁=%.2f', lambda_values(j)));
    end
    
    % OLS for reference
    plot(horizons, IRF_ols(:,i,shock_idx), 'k--', 'LineWidth', 1.5, ...
         'DisplayName', 'OLS');
    
    yline(0, 'k-', 'LineWidth', 0.8);
    
    xlabel('Quarters after shock');
    ylabel('Percentage points');
    title(sprintf('Response of %s', var_labels{i}));
    xlim([0, H]);
    legend('Location', 'northeast', 'FontSize', 7);
    grid on;
    box off;
end

sgtitle('Prior Sensitivity: Effect of Overall Tightness (λ₁) on IRFs\n(Smaller λ₁ = More Shrinkage Toward Random Walk)', ...
        'FontSize', 13, 'FontWeight', 'bold');

saveas(gcf, 'figure3_bvar_sensitivity.png');

% --- FIGURE 4: Posterior Distribution of Impact Effects ---
fprintf('   Figure 4: Posterior distributions of impact effects...\n');

figure('Position', [100, 100, 1400, 400]);

for i = 1:K
    subplot(1, 3, i);
    
    impact_draws = squeeze(IRF_draws(1, i, shock_idx, :));
    
    histogram(impact_draws, 50, 'Normalization', 'pdf', ...
              'FaceColor', colors_var(i,:), 'FaceAlpha', 0.7, ...
              'EdgeColor', 'white');
    hold on;
    
    % Posterior mean and median
    post_mean = mean(impact_draws);
    post_median = median(impact_draws);
    
    xline(post_mean, 'Color', [0.5 0 0], 'LineWidth', 2, ...
          'DisplayName', sprintf('Mean: %.4f', post_mean));
    xline(post_median, 'Color', [0 0 0.5], 'LineStyle', '--', 'LineWidth', 2, ...
          'DisplayName', sprintf('Median: %.4f', post_median));
    xline(IRF_ols(1, i, shock_idx), 'k:', 'LineWidth', 2, ...
          'DisplayName', sprintf('OLS: %.4f', IRF_ols(1, i, shock_idx)));
    
    xlabel('Impact effect (h=0)');
    ylabel('Posterior density');
    title(var_labels{i});
    legend('Location', 'northeast', 'FontSize', 8);
    grid on;
    box off;
end

sgtitle('Posterior Distribution of Impact Effects (h=0)\n(Response to Monetary Policy Shock)', ...
        'FontSize', 13, 'FontWeight', 'bold');

saveas(gcf, 'figure4_bvar_posterior.png');

fprintf('\n   ✓ All figures saved!\n');

%% ========================================================================
%  SUMMARY STATISTICS
%  ========================================================================
fprintf('\n======================================================================\n');
fprintf('SUMMARY RESULTS\n');
fprintf('======================================================================\n');

fprintf('\n--- Coefficient Shrinkage (BVAR vs OLS) ---\n');
fprintf('%-25s %12s %12s %12s\n', 'Coefficient', 'OLS', 'BVAR', 'Shrinkage');
fprintf('%s\n', repmat('-', 1, 61));

for i = 1:K
    for lag = 1:2
        ols_c = B_ols(1 + (lag-1)*K + i, i);
        bvar_c = B_bvar(1 + (lag-1)*K + i, i);
        if abs(ols_c) > 0.01
            shrink = abs(bvar_c - ols_c) / abs(ols_c) * 100;
        else
            shrink = 0;
        end
        fprintf('%s lag %d                %12.4f %12.4f %11.1f%%\n', ...
                var_names{i}, lag, ols_c, bvar_c, shrink);
    end
end

fprintf('\n--- Impact Effects of MP Shock (h=0) ---\n');
fprintf('%-20s %12s %12s %20s\n', 'Variable', 'OLS', 'BVAR Median', '68% CI');
fprintf('%s\n', repmat('-', 1, 64));
for i = 1:K
    ols_impact = IRF_ols(1, i, shock_idx);
    bvar_impact = IRF_median(1, i, shock_idx);
    ci_low = IRF_lower_68(1, i, shock_idx);
    ci_high = IRF_upper_68(1, i, shock_idx);
    fprintf('%-20s %12.4f %12.4f [%7.4f, %7.4f]\n', ...
            var_labels{i}, ols_impact, bvar_impact, ci_low, ci_high);
end

fprintf('\n--- Peak Effects of MP Shock ---\n');
for i = 1:K
    if i < 3
        [peak_val, peak_idx] = min(IRF_median(:, i, shock_idx));
    else
        [peak_val, peak_idx] = max(IRF_median(1:20, i, shock_idx));
    end
    peak_idx = peak_idx - 1;  % Convert to 0-indexed horizon
    ci_low = IRF_lower_68(peak_idx+1, i, shock_idx);
    ci_high = IRF_upper_68(peak_idx+1, i, shock_idx);
    fprintf('%s: Peak at h=%d, value = %.4f [%.4f, %.4f]\n', ...
            var_labels{i}, peak_idx, peak_val, ci_low, ci_high);
end

fprintf('\n--- Price Puzzle Check ---\n');
inflation_irf_median = IRF_median(1:20, 2, shock_idx);
puzzle_quarters = find(inflation_irf_median > 0) - 1;
if ~isempty(puzzle_quarters)
    fprintf('Price puzzle in posterior median: quarters %s\n', mat2str(puzzle_quarters));
else
    fprintf('No price puzzle in posterior median\n');
end

% Check probability of positive inflation response
prob_positive = mean(squeeze(IRF_draws(1:20, 2, shock_idx, :)) > 0, 2);
[max_prob, max_prob_h] = max(prob_positive);
fprintf('Max probability of positive inflation response: %.2f%% at h=%d\n', ...
        max_prob*100, max_prob_h-1);

fprintf('\n======================================================================\n');
fprintf('FILES GENERATED:\n');
fprintf('======================================================================\n');
fprintf('   figure1_bvar_irf.png         - Bayesian IRFs with credible intervals\n');
fprintf('   figure2_bvar_vs_ols.png      - Comparison BVAR vs OLS\n');
fprintf('   figure3_bvar_sensitivity.png - Prior sensitivity analysis\n');
fprintf('   figure4_bvar_posterior.png   - Posterior distributions\n');

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function [Y_d, X_d, sigma] = create_minnesota_dummies(Y, p, lambda1, lambda2, lambda3)
%CREATE_MINNESOTA_DUMMIES Create Minnesota prior via dummy observations
%
%   Parameters:
%   -----------
%   Y : matrix (T x K) - Data matrix
%   p : int - Number of lags
%   lambda1 : float - Overall tightness (controls shrinkage toward prior mean)
%   lambda2 : float - Cross-variable shrinkage (0 < lambda2 <= 1)
%   lambda3 : float - Lag decay (higher = faster decay of prior variance with lag)
%
%   Returns:
%   --------
%   Y_d : matrix - Dummy observations for dependent variable
%   X_d : matrix - Dummy observations for regressors
%   sigma : vector - AR(1) residual standard deviations (for scaling)

    [T, K] = size(Y);
    lambda4 = 1e5;  % Constant term tightness (large = diffuse prior on constant)
    
    % Estimate AR(1) for each variable to get scaling factors
    sigma = zeros(K, 1);
    delta = ones(K, 1);  % Random walk prior (delta=1)
    
    for i = 1:K
        y_i = Y(2:end, i);
        y_lag = Y(1:end-1, i);
        X_ar = [ones(length(y_i), 1), y_lag];
        beta_ar = X_ar \ y_i;
        resid = y_i - X_ar * beta_ar;
        sigma(i) = std(resid, 1);  % Use N-1 normalization
    end
    
    % Number of dummy observations
    n_regressors = 1 + K * p;  % constant + K*p lag coefficients
    
    Y_d = zeros(K * p + K + 1, K);
    X_d = zeros(K * p + K + 1, n_regressors);
    
    % ----- Block 1: Prior on VAR coefficients -----
    row = 1;
    for l = 1:p
        for i = 1:K
            Y_d(row, i) = delta(i) * sigma(i) / (lambda1 * (l ^ lambda3));
            col_idx = 1 + (l - 1) * K + i;
            X_d(row, col_idx) = sigma(i) / (lambda1 * (l ^ lambda3));
            row = row + 1;
        end
    end
    
    % ----- Block 2: Sum-of-coefficients prior -----
    for i = 1:K
        Y_d(row, i) = delta(i) * sigma(i) / lambda1;
        for l = 1:p
            col_idx = 1 + (l - 1) * K + i;
            X_d(row, col_idx) = sigma(i) / lambda1;
        end
        row = row + 1;
    end
    
    % ----- Block 3: Prior on constant -----
    Y_d(row, :) = 0;
    X_d(row, 1) = lambda4;
end

function IRF = compute_irf_cholesky(B, Sigma, p, K, H)
%COMPUTE_IRF_CHOLESKY Compute IRFs using Cholesky identification
%
%   Parameters:
%   -----------
%   B : matrix - VAR coefficient matrix
%   Sigma : matrix - Covariance matrix
%   p : int - Number of lags
%   K : int - Number of variables
%   H : int - Horizon
%
%   Returns:
%   --------
%   IRF : array (H+1 x K x K) - Impulse response functions

    % Cholesky decomposition
    P = chol(Sigma, 'lower');
    
    % Companion form
    A_comp = zeros(K * p, K * p);
    for l = 1:p
        A_comp(1:K, (l-1)*K+1:l*K) = B(1 + (l-1)*K+1:1 + l*K, :)';
    end
    if p > 1
        A_comp(K+1:end, 1:K*(p-1)) = eye(K * (p - 1));
    end
    
    % IRFs
    IRF = zeros(H + 1, K, K);
    IRF(1, :, :) = P;
    
    A_power = eye(K * p);
    for h = 1:H
        A_power = A_power * A_comp;
        Phi_h = A_power(1:K, 1:K);
        IRF(h + 1, :, :) = Phi_h * P;
    end
end