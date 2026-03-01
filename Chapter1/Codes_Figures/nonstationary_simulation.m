% =========================================================
% Nonstationary Processes: Simulation and Visualization
% =========================================================
% This script generates artificial time series for:
%   - Deterministic Trend: y_t = alpha + beta*t + u_t (u_t is AR(1))
%   - Random Walk: y_t = y_{t-1} + epsilon_t
%   - Random Walk with Drift: y_t = delta + y_{t-1} + epsilon_t
%
% Author:   Alessia Paccagnini
% Textbook: Macroeconometrics
% =========================================================

clear; clc; close all;
rng(42);

out_dir = fileparts(mfilename('fullpath'));

% =========================================================
% Parameters
% =========================================================
T      = 250;
sigma  = 1.0;
alpha  = 2.0;
beta   = 0.1;
phi_u  = 0.7;
delta  = 0.15;
nlags  = 30;

% =========================================================
% Generate all processes
% =========================================================
[y_det, ~, t_index] = generate_deterministic_trend(T, alpha, beta, phi_u, sigma);
y_rw                = generate_random_walk(T, sigma);
y_rwd               = generate_random_walk_drift(T, delta, sigma);
wn                  = sigma * randn(T, 1);

% =========================================================
% Figure 1a: Deterministic Trend (individual)
% =========================================================
fig = figure('Position', [100,100,950,480]);
plot(t_index, y_det, 'Color', [0.27 0.51 0.71], 'LineWidth', 0.9, 'DisplayName', 'y_t = 2+0.1t+u_t');
hold on;
plot(t_index, alpha + beta*t_index, 'r--', 'LineWidth', 2, 'DisplayName', 'Trend: 2+0.1t');
yline(0, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5, 'Alpha', 0.7);
hold off;
legend('Location','northwest'); xlabel('Time'); ylabel('y_t');
title('Deterministic Trend: y_t = \alpha + \beta t + u_t, u_t \sim AR(1)');
exportgraphics(fig, fullfile(out_dir, 'deterministic_trend.png'), 'Resolution', 150);
exportgraphics(fig, fullfile(out_dir, 'deterministic_trend.pdf'));

% =========================================================
% Figure 1b: Random Walk (individual)
% =========================================================
fig = figure('Position', [100,100,950,480]);
fill([1:T, fliplr(1:T)], [y_rw', zeros(1,T)], [0 0.39 0], ...
     'FaceAlpha', 0.25, 'EdgeColor', 'none');
hold on;
plot(1:T, y_rw, 'Color', [0 0.39 0], 'LineWidth', 0.9);
yline(0, 'r--', 'LineWidth', 1.5, 'Alpha', 0.8, 'Label', 'Initial level y_0=0');
hold off;
xlabel('Time'); ylabel('y_t');
title('Random Walk: y_t = y_{t-1} + \epsilon_t');
exportgraphics(fig, fullfile(out_dir, 'random_walk.png'), 'Resolution', 150);
exportgraphics(fig, fullfile(out_dir, 'random_walk.pdf'));

% =========================================================
% Figure 1c: Random Walk with Drift (individual)
% =========================================================
fig = figure('Position', [100,100,950,480]);
plot(t_index, y_rwd, 'Color', [0.5 0 0.5], 'LineWidth', 0.9, 'DisplayName', 'y_t = 0.15 + y_{t-1} + \epsilon_t');
hold on;
plot(t_index, delta*t_index, 'r--', 'LineWidth', 2, 'DisplayName', 'Drift: 0.15t');
hold off;
legend('Location','northwest'); xlabel('Time'); ylabel('y_t');
title('Random Walk with Drift: y_t = \delta + y_{t-1} + \epsilon_t');
exportgraphics(fig, fullfile(out_dir, 'random_walk_drift.png'), 'Resolution', 150);
exportgraphics(fig, fullfile(out_dir, 'random_walk_drift.pdf'));

% =========================================================
% Figure 2: Combined Time Series
% =========================================================
fig = figure('Position', [100,100,1100,900]);

subplot(3,1,1);
plot(t_index, y_det, 'Color', [0.27 0.51 0.71], 'LineWidth', 0.8, 'DisplayName', 'Series');
hold on;
plot(t_index, alpha + beta*t_index, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Trend');
hold off; legend('Location','northwest');
ylabel('y_t');
title('Deterministic Trend: y_t = 2+0.1t+u_t, u_t = 0.7u_{t-1}+\epsilon_t');

subplot(3,1,2);
plot(1:T, y_rw, 'Color', [0 0.39 0], 'LineWidth', 0.8);
yline(0, 'r--', 'LineWidth', 1.2, 'Alpha', 0.7);
ylabel('y_t');
title('Random Walk: y_t = y_{t-1}+\epsilon_t (variance grows with t)');

subplot(3,1,3);
plot(t_index, y_rwd, 'Color', [0.5 0 0.5], 'LineWidth', 0.8, 'DisplayName', 'Series');
hold on;
plot(t_index, delta*t_index, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Drift');
hold off; legend('Location','northwest');
ylabel('y_t'); xlabel('Time');
title('Random Walk with Drift: y_t = 0.15+y_{t-1}+\epsilon_t');

exportgraphics(fig, fullfile(out_dir, 'nonstationary_time_series.png'), 'Resolution', 150);
exportgraphics(fig, fullfile(out_dir, 'nonstationary_time_series.pdf'));

% =========================================================
% Figure 3: ACF Comparison
% =========================================================
fig = figure('Position', [100,100,1100,750]);
subplot(2,2,1); plot_acf_manual(wn,    nlags, 'ACF: White Noise (Stationary)',          [-0.3 1.1]);
subplot(2,2,2); plot_acf_manual(y_det, nlags, 'ACF: Deterministic Trend (before detr.)',[-0.3 1.1]);
subplot(2,2,3); plot_acf_manual(y_rw,  nlags, 'ACF: Random Walk',                       [-0.3 1.1]);
subplot(2,2,4); plot_acf_manual(y_rwd, nlags, 'ACF: Random Walk with Drift',            [-0.3 1.1]);
exportgraphics(fig, fullfile(out_dir, 'nonstationary_acf_comparison.png'), 'Resolution', 150);
exportgraphics(fig, fullfile(out_dir, 'nonstationary_acf_comparison.pdf'));

% =========================================================
% Figure 4: PACF Comparison
% =========================================================
fig = figure('Position', [100,100,1100,750]);
subplot(2,2,1); plot_pacf_manual(wn,    nlags, 'PACF: White Noise (Stationary)');
subplot(2,2,2); plot_pacf_manual(y_det, nlags, 'PACF: Deterministic Trend');
subplot(2,2,3); plot_pacf_manual(y_rw,  nlags, 'PACF: Random Walk');
subplot(2,2,4); plot_pacf_manual(y_rwd, nlags, 'PACF: Random Walk with Drift');
exportgraphics(fig, fullfile(out_dir, 'nonstationary_pacf_comparison.png'), 'Resolution', 150);
exportgraphics(fig, fullfile(out_dir, 'nonstationary_pacf_comparison.pdf'));

% =========================================================
% Figure 5: ACF Levels vs First Differences
% =========================================================
dy_det = diff(y_det);
dy_rw  = diff(y_rw);
dy_rwd = diff(y_rwd);

fig = figure('Position', [100,100,1400,750]);
subplot(2,3,1); plot_acf_manual(y_det,  nlags, 'Det. Trend (levels)',             [-0.3 1.1]);
subplot(2,3,2); plot_acf_manual(y_rw,   nlags, 'Random Walk (levels)',             [-0.3 1.1]);
subplot(2,3,3); plot_acf_manual(y_rwd,  nlags, 'RW with Drift (levels)',           [-0.3 1.1]);
subplot(2,3,4); plot_acf_manual(dy_det, nlags, 'Det. Trend (first diff.)',         [-0.5 1.1]);
subplot(2,3,5); plot_acf_manual(dy_rw,  nlags, 'Random Walk (first diff.) = WN',  [-0.5 1.1]);
subplot(2,3,6); plot_acf_manual(dy_rwd, nlags, 'RW with Drift (first diff.) = WN',[-0.5 1.1]);
sgtitle('ACF in Levels vs. First Differences: Diagnosing Nonstationarity', 'FontSize', 13);
exportgraphics(fig, fullfile(out_dir, 'acf_levels_vs_differences.png'), 'Resolution', 150);
exportgraphics(fig, fullfile(out_dir, 'acf_levels_vs_differences.pdf'));

% =========================================================
% Figure 6: Variance Growth Comparison
% =========================================================
n_sim = 100;
T_var = 200;

rw_sims  = zeros(n_sim, T_var);
det_sims = zeros(n_sim, T_var);
for i = 1:n_sim
    rw_sims(i,:)  = generate_random_walk(T_var, sigma)';
    tmp           = generate_deterministic_trend(T_var, 0, 0, phi_u, sigma);
    det_sims(i,:) = tmp';
end

rw_var        = var(rw_sims, 0, 1);
det_var       = var(det_sims, 0, 1);
theo_rw_var   = sigma^2 * (1:T_var);
theo_det_var  = sigma^2 / (1 - phi_u^2);

fig = figure('Position', [100,100,1100,480]);
subplot(1,2,1);
plot(1:T_var, rw_var,       'Color', [0 0.39 0], 'LineWidth', 1.5, 'DisplayName', 'Sample variance');
hold on;
plot(1:T_var, theo_rw_var, 'r--', 'LineWidth', 2, 'DisplayName', 'Theoretical: t\sigma^2');
hold off; legend; xlabel('Time'); ylabel('Variance');
title('Random Walk: Variance Grows Linearly with Time');

subplot(1,2,2);
plot(1:T_var, det_var, 'Color', [0.27 0.51 0.71], 'LineWidth', 1.5, 'DisplayName', 'Sample variance');
hold on;
yline(theo_det_var, 'r--', 'LineWidth', 2, 'Label', 'Theoretical: \sigma^2/(1-\phi^2)');
hold off; legend; xlabel('Time'); ylabel('Variance');
ylim([0, max(det_var)*1.5]);
title('Stationary AR(1): Variance Remains Constant');

exportgraphics(fig, fullfile(out_dir, 'variance_growth_comparison.png'), 'Resolution', 150);
exportgraphics(fig, fullfile(out_dir, 'variance_growth_comparison.pdf'));

% =========================================================
% Figure 7: Multiple Realizations
% =========================================================
fig = figure('Position', [100,100,1100,480]);

subplot(1,2,1);
hold on;
for i = 1:30
    y_tmp = generate_random_walk(T, sigma);
    plot(1:T, y_tmp, 'Color', [0 0 0 0.3], 'LineWidth', 0.5);
end
yline(0, 'r--', 'LineWidth', 2);
hold off;
xlabel('Time'); ylabel('y_t');
title('Random Walk: Multiple Realizations (Variance Expansion)');

subplot(1,2,2);
hold on;
for i = 1:30
    y_tmp = generate_deterministic_trend(T, alpha, beta, phi_u, sigma);
    plot(t_index, y_tmp, 'Color', [0 0 0 0.3], 'LineWidth', 0.5);
end
plot(t_index, alpha + beta*t_index, 'r--', 'LineWidth', 2);
hold off;
xlabel('Time'); ylabel('y_t');
title('Deterministic Trend: Multiple Realizations (Constant Variance)');

exportgraphics(fig, fullfile(out_dir, 'multiple_realizations.png'), 'Resolution', 150);
exportgraphics(fig, fullfile(out_dir, 'multiple_realizations.pdf'));

% =========================================================
% Print Summary Statistics
% =========================================================
fprintf('%s\n', repmat('=',1,70));
fprintf('NONSTATIONARY PROCESSES: SUMMARY STATISTICS\n');
fprintf('%s\n', repmat('=',1,70));

acf_k = @(x,k) compute_acf(x, k);

fprintf('\n--- Deterministic Trend: y_t = 2 + 0.1t + u_t ---\n');
fprintf('Sample Mean: %.3f\n',     mean(y_det));
fprintf('Sample Variance: %.3f\n', var(y_det));
fprintf('ACF(1): %.3f\n',          acf_k(y_det, 1));
fprintf('ACF(10): %.3f\n',         acf_k(y_det, 10));

fprintf('\n--- Random Walk: y_t = y_{t-1} + epsilon_t ---\n');
fprintf('Sample Mean: %.3f\n',     mean(y_rw));
fprintf('Sample Variance: %.3f\n', var(y_rw));
fprintf('Theoretical Variance at T=%d: %.3f\n', T, T*sigma^2);
fprintf('ACF(1): %.3f\n',          acf_k(y_rw, 1));
fprintf('ACF(10): %.3f\n',         acf_k(y_rw, 10));

fprintf('\n--- Random Walk with Drift: y_t = 0.15 + y_{t-1} + epsilon_t ---\n');
fprintf('Sample Mean: %.3f\n',     mean(y_rwd));
fprintf('Sample Variance: %.3f\n', var(y_rwd));
fprintf('ACF(1): %.3f\n',          acf_k(y_rwd, 1));
fprintf('ACF(10): %.3f\n',         acf_k(y_rwd, 10));

fprintf('\n--- First Differences ---\n');
fprintf('Delta-y (Random Walk)   - Mean: %.3f, Var: %.3f\n', mean(dy_rw), var(dy_rw));
fprintf('Delta-y (RW with Drift) - Mean: %.3f, Var: %.3f\n', mean(dy_rwd), var(dy_rwd));

fprintf('\n%s\n', repmat('=',1,70));
fprintf('KEY INSIGHT: WHY CORRELOGRAM FAILS FOR UNIT ROOTS\n');
fprintf('%s\n', repmat('=',1,70));
fprintf(['\nFor a random walk, the sample ACF at lag k is approximately:\n\n'...
         '    rho_k ~ (T-k)/T -> 1 as T -> Inf\n\n'...
         'This means:\n'...
         '1. ALL autocorrelations appear close to 1, regardless of lag\n'...
         '2. The slow linear decay is an artifact, not a true pattern\n'...
         '3. Standard confidence bands (1/sqrt(T)) are INVALID\n'...
         '4. This is why we need FORMAL UNIT ROOT TESTS (ADF, PP, KPSS)!\n\n']);

fprintf('\nFigures saved (PNG + PDF):\n');
files = {'deterministic_trend','random_walk','random_walk_drift',...
         'nonstationary_time_series','nonstationary_acf_comparison',...
         'nonstationary_pacf_comparison','acf_levels_vs_differences',...
         'variance_growth_comparison','multiple_realizations'};
for i = 1:length(files)
    fprintf('  - %s\n', files{i});
end


% =========================================================
% Local functions
% =========================================================

function [y, u, t_index] = generate_deterministic_trend(T, alpha, beta, phi, sigma)
    epsilon = sigma * randn(T, 1);
    u       = zeros(T, 1);
    u(1)    = randn(1) * sigma / sqrt(1 - phi^2);
    for t = 2:T
        u(t) = phi * u(t-1) + epsilon(t);
    end
    t_index = (0:T-1)';
    y       = alpha + beta * t_index + u;
end

function y = generate_random_walk(T, sigma, y0)
    if nargin < 3, y0 = 0; end
    epsilon = sigma * randn(T, 1);
    y       = zeros(T, 1);
    y(1)    = y0 + epsilon(1);
    for t = 2:T
        y(t) = y(t-1) + epsilon(t);
    end
end

function y = generate_random_walk_drift(T, delta, sigma, y0)
    if nargin < 3, sigma = 1.0; end
    if nargin < 4, y0 = 0; end
    epsilon = sigma * randn(T, 1);
    y       = zeros(T, 1);
    y(1)    = y0 + delta + epsilon(1);
    for t = 2:T
        y(t) = delta + y(t-1) + epsilon(t);
    end
end

function rho_k = compute_acf(x, k)
    xc  = x - mean(x);
    n   = length(xc);
    c0  = xc' * xc / n;
    ck  = xc(1:end-k)' * xc(k+1:end) / n;
    rho_k = ck / c0;
end

function plot_acf_manual(x, lags, ttl, ylim_range)
    if nargin < 4, ylim_range = [-1 1]; end
    n   = length(x);
    xc  = x - mean(x);
    c0  = xc' * xc / n;
    rho = zeros(lags, 1);
    for k = 1:lags
        rho(k) = (xc(1:end-k)' * xc(k+1:end)) / (n * c0);
    end
    ci = 1.96 / sqrt(n);
    stem(1:lags, rho, 'filled', 'MarkerSize', 3, 'Color', [0.27 0.51 0.71]);
    hold on;
    yline(ci,  'b--', 'LineWidth', 0.7);
    yline(-ci, 'b--', 'LineWidth', 0.7);
    yline(0,   'k',   'LineWidth', 0.4);
    hold off;
    ylim(ylim_range); xlabel('Lag'); ylabel('ACF');
    title(ttl, 'FontSize', 9);
end

function plot_pacf_manual(x, lags, ttl)
    n    = length(x);
    xc   = x - mean(x);
    c0   = xc' * xc / n;
    gamma = zeros(lags, 1);
    for k = 1:lags
        gamma(k) = xc(1:end-k)' * xc(k+1:end) / n;
    end
    rho = gamma / c0;
    % Durbin-Levinson recursion
    pacf_vals = zeros(lags, 1);
    a = zeros(lags, lags);
    for k = 1:lags
        if k == 1
            a(1,1) = rho(1);
        else
            num = rho(k) - a(k-1,1:k-1) * rho(k-1:-1:1);
            den = 1 - a(k-1,1:k-1) * rho(1:k-1);
            a(k,k) = num / den;
            for j = 1:k-1
                a(k,j) = a(k-1,j) - a(k,k) * a(k-1,k-j);
            end
        end
        pacf_vals(k) = a(k,k);
    end
    ci = 1.96 / sqrt(n);
    stem(1:lags, pacf_vals, 'filled', 'MarkerSize', 3, 'Color', [1 0.55 0]);
    hold on;
    yline(ci,  'b--', 'LineWidth', 0.7);
    yline(-ci, 'b--', 'LineWidth', 0.7);
    yline(0,   'k',   'LineWidth', 0.4);
    hold off;
    ylim([-1 1]); xlabel('Lag'); ylabel('PACF');
    title(ttl, 'FontSize', 9);
end
