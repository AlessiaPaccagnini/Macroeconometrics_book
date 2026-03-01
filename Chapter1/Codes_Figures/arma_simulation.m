% =========================================================
% ARMA Process Simulation and Visualization
% =========================================================
% This script generates artificial time series for:
%   - White Noise
%   - AR(1): y_t = phi * y_{t-1} + epsilon_t
%   - MA(1): y_t = epsilon_t + theta * epsilon_{t-1}
%   - ARMA(1,1): y_t = phi * y_{t-1} + epsilon_t + theta * epsilon_{t-1}
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
T     = 500;
sigma = 1.0;
phi   = 0.7;
theta = 0.5;

% =========================================================
% Generate all processes
% =========================================================
wn     = generate_white_noise(T, sigma);
ar1    = generate_ar1(T, phi, sigma);
ma1    = generate_ma1(T, theta, sigma);
arma11 = generate_arma11(T, phi, theta, sigma);

% =========================================================
% Figure 1: Time Series Plots
% =========================================================
fig1 = figure('Position', [100, 100, 1100, 900]);

subplot(4,1,1);
plot(wn, 'Color', [0.27 0.51 0.71], 'LineWidth', 0.8);
yline(0, 'r--', 'LineWidth', 0.8, 'Alpha', 0.7);
title('White Noise: \epsilon_t ~ N(0,1)', 'FontSize', 12);
ylabel('\epsilon_t');

subplot(4,1,2);
plot(ar1, 'Color', [0 0.39 0], 'LineWidth', 0.8);
yline(0, 'r--', 'LineWidth', 0.8, 'Alpha', 0.7);
title('AR(1): y_t = 0.7 y_{t-1} + \epsilon_t', 'FontSize', 12);
ylabel('y_t');

subplot(4,1,3);
plot(ma1, 'Color', [1 0.55 0], 'LineWidth', 0.8);
yline(0, 'r--', 'LineWidth', 0.8, 'Alpha', 0.7);
title('MA(1): y_t = \epsilon_t + 0.5 \epsilon_{t-1}', 'FontSize', 12);
ylabel('y_t');

subplot(4,1,4);
plot(arma11, 'Color', [0.5 0 0.5], 'LineWidth', 0.8);
yline(0, 'r--', 'LineWidth', 0.8, 'Alpha', 0.7);
title('ARMA(1,1): y_t = 0.7 y_{t-1} + \epsilon_t + 0.5 \epsilon_{t-1}', 'FontSize', 12);
ylabel('y_t'); xlabel('Time');

exportgraphics(fig1, fullfile(out_dir, 'arma_time_series.png'), 'Resolution', 150);
exportgraphics(fig1, fullfile(out_dir, 'arma_time_series.pdf'));

% =========================================================
% Figure 2: ACF Comparison
% =========================================================
fig2 = figure('Position', [100, 100, 1100, 750]);
lags = 20;

subplot(2,2,1); plot_acf_manual(wn,     lags, 'ACF: White Noise');
subplot(2,2,2); plot_acf_manual(ar1,    lags, 'ACF: AR(1), \phi = 0.7');
subplot(2,2,3); plot_acf_manual(ma1,    lags, 'ACF: MA(1), \theta = 0.5');
subplot(2,2,4); plot_acf_manual(arma11, lags, 'ACF: ARMA(1,1), \phi=0.7, \theta=0.5');

exportgraphics(fig2, fullfile(out_dir, 'arma_acf_comparison.png'), 'Resolution', 150);
exportgraphics(fig2, fullfile(out_dir, 'arma_acf_comparison.pdf'));

% =========================================================
% Figure 3: PACF Comparison
% =========================================================
fig3 = figure('Position', [100, 100, 1100, 750]);

subplot(2,2,1); plot_pacf_manual(wn,     lags, 'PACF: White Noise');
subplot(2,2,2); plot_pacf_manual(ar1,    lags, 'PACF: AR(1), \phi = 0.7');
subplot(2,2,3); plot_pacf_manual(ma1,    lags, 'PACF: MA(1), \theta = 0.5');
subplot(2,2,4); plot_pacf_manual(arma11, lags, 'PACF: ARMA(1,1), \phi=0.7, \theta=0.5');

exportgraphics(fig3, fullfile(out_dir, 'arma_pacf_comparison.png'), 'Resolution', 150);
exportgraphics(fig3, fullfile(out_dir, 'arma_pacf_comparison.pdf'));

% =========================================================
% Print Theoretical vs Sample Statistics
% =========================================================
fprintf('%s\n', repmat('=',1,70));
fprintf('THEORETICAL vs SAMPLE STATISTICS\n');
fprintf('%s\n', repmat('=',1,70));

fprintf('\n--- White Noise ---\n');
fprintf('Theoretical Mean: 0.000 | Sample Mean: %.3f\n', mean(wn));
fprintf('Theoretical Var:  %.3f | Sample Var:  %.3f\n', sigma^2, var(wn));

ar1_theo_var = sigma^2 / (1 - phi^2);
fprintf('\n--- AR(1): phi = %.1f ---\n', phi);
fprintf('Theoretical Mean: 0.000 | Sample Mean: %.3f\n', mean(ar1));
fprintf('Theoretical Var:  %.3f | Sample Var:  %.3f\n', ar1_theo_var, var(ar1));
fprintf('Theoretical ACF(1): %.3f | Sample ACF(1): %.3f\n', phi, corr(ar1(1:end-1), ar1(2:end)));

ma1_theo_var  = sigma^2 * (1 + theta^2);
ma1_theo_acf1 = theta / (1 + theta^2);
fprintf('\n--- MA(1): theta = %.1f ---\n', theta);
fprintf('Theoretical Mean: 0.000 | Sample Mean: %.3f\n', mean(ma1));
fprintf('Theoretical Var:  %.3f | Sample Var:  %.3f\n', ma1_theo_var, var(ma1));
fprintf('Theoretical ACF(1): %.3f | Sample ACF(1): %.3f\n', ma1_theo_acf1, corr(ma1(1:end-1), ma1(2:end)));

arma_theo_var  = sigma^2 * (1 + theta^2 + 2*phi*theta) / (1 - phi^2);
arma_theo_acf1 = (phi + theta) * (1 + phi*theta) / (1 + theta^2 + 2*phi*theta);
fprintf('\n--- ARMA(1,1): phi = %.1f, theta = %.1f ---\n', phi, theta);
fprintf('Theoretical Mean: 0.000 | Sample Mean: %.3f\n', mean(arma11));
fprintf('Theoretical Var:  %.3f | Sample Var:  %.3f\n', arma_theo_var, var(arma11));
fprintf('Theoretical ACF(1): %.3f | Sample ACF(1): %.3f\n', arma_theo_acf1, corr(arma11(1:end-1), arma11(2:end)));

fprintf('\n%s\n', repmat('=',1,70));
fprintf('ACF/PACF IDENTIFICATION PATTERNS\n');
fprintf('%s\n', repmat('=',1,70));
fprintf(['\nProcess     | ACF Pattern                    | PACF Pattern\n' ...
         '------------|--------------------------------|----------------------------------\n' ...
         'White Noise | No significant autocorrelations| No significant partial autocorr.\n' ...
         'AR(p)       | Geometric decay                | Cuts off after lag p\n' ...
         'MA(q)       | Cuts off after lag q           | Geometric decay\n' ...
         'ARMA(p,q)   | Decays after lag q             | Decays after lag p\n\n']);

fprintf('\nFigures saved:\n');
fprintf('  - arma_time_series.png/pdf\n');
fprintf('  - arma_acf_comparison.png/pdf\n');
fprintf('  - arma_pacf_comparison.png/pdf\n');


% =========================================================
% Local functions
% =========================================================

function wn = generate_white_noise(T, sigma)
    wn = sigma * randn(T, 1);
end

function y = generate_ar1(T, phi, sigma)
    if abs(phi) >= 1
        error('AR(1) requires |phi| < 1. Got phi = %g', phi);
    end
    epsilon = sigma * randn(T, 1);
    y       = zeros(T, 1);
    y(1)    = randn(1) * sigma / sqrt(1 - phi^2);
    for t = 2:T
        y(t) = phi * y(t-1) + epsilon(t);
    end
end

function y = generate_ma1(T, theta, sigma)
    epsilon = sigma * randn(T+1, 1);
    y       = zeros(T, 1);
    for t = 1:T
        y(t) = epsilon(t+1) + theta * epsilon(t);
    end
end

function y = generate_arma11(T, phi, theta, sigma)
    if abs(phi) >= 1
        error('ARMA(1,1) requires |phi| < 1. Got phi = %g', phi);
    end
    epsilon = sigma * randn(T+1, 1);
    y       = zeros(T, 1);
    y(1)    = epsilon(2) + theta * epsilon(1);
    for t = 2:T
        y(t) = phi * y(t-1) + epsilon(t+1) + theta * epsilon(t);
    end
end

function plot_acf_manual(x, lags, ttl)
    % Compute and plot sample ACF with confidence bands
    n    = length(x);
    xc   = x - mean(x);
    c0   = xc' * xc / n;
    rho  = zeros(lags, 1);
    for k = 1:lags
        rho(k) = (xc(1:end-k)' * xc(k+1:end)) / (n * c0);
    end
    ci = 1.96 / sqrt(n);
    stem(1:lags, rho, 'filled', 'MarkerSize', 4, 'Color', [0.27 0.51 0.71]);
    hold on;
    yline(ci,  'b--', 'LineWidth', 0.8);
    yline(-ci, 'b--', 'LineWidth', 0.8);
    yline(0,   'k',   'LineWidth', 0.5);
    hold off;
    ylim([-1 1]); xlabel('Lag'); ylabel('ACF');
    title(ttl, 'FontSize', 10);
end

function plot_pacf_manual(x, lags, ttl)
    % Compute PACF via Yule-Walker and plot
    n    = length(x);
    xc   = x - mean(x);
    c0   = xc' * xc / n;
    % Build autocovariance vector
    gamma = zeros(lags, 1);
    for k = 1:lags
        gamma(k) = xc(1:end-k)' * xc(k+1:end) / n;
    end
    rho = gamma / c0;
    % Durbin-Levinson recursion for PACF
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
    stem(1:lags, pacf_vals, 'filled', 'MarkerSize', 4, 'Color', [1 0.55 0]);
    hold on;
    yline(ci,  'b--', 'LineWidth', 0.8);
    yline(-ci, 'b--', 'LineWidth', 0.8);
    yline(0,   'k',   'LineWidth', 0.5);
    hold off;
    ylim([-1 1]); xlabel('Lag'); ylabel('PACF');
    title(ttl, 'FontSize', 10);
end
