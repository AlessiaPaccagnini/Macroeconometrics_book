% =========================================================
% Spurious Regression: Nine Regressions of Independent Random Walks
% =========================================================
% Author:   Alessia Paccagnini
% Textbook: Macroeconometrics
% =========================================================

clear; clc; close all;

% Parameters
T           = 300;
max_attempts = 100000;
attempt     = 0;
n_found     = 0;
target      = 9;

% Pre-allocate storage
res_x      = zeros(T, target);
res_y      = zeros(T, target);
res_fitted = zeros(T, target);
res_r2     = zeros(1, target);
res_t      = zeros(1, target);
res_dw     = zeros(1, target);

% Search for 9 pairs with R2 > 0.80 and DW < 0.20
while n_found < target && attempt < max_attempts
    attempt = attempt + 1;

    % Generate two independent random walks
    x = cumsum(randn(T, 1));
    y = cumsum(randn(T, 1));

    % OLS regression: y = alpha + beta*x + e
    X     = [ones(T,1), x];
    bhat  = X \ y;
    yhat  = X * bhat;
    resid = y - yhat;

    % R-squared
    ss_res = sum(resid.^2);
    ss_tot = sum((y - mean(y)).^2);
    r2     = 1 - ss_res / ss_tot;

    % t-statistic for beta
    s2      = ss_res / (T - 2);
    se_beta = sqrt(s2 * inv(X'*X));
    t_stat  = bhat(2) / se_beta(2,2);

    % Durbin-Watson statistic
    dw = sum(diff(resid).^2) / ss_res;

    if r2 > 0.80 && dw < 0.20
        n_found = n_found + 1;
        res_x(:, n_found)      = x;
        res_y(:, n_found)      = y;
        res_fitted(:, n_found) = yhat;
        res_r2(n_found)        = r2;
        res_t(n_found)         = t_stat;
        res_dw(n_found)        = dw;
        fprintf('Found pair %d: R2=%.2f, t=%.1f, DW=%.2f\n', n_found, r2, t_stat, dw);
    end
end

fprintf('\nTotal attempts: %d\n', attempt);
fprintf('Pairs found: %d\n', n_found);

% --------------------------------------------------------
% Plot: 3x3 grid of scatter plots with regression lines
% --------------------------------------------------------
colors_scatter = [0.608, 0.349, 0.714];   % #9B59B6
colors_line    = [1, 0, 0];

fig = figure('Position', [100, 100, 1200, 1000]);
sgtitle({'Nine Regressions of Independent Random Walks', ...
         '(All relationships are SPURIOUS)'}, ...
         'FontSize', 14, 'FontWeight', 'bold');

for i = 1:n_found
    subplot(3, 3, i);

    x_i      = res_x(:, i);
    y_i      = res_y(:, i);
    yhat_i   = res_fitted(:, i);
    r2_i     = res_r2(i);
    t_i      = res_t(i);
    dw_i     = res_dw(i);

    % Scatter
    scatter(x_i, y_i, 15, colors_scatter, 'filled', 'MarkerFaceAlpha', 0.45);
    hold on;

    % Regression line (sorted by x)
    [x_sort, idx] = sort(x_i);
    plot(x_sort, yhat_i(idx), 'Color', colors_line, 'LineWidth', 2);
    hold off;

    % Significance stars
    abs_t = abs(t_i);
    if abs_t > 3.29
        stars = '***';
    elseif abs_t > 2.58
        stars = '**';
    elseif abs_t > 1.96
        stars = '*';
    else
        stars = '';
    end

    if t_i < 0
        t_sign = '-';
    else
        t_sign = '';
    end

    title(sprintf('R^2=%.2f, t=%s%.1f%s, DW=%.2f', r2_i, t_sign, abs_t, stars), ...
          'FontSize', 10);
    xlabel('x_t', 'FontSize', 9);
    ylabel('y_t', 'FontSize', 9);
end

% Save figure
out_dir = fileparts(mfilename('fullpath'));
exportgraphics(fig, fullfile(out_dir, 'spurious_regression_multiple.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(out_dir, 'spurious_regression_multiple.pdf'));
fprintf('\nPlots saved!\n');
