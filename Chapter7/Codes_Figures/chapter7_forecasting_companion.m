% ============================================================================
% CHAPTER 7: FORECASTING — Figures 7.1–7.4 (MATLAB Companion)
% Macroeconometrics Textbook
% Author: Alessia Paccagnini
% ============================================================================
%
% This script generates four publication-ready figures for Chapter 7:
%   Figure 7.1: bias_variance_tradeoff.pdf
%   Figure 7.2: bias_variance_examples.pdf
%   Figure 7.3: forecast_errors_example.pdf  (needs GDPC1.xlsx)
%   Figure 7.4: giacomini_rossi_fluctuation_test.pdf
%
% Run: run all sections, or call individual functions at the bottom.
% ============================================================================

clear; clc; close all;

fprintf('************************************************************\n');
fprintf('*  CHAPTER 7: FORECASTING — MATLAB Companion (Fig 7.1–7.4)\n');
fprintf('*  Macroeconometrics Textbook — Alessia Paccagnini\n');
fprintf('************************************************************\n\n');

run_figure_7_1();
run_figure_7_2();
run_figure_7_3('GDPC1.xlsx');
run_figure_7_4();

fprintf('************************************************************\n');
fprintf('*  ALL FIGURES COMPLETE\n');
fprintf('************************************************************\n');


% ============================================================================
%  FIGURE 7.1 — The Bias-Variance Tradeoff in Forecasting
% ============================================================================

function run_figure_7_1()
    fprintf('==== FIGURE 7.1 — Bias-Variance Tradeoff ====\n');

    complexity = linspace(0, 10, 200);
    bias_sq    = 8 * exp(-0.5 * complexity) + 0.5;
    variance   = 0.1 + 0.15 * complexity.^1.5;
    irreducible = 2.0 * ones(size(complexity));
    total      = bias_sq + variance + irreducible;

    [opt_y, opt_idx] = min(total);
    opt_x = complexity(opt_idx);

    fig = figure('Position', [100 100 900 500]);

    plot(complexity, bias_sq, 'b-', 'LineWidth', 2.5); hold on;
    plot(complexity, variance, 'r-', 'LineWidth', 2.5);
    plot(complexity, irreducible, 'k--', 'LineWidth', 1.5);
    plot(complexity, total, 'Color', [0 0.7 0], 'LineWidth', 3);
    plot(opt_x, opt_y, 'o', 'Color', [0 0.5 0], 'MarkerSize', 12, ...
         'MarkerFaceColor', [0 0.5 0], 'LineWidth', 2);
    xline(opt_x, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);

    ylim([0 12]);
    xticks([0 2.5 5 7.5 10]);
    xticklabels({'Random Walk', 'AR(1)', 'AR(4)/BVAR', 'Unrestr. VAR', ...
                 'High-Dim Model'});
    xlabel('Model Complexity', 'FontWeight', 'bold', 'FontSize', 12);
    ylabel('Mean Squared Error', 'FontWeight', 'bold', 'FontSize', 12);
    title('The Bias-Variance Tradeoff in Forecasting', ...
          'FontSize', 14, 'FontWeight', 'bold');
    legend('Bias^2', 'Variance', 'Irreducible Error', 'Total Error', ...
           'Optimal', 'Location', 'north', 'NumColumns', 5);
    grid on; set(gca, 'GridAlpha', 0.3, 'FontSize', 10);
    hold off;

    exportgraphics(fig, 'bias_variance_tradeoff.pdf', 'ContentType', 'vector');
    saveas(fig, 'bias_variance_tradeoff.png');
    fprintf('Saved: bias_variance_tradeoff.pdf\n\n');
    close(fig);
end


% ============================================================================
%  FIGURE 7.2 — Bias-Variance Decomposition in Practice
% ============================================================================

function run_figure_7_2()
    fprintf('==== FIGURE 7.2 — Bias-Variance Decomposition ====\n');

    rng(42);

    % --- Left panel data ---
    simple  = 1.5 + 0.5 * randn(1000, 1);
    complex = 0.3 + 1.8 * randn(1000, 1);
    optimal = 0.5 + 0.9 * randn(1000, 1);

    % --- Right panel data ---
    models   = {'RW','AR(1)','AR(2)','AR(4)','VAR(2)','VAR(4)','BVAR','VAR(8)'};
    rmse_val = [3.2, 2.9, 2.6, 2.3, 2.4, 2.5, 2.35, 2.8];
    bias_val = [2.8, 2.2, 1.5, 0.8, 0.7, 0.5, 0.6, 0.3];
    var_val  = [0.4, 0.7, 1.1, 1.5, 1.7, 2.0, 1.75, 2.5];

    fig = figure('Position', [100 100 1300 450]);

    % Left panel
    subplot(1, 2, 1);
    edges = -6:0.3:7;
    histogram(simple,  edges, 'Normalization', 'pdf', ...
              'FaceColor', 'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none'); hold on;
    histogram(complex, edges, 'Normalization', 'pdf', ...
              'FaceColor', 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    histogram(optimal, edges, 'Normalization', 'pdf', ...
              'FaceColor', 'g', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    xline(0, 'k--', 'LineWidth', 2);
    xlabel('Forecast Error', 'FontWeight', 'bold');
    ylabel('Density', 'FontWeight', 'bold');
    title('Distribution of Forecast Errors', 'FontWeight', 'bold');
    legend(sprintf('RW (Bias=%.2f)', mean(simple)), ...
           sprintf('VAR(8) (Bias=%.2f)', mean(complex)), ...
           sprintf('AR(4) (Bias=%.2f)', mean(optimal)), ...
           'Target', 'Location', 'northeast', 'FontSize', 8);
    grid on; hold off;

    % Right panel
    subplot(1, 2, 2);
    b = bar(1:8, [bias_val; var_val]', 'stacked', 'EdgeColor', 'none');
    b(1).FaceColor = [0.27 0.51 0.71];  % steelblue
    b(2).FaceColor = [1.0 0.5 0.31];    % coral
    hold on;

    yyaxis right;
    plot(1:8, rmse_val, 'o-', 'Color', [0 0.5 0], 'LineWidth', 2.5, ...
         'MarkerFaceColor', [0 0.5 0], 'MarkerSize', 8);
    [~, oi] = min(rmse_val);
    plot(oi, rmse_val(oi), 'p', 'Color', [0 0.5 0], 'MarkerSize', 18, ...
         'MarkerFaceColor', [0 0.5 0]);
    ylabel('Total RMSE', 'FontWeight', 'bold', 'Color', [0 0.5 0]);
    set(gca, 'YColor', [0 0.5 0]);

    yyaxis left;
    ylabel('Error Components', 'FontWeight', 'bold');
    set(gca, 'XTickLabel', models, 'XTickLabelRotation', 45);
    title('Bias-Variance Decomposition by Model', 'FontWeight', 'bold');
    legend('Bias^2', 'Variance', 'Total RMSE', 'Location', 'northwest');
    grid on; hold off;

    exportgraphics(fig, 'bias_variance_examples.pdf', 'ContentType', 'vector');
    saveas(fig, 'bias_variance_examples.png');
    fprintf('Saved: bias_variance_examples.pdf\n\n');
    close(fig);
end


% ============================================================================
%  FIGURE 7.3 — Visualising Forecast Performance
% ============================================================================

function run_figure_7_3(file_path)
    fprintf('==== FIGURE 7.3 — Visualising Forecast Performance ====\n');

    % --- Load data ---
    use_real = false;
    try
        T_raw = readtable(file_path, 'Sheet', 'Quarterly');
        dates_raw = datetime(T_raw{:,1});
        gdpc1     = T_raw{:,2};
        growth_q  = [NaN; 100 * diff(log(gdpc1))];
        valid     = ~isnan(growth_q);
        dates_raw = dates_raw(valid);
        growth_q  = growth_q(valid);

        % 1985 to 2023
        sel = dates_raw >= datetime(1985,1,1) & dates_raw <= datetime(2023,12,31);
        dates = dates_raw(sel);
        y     = growth_q(sel);
        fprintf('   Loaded GDPC1: %s to %s (%d obs)\n', ...
                datestr(dates(1)), datestr(dates(end)), length(y));
        use_real = true;
    catch ME
        fprintf('   Could not load %s: %s\n', file_path, ME.message);
        fprintf('   Using simulated data.\n');
    end

    if ~use_real
        rng(2026);
        dates = (datetime(1985,1,1):calmonths(3):datetime(2023,10,1))';
        T = length(dates);
        y = zeros(T, 1);
        y(1) = 0.8; y(2) = 0.6;
        for t = 3:T
            y(t) = 0.15 + 0.30*y(t-1) + 0.12*y(t-2) + 0.5*randn;
        end
        covid_i = find(dates >= datetime(2020,4,1), 1);
        y(covid_i) = -8.2; y(covid_i+1) = 7.5; y(covid_i+2) = 1.1;
    end

    T = length(y);

    % --- Expanding-window AR(2) forecasts ---
    R = 95;
    forecasts = NaN(T, 1);

    for t = R:(T-1)
        Y_dep = y(3:t);
        X_reg = [ones(t-2,1), y(2:t-1), y(1:t-2)];
        beta = X_reg \ Y_dep;
        forecasts(t+1) = beta(1) + beta(2)*y(t) + beta(3)*y(t-1);
    end

    fmask  = ~isnan(forecasts);
    errors = NaN(T, 1);
    errors(fmask) = y(fmask) - forecasts(fmask);
    valid_err = errors(fmask);
    rmse = sqrt(mean(valid_err.^2));
    mae  = mean(abs(valid_err));

    fc_idx = find(fmask);
    fprintf('   Forecasts: %s to %s (%d obs)\n', ...
            datestr(dates(fc_idx(1))), datestr(dates(fc_idx(end))), sum(fmask));
    fprintf('   RMSE = %.2f,  MAE = %.2f\n', rmse, mae);

    % --- Recession dates ---
    rec_s = [datetime(2007,12,1), datetime(2020,2,1)];
    rec_e = [datetime(2009,6,30), datetime(2020,4,30)];

    % --- Plot from 2009 ---
    pi = dates >= datetime(2009,1,1);

    fig = figure('Position', [100 100 1100 800]);

    % Panel (a): Actual vs Forecast
    subplot(2, 1, 1);
    hold on;
    for k = 1:length(rec_s)
        fill([rec_s(k) rec_e(k) rec_e(k) rec_s(k)], ...
             [-15 -15 15 15], [0.85 0.85 0.85], ...
             'EdgeColor', 'none', 'FaceAlpha', 0.5);
    end
    plot(dates(pi), y(pi), 'k-', 'LineWidth', 1.8);
    fc_pi = pi & fmask;
    plot(dates(fc_pi), forecasts(fc_pi), 'b--', 'LineWidth', 1.8);
    yline(0, 'Color', [0 0 0 0.3]);
    ylabel('GDP Growth (%)', 'FontWeight', 'bold');
    title('(a) Actual vs. Forecast', 'FontWeight', 'bold', 'FontSize', 13);
    legend('', '', 'Actual', 'Forecast', 'Location', 'southeast');
    grid on; set(gca, 'GridAlpha', 0.3);
    hold off;

    % Panel (b): Forecast Errors
    subplot(2, 1, 2);
    hold on;
    for k = 1:length(rec_s)
        fill([rec_s(k) rec_e(k) rec_e(k) rec_s(k)], ...
             [-12 -12 12 12], [0.85 0.85 0.85], ...
             'EdgeColor', 'none', 'FaceAlpha', 0.5);
    end

    pi_idx = find(pi);
    bar_w = 75;  % days
    for j = 1:length(pi_idx)
        ii = pi_idx(j);
        if ~isnan(errors(ii))
            if errors(ii) >= 0
                col = [0.27 0.51 0.71];
            else
                col = [1.0 0.5 0.31];
            end
            d = dates(ii);
            fill([d-days(bar_w/2), d+days(bar_w/2), ...
                  d+days(bar_w/2), d-days(bar_w/2)], ...
                 [0, 0, errors(ii), errors(ii)], col, ...
                 'EdgeColor', 'none');
        end
    end

    yline(0, 'k-', 'LineWidth', 0.8);
    yline(rmse,  '--', 'Color', [0.5 0 0], 'LineWidth', 1.5);
    yline(-rmse, '--', 'Color', [0.5 0 0], 'LineWidth', 1.5);
    ylim([-10 10]);
    ylabel('Forecast Error (%)', 'FontWeight', 'bold');
    xlabel('Time', 'FontWeight', 'bold');
    title('(b) Forecast Errors', 'FontWeight', 'bold', 'FontSize', 13);
    legend('', '', sprintf('+RMSE = %.2f', rmse), ...
           sprintf('-RMSE = -%.2f', rmse), ...
           'Location', 'southeast');
    grid on; set(gca, 'GridAlpha', 0.3);

    % Text box
    annotation('textbox', [0.02 0.35 0.13 0.08], ...
               'String', sprintf('RMSE = %.2f\nMAE = %.2f', rmse, mae), ...
               'FitBoxToText', 'on', 'BackgroundColor', 'w', ...
               'EdgeColor', [0.5 0.5 0.5], 'FontSize', 10);
    hold off;

    exportgraphics(fig, 'forecast_errors_example.pdf', 'ContentType', 'vector');
    saveas(fig, 'forecast_errors_example.png');
    fprintf('Saved: forecast_errors_example.pdf\n\n');
    close(fig);
end


% ============================================================================
%  FIGURE 7.4 — Giacomini-Rossi Fluctuation Test
% ============================================================================

function res = giacomini_rossi_test(loss_diff, window_size, alpha)
    T = length(loss_diff);
    n_win = T - window_size + 1;
    t_stats = zeros(n_win, 1);

    for i = 1:n_win
        window = loss_diff(i:(i + window_size - 1));
        m  = mean(window);
        se = std(window) / sqrt(window_size);
        if se > 0
            t_stats(i) = m / se;
        end
    end

    max_stat = max(abs(t_stats));

    switch alpha
        case 0.10, cv = 1.73;
        case 0.05, cv = 1.95;
        case 0.01, cv = 2.37;
        otherwise,  cv = 1.95;
    end

    p_val = 2 * (1 - normcdf(max_stat));
    periods = (floor(window_size/2):(floor(window_size/2) + n_win - 1))';

    res.t_stats = t_stats;
    res.periods = periods;
    res.max_stat = max_stat;
    res.critical_value = cv;
    res.p_value = p_val;
end


function run_figure_7_4()
    fprintf('==== FIGURE 7.4 — Giacomini-Rossi Fluctuation Test ====\n');

    rng(42);
    T = 200;
    t_vec = (0:(T-1))';
    trend = 0.5 * (t_vec - T/2) / T;
    loss_diff = trend + 0.3 * randn(T, 1);

    window_size = 40;
    alpha = 0.05;
    res = giacomini_rossi_test(loss_diff, window_size, alpha);

    t_stats = res.t_stats;
    periods = res.periods;
    cv      = res.critical_value;

    fig = figure('Position', [100 100 1100 500]);
    hold on;

    % Shade rejection regions
    upper = t_stats > cv;
    lower = t_stats < -cv;
    for i = 1:length(periods)
        if upper(i)
            fill([periods(i)-0.5, periods(i)+0.5, ...
                  periods(i)+0.5, periods(i)-0.5], ...
                 [cv, cv, t_stats(i), t_stats(i)], ...
                 [0 0.8 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        end
        if lower(i)
            fill([periods(i)-0.5, periods(i)+0.5, ...
                  periods(i)+0.5, periods(i)-0.5], ...
                 [t_stats(i), t_stats(i), -cv, -cv], ...
                 [1 0.65 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        end
    end

    plot(periods, t_stats, 'b-', 'LineWidth', 2);
    yline(cv,  'r--', 'LineWidth', 1.5);
    yline(-cv, 'r--', 'LineWidth', 1.5);
    yline(0,   'k-',  'LineWidth', 0.5);

    xlabel('Time Period (centered on rolling window)', 'FontWeight', 'bold');
    ylabel('Rolling t-statistic', 'FontWeight', 'bold');
    title('Giacomini-Rossi Fluctuation Test: AR(1) vs AR(4)', ...
          'FontSize', 14, 'FontWeight', 'bold');
    legend('AR(4) better', 'AR(1) better', 'Rolling t-stat', ...
           sprintf('95%% CV (+/-%.2f)', cv), '', '', ...
           'Location', 'best');
    grid on; set(gca, 'GridAlpha', 0.3);

    % Text box
    annotation('textbox', [0.02 0.82 0.2 0.12], ...
               'String', sprintf('Test stat: %.3f\np-value: %.3f', ...
                                 res.max_stat, res.p_value), ...
               'FitBoxToText', 'on', 'BackgroundColor', [0.96 0.87 0.70], ...
               'EdgeColor', [0.5 0.5 0.5], 'FontSize', 10);
    hold off;

    exportgraphics(fig, 'giacomini_rossi_fluctuation_test.pdf', ...
                   'ContentType', 'vector');
    saveas(fig, 'giacomini_rossi_fluctuation_test.png');

    fprintf('   Max |t| = %.3f,  CV(5%%) = %.3f,  p = %.3f\n', ...
            res.max_stat, cv, res.p_value);
    if res.max_stat > cv
        fprintf('   => REJECT: time-varying forecast performance\n');
    else
        fprintf('   => Do not reject equal predictive ability\n');
    end
    fprintf('Saved: giacomini_rossi_fluctuation_test.pdf\n\n');
    close(fig);
end
