%% ========================================================================
%  Figure 5.1: Common Prior Distributions for Bayesian Estimation
%  ========================================================================
%  Author: Alessia Paccagnini
%  Textbook: Macroeconometrics
%  ========================================================================
clear; clc; close all;

colors = [0.122 0.467 0.706;   % #1f77b4
          1.000 0.498 0.055;   % #ff7f0e
          0.173 0.627 0.173;   % #2ca02c
          0.839 0.153 0.157;   % #d62728
          0.580 0.404 0.741];  % #9467bd

figure('Position', [100 100 1400 450]);

%% Panel (a): Normal Prior
subplot(1,3,1); hold on;
x = linspace(-6, 6, 500);

normal_params = {0, 0.5, 'N(0, 0.5^2) — Tight';
                 0, 1.0, 'N(0, 1^2) — Moderate';
                 0, 2.0, 'N(0, 2^2) — Diffuse';
                 1, 1.0, 'N(1, 1^2) — Shifted'};

for i = 1:size(normal_params, 1)
    mu_i = normal_params{i, 1};
    sig_i = normal_params{i, 2};
    plot(x, normpdf(x, mu_i, sig_i), 'Color', colors(i,:), 'LineWidth', 2.5);
end

xlabel('\theta', 'FontSize', 12);
ylabel('Density', 'FontSize', 12);
title({'(a) Normal Prior', 'Location parameters'}, 'FontSize', 12, 'FontWeight', 'bold');
legend(normal_params{:, 3}, 'Location', 'northeast', 'FontSize', 9);
xlim([-6 6]); ylim([0 0.85]);
grid on; box on;

%% Panel (b): Inverse Gamma Prior
subplot(1,3,2); hold on;
x_ig = linspace(0.01, 5, 500);

% Manual Inverse Gamma PDF: beta^alpha / Gamma(alpha) * x^(-alpha-1) * exp(-beta/x)
dinvgamma = @(x, alpha, beta) (beta.^alpha ./ gamma(alpha)) .* x.^(-alpha-1) .* exp(-beta./x);

ig_params = {3, 2, 'IG(3, 2) — Moderate';
             2, 1, 'IG(2, 1) — Diffuse';
             1, 1, 'IG(1, 1) — Heavy tail';
             8, 3, 'IG(8, 3) — Informative'};

for i = 1:size(ig_params, 1)
    a_i = ig_params{i, 1};
    b_i = ig_params{i, 2};
    plot(x_ig, dinvgamma(x_ig, a_i, b_i), 'Color', colors(i,:), 'LineWidth', 2.5);
end

xlabel('\sigma^2', 'FontSize', 12);
ylabel('Density', 'FontSize', 12);
title({'(b) Inverse Gamma Prior', 'Variance parameters'}, 'FontSize', 12, 'FontWeight', 'bold');
legend(ig_params{:, 3}, 'Location', 'northeast', 'FontSize', 9);
xlim([0 5]); ylim([0 4]);
grid on; box on;

%% Panel (c): Beta Prior
subplot(1,3,3); hold on;
x_beta = linspace(0.001, 0.999, 500);

beta_params = {1,   1, 'Beta(1,1) — Uniform';
               0.5, 0.5, 'Beta(0.5,0.5) — U-shaped';
               2,   5, 'Beta(2,5) — Skewed right';
               5,   2, 'Beta(5,2) — Skewed left';
               5,   5, 'Beta(5,5) — Symmetric'};

for i = 1:size(beta_params, 1)
    a_i = beta_params{i, 1};
    b_i = beta_params{i, 2};
    plot(x_beta, betapdf(x_beta, a_i, b_i), 'Color', colors(i,:), 'LineWidth', 2.5);
end

xlabel('p', 'FontSize', 12);
ylabel('Density', 'FontSize', 12);
title({'(c) Beta Prior', 'Probabilities & proportions'}, 'FontSize', 12, 'FontWeight', 'bold');
legend(beta_params{:, 3}, 'Location', 'northeast', 'FontSize', 9);
xlim([0 1]); ylim([0 4]);
grid on; box on;

%% Save
exportgraphics(gcf, 'prior_distributions.png', 'Resolution', 300);
fprintf('Saved: prior_distributions.png\n');
