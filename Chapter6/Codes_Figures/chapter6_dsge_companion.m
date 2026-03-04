%% ========================================================================
%  Chapter 6: DSGE Models — Companion MATLAB Code
%  Macroeconometrics Textbook
%  Author: Alessia Paccagnini
%
%  This file contains three self-contained sections:
%    SECTION 1: Table 6.1  — Business Cycle Statistics (HP filter)
%    SECTION 2: Figure 6.1 — RBC Impulse Responses (Blanchard-Kahn)
%    SECTION 3: Figure 6.2 — Prior vs Posterior Distributions (SW 2007)
%
%  No external toolboxes required (Statistics Toolbox recommended for
%  Section 3, but a fallback is provided).
%
%  Calibration note:
%    delta = 0.025 throughout (consistent with BK example, exercises,
%    and calibration discussion in Chapter 6)
%  ========================================================================

clear; clc; close all;
fprintf('================================================================\n');
fprintf('CHAPTER 6: DSGE MODELS — MATLAB COMPANION CODE\n');
fprintf('================================================================\n\n');


%% ========================================================================
%  SECTION 1: TABLE 6.1 — Business Cycle Statistics
%  Data: FRED-QD dataset (McCracken & Ng, 2016)
%  File: 2026-01-QD.xlsx
%  Sample: 1960:Q1 to 2019:Q4 (pre-COVID, 240 observations)
%  Consumption = nondurable goods + services (standard definition)
%  ========================================================================

fprintf('================================================================\n');
fprintf('SECTION 1: TABLE 6.1 — BUSINESS CYCLE STATISTICS\n');
fprintf('================================================================\n\n');

%%% Uncomment and set file_path to run with real data:
%
% file_path = '2026-01-QD.xlsx';
% [~, ~, raw] = xlsread(file_path, 'in');
% colnames = raw(1, :);
%
% get_col = @(code) find(strcmp(colnames, code));
%
% data_start = 6;  % data begins at row 6
%
% GDP    = cell2mat(raw(data_start:end, get_col('GDPC1')));
% PCNDx  = cell2mat(raw(data_start:end, get_col('PCNDx')));
% PCESVx = cell2mat(raw(data_start:end, get_col('PCESVx')));
% Cons   = PCNDx + PCESVx;   % nondurables + services
% Inv    = cell2mat(raw(data_start:end, get_col('GPDIC1')));
% Hours  = cell2mat(raw(data_start:end, get_col('HOANBS')));
% Prod   = cell2mat(raw(data_start:end, get_col('OPHNFB')));
% Wages  = cell2mat(raw(data_start:end, get_col('COMPRNFB')));
%
% % Sample: 1960Q1 to 2019Q4 = first 240 obs
% T_s = 240;
% series = [GDP(1:T_s), Cons(1:T_s), Inv(1:T_s), ...
%           Hours(1:T_s), Prod(1:T_s), Wages(1:T_s)];
% var_names = {'GDP', 'Consumption', 'Investment', ...
%              'Hours worked', 'Labor productivity', 'Real wages'};
%
% % HP-filter each log series (lambda = 1600)
% cycles = zeros(T_s, 6);
% for j = 1:6
%     [~, cycles(:,j)] = hp_filter_fn(log(series(:,j)), 1600);
%     cycles(:,j) = cycles(:,j) * 100;  % percent
% end
%
% % Compute statistics
% gdp_sd = std(cycles(:,1));
% fprintf('TABLE 6.1: Business Cycle Statistics for the United States\n');
% fprintf('%-25s %10s %12s %18s\n', 'Variable', 'Std.Dev(%)', ...
%         'Rel.to GDP', 'Corr w/ GDP');
% fprintf('%s\n', repmat('-', 1, 70));
% for j = 1:6
%     sd_j   = std(cycles(:,j));
%     rel_j  = sd_j / gdp_sd;
%     corr_j = corr(cycles(:,j), cycles(:,1));
%     fprintf('%-25s %10.2f %12.2f %18.2f\n', var_names{j}, sd_j, rel_j, corr_j);
% end
% fprintf('%s\n\n', repmat('-', 1, 70));
%
% % GDP autocorrelation
% rho1 = corr(cycles(2:end,1), cycles(1:end-1,1));
% rho4 = corr(cycles(5:end,1), cycles(1:end-4,1));
% fprintf('GDP autocorrelation: rho(1) = %.2f, rho(4) = %.2f\n\n', rho1, rho4);
%
% % Figures
% figure('Position', [100 100 1200 800]);
% subplot(2,1,1);
% plot(log(series(:,1)), 'b'); hold on;
% [tr, ~] = hp_filter_fn(log(series(:,1)), 1600);
% plot(tr, 'r', 'LineWidth', 2);
% title('Log GDP and HP Trend'); legend('Log GDP', 'HP Trend'); grid on;
% subplot(2,1,2);
% plot(cycles(:,1), 'b'); hold on; yline(0, 'k--');
% title('GDP: Cyclical Component (% deviation)'); grid on;
% print('hp_decomposition_gdp', '-dpng', '-r300');
%
% figure('Position', [100 100 1400 600]);
% plot(cycles); legend(var_names); yline(0, 'k--');
% title('Business Cycle Components'); grid on;
% print('all_cycles', '-dpng', '-r300');
%
% figure('Position', [100 100 1500 1000]);
% for j = 2:6
%     subplot(2,3,j-1);
%     scatter(cycles(:,1), cycles(:,j), 10, 'filled');
%     hold on; lsline; grid on;
%     xlabel('GDP Cycle (%)');
%     ylabel([var_names{j} ' Cycle (%)']);
%     r = corr(cycles(:,1), cycles(:,j));
%     title(sprintf('%s vs GDP\nCorr = %.2f', var_names{j}, r));
% end
% print('correlations_with_gdp', '-dpng', '-r300');

fprintf('Table 6.1 code ready. Uncomment and set file_path to run.\n\n');


%% ========================================================================
%  SECTION 2: FIGURE 6.1 — RBC Impulse Responses
%  Calibration: beta=0.99, alpha=0.33, delta=0.025, rho_A=0.95
%  sigma=1 (log utility), eta=1 (unit Frisch elasticity)
%  Solved via Blanchard-Kahn
%  ========================================================================

fprintf('================================================================\n');
fprintf('SECTION 2: FIGURE 6.1 — RBC IMPULSE RESPONSES\n');
fprintf('Calibration: beta=0.99, alpha=0.33, delta=0.025, rho_A=0.95\n');
fprintf('================================================================\n\n');

% --- 2.1 Calibration ---
beta_  = 0.99;
alpha_ = 0.33;
delta_ = 0.025;
sigma_ = 1.0;
eta_   = 1.0;
rho_a  = 0.95;

% --- 2.2 Steady State ---
r_ss = 1/beta_ - 1 + delta_;
KY   = alpha_ / r_ss;
IY   = delta_ * KY;
CY   = 1 - IY;

K_per_N = KY^(1/(1 - alpha_));
YN      = K_per_N^alpha_;
w_ss    = (1 - alpha_) * YN;
CN      = CY * YN;
N_ss    = (w_ss * CN^(-sigma_))^(1/(eta_ + sigma_));
K_ss    = K_per_N * N_ss;
Y_ss    = K_ss^alpha_ * N_ss^(1 - alpha_);
C_ss    = CY * Y_ss;
I_ss    = IY * Y_ss;

fprintf('Steady state:\n');
fprintf('   Y=%.4f, C=%.4f, I=%.4f, K=%.4f, N=%.4f\n', ...
        Y_ss, C_ss, I_ss, K_ss, N_ss);
fprintf('   C/Y=%.3f, I/Y=%.3f, K/Y=%.2f\n\n', CY, IY, KY);

% --- 2.3 Blanchard-Kahn ---

% Output elasticities (after substituting intratemporal FOC for n)
phi_yk = alpha_ + alpha_*(1 - alpha_)/(eta_ + alpha_);
phi_ya = 1 + (1 - alpha_)/(eta_ + alpha_);
phi_yc = -(1 - alpha_)*sigma_/(eta_ + alpha_);

% Capital accumulation coefficients
Akk = (1 - delta_) + delta_*phi_yk/IY;
Aka = delta_*phi_ya/IY;
Akc = delta_*(phi_yc - CY)/IY;

% Euler equation coefficients
rb  = r_ss * beta_;
pk1 = phi_yk - 1;

A_lhs = [1, 0, 0; ...
         0, 1, 0; ...
         0, 0, sigma_ - rb*phi_yc];

A_rhs = [Akk,          Aka,                              Akc; ...
         0,            rho_a,                            0; ...
         rb*pk1*Akk,   rb*pk1*Aka + rb*phi_ya*rho_a,     sigma_ + rb*pk1*Akc];

% Generalised eigenvalue problem: M = A_lhs \ A_rhs
M = A_lhs \ A_rhs;
[V, D] = eig(M);
eig_abs = abs(diag(D));

fprintf('Blanchard-Kahn:\n');
fprintf('   Eigenvalues: |%.4f|, |%.4f|, |%.4f|\n', sort(eig_abs));

% Sort: stable first
[eig_sorted, ord] = sort(eig_abs);
V_sorted = V(:, ord);

n_stable   = sum(eig_sorted < 1);
n_unstable = sum(eig_sorted > 1);
fprintf('   Stable: %d, Unstable: %d\n', n_stable, n_unstable);

% Partition eigenvectors
Z11 = V_sorted(1:2, 1:2);
Z21 = V_sorted(3, 1:2);

% Policy function: c_hat = F * [k_hat; a_hat]
F_mat = real(Z21 / Z11);

% State transition: [k'; a'] = P [k; a]
P_mat = [Akk + Akc*F_mat(1), Aka + Akc*F_mat(2); ...
         0,                   rho_a];

% --- 2.4 IRFs ---
periods = 40;
s = zeros(periods + 1, 2);
s(1, 2) = 1.0;   % 1% technology shock at t = 0

for t = 2:periods+1
    s(t,:) = (P_mat * s(t-1,:)')';
end

k_hat   = s(1:periods, 1);
a_hat   = s(1:periods, 2);
c_hat   = (F_mat * s(1:periods,:)')';
n_hat   = (a_hat + alpha_*k_hat - sigma_*c_hat) / (eta_ + alpha_);
y_hat   = phi_yk*k_hat + phi_ya*a_hat + phi_yc*c_hat;
i_hat   = (y_hat - CY*c_hat) / IY;
cap_hat = s(2:periods+1, 1);

quarters = (0:periods-1)';

fprintf('\nImpact effects (t=0):\n');
fprintf('   Y: %+.4f%%\n', y_hat(1));
fprintf('   C: %+.4f%%\n', c_hat(1));
fprintf('   I: %+.4f%%\n', i_hat(1));
fprintf('   N: %+.4f%%\n', n_hat(1));
fprintf('   K: %+.4f%%\n', cap_hat(1));
fprintf('   A: %+.4f%%\n\n', a_hat(1));

fprintf('Peak effects:\n');
vnames = {'Y','C','I','K'};
vdata  = {y_hat, c_hat, i_hat, cap_hat};
for j = 1:4
    [pv, pq] = max(vdata{j});
    fprintf('   %s: %.4f%% at quarter %d\n', vnames{j}, pv, pq-1);
end

% --- 2.5 Plot Figure 6.1 ---
vars_data = {y_hat, c_hat, i_hat, n_hat, cap_hat, a_hat};
titles_   = {'Output (Y)', 'Consumption (C)', 'Investment (I)', ...
             'Hours (N)', 'Capital (K)', 'Technology (A)'};
labels_   = {'(1)', '(2)', '(3)', '(4)', '(5)', '(6)'};

figure('Position', [100 100 1200 800]);
sgtitle('Impulse Responses to a One Percent Technology Shock', ...
        'FontSize', 16, 'FontWeight', 'bold');

for i = 1:6
    subplot(2, 3, i);
    plot(quarters, vars_data{i}, 'LineWidth', 2.5, ...
         'Color', [0.12 0.47 0.71]);
    hold on;
    yline(0, 'k-', 'LineWidth', 0.8, 'Alpha', 0.3);
    xlabel('Quarters'); ylabel('% deviation');
    title(titles_{i}, 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    xlim([0 periods-1]);
    text(0.92, 0.92, labels_{i}, 'Units', 'normalized', ...
         'FontSize', 14, 'FontWeight', 'bold', 'Color', 'r', ...
         'HorizontalAlignment', 'right');
    hold off;
end

print('figure_6_1_rbc_irf', '-dpng', '-r300');
fprintf('\nFigure saved: figure_6_1_rbc_irf.png\n\n');


%% ========================================================================
%  SECTION 3: FIGURE 6.2 — Prior vs Posterior Distributions
%  Smets-Wouters (2007, AER), Table 1A
%  Grey = prior, Blue = posterior (normal approx around mode)
%  ========================================================================

fprintf('================================================================\n');
fprintf('SECTION 3: FIGURE 6.2 — PRIOR vs POSTERIOR DISTRIBUTIONS\n');
fprintf('Smets-Wouters (2007, AER)\n');
fprintf('================================================================\n\n');

% --- 3.1 Parameter specifications from Table 6.4 ---
% Struct array: label, prior_type, pm, ps, mode, p5, p95

P = struct();
P(1).label = '\xi_p (Price stickiness)';
P(1).prior = 'B'; P(1).pm = 0.50; P(1).ps = 0.10;
P(1).mode = 0.65; P(1).p5 = 0.56; P(1).p95 = 0.74;

P(2).label = '\xi_w (Wage stickiness)';
P(2).prior = 'B'; P(2).pm = 0.50; P(2).ps = 0.10;
P(2).mode = 0.73; P(2).p5 = 0.60; P(2).p95 = 0.81;

P(3).label = 'h (Habit formation)';
P(3).prior = 'B'; P(3).pm = 0.70; P(3).ps = 0.10;
P(3).mode = 0.71; P(3).p5 = 0.64; P(3).p95 = 0.78;

P(4).label = '\varphi (Inv. adjustment)';
P(4).prior = 'N'; P(4).pm = 4.00; P(4).ps = 1.50;
P(4).mode = 5.48; P(4).p5 = 3.97; P(4).p95 = 7.42;

P(5).label = 'r_\pi (Taylor: inflation)';
P(5).prior = 'N'; P(5).pm = 1.50; P(5).ps = 0.25;
P(5).mode = 2.03; P(5).p5 = 1.74; P(5).p95 = 2.33;

P(6).label = '\rho (Interest smoothing)';
P(6).prior = 'B'; P(6).pm = 0.75; P(6).ps = 0.10;
P(6).mode = 0.81; P(6).p5 = 0.77; P(6).p95 = 0.85;

% --- 3.2 Print summary ---
fprintf('TABLE 6.4 (selected): Prior and Posterior\n');
fprintf('%s\n', repmat('=', 1, 80));
fprintf('%-28s %5s %8s %8s %10s %6s %6s\n', ...
        'Parameter', 'Dist', 'Prior m', 'Prior s', 'Post mode', '[5%', '95%]');
fprintf('%s\n', repmat('-', 1, 80));
for i = 1:6
    fprintf('%-28s %5s %8.2f %8.2f %10.2f %6.2f %6.2f\n', ...
            P(i).label, P(i).prior, P(i).pm, P(i).ps, ...
            P(i).mode, P(i).p5, P(i).p95);
end
fprintf('%s\n\n', repmat('-', 1, 80));

% Learning diagnostics
fprintf('Prior-to-Posterior Learning:\n');
fprintf('%s\n', repmat('-', 1, 70));
for i = 1:6
    post_sd = (P(i).p95 - P(i).p5) / 3.29;
    shrinkage = 1 - post_sd / P(i).ps;
    shift = P(i).mode - P(i).pm;
    fprintf('  %-28s shift = %+.2f, variance reduction = %.0f%%\n', ...
            P(i).label, shift, shrinkage*100);
end
fprintf('\n');

% --- 3.3 Plot Figure 6.2 ---
figure('Position', [100 100 1200 650]);
sgtitle('Bayesian Estimation: Prior vs Posterior Distributions', ...
        'FontSize', 15, 'FontWeight', 'bold');

for i = 1:6
    p = P(i);
    post_sd = (p.p95 - p.p5) / (2 * 1.645);

    subplot(2, 3, i);

    if strcmp(p.prior, 'B')
        x = linspace(0.001, 0.999, 500);
        % Beta parameters from mean/std
        v = p.ps^2;
        k_par = p.pm*(1-p.pm)/v - 1;
        a_b = p.pm * k_par;
        b_b = (1 - p.pm) * k_par;
        prior_pdf = betapdf(x, a_b, b_b);
    else
        lo = min(p.pm - 4*p.ps, p.mode - 4*post_sd);
        hi = max(p.pm + 4*p.ps, p.mode + 4*post_sd);
        x = linspace(lo, hi, 500);
        prior_pdf = normpdf(x, p.pm, p.ps);
    end

    post_pdf = normpdf(x, p.mode, post_sd);

    % Prior (grey)
    fill([x fliplr(x)], [prior_pdf zeros(size(x))], ...
         [0.6 0.6 0.6], 'FaceAlpha', 0.35, 'EdgeColor', 'none');
    hold on;
    plot(x, prior_pdf, 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);

    % Posterior (blue)
    fill([x fliplr(x)], [post_pdf zeros(size(x))], ...
         [0.13 0.44 0.71], 'FaceAlpha', 0.45, 'EdgeColor', 'none');
    plot(x, post_pdf, 'Color', [0.13 0.44 0.71], 'LineWidth', 1.8);

    % Posterior mode
    xline(p.mode, '--', 'Color', [0.03 0.32 0.61], 'LineWidth', 1.5);

    title(p.label, 'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'tex');
    set(gca, 'YTick', []);
    grid on;

    if strcmp(p.prior, 'B')
        xlim([0 1]);
    end

    hold off;
end

% Add a shared legend in the first panel
subplot(2,3,1);
legend('Prior', '', 'Posterior', '', 'Posterior mode', ...
       'Location', 'northwest', 'FontSize', 8);

print('figure_6_2_prior_posterior', '-dpng', '-r300');
fprintf('Figure saved: figure_6_2_prior_posterior.png\n\n');


fprintf('================================================================\n');
fprintf('ALL SECTIONS COMPLETE\n');
fprintf('================================================================\n');


%% ========================================================================
%  HELPER FUNCTION: HP Filter
%  (placed at end of file for MATLAB compatibility)
%  ========================================================================

function [trend, cycle] = hp_filter_fn(y, lambda)
    % Hodrick-Prescott filter via penalised least squares
    % Input:  y      - Tx1 vector
    %         lambda - smoothing parameter (1600 for quarterly)
    % Output: trend  - Tx1 trend component
    %         cycle  - Tx1 cyclical component
    
    n = length(y);
    y = y(:);
    I_mat = eye(n);
    
    % Second-difference matrix
    D2 = zeros(n-2, n);
    for i = 1:n-2
        D2(i, i)   =  1;
        D2(i, i+1) = -2;
        D2(i, i+2) =  1;
    end
    
    A = I_mat + lambda * (D2' * D2);
    trend = A \ y;
    cycle = y - trend;
end
