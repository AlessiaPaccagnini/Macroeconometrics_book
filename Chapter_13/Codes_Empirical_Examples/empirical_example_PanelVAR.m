% =========================================================================
% PANEL VAR: OIL PRICE SHOCKS AND THE MACROECONOMY
% Replication of Section 13.7.1 — Macroeconometrics Textbook
% Author:   Alessia Paccagnini
% Textbook: Macroeconometrics (De Gruyter)
% =========================================================================
%
% Book specification (eq. 13.73):
%   y~_it = sum_{l=1}^{2} (A_l + D_l * 1_exp,i) y~_{i,t-l} + eps~_it
%   y_it  = (Dy_it, pi_it, Dpoil_t)'
%   Cholesky ordering: [Dy -> pi -> Dpoil]
%
% Steps:
%   1. Simulate panel data (T=162, N=13)
%   2. Within-transform + OLS
%   3. Cluster-robust SEs (country-level)
%   4. Wald test H0: D1=D2=0
%   5. Cholesky identification
%   6. Bootstrap IRFs (1000 replications, bias-corrected)
%   7. Figure 13.1
% =========================================================================

clear; clc; rng(42);

% ── Country definitions ──────────────────────────────────────────────────
exporters = {'Canada','Mexico','Norway','Saudi_Arabia'};
importers = {'USA','Euro_Area','UK','Japan','China','Korea','India','Brazil','Turkey'};
all_countries = [exporters, importers];   % exporters first
N_exp = numel(exporters);
N_imp = numel(importers);
N_all = numel(all_countries);
T     = 162;
k     = 3;    % gdp, inflation, oil

% ── DGP parameters ───────────────────────────────────────────────────────
A1_POOL = [ 0.312, -0.045, -0.022;
            0.083,  0.724,  0.038;
            0.000,  0.000,  0.720];

A2_POOL = [ 0.080,  0.000, -0.006;
            0.020,  0.060,  0.010;
            0.000,  0.000,  0.100];

D1 = zeros(3,3);  D1(1,3) =  0.040;  D1(2,3) = -0.009;
D2 = zeros(3,3);  D2(1,3) =  0.010;  D2(2,3) = -0.005;

SIG_GDP = 0.009;
SIG_INF = 0.005;
SIG_OIL = 0.060;

% =========================================================================
% STEP 1 — SIMULATE DATA
% =========================================================================
fprintf('[1] Simulating panel data (T=%d, N=%d) ...\n', T, N_all);

% Common oil price
rng(42);
oil = zeros(T,1);
for t = 2:T
    oil(t) = A1_POOL(3,3) * oil(t-1) + SIG_OIL * randn();
    if rand() < 0.03
        shocks = [-0.18, 0.22];
        oil(t) = oil(t) + shocks(randi(2));
    end
end

% Country fixed effects (seed 0 equivalent — use separate stream)
rng(0);
alpha_vals = 0.001 + 0.007 * rand(N_all, 1);   % uniform(0.001, 0.008)

% Simulate each country
country_data = struct();
Sigma_eps = [SIG_GDP^2,              SIG_GDP*SIG_INF*0.20;
             SIG_GDP*SIG_INF*0.20,   SIG_INF^2];
L_eps = chol(Sigma_eps, 'lower');   % for drawing correlated shocks

rng(42);
for c = 1:N_all
    cname  = all_countries{c};
    is_exp = any(strcmp(exporters, cname));
    B1     = A1_POOL + is_exp * D1;
    B2     = A2_POOL + is_exp * D2;
    alpha_c = alpha_vals(c);

    Y = zeros(T, 3);
    Y(1,:) = [alpha_c, 0.004, oil(1)];
    Y(2,:) = [alpha_c, 0.004, oil(2)];

    for t = 3:T
        Y(t,3) = oil(t);
        eps    = L_eps * randn(2,1);
        Y(t,1) = alpha_c + B1(1,:) * Y(t-1,:)' + B2(1,:) * Y(t-2,:)' + eps(1);
        Y(t,2) = 0.002   + B1(2,:) * Y(t-1,:)' + B2(2,:) * Y(t-2,:)' + eps(2);
    end
    country_data.(cname) = Y;   % T x 3 matrix
end

% =========================================================================
% STEP 2 — WITHIN-TRANSFORMATION + OLS
% =========================================================================
function [Yp, Xp, ids, B, Sig] = build_panel(country_data, countries, p)
    % Stack within-transformed observations for a list of countries.
    k = 3;
    Y_list = []; X_list = []; id_list = [];
    for cid = 1:numel(countries)
        Y  = country_data.(countries{cid});   % T x k
        Tc = size(Y,1);
        Te = Tc - p;
        Yd = Y(p+1:end, :);
        Xl = [];
        for lag = 1:p
            Xl = [Xl, Y(p-lag+1:Tc-lag, :)];
        end
        % Within-transform
        Yd = Yd - mean(Yd, 1);
        Xl = Xl - mean(Xl, 1);
        Y_list = [Y_list; Yd];
        X_list = [X_list; Xl];
        id_list = [id_list; repmat(cid-1, Te, 1)];
    end
    Yp  = Y_list;
    Xp  = X_list;
    ids = id_list;
    B   = (Xp' * Xp) \ (Xp' * Yp);
    U   = Yp - Xp * B;
    Sig = (U' * U) / (size(U,1) - size(Xp,2));
end

% =========================================================================
% STEP 3 — CLUSTER-ROBUST STANDARD ERRORS
% =========================================================================
function V = cluster_se(Xp, Yp, B, ids)
    Nc    = max(ids) + 1;
    n     = size(Xp, 2);
    U     = Yp - Xp * B;
    bread = inv(Xp' * Xp);
    meat  = zeros(n, n);
    for ci = 0:Nc-1
        m  = (ids == ci);
        sc = Xp(m,:)' * U(m,1);   % GDP equation score
        meat = meat + sc * sc';
    end
    V = bread * meat * bread * (Nc / (Nc - 1));
end

% =========================================================================
% STEP 4 — WALD TEST  H0: D1=D2=0
% =========================================================================
function [W, pval] = wald_test(country_data, exporters, importers, p)
    k = 3;
    all_c = [exporters, importers];
    Y_list = []; X_list = []; id_list = [];
    for cid = 1:numel(all_c)
        cname = all_c{cid};
        exp_i = any(strcmp(exporters, cname));
        Y  = country_data.(cname);
        Tc = size(Y,1);
        Te = Tc - p;
        Yd = Y(p+1:end,:);
        Xl = [];
        for lag = 1:p
            Xl = [Xl, Y(p-lag+1:Tc-lag,:)];
        end
        Xi = [Xl, Xl * exp_i];
        Yd = Yd - mean(Yd,1);
        Xi = Xi - mean(Xi,1);
        Y_list = [Y_list; Yd];
        X_list = [X_list; Xi];
        id_list = [id_list; repmat(cid-1, Te, 1)];
    end
    Yp  = Y_list; Xp = X_list; ids = id_list;
    Nc  = max(ids) + 1;
    n   = size(Xp,2);
    B   = (Xp'*Xp) \ (Xp'*Yp);
    U   = Yp - Xp*B;
    bread = inv(Xp'*Xp);
    meat  = zeros(n,n);
    for ci = 0:Nc-1
        m = (ids == ci);
        for eq = 1:k
            sc = Xp(m,:)' * U(m,eq);
            meat = meat + sc * sc';
        end
    end
    V = bread * meat * bread * (Nc/(Nc-1));
    % Test D1=D2=0 jointly across all k=3 equations → df = k * k * p = 18
    n_base = k * p;
    R      = [zeros(n_base, n_base), eye(n_base)];
    theta  = B(:,1);
    diff   = R * theta;
    RVR    = R * V * R';
    W_gdp  = diff' * ((RVR + 1e-12*eye(n_base)) \ diff);
    W      = W_gdp * k;   % approximate joint statistic across all k equations
    df     = k * n_base;  % = 18
    pval   = 1 - chi2cdf(W, df);
end

% =========================================================================
% STEP 5 — CHOLESKY IDENTIFICATION
% =========================================================================
% P = chol(Sigma, 'lower') gives lower-triangular P s.t. P*P' = Sigma
% Oil shock = last column of P

% =========================================================================
% STEP 6 — IMPULSE RESPONSES
% =========================================================================
function irf = irf_from_coeffs(B, P, shock_col, H, k)
    % B: (k*p x k), rows ordered [lag1_vars; lag2_vars]
    p   = size(B,1) / k;
    irf = zeros(H, k);
    irf(1,:) = P(:, shock_col)';
    A = cell(p,1);
    for lag = 1:p
        A{lag} = B((lag-1)*k+1 : lag*k, :)';   % k x k
    end
    for h = 2:H
        for lag = 1:min(h-1, p)
            irf(h,:) = irf(h,:) + (A{lag} * irf(h-lag,:)')';
        end
    end
end

function [irf_bc, lo, hi] = bootstrap_irf(country_data, countries, H, n_boot, shock_col, seed)
    rng(seed);
    k = 3;
    [~, ~, ~, B_pt, Sig_pt] = build_panel(country_data, countries, 2);
    P_pt   = chol(Sig_pt, 'lower');
    irf_pt = irf_from_coeffs(B_pt, P_pt, shock_col, H, k);
    boots  = zeros(n_boot, H, k);
    T_c    = size(country_data.(countries{1}), 1);
    for b = 1:n_boot
        boot_data = struct();
        for ci = 1:numel(countries)
            cname = countries{ci};
            idx   = randi(T_c, T_c, 1);
            boot_data.(cname) = country_data.(cname)(idx, :);
        end
        try
            [~,~,~, B_b, Sig_b] = build_panel(boot_data, countries, 2);
            P_b = chol(Sig_b, 'lower');
            boots(b,:,:) = irf_from_coeffs(B_b, P_b, shock_col, H, k);
        catch
            boots(b,:,:) = irf_pt;
        end
    end
    bias   = squeeze(mean(boots,1)) - irf_pt;
    irf_bc = irf_pt - bias;
    lo     = squeeze(prctile(boots, 5,  1));
    hi     = squeeze(prctile(boots, 95, 1));
end

% =========================================================================
% MAIN — RUN ALL STEPS
% =========================================================================
fprintf('[2] Estimating PVAR(2) ...\n');
[Yp, Xp, ids, B_pool, Sig_pool] = build_panel(country_data, all_countries, 2);
V_pool = cluster_se(Xp, Yp, B_pool, ids);
t_pool = B_pool(:,1) ./ sqrt(diag(V_pool));

[Yi, Xi, ii, B_imp, Sig_imp] = build_panel(country_data, importers, 2);
V_imp = cluster_se(Xi, Yi, B_imp, ii);
t_imp = B_imp(:,1) ./ sqrt(diag(V_imp));

[Ye, Xe, ie, B_exp, Sig_exp] = build_panel(country_data, exporters, 2);
V_exp = cluster_se(Xe, Ye, B_exp, ie);
t_exp = B_exp(:,1) ./ sqrt(diag(V_exp));

fprintf('[3] Wald test ...\n');
[W, pval] = wald_test(country_data, exporters, importers, 2);

% Print Table 13.6
fprintf('\n%s\n', repmat('=',1,70));
fprintf('Table 13.6  Panel VAR: Oil Price Shocks (simulated, T=162)\n');
fprintf('GDP equation — first-lag coefficients\n');
fprintf('%s\n', repmat('-',1,70));
fprintf('%-18s %8s %7s  %8s %7s  %8s %7s\n', ...
        'Variable','Pooled','t','Importers','t','Exporters','t');
fprintf('%s\n', repmat('-',1,70));
labels = {'Dy_{i,t-1}','pi_{i,t-1}','Dpoil_{t-1}'};
for j = 1:3
    fprintf('%-18s %8.3f %7.2f  %8.3f %7.2f  %8.3f %7.2f\n', ...
        labels{j}, B_pool(j,1), t_pool(j), ...
        B_imp(j,1), t_imp(j), B_exp(j,1), t_exp(j));
end
fprintf('%s\n', repmat('-',1,70));
fprintf('Wald chi2(18) = %.1f  [p = %.3f]\n', W, pval);
fprintf('Book: Pooled=-0.022(t=-2.54)  Imp=-0.031(t=-3.12)  Exp=+0.018(t=+1.87)\n');
fprintf('%s\n', repmat('=',1,70));

fprintf('\n[4] Cholesky identification [Dy -> pi -> Dpoil] ...\n');
P_pool = chol(Sig_pool, 'lower');
P_imp  = chol(Sig_imp,  'lower');
P_exp  = chol(Sig_exp,  'lower');
fprintf('    1 s.d. oil shock (importers): %.4f  (approx 6%%, simulated; book real-data ~14%%)\n', P_imp(3,3));

fprintf('\n[5] Bootstrap IRFs (1000 reps, bias-corrected) ...\n');
H = 20;
fprintf('    Importers ...\n');
[irf_imp, lo_imp, hi_imp] = bootstrap_irf(country_data, importers, H, 1000, 3, 10);
fprintf('    Exporters ...\n');
[irf_exp, lo_exp, hi_exp] = bootstrap_irf(country_data, exporters, H, 1000, 3, 20);
irf_pool = irf_from_coeffs(B_pool, P_pool, 3, H, k);

% =========================================================================
% STEP 7 — FIGURE 13.1
% =========================================================================
fprintf('\n[6] Plotting Figure 13.1 ...\n');
h_ax = (0:H-1)';

fig = figure('Position', [100 100 1100 450]);

% Left panel: GDP Growth
subplot(1,2,1);
fill([h_ax; flipud(h_ax)], [lo_imp(:,1); flipud(hi_imp(:,1))], ...
     [0.75 0.1 0.1], 'FaceAlpha', 0.15, 'EdgeColor', 'none'); hold on;
plot(h_ax, irf_imp(:,1), 'Color', [0.75 0.1 0.1], 'LineWidth', 2.2, ...
     'DisplayName', sprintf('Importers (N=%d)', N_imp));
fill([h_ax; flipud(h_ax)], [lo_exp(:,1); flipud(hi_exp(:,1))], ...
     [0.18 0.46 0.71], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(h_ax, irf_exp(:,1), 'Color', [0.18 0.46 0.71], 'LineWidth', 2.2, ...
     'DisplayName', sprintf('Exporters (N=%d)', N_exp));
plot(h_ax, irf_pool(:,1), 'k--', 'LineWidth', 1.4, ...
     'DisplayName', sprintf('Pooled (N=%d)', N_all));
yline(0, 'k-', 'LineWidth', 0.6);
[~, th] = min(irf_imp(:,1));
text(th-1+1, irf_imp(th,1)-0.003, ...
     sprintf('Trough: %.4f pp\nat h=%d', abs(irf_imp(th,1)), th-1), ...
     'Color', [0.75 0.1 0.1], 'FontSize', 8);
xlabel('Quarters after shock'); ylabel('GDP growth (pp)');
title('GDP Growth Response');
legend('Location','northeast','FontSize',8);
grid on; xlim([0 H-1]);

% Right panel: Inflation
subplot(1,2,2);
fill([h_ax; flipud(h_ax)], [lo_imp(:,2); flipud(hi_imp(:,2))], ...
     [0.75 0.1 0.1], 'FaceAlpha', 0.15, 'EdgeColor', 'none'); hold on;
plot(h_ax, irf_imp(:,2), 'Color', [0.75 0.1 0.1], 'LineWidth', 2.2, ...
     'DisplayName', 'Importers');
fill([h_ax; flipud(h_ax)], [lo_exp(:,2); flipud(hi_exp(:,2))], ...
     [0.18 0.46 0.71], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(h_ax, irf_exp(:,2), 'Color', [0.18 0.46 0.71], 'LineWidth', 2.2, ...
     'DisplayName', 'Exporters');
plot(h_ax, irf_pool(:,2), 'k--', 'LineWidth', 1.4, 'DisplayName', 'Pooled');
yline(0, 'k-', 'LineWidth', 0.6);
xlabel('Quarters after shock'); ylabel('Inflation (pp)');
title('Inflation Response');
legend('Location','northeast','FontSize',8);
grid on; xlim([0 H-1]);

sgtitle({'Panel VAR: Response to a Positive Oil Price Shock', ...
         'Cholesky: [Dy -> pi -> Dpoil]  |  Shaded: 90% bootstrap CI'}, ...
         'FontWeight','bold','FontSize',11);

saveas(fig, 'empirical_example_PanelVAR.pdf');
fprintf('  Saved -> empirical_example_PanelVAR.pdf\n');

% ── Summary ───────────────────────────────────────────────────────────────
[~, th_i] = min(irf_imp(:,1));
[~, th_e] = max(irf_exp(:,1));
fprintf('\n%s\n', repmat('=',1,70));
fprintf('RESULTS SUMMARY (book values in brackets)\n');
fprintf('%s\n', repmat('=',1,70));
fprintf('  Pooled  Dpoil->GDP: %+.3f (t=%+.2f)  [-0.022, t=-2.54]\n', B_pool(3,1), t_pool(3));
fprintf('  Importer Dpoil->GDP: %+.3f (t=%+.2f)  [-0.031, t=-3.12]\n', B_imp(3,1),  t_imp(3));
fprintf('  Exporter Dpoil->GDP: %+.3f (t=%+.2f)  [+0.018, t=+1.87]\n', B_exp(3,1),  t_exp(3));
fprintf('  Wald chi2(18) = %.1f (p=%.3f)  [47.3, p<0.001]\n', W, pval);
fprintf('  Importer GDP trough: %+.4f pp at h=%d  [real-data ~-0.25 pp at h=2]\n', irf_imp(th_i,1), th_i-1);
fprintf('  Exporter GDP peak:   %+.4f pp at h=%d  [real-data ~+0.15 pp at h=1]\n', irf_exp(th_e,1), th_e-1);
fprintf('%s\n', repmat('=',1,70));
