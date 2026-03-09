% =========================================================================
% GVAR: US MONETARY POLICY SPILLOVERS
% Replication of Section 13.7.2 — Macroeconometrics Textbook
% Author:   Alessia Paccagnini
% Textbook: Macroeconometrics (De Gruyter)
% =========================================================================
%
% Book specification (eq. 13.74):
%   y_it = alpha_i + Phi_i y_{i,t-1} + Lam0_i y*_it
%                  + Lam1_i y*_{i,t-1} + eps_it
%   USA: closed VAR(2) + weakly exogenous oil (dominant unit)
%   y_it = (Dy_it, pi_it, rs_it)'
%
% Steps:
%   1. Simulate GVAR data (T=162, N=8)
%   2. Construct trade-weighted foreign variables
%   3. Estimate VARX*(1,1) per country
%   4. Weak exogeneity tests  (Table 13.7)
%   5. Stack global VAR  (eq. 13.71)
%   6. Stability check + Figure 13.2
%   7. GIRF to US rate shock + Figure 13.3
%   8. FEVD + spillover network + Figure 13.4
% =========================================================================

clear; clc; rng(42);

% ── Dimensions ───────────────────────────────────────────────────────────
countries = {'USA','Euro_Area','UK','Japan','China','Canada','Korea','Brazil'};
N = numel(countries);
K = 3;     % gdp, inflation, interest rate
T = 162;   % 1979Q2-2019Q4

% ── Trade weight matrix (wCAN,USA = 0.55) ────────────────────────────────
W_raw = [
    0.00, 0.18, 0.10, 0.12, 0.14, 0.22, 0.13, 0.11;   % USA
    0.26, 0.00, 0.18, 0.10, 0.14, 0.08, 0.12, 0.12;   % Euro Area
    0.22, 0.36, 0.00, 0.08, 0.10, 0.07, 0.08, 0.09;   % UK
    0.28, 0.14, 0.07, 0.00, 0.21, 0.08, 0.13, 0.09;   % Japan
    0.20, 0.18, 0.08, 0.18, 0.00, 0.08, 0.18, 0.10;   % China
    0.55, 0.09, 0.07, 0.07, 0.08, 0.00, 0.07, 0.07;   % Canada
    0.25, 0.14, 0.08, 0.20, 0.18, 0.07, 0.00, 0.08;   % Korea
    0.22, 0.20, 0.10, 0.08, 0.14, 0.08, 0.08, 0.00;   % Brazil
];
W = W_raw ./ sum(W_raw, 2);   % row-normalise

% ── DGP parameters ───────────────────────────────────────────────────────
A1_US = [ 0.50,  0.00, -0.28;
          0.09,  0.55,  0.06;
          0.22,  0.12,  0.80];
A2_US = [ 0.09,  0.00, -0.12;
          0.02,  0.07,  0.00;
          0.06,  0.04,  0.08];
GAMMA_OIL_US = [-0.06; 0.08; 0.02];

% Phi_i, Lambda0_i, Lambda1_i for non-US countries
PHI  = struct();
LAM0 = struct();

PHI.Euro_Area  = diag([0.50, 0.62, 0.82]);
PHI.UK         = diag([0.48, 0.58, 0.80]);
PHI.Japan      = diag([0.44, 0.52, 0.85]);
PHI.China      = diag([0.60, 0.65, 0.84]);
PHI.Canada     = diag([0.52, 0.58, 0.81]);
PHI.Korea      = diag([0.46, 0.54, 0.78]);
PHI.Brazil     = diag([0.40, 0.68, 0.74]);

LAM0.Euro_Area = [-0.10, 0.04, -0.08; 0.04, 0.08, 0.04; 0.06, 0.04, 0.14];
LAM0.UK        = [-0.12, 0.04, -0.10; 0.04, 0.09, 0.04; 0.07, 0.04, 0.16];
LAM0.Japan     = [-0.05, 0.02, -0.04; 0.02, 0.04, 0.02; 0.04, 0.02, 0.10];
LAM0.China     = [-0.04, 0.02, -0.03; 0.02, 0.05, 0.02; 0.03, 0.02, 0.08];
LAM0.Canada    = [-0.18, 0.06, -0.14; 0.06, 0.12, 0.06; 0.09, 0.06, 0.20];
LAM0.Korea     = [-0.07, 0.03, -0.06; 0.03, 0.07, 0.03; 0.05, 0.03, 0.12];
LAM0.Brazil    = [-0.06, 0.04, -0.05; 0.04, 0.09, 0.03; 0.05, 0.04, 0.10];

% LAM1 = 0.35 * LAM0
LAM1 = struct();
non_us = countries(2:end);
for i = 1:numel(non_us)
    c = non_us{i};
    LAM1.(c) = 0.35 * LAM0.(c);
end

SIG = struct();
SIG.USA      = [0.010, 0.005, 0.006];
SIG.Euro_Area= [0.009, 0.004, 0.005];
SIG.UK       = [0.010, 0.005, 0.006];
SIG.Japan    = [0.009, 0.003, 0.004];
SIG.China    = [0.013, 0.005, 0.006];
SIG.Canada   = [0.010, 0.004, 0.005];
SIG.Korea    = [0.012, 0.005, 0.008];
SIG.Brazil   = [0.018, 0.008, 0.012];

ALPHA = struct();
ALPHA.USA      = [0.006; 0.004; 0.010];
ALPHA.Euro_Area= [0.004; 0.003; 0.008];
ALPHA.UK       = [0.005; 0.004; 0.009];
ALPHA.Japan    = [0.003; 0.001; 0.005];
ALPHA.China    = [0.013; 0.004; 0.008];
ALPHA.Canada   = [0.005; 0.004; 0.010];
ALPHA.Korea    = [0.008; 0.004; 0.060];
ALPHA.Brazil   = [0.004; 0.006; 0.055];

% =========================================================================
% STEP 1 — SIMULATE DATA
% =========================================================================
fprintf('[1] Simulating GVAR data (T=%d, N=%d) ...\n', T, N);

oil = zeros(T,1);
for t = 2:T
    oil(t) = 0.75*oil(t-1) + 0.045*randn();
end

Y_data = struct();
for i = 1:N
    Y_data.(countries{i}) = zeros(T, K);
end

% Initialise
for i = 1:N
    c = countries{i};
    sig_c = SIG.(c);
    Y_data.(c)(1,:) = ALPHA.(c)' + sig_c .* randn(1,K) * 0.5;
    Y_data.(c)(2,:) = ALPHA.(c)' + sig_c .* randn(1,K) * 0.5;
end

for t = 3:T
    % USA (dominant unit, closed VAR(2) + oil)
    eps_us = (SIG.USA .* randn(1,K))';
    Y_data.USA(t,:) = (ALPHA.USA + A1_US * Y_data.USA(t-1,:)' ...
                       + A2_US * Y_data.USA(t-2,:)' ...
                       + GAMMA_OIL_US * oil(t) + eps_us)';

    % Non-US countries
    for i = 2:N
        c   = countries{i};
        y_star   = zeros(K,1);
        y_star_l = zeros(K,1);
        for j = 1:N
            if j ~= i
                c2 = countries{j};
                wij = W(i,j);
                if j == 1   % USA: use contemporaneous
                    y_star   = y_star   + wij * Y_data.(c2)(t,:)';
                else
                    y_star   = y_star   + wij * Y_data.(c2)(t-1,:)';
                end
                y_star_l = y_star_l + wij * Y_data.(c2)(t-1,:)';
            end
        end
        eps_c = (SIG.(c) .* randn(1,K))';
        Y_data.(c)(t,:) = (ALPHA.(c) + PHI.(c)*Y_data.(c)(t-1,:)' ...
                           + LAM0.(c)*y_star + LAM1.(c)*y_star_l + eps_c)';
    end
end
fprintf('    wCAN,USA = %.2f  (book: 0.55)\n', W(6,1));

% =========================================================================
% STEP 2 — FOREIGN VARIABLES
% =========================================================================
foreign = struct();
for i = 1:N
    c  = countries{i};
    Ys = zeros(T, K);
    for j = 1:N
        if j ~= i
            Ys = Ys + W(i,j) * Y_data.(countries{j});
        end
    end
    foreign.(c) = Ys;
end

% =========================================================================
% STEP 3 — ESTIMATE VARX* COUNTRY MODELS
% =========================================================================
function model = estimate_varx(Y, Y_star, p, q)
    [Tc, k]  = size(Y);
    ks = size(Y_star, 2);
    ml = max(p, q);
    Te = Tc - ml;
    X  = [ones(Te,1), Y(ml:Tc-1,:), Y_star(ml+1:Tc,:), Y_star(ml:Tc-1,:)];
    Yd = Y(ml+1:Tc,:);
    B  = (X'*X) \ (X'*Yd);
    U  = Yd - X*B;
    Sig = (U'*U) / Te;
    ss_res = sum(U.^2, 1);
    ss_tot = sum((Yd - mean(Yd,1)).^2, 1) + 1e-12;
    r2 = max(0, 1 - ss_res ./ ss_tot);
    % Extract component matrices
    n1=1; n2=n1+k; n3=n2+ks; n4=n3+ks;
    model.B     = B;
    model.Phi   = B(n1+1:n2, :)';   % k x k
    model.Lam0  = B(n2+1:n3, :)';   % k x ks
    model.Lam1  = B(n3+1:n4, :)';   % k x ks
    model.Sigma = Sig;
    model.U     = U;
    model.r2    = r2;
    model.r2_avg = mean(r2);
end

function model = estimate_var_usa(Y, p)
    % Pure VAR(p) for USA — no foreign variables
    [Tc, k] = size(Y);
    Te = Tc - p;
    % Build X: [const, y_{t-1}, y_{t-2}, ..., y_{t-p}]
    X = ones(Te, 1);
    for lag = 1:p
        X = [X, Y(p-lag+1:Tc-lag, :)];
    end
    Yd = Y(p+1:Tc, :);
    B  = (X'*X) \ (X'*Yd);
    U  = Yd - X*B;
    Sig = (U'*U) / Te;
    ss_res = sum(U.^2, 1);
    ss_tot = sum((Yd - mean(Yd,1)).^2, 1) + 1e-12;
    r2 = max(0, 1 - ss_res ./ ss_tot);
    model.B     = B;
    model.Phi   = B(2:k+1, :)';        % k x k  (lag-1 block)
    model.Lam0  = zeros(k, k);         % no foreign contemporaneous
    model.Lam1  = zeros(k, k);         % no foreign lagged
    model.Sigma = Sig;
    model.U     = U;
    model.r2    = r2;
    model.r2_avg = mean(r2);
end

fprintf('\n[3] Estimating VARX*(1,1) country models ...\n');
models = struct();
for i = 1:N
    c = countries{i};
    Y  = Y_data.(c);
    if strcmp(c, 'USA')
        models.(c) = estimate_var_usa(Y, 2);
    else
        models.(c) = estimate_varx(Y, foreign.(c), 1, 1);
    end
    fprintf('  %-12s: avg R2=%.3f\n', c, models.(c).r2_avg);
end

% =========================================================================
% STEP 4 — WEAK EXOGENEITY TESTS
% =========================================================================
fprintf('\n[4] Weak exogeneity tests ...\n');
fprintf('%s\n', repmat('=',1,52));
fprintf('Table 13.7  Weak Exogeneity Tests: F-Statistics\n');
fprintf('%s\n', repmat('=',1,52));
fprintf('%-14s %8s %8s %8s\n', 'Country', 'Dy*', 'pi*', 'rs*');
fprintf('%s\n', repmat('-',1,52));
crit = 3.07;

for i = 2:N
    c    = countries{i};
    U_hat = models.(c).U;
    Ys    = foreign.(c);
    Te    = size(U_hat,1);
    dYs   = diff(Ys(end-Te:end,:), 1, 1);
    dYs_  = dYs(2:end,:);
    ec    = U_hat(1:end-1,:);
    n_obs = size(dYs_,1);
    f_stats = zeros(1,K);
    for v = 1:K
        y  = dYs_(:,v);
        lag_y = [0; y(1:end-1)];
        X_u = [ones(n_obs,1), lag_y, ec(1:n_obs,1)];
        B_u = (X_u'*X_u) \ (X_u'*y);
        ss_u = sum((y - X_u*B_u).^2);
        X_r = X_u(:,1:end-1);
        B_r = (X_r'*X_r) \ (X_r'*y);
        ss_r = sum((y - X_r*B_r).^2);
        df2 = max(n_obs - size(X_u,2), 1);
        f_stats(v) = ((ss_r-ss_u)/1) / (ss_u/df2);
    end
    flags = repmat(' ',1,3);
    for v=1:3; if f_stats(v)>crit; flags(v)='*'; end; end
    fprintf('%-14s %7.2f%s %7.2f%s %7.2f%s\n', ...
        strrep(c,'_',' '), f_stats(1),flags(1), f_stats(2),flags(2), f_stats(3),flags(3));
end
fprintf('%s\n', repmat('-',1,52));
fprintf('5%% critical value ~%.2f  (* exceeds)\n', crit);
fprintf('%s\n', repmat('=',1,52));

% =========================================================================
% STEP 5 — STACK GLOBAL VAR
% =========================================================================
fprintf('\n[5] Stacking global VAR ...\n');
Kg = N * K;
G0 = eye(Kg);
G1 = zeros(Kg);

for i = 1:N
    c    = countries{i};
    rs   = (i-1)*K + 1;
    re   = i*K;
    Phi  = models.(c).Phi;
    G1(rs:re, rs:re) = Phi;
    % USA is the dominant unit: no foreign variables in its equation
    if strcmp(c, 'USA'); continue; end
    Lam0 = models.(c).Lam0;
    Lam1 = models.(c).Lam1;
    for j = 1:N
        if j ~= i
            cs = (j-1)*K+1; ce = j*K;
            wij = W(i,j);
            G0(rs:re, cs:ce) = G0(rs:re, cs:ce) - Lam0 * wij;
            G1(rs:re, cs:ce) = G1(rs:re, cs:ce) + Lam1 * wij;
        end
    end
end

G0_inv  = inv(G0);
F_mat   = G0_inv * G1;

% Global shock covariance
Sig_blocks = zeros(Kg, Kg);
for i = 1:N
    rs = (i-1)*K+1; re = i*K;
    Sig_blocks(rs:re, rs:re) = models.(countries{i}).Sigma;
end
Sigma_e = G0_inv * Sig_blocks * G0_inv';
fprintf('    Global system: %dx%d  (%d countries x %d vars)\n', Kg, Kg, N, K);

% =========================================================================
% STEP 6 — STABILITY + FIGURE 13.2
% =========================================================================
fprintf('\n[6] Stability check + Figure 13.2 ...\n');
eigs_F   = eig(F_mat);
max_mod  = max(abs(eigs_F));
fprintf('    Max |lambda| = %.3f  (book: 0.973)\n', max_mod);

fig2 = figure('Position',[100 100 550 550]);
theta_c = linspace(0, 2*pi, 300);
plot(cos(theta_c), sin(theta_c), 'k-', 'LineWidth', 1, 'DisplayName','Unit circle'); hold on;
scatter(real(eigs_F), imag(eigs_F), 45, [0.18 0.46 0.71], 'filled', ...
        'DisplayName','Eigenvalues');
xline(0,'Color',[0.5 0.5 0.5],'LineWidth',0.4);
yline(0,'Color',[0.5 0.5 0.5],'LineWidth',0.4);
axis equal; xlim([-1.3 1.3]); ylim([-1.3 1.3]);
xlabel('Real'); ylabel('Imaginary');
title(sprintf('Eigenvalues of the GVAR Companion Matrix\nMax |\\lambda| = %.3f', max_mod), ...
      'FontWeight','bold');
legend('Location','best','FontSize',9);
text(0.98, 0.02, sprintf('Max |\\lambda| = %.3f', max_mod), ...
     'Units','normalized','HorizontalAlignment','right', ...
     'Color',[0.75 0.1 0.1],'FontSize',9);
grid on;
saveas(fig2, 'empirical_example_GlobalVAR_eigenvalues.pdf');
fprintf('    Saved -> empirical_example_GlobalVAR_eigenvalues.pdf\n');

% =========================================================================
% STEP 7 — GIRF
% =========================================================================
function irf_out = compute_girf(F_mat, Sigma_e, shock_cid, shock_vid, H, K)
    % shock_cid: 1-based country index, shock_vid: 1-based variable index
    Kg  = size(F_mat,1);
    j   = (shock_cid-1)*K + shock_vid;
    e_j = zeros(Kg,1); e_j(j) = 1;
    sig_jj = Sigma_e(j,j);
    b      = Sigma_e * e_j / sqrt(sig_jj);
    irf_raw = zeros(H, Kg);
    Cs = eye(Kg);
    irf_raw(1,:) = (Cs * b)';
    for h = 2:H
        Cs = Cs * F_mat;
        irf_raw(h,:) = (Cs * b)';
    end
    irf_out = struct();
    for i = 1:N_
        c = countries_{i};
        irf_out.(c) = irf_raw(:, (i-1)*K+1 : i*K);
    end
end

% Bootstrap GIRF
function [irf_lo, irf_hi] = bootstrap_girf(F_mat, Sigma_e, models, G0_inv, ...
                                             countries, N, K, ...
                                             shock_cid, shock_vid, H, n_boot, seed)
    rng(seed);
    Kg = size(F_mat,1);
    boots = struct();
    for i = 1:N
        boots.(countries{i}) = zeros(n_boot, H, K);
    end
    for b = 1:n_boot
        Sig_b = zeros(Kg);
        for i = 1:N
            c = countries{i};
            U_b = models.(c).U;
            idx = randi(size(U_b,1), size(U_b,1), 1);
            Ub  = U_b(idx,:);
            Sb  = (Ub'*Ub) / size(Ub,1);
            rs=(i-1)*K+1; re=i*K;
            Sig_b(rs:re,rs:re) = Sb;
        end
        Sigma_eb = G0_inv * Sig_b * G0_inv';
        F_b = F_mat + 0.02*std(F_mat(:))*randn(Kg,Kg);
        try
            irf_b = girf_internal(F_b, Sigma_eb, shock_cid, shock_vid, H, K, countries, N);
            for i=1:N
                c = countries{i};
                boots.(c)(b,:,:) = irf_b.(c);
            end
        catch; end
    end
    irf_lo = struct(); irf_hi = struct();
    for i = 1:N
        c = countries{i};
        irf_lo.(c) = squeeze(prctile(boots.(c), 5,  1));
        irf_hi.(c) = squeeze(prctile(boots.(c), 95, 1));
    end
end

function irf_out = girf_internal(F_mat, Sigma_e, shock_cid, shock_vid, H, K, countries, N)
    Kg  = size(F_mat,1);
    j   = (shock_cid-1)*K + shock_vid;
    e_j = zeros(Kg,1); e_j(j) = 1;
    b   = Sigma_e * e_j / sqrt(max(Sigma_e(j,j),1e-12));
    irf_raw = zeros(H, Kg);
    Cs = eye(Kg);
    irf_raw(1,:) = (Cs * b)';
    for h = 2:H
        Cs = Cs * F_mat;
        irf_raw(h,:) = (Cs * b)';
    end
    irf_out = struct();
    for i = 1:N
        c = countries{i};
        irf_out.(c) = irf_raw(:, (i-1)*K+1 : i*K);
    end
end

% Since nested functions can't use workspace vars directly in MATLAB scripts,
% call with explicit args:
fprintf('\n[7] GIRF to US rate shock + Figure 13.3 ...\n');
H = 20;
shock_cid = 1;   % USA = country 1
shock_vid = 3;   % rs  = variable 3

irf = girf_internal(F_mat, Sigma_e, shock_cid, shock_vid, H, K, countries, N);
fprintf('    Bootstrap CIs (500 reps) ...\n');
[irf_lo, irf_hi] = bootstrap_girf(F_mat, Sigma_e, models, G0_inv, ...
                                   countries, N, K, ...
                                   shock_cid, shock_vid, H, 500, 20);

% Figure 13.3
fig3 = figure('Position', [100 100 1300 650]);
h_ax = (0:H-1)';
col_imp = [0.18 0.46 0.71];
for idx = 1:N
    c   = countries{idx};
    gdp = irf.(c)(:,1);
    lo_ = irf_lo.(c)(:,1);
    hi_ = irf_hi.(c)(:,1);
    subplot(2,4,idx);
    fill([h_ax; flipud(h_ax)], [lo_; flipud(hi_)], col_imp, ...
         'FaceAlpha',0.20,'EdgeColor','none'); hold on;
    plot(h_ax, gdp, 'Color', col_imp, 'LineWidth', 2.0);
    yline(0,'k-','LineWidth',0.6);
    [~,th] = min(gdp);
    text(th, gdp(th)-0.005, sprintf('%.3f',gdp(th)), ...
         'Color',[0.75 0.1 0.1],'FontSize',8,'HorizontalAlignment','center');
    title(strrep(c,'_',' '),'FontWeight','bold','FontSize',9);
    xlabel('Quarters','FontSize',8); ylabel('GDP growth (pp)','FontSize',8);
    xlim([0 H-1]); grid on;
end
%sgtitle({'Figure 13.3: GVAR — GDP Response to US Monetary Policy Tightening', ...
        % 'GIRF to 1 s.d. increase in rs_{US} (~50 bp)  |  90% bootstrap CI'}, ...
        %'FontWeight','bold','FontSize',10);
saveas(fig3, 'empirical_example_GlobalVAR_girf.pdf');
fprintf('    Saved -> empirical_example_GlobalVAR_girf.pdf\n');

% =========================================================================
% STEP 8 — FEVD + NETWORK  (Figure 13.4)
% =========================================================================
fprintf('\n[8] FEVD + Spillover network + Figure 13.4 ...\n');

H_fevd = 8;
Cs_all = zeros(Kg, Kg, H_fevd);
Cs_all(:,:,1) = eye(Kg);
for h = 2:H_fevd
    Cs_all(:,:,h) = Cs_all(:,:,h-1) * F_mat;
end

fevd_gdp = zeros(N, N);
for i = 1:N
    v = (i-1)*K + 1;   % GDP index
    e_v = zeros(Kg,1); e_v(v) = 1;
    fev_tot = 0;
    for h = 1:H_fevd
        Cs = Cs_all(:,:,h);
        fev_tot = fev_tot + e_v' * Cs * Sigma_e * Cs' * e_v;
    end
    fev_tot = max(fev_tot, 1e-12);
    for j = 1:N
        num = 0;
        for s = 1:K
            js = (j-1)*K + s;
            e_js = zeros(Kg,1); e_js(js) = 1;
            sig_js = max(Sigma_e(js,js), 1e-12);
            for h = 1:H_fevd
                Cs = Cs_all(:,:,h);
                num = num + (e_v' * Cs * Sigma_e * e_js)^2 / sig_js;
            end
        end
        fevd_gdp(i,j) = num / fev_tot;
    end
end
fevd_gdp = fevd_gdp ./ sum(fevd_gdp,2);   % row normalise

short_names = {'USA','EUR','GBR','JPN','CHN','CAN','KOR','BRA'};
long_names  = {'United States','Euro Area','United Kingdom','Japan', ...
               'China','Canada','Korea','Brazil'};
pct = round(fevd_gdp * 100);

fig4 = figure('Position',[100 100 1300 520]);

% Heatmap
subplot(1,2,1);
blue_cmap = [linspace(1,0.03,100)', linspace(1,0.27,100)', linspace(1,0.58,100)'];
imagesc(pct); colormap(gca, blue_cmap); caxis([0 100]);
cb = colorbar; cb.Label.String = 'Share of GDP FEV (%)';
set(gca,'XTick',1:N,'XTickLabel',short_names,'FontSize',8, ...
        'YTick',1:N,'YTickLabel',long_names);
xlabel('Shock source (country j)');
ylabel('Affected country (country i)');
title('Forecast Error Variance Decomposition (h=8)');
for i=1:N
    for j=1:N
        v_ = pct(i,j);
        col_ = 'k'; if v_>55; col_='w'; end
        text(j,i,num2str(v_),'HorizontalAlignment','center','FontSize',7,'Color',col_);
    end
end

% Network
subplot(1,2,2);
axis off; hold on;
title('Spillover Network (edges > 3% of GDP FEV)');
theta_n = linspace(0, 2*pi, N+1); theta_n = theta_n(1:end-1);
pos_x = 0.85 * cos(theta_n);
pos_y = 0.85 * sin(theta_n);
out_conn = sum(fevd_gdp,2) - diag(fevd_gdp);

% Edges
for i=1:N
    for j=1:N
        if i~=j && fevd_gdp(i,j)>0.03
            lw_ = 1 + 6*fevd_gdp(i,j);
            annotation_arrow = plot([pos_x(i), pos_x(j)], [pos_y(i), pos_y(j)], ...
                '-','Color',[0.18 0.46 0.71 0.55],'LineWidth',lw_);
        end
    end
end
% Nodes
for i=1:N
    r_ = 0.06 + 0.14*out_conn(i);
    theta_circle = linspace(0,2*pi,50);
    xc = pos_x(i) + r_*cos(theta_circle);
    yc = pos_y(i) + r_*sin(theta_circle);
    fill(xc, yc, [0.18 0.46 0.71], 'FaceAlpha',0.85,'EdgeColor','none');
    text(pos_x(i), pos_y(i), short_names{i}, 'HorizontalAlignment','center', ...
         'Color','w','FontWeight','bold','FontSize',7);
end
xlim([-1.4 1.4]); ylim([-1.4 1.4]); axis equal;

%sgtitle({'Figure 13.4: GVAR — FEVD and Spillover Network (h=8 quarters)'}, ...
        %'FontWeight','bold','FontSize',10);
saveas(fig4, 'empirical_example_GlobalVAR_fevd.pdf');
fprintf('    Saved -> empirical_example_GlobalVAR_fevd.pdf\n');

% ── Summary ───────────────────────────────────────────────────────────────
fprintf('\n%s\n', repmat('=',1,70));
fprintf('RESULTS SUMMARY (book values in brackets)\n');
fprintf('%s\n', repmat('=',1,70));
fprintf('  Max eigenvalue: %.3f  [0.973]\n', max_mod);
fprintf('\n  Peak GDP response to 1 s.d. US rate shock:\n');
book_peaks = [-0.141,-0.022,-0.034,-0.009,-0.003,-0.093,-0.004,NaN];
for i=1:N
    c   = countries{i};
    gdp = irf.(c)(:,1);
    [~,th] = min(gdp);
    if ~isnan(book_peaks(i))
        fprintf('    %-12s: %+.3f pp at h=%d  [%.3f]\n', ...
                strrep(c,'_',' '), gdp(th), th-1, book_peaks(i));
    else
        fprintf('    %-12s: %+.3f pp at h=%d  [n/a]\n', ...
                strrep(c,'_',' '), gdp(th), th-1);
    end
end
can_i=6; usa_i=1; eur_i=2; uk_i=3;
fprintf('\n  GDP FEVD at h=8:\n');
fprintf('    Own shocks: %d-%d%%  [book: 91-97%%]\n', ...
        min(pct(sub2ind([N,N],1:N,1:N))), max(pct(sub2ind([N,N],1:N,1:N))));
fprintf('    USA -> Canada: %d%%  [book: ~6%%]\n', pct(can_i,usa_i));
fprintf('    EUR -> UK:     %d%%  [book: ~3%%]\n', pct(uk_i,eur_i));
fprintf('%s\n', repmat('=',1,70));
