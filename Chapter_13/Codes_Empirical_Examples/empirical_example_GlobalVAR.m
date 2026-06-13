% =========================================================================
% GVAR: US MONETARY POLICY SPILLOVERS
% Replication of Section 13.7.2 — Macroeconometrics Textbook
% Author:   Alessia Paccagnini
% Textbook: Macroeconometrics (De Gruyter)
% =========================================================================
%
% -------------------------------------------------------------------------
% NOTE ON REPRODUCIBILITY ACROSS LANGUAGES (Python / MATLAB / R)
% -------------------------------------------------------------------------
% The Python, MATLAB, and R companion scripts are designed to produce
% IDENTICAL numbers. The dataset is generated ONCE by the Python script and
% written to a shared CSV (gvar_simulated_data.csv). THIS MATLAB script READS
% that CSV rather than simulating, so estimation, the global stacking, the
% GIRF, the FEVD, and the Table 13.7 weak-exogeneity tests run on exactly the
% same data in all three languages.
%
% >> Run empirical_example_GlobalVAR.py first (it writes the CSV), or copy
%    gvar_simulated_data.csv into this working directory before running.
%
% IMPORTANT (MATLAB script structure): in a MATLAB *script*, ALL local
% functions must be defined at the END of the file, after the last line of
% script code. They are therefore collected in the "LOCAL FUNCTIONS" section
% at the bottom. Defining them mid-script (as in an earlier draft) makes
% MATLAB refuse to run the file at all ("All functions in a script must be
% defined at the end of the file"), which is why no figures/PDFs were saved.
%
% Model (book eq. 13.74): each non-US country is a VARX*(1,1); the USA is the
% dominant unit and a closed VAR(1) with weakly exogenous oil.
% Impulse responses are in the variables' own decimal units (x10^-3 / x10^-4).
% Figures are saved WITHOUT internal titles or book figure numbers, so the
% LaTeX captions carry the numbering.
% -------------------------------------------------------------------------
%
% Book specification (eq. 13.74):
%   y_it = alpha_i + Phi_i y_{i,t-1} + Lam0_i y*_it
%                  + Lam1_i y*_{i,t-1} + eps_it
%   USA: closed VAR(1) + weakly exogenous oil (dominant unit)
%   y_it = (Dy_it, pi_it, rs_it)'
%
% Steps:
%   1. Load shared GVAR data (T=162, N=8)
%   2. Construct trade-weighted foreign variables
%   3. Estimate VARX*(1,1) per country
%   4. Weak exogeneity tests  (Table 13.7)
%   5. Stack global VAR  (eq. 13.71)
%   6. Stability check — eigenvalue plot
%   7. GIRF to US rate shock — 8-panel plot
%   8. FEVD + spillover network
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

% =========================================================================
% STEP 1 — LOAD SHARED DATA (written by the Python script)
% =========================================================================
csv_path = 'gvar_simulated_data.csv';
fprintf('[1] Loading shared GVAR data <- %s ...\n', csv_path);
if exist(csv_path, 'file') ~= 2
    error(['Shared dataset not found: %s\n' ...
           'Run empirical_example_GlobalVAR.py first (it writes the CSV), ' ...
           'or copy the CSV into this directory.'], csv_path);
end
% Read by COLUMN POSITION (not header name) so it works across MATLAB
% versions regardless of how readtable would sanitise the header.
% CSV layout: date(text), country(text), gdp, inf, rs
fid = fopen(csv_path, 'r');
C   = textscan(fid, '%s%s%f%f%f', 'Delimiter', ',', ...
               'HeaderLines', 1, 'TreatAsEmpty', {'NA','NaN'});
fclose(fid);
csv_country = C{2};
csv_vals    = [C{3}, C{4}, C{5}];   % gdp, inf, rs
Y_data = struct();
for i = 1:N
    c = countries{i};
    rows = strcmp(csv_country, c);
    if ~any(rows)
        error('Country "%s" not found in %s', c, csv_path);
    end
    Y_data.(c) = csv_vals(rows, :);
end
fprintf('    Loaded %d countries x %d obs\n', N, size(Y_data.(countries{1}),1));
fprintf('    wCAN,USA = %.2f\n', W(6,1));

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
fprintf('\n[3] Estimating VARX*(1,1) country models ...\n');
models = struct();
for i = 1:N
    c = countries{i};
    Y  = Y_data.(c);
    if strcmp(c, 'USA')
        models.(c) = estimate_var_usa(Y, 1);
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
% STEP 6 — STABILITY + EIGENVALUE PLOT
% =========================================================================
fprintf('\n[6] Stability check + eigenvalue plot ...\n');
eigs_F   = eig(F_mat);
max_mod  = max(abs(eigs_F));
fprintf('    Max |lambda| = %.3f\n', max_mod);

fig2 = figure('Position',[100 100 550 550]);
theta_c = linspace(0, 2*pi, 300);
plot(cos(theta_c), sin(theta_c), 'k-', 'LineWidth', 1, 'DisplayName','Unit circle'); hold on;
scatter(real(eigs_F), imag(eigs_F), 45, [0.18 0.46 0.71], 'filled', ...
        'DisplayName','Eigenvalues');
xline(0,'Color',[0.5 0.5 0.5],'LineWidth',0.4,'HandleVisibility','off');
yline(0,'Color',[0.5 0.5 0.5],'LineWidth',0.4,'HandleVisibility','off');
axis equal; xlim([-1.3 1.3]); ylim([-1.3 1.3]);
xlabel('Real'); ylabel('Imaginary');
legend('Location','best','FontSize',9);
text(0.98, 0.02, sprintf('Max |\\lambda| = %.3f', max_mod), ...
     'Units','normalized','HorizontalAlignment','right', ...
     'Color',[0.75 0.1 0.1],'FontSize',9);
grid on;
saveas(fig2, 'GlobalVAR_eigenvalues.pdf');
print(fig2, 'GlobalVAR_eigenvalues.png', '-dpng', '-r300');
fprintf('    Saved -> GlobalVAR_eigenvalues.pdf / .png\n');

% =========================================================================
% STEP 7 — GIRF
% =========================================================================
fprintf('\n[7] GIRF to US rate shock ...\n');
H = 20;
shock_cid = 1;   % USA = country 1
shock_vid = 3;   % rs  = variable 3

irf = girf_internal(F_mat, Sigma_e, shock_cid, shock_vid, H, K, countries, N);
fprintf('    Bootstrap CIs (500 reps) ...\n');
[irf_lo, irf_hi] = bootstrap_girf(F_mat, Sigma_e, models, G0_inv, ...
                                   countries, N, K, ...
                                   shock_cid, shock_vid, H, 500, 20);

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
    span = max(max(hi_) - min(lo_), 1e-9);
    text(th, gdp(th)-0.18*span, sprintf('%.1e',gdp(th)), ...
         'Color',[0.75 0.1 0.1],'FontSize',7,'HorizontalAlignment','center');
    title(strrep(c,'_',' '),'FontWeight','bold','FontSize',9);
    xlabel('Quarters','FontSize',8); ylabel('GDP growth (pp)','FontSize',8);
    xlim([0 H-1]); grid on;
end
saveas(fig3, 'GlobalVAR_girf.pdf');
print(fig3, 'GlobalVAR_girf.png', '-dpng', '-r300');
fprintf('    Saved -> GlobalVAR_girf.pdf / .png\n');

% =========================================================================
% STEP 8 — FEVD + NETWORK
% =========================================================================
fprintf('\n[8] FEVD + Spillover network ...\n');

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
theta_n = linspace(0, 2*pi, N+1); theta_n = theta_n(1:end-1);
pos_x = 0.85 * cos(theta_n);
pos_y = 0.85 * sin(theta_n);
out_conn = sum(fevd_gdp,2) - diag(fevd_gdp);

% Edges
for i=1:N
    for j=1:N
        if i~=j && fevd_gdp(i,j)>0.03
            lw_ = 1 + 6*fevd_gdp(i,j);
            plot([pos_x(i), pos_x(j)], [pos_y(i), pos_y(j)], ...
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

saveas(fig4, 'GlobalVAR_fevd.pdf');
print(fig4, 'GlobalVAR_fevd.png', '-dpng', '-r300');
fprintf('    Saved -> GlobalVAR_fevd.pdf / .png\n');

% ── Summary ───────────────────────────────────────────────────────────────
fprintf('\n%s\n', repmat('=',1,70));
fprintf('RESULTS SUMMARY\n');
fprintf('%s\n', repmat('=',1,70));
fprintf('  Max eigenvalue: %.3f\n', max_mod);
fprintf('\n  Trough GDP response to 1 s.d. US rate shock (decimal units):\n');
for i=1:N
    c   = countries{i};
    gdp = irf.(c)(:,1);
    [~,th] = min(gdp);
    fprintf('    %-12s: %+.2e at h=%d\n', strrep(c,'_',' '), gdp(th), th-1);
end
can_i=6; usa_i=1; eur_i=2; uk_i=3;
fprintf('\n  GDP FEVD at h=8:\n');
fprintf('    Own shocks: %d-%d%%\n', ...
        min(pct(sub2ind([N,N],1:N,1:N))), max(pct(sub2ind([N,N],1:N,1:N))));
fprintf('    USA -> Canada: %d%%\n', pct(can_i,usa_i));
fprintf('    EUR -> UK:     %d%%\n', pct(uk_i,eur_i));
fprintf('%s\n', repmat('=',1,70));

% =========================================================================
% LOCAL FUNCTIONS
% (MATLAB requires ALL local functions in a script to be defined here, at
%  the very end of the file, after the last line of script code.)
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
    % Pure VAR(p) for USA — no foreign variables (dominant unit)
    [Tc, k] = size(Y);
    Te = Tc - p;
    % Build X: [const, y_{t-1}, y_{t-2}, ..., y_{t-p}]
    X = ones(Te, 1);
    for lag = 1:p
        X = [X, Y(p-lag+1:Tc-lag, :)]; %#ok<AGROW>
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

function irf_out = girf_internal(F_mat, Sigma_e, shock_cid, shock_vid, H, K, countries, N)
    % GIRF (Pesaran & Shin 1998): all inputs passed explicitly.
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

function [irf_lo, irf_hi] = bootstrap_girf(F_mat, Sigma_e, models, G0_inv, ...
                                             countries, N, K, ...
                                             shock_cid, shock_vid, H, n_boot, seed)
    % Residual-resampling bootstrap confidence bands for the GIRF.
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
        catch
        end
    end
    irf_lo = struct(); irf_hi = struct();
    for i = 1:N
        c = countries{i};
        irf_lo.(c) = squeeze(pctile_base(boots.(c), 5));
        irf_hi.(c) = squeeze(pctile_base(boots.(c), 95));
    end
end

function out = pctile_base(A, p)
    % Percentile along dim 1 with linear interpolation (base MATLAB only;
    % avoids the Statistics Toolbox dependency of prctile).
    [n, H, K] = size(A);
    As = sort(A, 1);
    pos = 1 + (p/100) * (n - 1);    % matches numpy 'linear' / prctile default
    lo  = floor(pos); hi = ceil(pos);
    w   = pos - lo;
    out = (1-w) * reshape(As(lo,:,:), [1, H, K]) ...
        +     w * reshape(As(hi,:,:), [1, H, K]);
end
