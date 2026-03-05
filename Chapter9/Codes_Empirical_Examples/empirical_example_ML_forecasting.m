% =========================================================================
% Empirical Application: Forecasting U.S. GDP Growth and Inflation
% Traditional Methods vs. Machine Learning
% =========================================================================
%
% Author   : Alessia Paccagnini
% Textbook : Macroeconometrics
%
% DATA  (FRED):
%   GDP.xlsx      — Real GDP (GDPC1), quarterly
%   GDPDEFL.xlsx  — GDP Deflator (GDPDEF), quarterly
%   FFR.xlsx      — Federal Funds Rate (FEDFUNDS), monthly -> quarterly avg
%
% SAMPLE: 1954:Q4 - 2025:Q3  (284 quarterly observations)
% TRANSFORMATIONS:
%   GDP Growth = 400 * diff(log(GDPC1))   [annualised %]
%   Inflation  = 400 * diff(log(GDPDEF))  [annualised %]
%   FedFunds   = level                    [%]
%
% METHODS:
%   Traditional : VAR(4), BVAR(4) with Minnesota prior (lambda1=0.2)
%   ML          : LASSO, Ridge, Elastic Net (5-fold CV), Random Forest
%
% EVALUATION:
%   Window    : expanding
%   OOS       : final 60 observations ~2010:Q1-2025:Q3
%   Horizons  : h = 1, 4 quarters ahead
%   Point     : RMSE, MAE
%   Density   : 90% PI coverage, width
%   Sig.      : Diebold-Mariano vs VAR
%
% REQUIREMENTS:
%   Statistics and Machine Learning Toolbox  (lasso, TreeBagger)
%   ridge() is in base MATLAB
% =========================================================================

clc; clear; close all;
rng(42);                    % reproducibility

% =========================================================================
% SECTION 1: LOAD AND PREPARE DATA
% =========================================================================
fprintf('=================================================================\n');
fprintf('Forecasting U.S. GDP Growth and Inflation\n');
fprintf('Traditional Methods vs. Machine Learning\n');
fprintf('=================================================================\n\n');

fprintf('[1/4] Loading data...\n');

% --- File paths (update if running locally) ---
gdp_path      = 'GDP.xlsx';
deflator_path = 'GDPDEFL.xlsx';
fedfunds_path = 'FFR.xlsx';

% --- Real GDP (quarterly) ---
gdp_raw        = readtable(gdp_path,  'VariableNamingRule','preserve');
gdp_raw        = gdp_raw(:, 1:2);
gdp_raw.Properties.VariableNames = {'date','GDPC1'};
gdp_raw.date   = datetime(gdp_raw.date);
gdp_raw.GDPC1  = double(gdp_raw.GDPC1);
gdp_raw.quarter = dateshift(gdp_raw.date, 'start', 'quarter');

% --- GDP Deflator (quarterly) ---
defl_raw       = readtable(deflator_path, 'VariableNamingRule','preserve');
defl_raw       = defl_raw(:, 1:2);
defl_raw.Properties.VariableNames = {'date','GDPDEF'};
defl_raw.date  = datetime(defl_raw.date);
defl_raw.GDPDEF = double(defl_raw.GDPDEF);
defl_raw.quarter = dateshift(defl_raw.date, 'start', 'quarter');

% --- Federal Funds Rate (monthly -> quarterly average) ---
ffr_raw        = readtable(fedfunds_path, 'VariableNamingRule','preserve');
ffr_raw        = ffr_raw(:, 1:2);
ffr_raw.Properties.VariableNames = {'date','FEDFUNDS'};
ffr_raw.date   = datetime(ffr_raw.date);
ffr_raw.FEDFUNDS = double(ffr_raw.FEDFUNDS);
ffr_raw.quarter  = dateshift(ffr_raw.date, 'start', 'quarter');

% Quarterly average of FFR
[ffr_q_dates, ~, ic] = unique(ffr_raw.quarter);
ffr_q_vals = accumarray(ic, ffr_raw.FEDFUNDS, [], @mean);

% --- Align on common quarterly dates ---
[common_dates, ia, ib] = intersect(gdp_raw.quarter, defl_raw.quarter);
GDPC1  = gdp_raw.GDPC1(ia);
GDPDEF = defl_raw.GDPDEF(ib);

[common_dates, ic2, id2] = intersect(common_dates, ffr_q_dates);
GDPC1  = GDPC1(ic2);
GDPDEF = GDPDEF(ic2);
FFR    = ffr_q_vals(id2);

% --- Transformations ---
GDP_growth = 400 * diff(log(GDPC1));   % annualised %
Inflation  = 400 * diff(log(GDPDEF));  % annualised %
FedFunds   = FFR(2:end);               % level %
dates      = common_dates(2:end);      % drop first obs (lost to diff)

% --- Restrict to sample: 1954:Q4 - 2025:Q3 ---
start_date = datetime(1954, 10, 1);    % 1954:Q4
end_date   = datetime(2025,  7, 1);    % 2025:Q3
mask       = dates >= start_date & dates <= end_date;
GDP_growth = GDP_growth(mask);
Inflation  = Inflation(mask);
FedFunds   = FedFunds(mask);
dates      = dates(mask);

T_total = length(GDP_growth);
fprintf('Sample: %s - %s  (%d observations)\n', ...
    datestr(dates(1),'yyyy:QQ'), datestr(dates(end),'yyyy:QQ'), T_total);
fprintf('Variables: GDP_growth [ann. %%], Inflation [ann. %%], FedFunds [%%]\n\n');

% Data matrix: columns = [GDP_growth, Inflation, FedFunds]
DATA = [GDP_growth, Inflation, FedFunds];
var_names  = {'GDP_growth','Inflation','FedFunds'};
n_vars     = 3;

% Summary statistics
fprintf('Summary Statistics:\n');
fprintf('%-15s %8s %8s %8s %8s %8s\n','Variable','Mean','Std','Min','Median','Max');
fprintf('%s\n', repmat('-',1,55));
for v = 1:n_vars
    fprintf('%-15s %8.2f %8.2f %8.2f %8.2f %8.2f\n', ...
        var_names{v}, mean(DATA(:,v)), std(DATA(:,v)), ...
        min(DATA(:,v)), median(DATA(:,v)), max(DATA(:,v)));
end
fprintf('\n');

% =========================================================================
% SECTION 2: CONFIGURATION
% =========================================================================
p         = 4;          % lags
horizons  = [1, 4];     % forecast horizons
test_size = 60;         % OOS observations (~2010:Q1-2025:Q3)
alpha     = 0.10;       % 90% prediction interval
z_alpha   = norminv(1 - alpha/2);   % 1.645

% Minnesota prior hyperparameters
lambda1   = 0.2;
lambda2   = 0.5;
lambda3   = 1.0;

methods_all = {'VAR','BVAR','LASSO','Ridge','ElasticNet','RandomForest'};
n_methods   = numel(methods_all);

% =========================================================================
% SECTION 3: HELPER FUNCTIONS (defined at end of file as local functions)
% =========================================================================
% prepare_forecast_data  — build y, X for direct h-step forecasting
% fit_var                — OLS regression with intercept
% fit_bvar               — Bayesian regression with Minnesota prior
% fit_lasso_cv           — LASSO with 5-fold CV
% fit_ridge_cv           — Ridge with 5-fold CV
% fit_elasticnet_cv      — Elastic Net with 5-fold CV
% fit_random_forest      — Random Forest (TreeBagger)
% standardise / unstandardise — z-score scaling for ML methods
% diebold_mariano        — DM test, HAC bandwidth = h-1

% =========================================================================
% SECTION 4: EXPANDING-WINDOW FORECASTING LOOP
% =========================================================================
fprintf('[2/4] Running forecasting comparison (expanding window, 60 OOS obs)...\n');

% Pre-allocate result containers
%   struct arrays indexed by horizon then method
for hi = 1:numel(horizons)
    h = horizons(hi);
    for target_idx = 1:2          % 1 = GDP growth, 2 = Inflation
        target_label = var_names{target_idx};

        fprintf('\n%s\n', repmat('=',1,60));
        fprintf('Forecasting %s at horizon h=%d\n', target_label, h);
        fprintf('%s\n', repmat('=',1,60));

        % Build y and X for direct h-step forecasting
        [y_full, X_full] = prepare_forecast_data(DATA, target_idx, h, p, n_vars);
        T = length(y_full);
        T0 = T - test_size;       % initial training size

        % Storage
        fc   = zeros(test_size, n_methods);   % point forecasts
        lb   = zeros(test_size, n_methods);   % lower 90% PI bound
        ub   = zeros(test_size, n_methods);   % upper 90% PI bound
        act  = zeros(test_size, 1);           % actuals

        for t = 1:test_size
            train_end = T0 + t - 1;           % last training index (1-based)

            y_tr  = y_full(1:train_end);
            X_tr  = X_full(1:train_end, :);
            y_te  = y_full(train_end + 1);
            X_te  = X_full(train_end + 1, :);
            act(t) = y_te;

            if mod(t-1, 20) == 0
                fprintf('  Forecast origin %d/%d\n', t, test_size);
            end

            % ---- VAR (OLS with intercept) ----
            [beta_v, sigma_v] = fit_var(y_tr, X_tr);
            x_aug = [1, X_te];
            fc(t,1) = x_aug * beta_v;
            lb(t,1) = fc(t,1) - z_alpha * sigma_v;
            ub(t,1) = fc(t,1) + z_alpha * sigma_v;

            % ---- BVAR (Minnesota prior) ----
            [beta_b, sigma_b, V_post] = fit_bvar(y_tr, X_tr, p, n_vars, ...
                                                  lambda1, lambda2, lambda3);
            x_aug_b = [1, X_te];
            fc(t,2)  = x_aug_b * beta_b;
            pred_var_b = x_aug_b * V_post * x_aug_b' + sigma_b^2;
            lb(t,2) = fc(t,2) - z_alpha * sqrt(pred_var_b);
            ub(t,2) = fc(t,2) + z_alpha * sqrt(pred_var_b);

            % ---- Standardise features for ML methods ----
            [X_tr_s, mu_X, sd_X] = standardise(X_tr);
            X_te_s = (X_te - mu_X) ./ sd_X;

            % ---- LASSO (5-fold CV) ----
            [beta_l, sigma_l] = fit_lasso_cv(y_tr, X_tr_s);
            fc(t,3) = [1, X_te_s] * beta_l;
            lb(t,3) = fc(t,3) - z_alpha * sigma_l;
            ub(t,3) = fc(t,3) + z_alpha * sigma_l;

            % ---- Ridge (5-fold CV) ----
            [beta_r, sigma_r] = fit_ridge_cv(y_tr, X_tr_s);
            fc(t,4) = [1, X_te_s] * beta_r;
            lb(t,4) = fc(t,4) - z_alpha * sigma_r;
            ub(t,4) = fc(t,4) + z_alpha * sigma_r;

            % ---- Elastic Net (5-fold CV) ----
            [beta_e, sigma_e] = fit_elasticnet_cv(y_tr, X_tr_s);
            fc(t,5) = [1, X_te_s] * beta_e;
            lb(t,5) = fc(t,5) - z_alpha * sigma_e;
            ub(t,5) = fc(t,5) + z_alpha * sigma_e;

            % ---- Random Forest (500 trees) ----
            [fc_rf, lb_rf, ub_rf] = fit_random_forest(y_tr, X_tr_s, ...
                                                       X_te_s, alpha);
            fc(t,6) = fc_rf;
            lb(t,6) = lb_rf;
            ub(t,6) = ub_rf;
        end

        % ---- Metrics ----
        RMSE_vec = sqrt(mean((act - fc).^2, 1));
        MAE_vec  = mean(abs(act - fc), 1);
        COV_vec  = mean((act >= lb) & (act <= ub), 1);
        WID_vec  = mean(ub - lb, 1);

        % Store for table printing and DM tests
        RES.(sprintf('%s_h%d', strrep(target_label,'_',''), h)).fc   = fc;
        RES.(sprintf('%s_h%d', strrep(target_label,'_',''), h)).act  = act;
        RES.(sprintf('%s_h%d', strrep(target_label,'_',''), h)).lb   = lb;
        RES.(sprintf('%s_h%d', strrep(target_label,'_',''), h)).ub   = ub;
        RES.(sprintf('%s_h%d', strrep(target_label,'_',''), h)).RMSE = RMSE_vec;
        RES.(sprintf('%s_h%d', strrep(target_label,'_',''), h)).MAE  = MAE_vec;
        RES.(sprintf('%s_h%d', strrep(target_label,'_',''), h)).COV  = COV_vec;
        RES.(sprintf('%s_h%d', strrep(target_label,'_',''), h)).WID  = WID_vec;

        % Print progress
        fprintf('\n  Results for h=%d:\n', h);
        fprintf('  %-15s %8s %8s %10s %10s\n', ...
            'Method','RMSE','MAE','Coverage','AvgWidth');
        fprintf('  %s\n', repmat('-',1,53));
        for m = 1:n_methods
            fprintf('  %-15s %8.3f %8.3f %9.1f%% %10.2f\n', ...
                methods_all{m}, RMSE_vec(m), MAE_vec(m), ...
                COV_vec(m)*100, WID_vec(m));
        end
    end
end

% =========================================================================
% SECTION 5: PRINT TABLES 9.6, 9.7, 9.8
% =========================================================================
fprintf('\n[3/4] Building results tables...\n\n');

target_keys   = {'GDPgrowth','Inflation'};
target_labels = {'GDP Growth','Inflation'};

% ---- Point Forecast Accuracy: RMSE and MAE ----
fprintf('%s\n', repmat('=',1,72));
fprintf('Point Forecast Accuracy: RMSE and MAE\n');
fprintf('%s\n', repmat('=',1,72));
fprintf('%-15s  %16s  %16s\n', '', 'h=1', 'h=4');
fprintf('%-15s  %8s %8s  %8s %8s\n', 'Method','RMSE','MAE','RMSE','MAE');
fprintf('%s\n', repmat('-',1,55));
for tl = 1:2
    fprintf('\n  %s\n', target_labels{tl});
    for m = 1:n_methods
        r1 = RES.(sprintf('%s_h1', target_keys{tl}));
        r4 = RES.(sprintf('%s_h4', target_keys{tl}));
        fprintf('  %-13s  %8.2f %8.2f  %8.2f %8.2f\n', ...
            methods_all{m}, r1.RMSE(m), r1.MAE(m), r4.RMSE(m), r4.MAE(m));
    end
end

% ---- Density Forecast Accuracy: Coverage and Width ----
fprintf('\n%s\n', repmat('=',1,72));
fprintf('Density Forecast Accuracy: Coverage and Interval Width\n');
fprintf('(Nominal coverage = 90%%)\n');
fprintf('%s\n', repmat('=',1,72));
fprintf('%-15s  %22s  %22s\n', '', 'h=1', 'h=4');
fprintf('%-15s  %8s %10s  %8s %10s\n','Method','Cov.','Width','Cov.','Width');
fprintf('%s\n', repmat('-',1,60));
for tl = 1:2
    fprintf('\n  %s\n', target_labels{tl});
    for m = 1:n_methods
        r1 = RES.(sprintf('%s_h1', target_keys{tl}));
        r4 = RES.(sprintf('%s_h4', target_keys{tl}));
        fprintf('  %-13s  %7.1f%%  %10.2f  %7.1f%%  %10.2f\n', ...
            methods_all{m}, r1.COV(m)*100, r1.WID(m), ...
            r4.COV(m)*100, r4.WID(m));
    end
end

% ---- Diebold-Mariano Tests: Comparison to VAR Benchmark ----
fprintf('\n%s\n', repmat('=',1,95));
fprintf('Diebold-Mariano Tests: Comparison to VAR Benchmark\n');
fprintf('(Negative DM -> method outperforms VAR; HAC bandwidth = h-1)\n');
fprintf('%s\n', repmat('=',1,95));
fprintf('%-15s  %30s  %30s\n','','GDP Growth','Inflation');
fprintf('%-15s  %9s %8s  %9s %8s  %9s %8s  %9s %8s\n', ...
    'Method','h=1 DM','p-val','h=4 DM','p-val', ...
           'h=1 DM','p-val','h=4 DM','p-val');
fprintf('%s\n', repmat('-',1,95));

non_var = {'BVAR','LASSO','Ridge','ElasticNet','RandomForest'};
non_var_idx = [2,3,4,5,6];   % column indices in fc matrix

for mi = 1:numel(non_var)
    m_idx = non_var_idx(mi);
    row_str = sprintf('%-15s', non_var{mi});
    for tl = 1:2
        for h = [1, 4]
            r = RES.(sprintf('%s_h%d', target_keys{tl}, h));
            e_var = r.act - r.fc(:,1);   % VAR errors
            e_m   = r.act - r.fc(:,m_idx);
            [dm, pv] = diebold_mariano(e_m, e_var, h);
            if     pv < 0.01;  sig = '***';
            elseif pv < 0.05;  sig = '** ';
            elseif pv < 0.10;  sig = '*  ';
            else;              sig = '   ';
            end
            row_str = [row_str, sprintf('  %+8.2f%s %7.3f', dm, sig, pv)]; %#ok
        end
    end
    fprintf('%s\n', row_str);
end
fprintf('\n  * p<0.10  ** p<0.05  *** p<0.01\n');

% =========================================================================
% SECTION 6: FIGURES
% =========================================================================
fprintf('\n[4/4] Generating figures...\n');

fig_colors = {[0.122 0.467 0.706], [1.000 0.498 0.055], [0.173 0.627 0.173], ...
              [0.839 0.153 0.157], [0.580 0.404 0.741], [0.549 0.337 0.294]};

for tl = 1:2
    for hi = 1:numel(horizons)
        h = horizons(hi);
        r = RES.(sprintf('%s_h%d', target_keys{tl}, h));

        % Reconstruct forecast-origin dates for x-axis
        [y_full_tmp, ~] = prepare_forecast_data(DATA, tl, h, p, n_vars);
        T_tmp = length(y_full_tmp);
        T0_tmp = T_tmp - test_size;
        % forecast origin dates = dates aligned to y_full index
        % y_full starts at index p+1 of DATA
        data_start_idx = p + 1;
        origin_indices = (data_start_idx + T0_tmp) : (data_start_idx + T0_tmp + test_size - 1);
        % cap to available dates
        origin_indices = min(origin_indices, length(dates));
        fc_dates = dates(origin_indices);

        % Convert datetime to datenum for fill() compatibility
        x_num = datenum(fc_dates);

        figure('Name', sprintf('%s h=%d', target_labels{tl}, h), ...
               'Position', [100 100 1400 900]);
        for m = 1:n_methods
            subplot(2,3,m);
            hold on;
            % fill() requires numeric x; use datenum then set XAxis to datetime
            fill([x_num; flipud(x_num)], ...
                 [r.lb(:,m); flipud(r.ub(:,m))], ...
                 fig_colors{m}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
            plot(x_num, r.act,     'k-',  'LineWidth', 1.5);
            plot(x_num, r.fc(:,m), '--',  'Color', fig_colors{m}, 'LineWidth', 1.2);
            % Format x-axis as dates
            ax = gca;
            ax.XTick = x_num(1 : round(test_size/5) : end);
            ax.XTickLabel = datestr(ax.XTick, 'yyyy');
            ax.XTickLabelRotation = 45;
            title(methods_all{m}, 'FontWeight','bold');
            xlabel('Date'); ylabel(target_labels{tl});
            legend('90% PI', 'Actual', 'Forecast', 'Location','best','FontSize',8);
            text(0.02, 0.97, sprintf('RMSE: %.3f\nCov: %.1f%%', ...
                r.RMSE(m), r.COV(m)*100), ...
                'Units','normalized','VerticalAlignment','top','FontSize',9, ...
                'BackgroundColor','white','EdgeColor','black');
            grid on; box on;
        end
        sgtitle(sprintf('%s Forecasts (h=%d)', target_labels{tl}, h), ...
                'FontSize',13,'FontWeight','bold');
        saveas(gcf, sprintf('figure9_%s_h%d.png', ...
               lower(strrep(target_keys{tl},'_','')), h));
    end
end

% Bar chart: RMSE and coverage across methods (one figure per target)
for tl = 1:2
    figure('Name', sprintf('Metrics %s', target_labels{tl}), ...
           'Position', [100 100 1200 480]);

    subplot(1,2,1);
    rmse_mat = zeros(n_methods, numel(horizons));
    for hi = 1:numel(horizons)
        r = RES.(sprintf('%s_h%d', target_keys{tl}, horizons(hi)));
        rmse_mat(:,hi) = r.RMSE';
    end
    bar(rmse_mat); set(gca,'XTickLabel', methods_all, 'XTickLabelRotation',30);
    legend(arrayfun(@(h) sprintf('h=%d',h), horizons, 'UniformOutput',false));
    ylabel('RMSE'); title('Point Forecast Accuracy (RMSE)','FontWeight','bold');
    grid on;

    subplot(1,2,2);
    cov_mat = zeros(n_methods, numel(horizons));
    for hi = 1:numel(horizons)
        r = RES.(sprintf('%s_h%d', target_keys{tl}, horizons(hi)));
        cov_mat(:,hi) = r.COV' * 100;
    end
    bar(cov_mat); set(gca,'XTickLabel', methods_all, 'XTickLabelRotation',30);
    yline(90, 'r--', 'LineWidth',1.5, 'Label','Nominal 90%');
    legend(arrayfun(@(h) sprintf('h=%d',h), horizons, 'UniformOutput',false));
    ylabel('Coverage (%)'); ylim([0 110]);
    title('Density Forecast Accuracy (Coverage)','FontWeight','bold');
    grid on;

    sgtitle(sprintf('Forecasting Comparison: %s', target_labels{tl}), ...
            'FontSize',13,'FontWeight','bold');
    saveas(gcf, sprintf('figure9_metrics_%s.png', ...
           lower(strrep(target_keys{tl},'_',''))));
end

fprintf('\n%s\n', repmat('=',1,70));
fprintf('Replication complete.\n');
fprintf('Outputs: figure9_*.png\n');
fprintf('%s\n', repmat('=',1,70));


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function [y, X] = prepare_forecast_data(DATA, target_idx, h, p, n_vars)
% Build direct h-step-ahead forecast dataset.
%
%  y(t) = DATA(t + h, target_idx)      (what we forecast)
%  X(t) = [DATA(t,1..n), ..., DATA(t-p+1,1..n)]  (lagged features)
%
% Valid rows: t = p, ..., T - h   (1-based, so indices p+1..T-h in MATLAB)

    T   = size(DATA, 1);
    % t runs from p+1 to T-h (MATLAB 1-based)
    t_start = p + 1;
    t_end   = T - h;
    N       = t_end - t_start + 1;

    y = DATA(t_start + h - 1 : t_end + h - 1, target_idx);
    % Note: y(i) corresponds to t = t_start + i - 1,
    %       target at t+h = t_start + i - 1 + h

    X = zeros(N, p * n_vars);
    for lag = 1:p
        col_start = (lag - 1) * n_vars + 1;
        col_end   = lag * n_vars;
        % lag-l values for each row i: DATA(t - lag + 1) = DATA(t_start+i-1-lag, :)
        X(:, col_start:col_end) = DATA(t_start - lag : t_end - lag, :);
    end
end


function [beta, sigma] = fit_var(y, X)
% OLS regression: y = [1, X] * beta + e
% Returns beta (k+1 x 1) and residual std sigma.
    X_aug = [ones(length(y),1), X];
    beta  = X_aug \ y;
    resid = y - X_aug * beta;
    k     = size(X_aug, 2);
    sigma = std(resid, 0);    % ddof = size(X_aug,2) equivalent: use 0 flag
    % Match Python ddof = X_aug.shape[1]:
    sigma = sqrt(sum(resid.^2) / (length(resid) - k));
end


function [beta, sigma, V_post] = fit_bvar(y, X, p, n_vars, lam1, lam2, lam3)
% Bayesian regression with Minnesota prior (conjugate normal-normal).
%
%  Prior on slope j:
%    lag  = ceil(j / n_vars)
%    var  = (lam1 / lag^lam3)^2   for own-variable lags
%           * lam2^2               for cross-variable lags
%  Intercept: diffuse prior variance = 100
%
%  Posterior mean: beta_post = V_post * (V_prior_inv * prior_mean + X'y)

    T   = length(y);
    k   = size(X, 2);        % = p * n_vars

    % Prior variances
    prior_var = ones(k, 1);
    for j = 1:k
        lag     = ceil(j / n_vars);
        var_idx = mod(j-1, n_vars);          % 0-based variable index
        prior_var(j) = (lam1 / (lag^lam3))^2;
        if var_idx > 0
            prior_var(j) = prior_var(j) * lam2^2;
        end
    end

    % Augmented system (intercept first)
    X_aug           = [ones(T,1), X];
    prior_var_full  = [100; prior_var];
    prior_mean_full = zeros(k+1, 1);

    V_prior_inv = diag(1 ./ prior_var_full);
    V_post_inv  = V_prior_inv + X_aug' * X_aug;
    V_post      = inv(V_post_inv);

    beta = V_post * (V_prior_inv * prior_mean_full + X_aug' * y);

    resid = y - X_aug * beta;
    sigma = std(resid, 1);   % ddof=1, matches Python
end


function [beta_full, sigma] = fit_lasso_cv(y, X_s)
% LASSO with 5-fold CV using MATLAB's lasso().
% Returns [intercept; coefficients] and residual sigma.
    [B, FitInfo] = lasso(X_s, y, 'CV', 5, 'NumLambda', 50, 'MaxIter', 10000);
    idx_min      = FitInfo.IndexMinMSE;
    coef         = B(:, idx_min);
    intercept    = FitInfo.Intercept(idx_min);
    beta_full    = [intercept; coef];
    resid        = y - (intercept + X_s * coef);
    sigma        = std(resid, 1);
end


function [beta_full, sigma] = fit_ridge_cv(y, X_s)
% Ridge with 5-fold CV.
% MATLAB's ridge() takes lambda directly; we choose via CV over a grid.
    lambdas  = logspace(-3, 3, 50);
    k        = size(X_s, 2);
    n        = length(y);
    cv_mse   = zeros(numel(lambdas), 1);

    % 5-fold CV (manual)
    fold_size = floor(n / 5);
    for li = 1:numel(lambdas)
        mse_folds = zeros(5,1);
        for fold = 1:5
            val_idx   = (fold-1)*fold_size+1 : min(fold*fold_size, n);
            train_idx = setdiff(1:n, val_idx);
            X_tr = X_s(train_idx,:); y_tr = y(train_idx);
            X_va = X_s(val_idx,:);   y_va = y(val_idx);
            % ridge() standardises internally; we pass already-standardised X
            b   = (X_tr'*X_tr + lambdas(li)*eye(k)) \ (X_tr'*y_tr);
            int = mean(y_tr) - mean(X_tr)*b;
            mse_folds(fold) = mean((y_va - int - X_va*b).^2);
        end
        cv_mse(li) = mean(mse_folds);
    end
    [~, best_li] = min(cv_mse);
    lam_best = lambdas(best_li);
    coef     = (X_s'*X_s + lam_best*eye(k)) \ (X_s'*y);
    intercept = mean(y) - mean(X_s)*coef;
    beta_full = [intercept; coef];
    resid     = y - (intercept + X_s * coef);
    sigma     = std(resid, 1);
end


function [beta_full, sigma] = fit_elasticnet_cv(y, X_s)
% Elastic Net with 5-fold CV over (lambda, l1_ratio) grid.
% Uses MATLAB's lasso() with Alpha parameter (= l1_ratio in sklearn).
    l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99];
    best_mse  = Inf;
    best_coef = zeros(size(X_s,2),1);
    best_int  = 0;

    for ai = 1:numel(l1_ratios)
        alph = l1_ratios(ai);
        try
            [B, FI] = lasso(X_s, y, 'Alpha', alph, 'CV', 5, ...
                            'NumLambda', 50, 'MaxIter', 10000);
            idx = FI.IndexMinMSE;
            mse_val = FI.MSE(idx);
            if mse_val < best_mse
                best_mse  = mse_val;
                best_coef = B(:, idx);
                best_int  = FI.Intercept(idx);
            end
        catch
            % skip if lasso fails for this alpha
        end
    end
    beta_full = [best_int; best_coef];
    resid     = y - (best_int + X_s * best_coef);
    sigma     = std(resid, 1);
end


function [fc, lb, ub] = fit_random_forest(y_tr, X_tr_s, X_te_s, alpha)
% Random Forest using TreeBagger (500 trees, MinLeafSize=5).
% Prediction interval: tree variance + residual variance (normal approx).
    n_trees  = 500;
    z_a      = norminv(1 - alpha/2);
    mdl      = TreeBagger(n_trees, X_tr_s, y_tr, ...
                          'Method',      'regression', ...
                          'MinLeafSize',  5, ...
                          'NumPredictorsToSample', 'all');
    % In-sample residuals
    y_fitted = predict(mdl, X_tr_s);
    resid    = y_tr - y_fitted;
    sigma_rf = std(resid, 1);

    % Per-tree predictions on test point
    tree_preds = zeros(n_trees, 1);
    for tr = 1:n_trees
        tree_preds(tr) = predict(mdl.Trees{tr}, X_te_s);
    end
    fc       = mean(tree_preds);
    tree_var = var(tree_preds, 1);
    total_std = sqrt(tree_var + sigma_rf^2);
    lb = fc - z_a * total_std;
    ub = fc + z_a * total_std;
end


function [X_s, mu, sd] = standardise(X)
% Z-score standardisation column-wise.
    mu  = mean(X, 1);
    sd  = std(X,  0, 1);
    sd(sd == 0) = 1;          % avoid division by zero
    X_s = (X - mu) ./ sd;
end


function [DM, p_value] = diebold_mariano(e1, e2, h)
% Diebold-Mariano (1995) test: H0 E[d_t]=0, d_t = e1_t^2 - e2_t^2.
% HAC variance: Newey-West, bandwidth = h-1.
% Negative DM -> model 1 outperforms model 2.
    d     = e1.^2 - e2.^2;
    d     = d(~isnan(d));
    T     = length(d);
    d_bar = mean(d);

    gamma0    = var(d, 1);       % biased variance (ddof=0), then scale
    gamma0    = var(d, 0);       % unbiased (ddof=1), matches Python ddof=1
    gamma_sum = 0;
    bw        = h - 1;           % bandwidth = h-1
    for k = 1:bw
        dc    = d - d_bar;
        gk    = mean(dc(k+1:end) .* dc(1:end-k));
        w     = 1 - k / (bw + 1);   % Bartlett kernel
        gamma_sum = gamma_sum + 2 * w * gk;
    end

    var_dbar = (gamma0 + gamma_sum) / T;
    if var_dbar > 0
        DM      = d_bar / sqrt(var_dbar);
        p_value = 2 * (1 - normcdf(abs(DM)));
    else
        DM      = NaN;
        p_value = NaN;
    end
end
