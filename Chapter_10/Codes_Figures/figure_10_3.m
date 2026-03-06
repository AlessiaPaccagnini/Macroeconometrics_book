% =============================================================================
% figure_10_3.m
% Replication code for Figure 10.3
% US Macroeconomic Data with Regime Indicators (1960-2019)
%
% Author: Alessia Paccagnini
% Textbook: Macroeconometrics, Chapter 10
%
% Data file required (same folder): macro_data_1960_2019.csv
%   Columns: date, gdp_growth, inflation, ffr, nber_rec
% =============================================================================

clear; clc; close all;
cd(fileparts(mfilename('fullpath')));

%% ── 0. Load data ─────────────────────────────────────────────────────────────
opts = detectImportOptions('macro_data_1960_2019.csv');
opts = setvartype(opts, {'date'}, 'datetime');
df   = readtable('macro_data_1960_2019.csv', opts);
df.date.Format = 'yyyy-MM-dd';

dates    = df.date;
gdp      = df.gdp_growth;       % annualised quarterly GDP growth (%)
infl     = df.inflation;        % annualised quarterly inflation (%)
ffr      = df.ffr;              % federal funds rate (%)
rec      = df.nber_rec;         % NBER recession dummy (0/1)
T        = height(df);

%% ── 1. Regime indicators ─────────────────────────────────────────────────────
% Rolling 8-quarter std of GDP growth
win      = 8;
roll_std = NaN(T,1);
for t = win:T
    roll_std(t) = std(gdp(t-win+1:t), 'omitnan');
end
p75      = prctile(roll_std(~isnan(roll_std)), 75);
high_vol = double(roll_std > p75);
high_vol(isnan(roll_std)) = 0;

% Combined stress = NBER OR high-vol
stress  = max(rec, high_vol);

gm_date = datetime(1984,1,1);
ffr_med = median(ffr, 'omitnan');

%% ── 2. Colour scheme ─────────────────────────────────────────────────────────
C_REC  = [192  57  43]/255;   % red
C_HVOL = [ 36 113 163]/255;   % blue
C_STR  = [125  60 152]/255;   % purple
C_GM   = [ 30 132  73]/255;   % green
C_GDP  = [ 26  26  46]/255;   % dark
C_INF  = [183  96  10]/255;   % orange
C_FFR  = [ 21  67  96]/255;   % navy

%% ── 3. Helper: shade recession bands ────────────────────────────────────────
function shade_bands(ax, dates, ind, color, alpha_val)
    axes(ax)
    in_band = false;
    for t = 1:length(dates)
        if ind(t) && ~in_band
            s = dates(t); in_band = true;
        elseif ~ind(t) && in_band
            patch([s s dates(t) dates(t)], ...
                  [ax.YLim(1) ax.YLim(2) ax.YLim(2) ax.YLim(1)], ...
                  color, 'FaceAlpha', alpha_val, 'EdgeColor', 'none');
            in_band = false;
        end
    end
    if in_band
        patch([s s dates(end) dates(end)], ...
              [ax.YLim(1) ax.YLim(2) ax.YLim(2) ax.YLim(1)], ...
              color, 'FaceAlpha', alpha_val, 'EdgeColor', 'none');
    end
end

function draw_regime_bars(ax, dates, ind, y_center, h, color, alpha_val)
    axes(ax)
    in_band = false;
    for t = 1:length(dates)
        if ind(t) && ~in_band
            s = dates(t); in_band = true;
        elseif ~ind(t) && in_band
            patch([s s dates(t) dates(t)], ...
                  [y_center-h/2 y_center+h/2 y_center+h/2 y_center-h/2], ...
                  color, 'FaceAlpha', alpha_val, 'EdgeColor', 'none');
            in_band = false;
        end
    end
    if in_band
        patch([s s dates(end) dates(end)], ...
              [y_center-h/2 y_center+h/2 y_center+h/2 y_center-h/2], ...
              color, 'FaceAlpha', alpha_val, 'EdgeColor', 'none');
    end
end

%% ── 4. Build figure ──────────────────────────────────────────────────────────
fig = figure('Units','centimeters','Position',[2 2 33 28],'Color','w');

% Tile heights: 3 equal top + smaller bottom
ht = [1.8 1.8 1.8 1.1];
ht = ht / sum(ht);

gap = 0.01; lm = 0.09; rm = 0.02; bm = 0.07; tm = 0.05;
h_total = 1 - bm - tm;
ypos = cumsum([0 fliplr(ht)]); ypos = fliplr(ypos);

ax = gobjects(4,1);
for k = 1:4
    bot = bm + ypos(k+1)*h_total + gap/2;
    ht_k = ht(k)*h_total - gap;
    ax(k) = axes('Position',[lm bot (1-lm-rm) ht_k]);
end

%% Panel 1 — GDP Growth
axes(ax(1)); hold on
ylim([-11 11])
shade_bands(ax(1), dates, rec, C_REC, 0.20)
yline(0,'--k','LineWidth',0.6,'Alpha',0.35);
plot(dates, gdp, 'Color', C_GDP, 'LineWidth', 1.0)
xline(gm_date,'--','Color',C_GM,'LineWidth',1.5,'Alpha',0.85)
ylabel('GDP Growth (%)','FontSize',9)
title('US Macroeconomic Data with Regime Indicators (1960\20132019)', ...
      'FontSize',11,'FontWeight','bold')
ax(1).XTickLabel = []; grid on; grid minor; box off
patch(NaN,NaN,C_REC,'FaceAlpha',0.20,'EdgeColor','none');
legend('NBER Recession','Location','northeast','FontSize',8,'Box','off')
set(ax(1),'XColor','none')

%% Panel 2 — Inflation
axes(ax(2)); hold on
ylim([-0.5 10])
shade_bands(ax(2), dates, rec, C_REC, 0.20)
plot(dates, infl, 'Color', C_INF, 'LineWidth', 1.0)
xline(gm_date,'--','Color',C_GM,'LineWidth',1.5,'Alpha',0.85)
ylabel('Inflation (%)','FontSize',9)
ax(2).XTickLabel = []; grid on; box off
set(ax(2),'XColor','none')

%% Panel 3 — Fed Funds Rate
axes(ax(3)); hold on
ylim([-0.5 20.5])
shade_bands(ax(3), dates, rec, C_REC, 0.20)
plot(dates, ffr, 'Color', C_FFR, 'LineWidth', 1.0)
yline(ffr_med,':','Color',[0.5 0.5 0.5],'LineWidth',1.0,'Alpha',0.85, ...
      'Label',sprintf('Median (%.1f%%)',ffr_med),'LabelHorizontalAlignment','right', ...
      'FontSize',7)
xline(gm_date,'--','Color',C_GM,'LineWidth',1.5,'Alpha',0.85, ...
      'Label','Great Moderation','LabelOrientation','horizontal', ...
      'LabelVerticalAlignment','bottom','FontSize',7,'Color',C_GM)
ylabel('Interest Rate (%)','FontSize',9)
ax(3).XTickLabel = []; grid on; box off
set(ax(3),'XColor','none')

%% Panel 4 — Regime indicators
axes(ax(4)); hold on
ylim([-0.05 0.88])
y_r = 0.67; y_h = 0.37; y_s = 0.07; hw = 0.18;
draw_regime_bars(ax(4), dates, rec,      y_r, hw, C_REC,  0.75)
draw_regime_bars(ax(4), dates, high_vol, y_h, hw, C_HVOL, 0.75)
draw_regime_bars(ax(4), dates, stress,   y_s, hw, C_STR,  0.75)
xline(gm_date,'--','Color',C_GM,'LineWidth',1.5,'Alpha',0.85)
set(ax(4),'YTick',[y_s y_h y_r],'YTickLabel',{'Stress','High Vol','Recession'},'FontSize',8)
xlabel('Date','FontSize',9); grid on; box off

% Shared legend
ph = [patch(NaN,NaN,C_REC, 'FaceAlpha',0.75,'EdgeColor','none'), ...
      patch(NaN,NaN,C_HVOL,'FaceAlpha',0.75,'EdgeColor','none'), ...
      patch(NaN,NaN,C_STR, 'FaceAlpha',0.75,'EdgeColor','none'), ...
      plot(NaN,NaN,'--','Color',C_GM,'LineWidth',1.5)];
legend(ph,{'NBER Recession','High Uncertainty','Combined Stress','Great Moderation'}, ...
       'Location','southeast','FontSize',7.5,'NumColumns',2,'Box','on')

%% ── 5. Link x-axes and format ────────────────────────────────────────────────
linkaxes(ax,'x')
xlim([dates(1) dates(end)])

%% ── 6. Save ──────────────────────────────────────────────────────────────────
exportgraphics(fig,'Figure_10_3.pdf','ContentType','vector')
exportgraphics(fig,'Figure_10_3.png','Resolution',300)
fprintf('Saved Figure_10_3.pdf and Figure_10_3.png\n')
