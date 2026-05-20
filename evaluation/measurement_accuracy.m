% measurement_accuracy.m
% 2D 측정 파이프라인의 mm 정확도 평가 스크립트
%
% [입력] results.csv  컬럼:
%   image           - 이미지 파일명
%   object_id       - 같은 이미지 내 객체 번호
%   object_type     - card / coin / book / a4 / ...
%   gt_width_mm     - 실제 너비 (mm)
%   gt_height_mm    - 실제 높이 (mm)
%   pred_width_mm   - 파이프라인 예측 너비 (mm)
%   pred_height_mm  - 파이프라인 예측 높이 (mm)
%
% [출력]
%   콘솔 - 전체/객체별 MAE, RMSE, Bias, MaxAbs, MAPE
%   figure - 산점도, 오차 히스토그램, 객체별 boxplot, Bland-Altman
%   results_with_errors.csv - 오차 컬럼 추가된 결과

clear; clc; close all;

% 1. 데이터 로드
csv_path = 'sample_results.csv';   % 실제 결과 파일로 교체
T = readtable(csv_path);
fprintf('총 %d개 객체 평가 (%d개 이미지)\n', ...
    height(T), numel(unique(T.image)));

% 2. 오차 컬럼 추가
T.width_err   = T.pred_width_mm  - T.gt_width_mm;
T.height_err  = T.pred_height_mm - T.gt_height_mm;
T.width_pct   = 100 * T.width_err  ./ T.gt_width_mm;
T.height_pct  = 100 * T.height_err ./ T.gt_height_mm;

gt_diag   = sqrt(T.gt_width_mm.^2   + T.gt_height_mm.^2);
pred_diag = sqrt(T.pred_width_mm.^2 + T.pred_height_mm.^2);
T.diag_err = pred_diag - gt_diag;

% 3. 전체 지표
fprintf('\n=== 전체 지표 ===\n');
print_metrics('Width ', T.width_err,  T.width_pct);
print_metrics('Height', T.height_err, T.height_pct);

% 4. 객체 종류별 지표
fprintf('\n=== 객체 종류별 ===\n');
classes = unique(T.object_type);
for i = 1:numel(classes)
    mask = strcmp(T.object_type, classes{i});
    fprintf('[%s] n=%d\n', classes{i}, sum(mask));
    print_metrics('  Width ', T.width_err(mask),  T.width_pct(mask));
    print_metrics('  Height', T.height_err(mask), T.height_pct(mask));
end

% 5. 시각화

% (a) 예측 vs GT 산점도 (정확하면 y=x 위에 모여야 함)
figure('Name','Pred vs GT', 'Position',[100 100 900 420]);
subplot(1,2,1); plot_pred_vs_gt(T.gt_width_mm,  T.pred_width_mm,  'Width');
subplot(1,2,2); plot_pred_vs_gt(T.gt_height_mm, T.pred_height_mm, 'Height');

% (b) 오차 분포 (편향(bias)이 있는지, 정규분포인지 확인)
figure('Name','Error distribution', 'Position',[150 150 900 500]);
subplot(2,1,1);
histogram(T.width_err, 20); grid on;
xline(0, 'r--');
xlabel('Width error (mm)');
title(sprintf('Width  mean=%+.2f  std=%.2f', ...
    mean(T.width_err), std(T.width_err)));
subplot(2,1,2);
histogram(T.height_err, 20); grid on;
xline(0, 'r--');
xlabel('Height error (mm)');
title(sprintf('Height mean=%+.2f  std=%.2f', ...
    mean(T.height_err), std(T.height_err)));

% (c) 객체 종류별 오차 boxplot
figure('Name','Error by class', 'Position',[200 200 900 420]);
subplot(1,2,1);
boxplot(T.width_err, T.object_type); grid on;
yline(0, 'r--');
ylabel('Width error (mm)'); title('Width error by class');
subplot(1,2,2);
boxplot(T.height_err, T.object_type); grid on;
yline(0, 'r--');
ylabel('Height error (mm)'); title('Height error by class');

% (d) Bland-Altman plot (대각선 기준 일치도)
figure('Name','Bland-Altman', 'Position',[250 250 700 500]);
mean_diag = (gt_diag + pred_diag) / 2;
m  = mean(T.diag_err);
sd = std(T.diag_err);
scatter(mean_diag, T.diag_err, 40, 'filled'); hold on; grid on;
yline(m,            'r-',  sprintf('mean=%+.2fmm', m));
yline(m + 1.96*sd,  'r--', sprintf('+1.96\\sigma=%+.2fmm', m + 1.96*sd));
yline(m - 1.96*sd,  'r--', sprintf('-1.96\\sigma=%+.2fmm', m - 1.96*sd));
xlabel('Mean diagonal (mm)'); ylabel('Pred - GT (mm)');
title('Bland-Altman (diagonal)');

% 6. 결과 저장
writetable(T, 'results_with_errors.csv');
fprintf('\n→ results_with_errors.csv 저장 완료\n');


% --- 로컬 함수 ---
function print_metrics(label, err, pct)
    fprintf('%s  MAE=%6.2fmm  RMSE=%6.2fmm  Bias=%+6.2fmm  MaxAbs=%6.2fmm  MAPE=%5.2f%%\n', ...
        label, ...
        mean(abs(err)), ...
        sqrt(mean(err.^2)), ...
        mean(err), ...
        max(abs(err)), ...
        mean(abs(pct)));
end

function plot_pred_vs_gt(gt, pred, name)
    scatter(gt, pred, 40, 'filled'); hold on; grid on;
    lo = min([gt; pred]) * 0.95;
    hi = max([gt; pred]) * 1.05;
    plot([lo hi], [lo hi], 'r--');
    xlabel(sprintf('GT %s (mm)', name));
    ylabel(sprintf('Pred %s (mm)', name));
    title(name);
    axis equal; xlim([lo hi]); ylim([lo hi]);
end
