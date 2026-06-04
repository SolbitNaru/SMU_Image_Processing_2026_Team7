% run_denoise.m
% 입력 이미지의 노이즈를 제거하고 여러 방법을 비교하는 데모 스크립트.
%
% 사용법:
%   1) input_path 를 실제 이미지 경로로 바꾼다.
%      비워두면 uigetfile 다이얼로그가 뜬다.
%   2) 실행하면 5가지 필터 결과를 한 화면에 비교하고
%      가장 추천되는 결과를 results/ 폴더에 저장한다.

clear; clc; close all;

% 1. 입력 이미지 경로
input_path = '';   % 예: '../images/input/train_001.jpg'

if isempty(input_path)
    [fname, fpath] = uigetfile( ...
        {'*.jpg;*.jpeg;*.png;*.bmp;*.tif', '이미지 파일'}, ...
        '노이즈 제거할 이미지 선택');
    if isequal(fname, 0)
        disp('취소됨'); return;
    end
    input_path = fullfile(fpath, fname);
end

img = imread(input_path);
fprintf('입력: %s  (%dx%dx%d, %s)\n', input_path, ...
    size(img,1), size(img,2), size(img,3), class(img));

% 2. 5가지 방법으로 노이즈 제거
methods = {'median', 'wiener', 'gaussian', 'bilateral', 'auto'};
results = cell(1, numel(methods));
infos   = cell(1, numel(methods));

for i = 1:numel(methods)
    fprintf('  - %s ... ', methods{i});
    tic;
    [results{i}, infos{i}] = denoise_image(img, methods{i});
    fprintf('%.2fs\n', toc);
end

% 3. 결과 품질 지표 (원본 대비 PSNR/SSIM — 노이즈 제거량의 척도)
fprintf('\n[원본 대비 변화량 — 클수록 더 많이 평활화됨]\n');
for i = 1:numel(methods)
    p = psnr(results{i}, img);
    s = ssim(results{i}, img);
    fprintf('  %-10s  PSNR=%.2f dB  SSIM=%.4f\n', methods{i}, p, s);
end

% 4. 비교 figure
figure('Name', '노이즈 제거 결과 비교', ...
    'Position', [100, 100, 1400, 700]);
subplot(2, 3, 1); imshow(img);  title('Original (입력)');
for i = 1:numel(methods)
    subplot(2, 3, i + 1);
    imshow(results{i});
    if isfield(infos{i}, 'auto_chosen')
        ttl = sprintf('%s (auto→%s)', methods{i}, infos{i}.auto_chosen);
    else
        ttl = methods{i};
    end
    title(ttl, 'Interpreter', 'none');
end

% 5. 'auto' 결과를 저장
out_dir = fullfile(fileparts(input_path), 'denoised');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
[~, name, ext] = fileparts(input_path);
out_path = fullfile(out_dir, [name, '_denoised', ext]);

auto_idx = find(strcmp(methods, 'auto'), 1);
imwrite(results{auto_idx}, out_path);
fprintf('\n저장됨: %s  (auto → %s)\n', out_path, infos{auto_idx}.auto_chosen);
