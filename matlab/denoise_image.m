function [denoised, info] = denoise_image(img, method, params)
% denoise_image  이미지 노이즈 제거 함수.
%
% [denoised, info] = denoise_image(img, method, params)
%
% [입력]
%   img    - 입력 이미지. 문자열(경로) 또는 uint8/double 행렬.
%   method - 'median' | 'wiener' | 'gaussian' | 'bilateral' | 'nlm' | 'auto'
%            (기본값: 'auto')
%   params - 옵션 구조체 (전부 선택적)
%       .winSize       - 미디언/위너 윈도우 크기 (기본 3)
%       .sigma         - 가우시안/양방향 표준편차 (기본 1.5)
%       .nlmSearch     - NLM 탐색 윈도우 크기 (기본 21)
%       .nlmPatch      - NLM 패치 크기 (기본 5)
%       .salt_pepper_threshold - auto 모드에서 S&P 판정 임계값 (기본 0.005)
%
% [출력]
%   denoised - 결과 이미지 (입력과 동일 클래스)
%   info     - 처리 메타데이터 (사용된 method, 파라미터, 추정 노이즈)
%
% 예:
%   out = denoise_image('noisy.jpg', 'auto');
%   imshow(out);

    if nargin < 2 || isempty(method)
        method = 'auto';
    end
    if nargin < 3 || isempty(params)
        params = struct();
    end

    params = apply_defaults(params);

    % 1. 입력 정규화
    if ischar(img) || isstring(img)
        info.source_path = char(img);
        img = imread(img);
    else
        info.source_path = '';
    end
    orig_class = class(img);
    info.size  = size(img);
    info.dtype = orig_class;

    % 2. 자동 모드: 노이즈 종류 추정 후 적절한 method 선택
    if strcmpi(method, 'auto')
        sp_ratio = estimate_salt_pepper(img);
        info.salt_pepper_ratio = sp_ratio;
        if sp_ratio > params.salt_pepper_threshold
            method = 'median';
        else
            method = 'wiener';
        end
        info.auto_chosen = method;
    end
    info.method = method;
    info.params = params;

    % 3. method 적용
    switch lower(method)
        case 'median'
            denoised = apply_median(img, params.winSize);
        case 'wiener'
            denoised = apply_wiener(img, params.winSize);
        case 'gaussian'
            denoised = apply_gaussian(img, params.sigma);
        case 'bilateral'
            denoised = apply_bilateral(img, params.sigma);
        case 'nlm'
            denoised = apply_nlm(img, params.nlmSearch, params.nlmPatch);
        otherwise
            error('denoise_image:unknownMethod', ...
                'Unknown method: %s', method);
        end

    % 4. 출력 클래스 복원
    denoised = cast(denoised, orig_class);
end


% =====================================================
% 기본 파라미터
% =====================================================
function p = apply_defaults(p)
    if ~isfield(p, 'winSize'),                p.winSize                = 3;     end
    if ~isfield(p, 'sigma'),                  p.sigma                  = 1.5;   end
    if ~isfield(p, 'nlmSearch'),              p.nlmSearch              = 21;    end
    if ~isfield(p, 'nlmPatch'),               p.nlmPatch               = 5;     end
    if ~isfield(p, 'salt_pepper_threshold'),  p.salt_pepper_threshold  = 0.005; end
end


% =====================================================
% 미디언 필터 — salt & pepper 노이즈에 강함
% =====================================================
function out = apply_median(img, w)
    out = process_per_channel(img, @(c) medfilt2(c, [w w], 'symmetric'));
end


% =====================================================
% 적응 위너 필터 — 가우시안/균일 잡음에 적합
% =====================================================
function out = apply_wiener(img, w)
    out = process_per_channel(img, @(c) wiener2(c, [w w]));
end


% =====================================================
% 가우시안 블러 — 가장 단순한 평활화
% =====================================================
function out = apply_gaussian(img, sigma)
    out = imgaussfilt(img, sigma);
end


% =====================================================
% 양방향 필터 — 엣지 보존 평활화
% =====================================================
function out = apply_bilateral(img, sigma)
    img_d = im2double(img);
    % imbilatfilt: degreeOfSmoothing은 노이즈 분산을 추정해 자동 결정 가능
    out_d = imbilatfilt(img_d, 2 * sigma^2, sigma);
    out = out_d;
end


% =====================================================
% Non-Local Means — 강력하지만 느림 (R2018b+)
% =====================================================
function out = apply_nlm(img, searchWin, patchSize)
    if exist('imnlmfilt', 'file') ~= 2
        warning('denoise_image:nlmUnavailable', ...
            'imnlmfilt 미지원 MATLAB 버전. wiener로 대체.');
        out = apply_wiener(img, 3);
        return;
    end
    out = imnlmfilt(img, ...
        'SearchWindowSize', searchWin, ...
        'ComparisonWindowSize', patchSize);
end


% =====================================================
% 채널별 처리 헬퍼 (그레이/RGB 모두 지원)
% =====================================================
function out = process_per_channel(img, fn)
    if ndims(img) == 3
        out = zeros(size(img), 'like', img);
        for k = 1:size(img, 3)
            out(:, :, k) = fn(img(:, :, k));
        end
    else
        out = fn(img);
    end
end


% =====================================================
% Salt & Pepper 비율 추정
% 픽셀 값이 0 또는 max에 극단적으로 몰린 비율을 측정
% =====================================================
function ratio = estimate_salt_pepper(img)
    if ndims(img) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end
    if isinteger(gray)
        lo = (gray == 0);
        hi = (gray == intmax(class(gray)));
    else
        lo = (gray <= 0.01);
        hi = (gray >= 0.99);
    end
    ratio = (nnz(lo) + nnz(hi)) / numel(gray);
end
