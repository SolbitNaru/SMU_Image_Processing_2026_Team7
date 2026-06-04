# denoise.py
# MATLAB denoise_image.m의 Python(OpenCV) 포트.
# 5가지 필터 + auto 모드(salt&pepper 비율로 median/wiener 자동 선택).

import cv2
import numpy as np

try:
    from scipy.signal import wiener as _scipy_wiener
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


METHODS = ("median", "wiener", "gaussian", "bilateral", "nlm", "auto", "off")


def estimate_salt_pepper(image):
    """이미지에서 salt & pepper 노이즈 비율을 추정한다."""
    if image is None:
        return 0.0
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if gray.dtype == np.uint8:
        lo = (gray == 0).sum()
        hi = (gray == 255).sum()
    else:
        lo = (gray <= 0.01).sum()
        hi = (gray >= 0.99).sum()
    return float(lo + hi) / gray.size


def _apply_median(image, ksize=3):
    k = max(3, ksize | 1)  # OpenCV는 홀수 ksize 요구
    return cv2.medianBlur(image, k)


def _apply_wiener(image, ksize=3):
    if not HAS_SCIPY:
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    # scipy wiener는 그레이만 처리. 채널별 적용.
    if image.ndim == 3:
        out = np.zeros_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            out[..., c] = _scipy_wiener(image[..., c].astype(np.float64), mysize=ksize)
        return np.clip(out, 0, 255).astype(image.dtype)
    return np.clip(
        _scipy_wiener(image.astype(np.float64), mysize=ksize), 0, 255
    ).astype(image.dtype)


def _apply_gaussian(image, sigma=1.5):
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)


def _apply_bilateral(image, sigma=1.5):
    # d=0 → sigmaSpace에서 자동 추정. sigmaColor는 노이즈 강도에 비례.
    return cv2.bilateralFilter(image, d=0,
                                sigmaColor=sigma * 25,
                                sigmaSpace=sigma * 5)


def _apply_nlm(image, h=10):
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(
            image, None, h=h, hColor=h,
            templateWindowSize=7, searchWindowSize=21,
        )
    return cv2.fastNlMeansDenoising(
        image, None, h=h,
        templateWindowSize=7, searchWindowSize=21,
    )


def denoise(image, method="auto", params=None):
    """이미지 노이즈 제거.

    method: 'median' | 'wiener' | 'gaussian' | 'bilateral' | 'nlm' | 'auto' | 'off'
    params: dict, 키는 winSize, sigma, h, salt_pepper_threshold
    반환: (denoised_image, info_dict)
    """
    if image is None:
        return None, {"method": "off", "skipped": True}

    p = {
        "winSize": 3,
        "sigma": 1.5,
        "h": 10,
        "salt_pepper_threshold": 0.005,
    }
    if params:
        p.update(params)

    info = {"params": dict(p)}

    if method == "off":
        info["method"] = "off"
        return image.copy(), info

    if method == "auto":
        sp = estimate_salt_pepper(image)
        info["salt_pepper_ratio"] = sp
        method = "median" if sp > p["salt_pepper_threshold"] else "wiener"
        info["auto_chosen"] = method

    info["method"] = method

    if method == "median":
        out = _apply_median(image, p["winSize"])
    elif method == "wiener":
        out = _apply_wiener(image, p["winSize"])
    elif method == "gaussian":
        out = _apply_gaussian(image, p["sigma"])
    elif method == "bilateral":
        out = _apply_bilateral(image, p["sigma"])
    elif method == "nlm":
        out = _apply_nlm(image, p["h"])
    else:
        raise ValueError(f"unknown denoise method: {method}")

    return out, info
