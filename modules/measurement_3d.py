# measurement_3d.py
# 카메라 내부 파라미터(K) + 촬영 거리를 사용한 단안 측정.
# 기준 객체(예: 카드)가 없어도 핀홀 모델로 mm 크기를 계산할 수 있다.
#
# 핀홀 모델:
#   real_size_mm = pixel_size_px × distance_mm / focal_length_px

import os
import numpy as np

try:
    import scipy.io
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_camera_matrix(path="camparams.mat"):
    """MATLAB .mat 파일에서 K(3x3) 로드. 실패 시 None."""
    if not os.path.exists(path):
        return None
    if not HAS_SCIPY:
        return None
    try:
        mat = scipy.io.loadmat(path)
    except Exception:
        return None
    if "K" not in mat:
        return None
    return np.array(mat["K"], dtype=float)


def get_focal_lengths(K, scale=1.0):
    """K에서 (fx, fy) 추출.
    K는 원본 해상도 기준이므로 리사이즈 스케일을 곱해서 돌려준다.
    """
    if K is None:
        return None, None
    return K[0, 0] * scale, K[1, 1] * scale


def measure_from_distance(pixel_size, focal_length_px, distance_mm):
    """real_size = pixel × Z / f"""
    if focal_length_px is None or focal_length_px <= 0:
        return float("nan")
    return pixel_size * distance_mm / focal_length_px


def measure_object_3d(width_px, height_px, fx, fy, distance_mm):
    """객체의 (width_mm, height_mm) 반환"""
    return (
        measure_from_distance(width_px, fx, distance_mm),
        measure_from_distance(height_px, fy, distance_mm),
    )
