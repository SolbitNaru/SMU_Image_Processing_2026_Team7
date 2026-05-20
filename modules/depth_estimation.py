# depth_estimation.py
# MiDaS 단안 깊이 추정. 기준 객체 없이 객체별 거리(Z)를 추정한다.
#
# MiDaS는 disparity(역깊이)를 반환하므로 값이 클수록 카메라에 가깝다.
# 절대 스케일이 없으므로 anchor 거리 한 점을 받아 mm 거리로 환산한다.

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_model = None
_transform = None
_device = None
_load_error = None


def is_available():
    return HAS_TORCH


def load_error():
    return _load_error


def _ensure_loaded():
    global _model, _transform, _device, _load_error
    if _model is not None:
        return True
    if not HAS_TORCH:
        _load_error = "torch 미설치"
        return False
    try:
        import cv2  # noqa: F401  (사용처 확인용)
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
        )
        _model.to(_device).eval()
        transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        _transform = transforms.small_transform
        _load_error = None
        return True
    except Exception as e:
        _model = None
        _transform = None
        _load_error = str(e)
        return False


def estimate_depth(image_bgr):
    """이미지 BGR → 상대 disparity 맵 (float32, H×W). 실패 시 None.
    값이 클수록 카메라에 가깝다.
    """
    if not _ensure_loaded():
        return None
    import cv2
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    batch = _transform(rgb).to(_device)
    with torch.no_grad():
        prediction = _model(batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy().astype(np.float32)


def box_disparity(depth_map, box):
    """박스 내부 disparity 중앙값."""
    import cv2
    h, w = depth_map.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(box)], 1)
    vals = depth_map[mask > 0]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def disparity_to_distance_mm(disparity, anchor_disparity, anchor_distance_mm):
    """anchor 기준으로 disparity → mm 거리.
    Z = anchor_Z * (anchor_disparity / disparity)
    """
    if disparity is None or anchor_disparity is None:
        return float("nan")
    if not np.isfinite(disparity) or disparity <= 1e-6:
        return float("nan")
    return anchor_distance_mm * (anchor_disparity / disparity)


def disparity_to_visual(depth_map):
    """디버그용 시각화: disparity → 0~255 그레이스케일."""
    if depth_map is None:
        return None
    d = depth_map.astype(np.float32)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    if hi - lo < 1e-6:
        return np.zeros_like(d, dtype=np.uint8)
    norm = np.clip((d - lo) / (hi - lo), 0, 1)
    return (norm * 255).astype(np.uint8)
