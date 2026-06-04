# coin_classifier.py
# 한국 동전(10/50/100/500원) 분류기.
# 학습된 YOLOv8-cls 모델로 박스 영역 crop을 분류한다.

import os
import cv2

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

MODEL_PATH = "runs/classify/coin_classifier/weights/best.pt"

_model = None
_load_error = None


def is_available():
    return HAS_ULTRALYTICS and os.path.exists(MODEL_PATH)


def load_error():
    return _load_error


def _ensure_loaded():
    global _model, _load_error
    if _model is not None:
        return True
    if not HAS_ULTRALYTICS:
        _load_error = "ultralytics 미설치"
        return False
    if not os.path.exists(MODEL_PATH):
        _load_error = f"모델 없음: {MODEL_PATH}"
        return False
    try:
        _model = YOLO(MODEL_PATH)
        _load_error = None
        return True
    except Exception as e:
        _model = None
        _load_error = str(e)
        return False


def classify_crop(crop_bgr, top_k=1):
    """크롭된 BGR 이미지 → [(class_name, confidence), ...] (top_k개).
    실패 시 빈 리스트.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    if not _ensure_loaded():
        return []
    results = _model(crop_bgr, imgsz=224, verbose=False)
    out = []
    for r in results:
        probs = r.probs
        if probs is None:
            continue
        names = r.names
        # 상위 top_k
        topk = probs.top5[:top_k] if hasattr(probs, "top5") else [int(probs.top1)]
        for idx in topk:
            out.append((names[int(idx)], float(probs.data[int(idx)])))
    return out


def classify_box(image, box, top_k=1, padding=4):
    """원본 이미지와 (cv2 contour 박스 4점)로 ROI를 잘라 분류."""
    if image is None or box is None:
        return []
    x, y, w, h = cv2.boundingRect(box)
    H, W = image.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(W, x + w + padding)
    y1 = min(H, y + h + padding)
    if x1 <= x0 or y1 <= y0:
        return []
    crop = image[y0:y1, x0:x1]
    return classify_crop(crop, top_k=top_k)
