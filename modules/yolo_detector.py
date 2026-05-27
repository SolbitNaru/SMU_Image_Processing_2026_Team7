# modules/yolo_detector.py
# 다중 기준 객체 YOLO detection 모듈

from ultralytics import YOLO
import cv2
import os


MODEL_PATH = (
    "runs/detect/"
    "multi_reference_detector/"
    "weights/best.pt"
)

model = None


# -------------------------------------------------
# 모델 안전 로드
# -------------------------------------------------
if os.path.exists(MODEL_PATH):

    try:

        model = YOLO(MODEL_PATH)

        print(
            f"[YOLO] model loaded: {MODEL_PATH}"
        )

    except Exception as e:

        print(
            f"[YOLO] load failed: {e}"
        )

        model = None

else:

    print(
        f"[YOLO] model not found: {MODEL_PATH}"
    )


# -------------------------------------------------
# YOLO detect
# -------------------------------------------------
def detect_objects(image):

    # 모델이 없으면 빈 결과 반환
    if model is None:

        return []

    original_height, original_width = image.shape[:2]

    target_size = 1280

    scale = (
        target_size /
        max(
            original_width,
            original_height
        )
    )

    # 작은 이미지는 확대하지 않음
    scale = min(scale, 1.0)

    resized_width = int(
        original_width * scale
    )

    resized_height = int(
        original_height * scale
    )

    resized = cv2.resize(

        image,

        (
            resized_width,
            resized_height
        ),

        interpolation=cv2.INTER_AREA
    )

    results = model(

        resized,

        imgsz=1280,

        conf=0.4,

        verbose=False
    )

    detections = []

    for result in results:

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = box.xyxy[0]

            # 원본 좌표 복원
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)

            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            detections.append({

                "class_name": class_name,

                "confidence": confidence,

                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            })

    return detections
