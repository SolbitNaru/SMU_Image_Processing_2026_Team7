# yolo_detector.py
# 카드 전용 YOLO detection 모듈

from ultralytics import YOLO

import cv2


model = YOLO(
    "runs/detect/card_detector/weights/best.pt"
)


def detect_objects(image):

    original_height, original_width = image.shape[:2]

    # YOLO 전용 입력 크기
    target_size = 1280

    scale = (
        target_size /
        max(original_width, original_height)
    )

    # 너무 작은 이미지는 확대하지 않음
    scale = min(scale, 1.0)

    resized_width = int(
        original_width * scale
    )

    resized_height = int(
        original_height * scale
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
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

            detections.append({

                "class_name": "card",

                "confidence": confidence,

                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            })

    return detections