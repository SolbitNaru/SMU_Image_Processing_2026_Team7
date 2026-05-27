# main.py
# YOLO + OpenCV hybrid measurement pipeline

import cv2

from modules.preprocess import (
    load_image,
    preprocess
)

from modules.noise_filter import (
    remove_small_noise
)

from modules.edge_detection import (
    detect_edges,
    filter_edge_components
)

from modules.contour_detection import (
    find_contours,
    get_boxes
)

from modules.visualization import (
    draw_boxes
)

from modules.measurement_2d import (
    sort_dimensions,
    calculate_ratio,
    measure_object
)

from modules.yolo_detector import (
    detect_objects
)


# -------------------------------------------------
# 기준 객체 정보
# -------------------------------------------------
REFERENCE_OBJECTS = {

    "card": {

        "width_mm": 85.6,

        "ratio": 1.586
    },

    "battery_aa": {

        "width_mm": 50.5,

        "ratio": 50.5 / 14.5
    },

    "bill_10000": {

        "width_mm": 148.0,

        "ratio": 148.0 / 68.0
    },
}


# -------------------------------------------------
# YOLO + ratio 기반 기준 객체 선택
# -------------------------------------------------
def find_reference_box_with_yolo(

    boxes,
    detections,
    ratio_tolerance=0.25

):

    if not detections:

        return -1, None

    best_index = -1

    best_score = float("inf")

    best_name = None

    for index, (
        box,
        w,
        h,
        area,
        angle
    ) in enumerate(boxes):

        contour_ratio = (
            max(w, h) /
            min(w, h)
        )

        x, y, bw, bh = cv2.boundingRect(box)

        box_area = bw * bh

        for name, info in REFERENCE_OBJECTS.items():

            ratio_diff = abs(

                contour_ratio -
                info["ratio"]
            )

            if ratio_diff > ratio_tolerance:

                continue

            for detection in detections:

                x1, y1, x2, y2 = detection["bbox"]

                inter_x1 = max(x, x1)
                inter_y1 = max(y, y1)

                inter_x2 = min(x + bw, x2)
                inter_y2 = min(y + bh, y2)

                inter_w = max(
                    0,
                    inter_x2 - inter_x1
                )

                inter_h = max(
                    0,
                    inter_y2 - inter_y1
                )

                intersection = (
                    inter_w *
                    inter_h
                )

                detection_area = (
                    (x2 - x1) *
                    (y2 - y1)
                )

                union = (

                    box_area +
                    detection_area -
                    intersection
                )

                if union <= 0:

                    continue

                iou = intersection / union

                # ratio 우선
                # iou 보조
                score = ratio_diff - (iou * 0.5)

                if score < best_score:

                    best_score = score

                    best_index = index

                    best_name = name

    return best_index, best_name


# -------------------------------------------------
# ratio-only fallback
# -------------------------------------------------
def find_reference_box_by_ratio(

    boxes,
    tolerance=0.25

):

    best_index = -1

    best_score = float("inf")

    best_name = None

    for index, (
        box,
        w,
        h,
        area,
        angle
    ) in enumerate(boxes):

        ratio = (
            max(w, h) /
            min(w, h)
        )

        for name, info in REFERENCE_OBJECTS.items():

            score = abs(
                ratio - info["ratio"]
            )

            if (
                score < tolerance and
                score < best_score
            ):

                best_score = score

                best_index = index

                best_name = name

    return best_index, best_name


# -------------------------------------------------
# 이미지 로드
# -------------------------------------------------
image, scale = load_image(
    "images/input/test.jpg"
)

if image is None:

    raise ValueError(
        "Image loading failed"
    )


# -------------------------------------------------
# OpenCV pipeline
# -------------------------------------------------
adaptive = preprocess(
    image
)

filtered = remove_small_noise(

    adaptive,

    min_area=500
)

edges = detect_edges(
    filtered
)

filtered_edges = filter_edge_components(

    edges,

    min_area=1500,

    min_width=25,

    min_height=25,

    dilate_kernel=2,

    dilate_iterations=1,

    close_iterations=0
)

contours = find_contours(
    filtered_edges
)

boxes = get_boxes(

    contours,

    min_area=1500,

    min_width=25,

    min_height=25,

    min_solidity=0.7
)


# -------------------------------------------------
# YOLO detect
# -------------------------------------------------
detections = detect_objects(
    image
)

if not detections:

    print(
        "YOLO detections: none"
    )

else:

    print(detections)


labels = []


# -------------------------------------------------
# 측정
# -------------------------------------------------
if len(boxes) > 0:

    # 1차:
    # YOLO + ratio
    ref_index, ref_name = find_reference_box_with_yolo(

        boxes,
        detections
    )

    # 2차:
    # ratio-only fallback
    if ref_index == -1:

        print(
            "YOLO matching failed"
        )

        ref_index, ref_name = (
            find_reference_box_by_ratio(
                boxes
            )
        )

    # 3차:
    # 최종 fallback
    if ref_index == -1:

        print(
            "Fallback -> contour #0"
        )

        ref_index = 0

        ref_name = "card"

    ref_box = boxes[ref_index]

    ref_width_px, ref_height_px = (

        sort_dimensions(

            ref_box[1],
            ref_box[2]
        )
    )

    ref_info = REFERENCE_OBJECTS.get(

        ref_name,

        REFERENCE_OBJECTS["card"]
    )

    ratio = calculate_ratio(

        ref_info["width_mm"],

        ref_width_px
    )

    for index, (
        box,
        w,
        h,
        area,
        angle
    ) in enumerate(boxes):

        width_px, height_px = (

            sort_dimensions(w, h)
        )

        width_mm = measure_object(

            width_px,

            ratio
        )

        height_mm = measure_object(

            height_px,

            ratio
        )

        if index == ref_index:

            labels.append(

                f"REF({ref_name}) "
                f"{width_mm:.1f}x"
                f"{height_mm:.1f}mm"
            )

        else:

            labels.append(

                f"{width_mm:.1f}x"
                f"{height_mm:.1f}mm"
            )

else:

    print(
        "No objects detected"
    )


# -------------------------------------------------
# 결과 출력
# -------------------------------------------------
result = draw_boxes(

    image.copy(),

    boxes,

    labels=labels
)

cv2.imshow(
    "Adaptive",
    adaptive
)

cv2.imshow(
    "Filtered",
    filtered
)

cv2.imshow(
    "Edges",
    edges
)

cv2.imshow(
    "Filtered Edges",
    filtered_edges
)

cv2.imshow(
    "Result",
    result
)

cv2.waitKey(0)

cv2.destroyAllWindows()
