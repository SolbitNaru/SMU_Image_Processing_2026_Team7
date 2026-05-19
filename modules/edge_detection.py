# edge_detection.py
# blur + canny 에서 canny 부분만 따로 모듈화

import cv2
import numpy as np


def detect_edges(
    image,
    low_threshold=50,
    high_threshold=150
):

    if len(image.shape) == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    edges = cv2.Canny(
        image,
        low_threshold,
        high_threshold
    )

    return edges


def filter_edge_components(
    edges,
    min_area=1500,
    max_area=None,
    min_width=20,
    min_height=20,
    max_width=None,
    max_height=None,
    min_perimeter=0,
    max_perimeter=None,
    dilate_kernel=2,
    dilate_iterations=1,
    close_iterations=0
):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (dilate_kernel, dilate_kernel)
    )

    connected = cv2.dilate(
        edges,
        kernel,
        iterations=dilate_iterations
    )

    if close_iterations > 0:

        connected = cv2.morphologyEx(
            connected,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=close_iterations
        )

    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    mask = np.zeros_like(edges)

    for contour in contours:

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(
            contour,
            True
        )

        x, y, w, h = cv2.boundingRect(contour)

        if (
            area < min_area
            or w < min_width
            or h < min_height
            or perimeter < min_perimeter
        ):
            continue

        if max_area is not None and area > max_area:
            continue

        if max_width is not None and w > max_width:
            continue

        if max_height is not None and h > max_height:
            continue

        if max_perimeter is not None and perimeter > max_perimeter:
            continue

        cv2.drawContours(
            mask,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

    return mask