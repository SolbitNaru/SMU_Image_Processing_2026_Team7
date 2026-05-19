# contour_detection.py
# Contour 추출 모듈화, 면적 기준으로 필터링, bounding box 계산

import cv2
import numpy as np


def find_contours(edges):

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def normalize_rect(rect):

    (x, y), (w, h), angle = rect

    if w < h:
        w, h = h, w
        angle += 90

    if angle > 90:
        angle -= 180

    if angle <= -90:
        angle += 180

    return (x, y), (w, h), angle


def get_boxes(
    contours,
    min_area=1500,
    min_width=20,
    min_height=20,
    max_area=None,
    max_width=None,
    max_height=None,
    min_solidity=0.7
):

    boxes = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        if max_area is not None and area > max_area:
            continue

        rect = cv2.minAreaRect(contour)

        rect = normalize_rect(rect)

        (x, y), (w, h), angle = rect

        if w < min_width or h < min_height:
            continue

        if max_width is not None and w > max_width:
            continue

        if max_height is not None and h > max_height:
            continue

        rect_area = w * h

        solidity = area / rect_area if rect_area > 0 else 0

        if solidity < min_solidity:
            continue

        box = cv2.boxPoints(rect)

        box = np.int32(box)

        boxes.append(
            (box, w, h, area, angle)
        )

    return boxes