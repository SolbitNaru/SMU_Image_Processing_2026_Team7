# reference_detector.py
import cv2
import numpy as np

from modules.preprocess import preprocess
from modules.edge_detection import detect_edges
from modules.contour_detection import find_contours


def detect_reference_contour(image, bbox):

    x1, y1, x2, y2 = bbox

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    processed = preprocess(crop)

    edges = detect_edges(processed)

    contours = find_contours(edges)

    if not contours:
        return None

    largest = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(largest)

    if area < 500:
        return None

    rect = cv2.minAreaRect(largest)

    box = cv2.boxPoints(rect)

    box = np.int32(box)

    box[:, 0] += x1
    box[:, 1] += y1

    return box