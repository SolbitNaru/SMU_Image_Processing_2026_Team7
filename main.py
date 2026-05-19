from modules.preprocess import load_image, preprocess
from modules.noise_filter import remove_small_noise
from modules.edge_detection import detect_edges, filter_edge_components
from modules.contour_detection import find_contours, get_boxes
from modules.visualization import draw_boxes
from modules.measurement_2d import (
    sort_dimensions,
    calculate_ratio,
    measure_object
)

import cv2


def find_reference_box(boxes, target_ratio=1.586, tolerance=0.25):

    best_index = -1
    best_score = float("inf")

    for index, (_, w, h, area, _) in enumerate(boxes):

        ratio = max(w, h) / min(w, h)

        score = abs(ratio - target_ratio)

        if score < tolerance and score < best_score:
            best_score = score
            best_index = index

    return best_index


image, scale = load_image("images/input/test.jpg")

if image is None:
    raise ValueError("Image loading failed")


adaptive = preprocess(image)


filtered = remove_small_noise(
    adaptive,
    min_area=500
)


edges = detect_edges(filtered)


filtered_edges = filter_edge_components(
    edges,
    min_area=1500,
    min_width=25,
    min_height=25,
    dilate_kernel=2,
    dilate_iterations=1,
    close_iterations=0
)


contours = find_contours(filtered_edges)


boxes = get_boxes(
    contours,
    min_area=1500,
    min_width=25,
    min_height=25,
    min_solidity=0.7
)


labels = []


if len(boxes) > 0:

    ref_index = find_reference_box(boxes)

    if ref_index != -1:

        ref_box = boxes[ref_index]

        ref_width_px, ref_height_px = sort_dimensions(
            ref_box[1],
            ref_box[2]
        )

        ratio = calculate_ratio(
            85.6,
            ref_width_px
        )

        for index, (box, w, h, area, angle) in enumerate(boxes):

            width_px, height_px = sort_dimensions(w, h)

            width_mm = measure_object(width_px, ratio)
            height_mm = measure_object(height_px, ratio)

            if index == ref_index:
                labels.append(
                    f"REF {width_mm:.1f}x{height_mm:.1f}mm"
                )
            else:
                labels.append(
                    f"{width_mm:.1f}x{height_mm:.1f}mm"
                )

    else:

        for i in range(len(boxes)):
            labels.append("OBJECT")



result = draw_boxes(
    image.copy(),
    boxes,
    labels=labels
)


cv2.imshow("Adaptive", adaptive)
cv2.imshow("Filtered", filtered)
cv2.imshow("Edges", edges)
cv2.imshow("Filtered Edges", filtered_edges)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()