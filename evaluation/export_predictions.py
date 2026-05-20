# export_predictions.py
# 측정 파이프라인을 여러 이미지에 돌리고, 결과를 CSV로 저장한다.
# 저장된 CSV에 GT 컬럼만 채워 넣으면 measurement_accuracy.m 에 그대로 입력 가능.
#
# 사용법:
#   1) ground_truth.csv 를 먼저 만들어 둔다.
#      컬럼: image, object_id, object_type, gt_width_mm, gt_height_mm
#   2) python evaluation/export_predictions.py
#   3) 생성된 results.csv 를 MATLAB에서 measurement_accuracy.m 으로 평가

import csv
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

from modules.preprocess import load_image, preprocess
from modules.noise_filter import remove_small_noise
from modules.edge_detection import detect_edges, filter_edge_components
from modules.contour_detection import find_contours, get_boxes
from modules.measurement_2d import (
    sort_dimensions,
    calculate_ratio,
    measure_object,
)


REFERENCE_WIDTH_MM = 85.6   # 신용카드 긴 변
GT_CSV   = "evaluation/ground_truth.csv"
OUT_CSV  = "evaluation/results.csv"
IMAGE_DIR = "images/input"


def find_reference_box(boxes, target_ratio=1.586, tolerance=0.25):
    best_index = -1
    best_score = float("inf")
    for index, (_, w, h, _, _) in enumerate(boxes):
        ratio = max(w, h) / min(w, h)
        score = abs(ratio - target_ratio)
        if score < tolerance and score < best_score:
            best_score = score
            best_index = index
    return best_index


def run_pipeline(image_path):
    """이미지 한 장 → [(width_mm, height_mm), ...] 리스트 반환"""
    image, _ = load_image(image_path)
    if image is None:
        return []

    adaptive = preprocess(image)
    filtered = remove_small_noise(adaptive, min_area=500)
    edges = detect_edges(filtered)
    filtered_edges = filter_edge_components(
        edges, min_area=1500, min_width=25, min_height=25,
        dilate_kernel=2, dilate_iterations=1, close_iterations=0,
    )
    contours = find_contours(filtered_edges)
    boxes = get_boxes(
        contours, min_area=1500, min_width=25, min_height=25,
        min_solidity=0.7,
    )

    if not boxes:
        return []

    ref_index = find_reference_box(boxes)
    if ref_index == -1:
        return []

    ref_w_px, _ = sort_dimensions(boxes[ref_index][1], boxes[ref_index][2])
    ratio = calculate_ratio(REFERENCE_WIDTH_MM, ref_w_px)

    results = []
    for box, w, h, _, _ in boxes:
        w_px, h_px = sort_dimensions(w, h)
        results.append((
            measure_object(w_px, ratio),
            measure_object(h_px, ratio),
        ))
    return results


def load_ground_truth(path):
    """{image: [(object_id, type, gt_w, gt_h), ...]}"""
    gt = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt.setdefault(row["image"], []).append((
                int(row["object_id"]),
                row["object_type"],
                float(row["gt_width_mm"]),
                float(row["gt_height_mm"]),
            ))
    return gt


def main():
    gt = load_ground_truth(GT_CSV)
    rows = []

    for image_name, gt_objects in gt.items():
        image_path = os.path.join(IMAGE_DIR, image_name)
        preds = run_pipeline(image_path)

        # GT의 object_id 순서대로 매칭 (단순 인덱스 매칭).
        # 실제로는 IoU 매칭이 더 정확하지만 우선은 동일 순서 가정.
        for object_id, otype, gt_w, gt_h in gt_objects:
            if object_id - 1 < len(preds):
                pred_w, pred_h = preds[object_id - 1]
            else:
                pred_w, pred_h = float("nan"), float("nan")

            rows.append({
                "image": image_name,
                "object_id": object_id,
                "object_type": otype,
                "gt_width_mm": gt_w,
                "gt_height_mm": gt_h,
                "pred_width_mm": round(pred_w, 2),
                "pred_height_mm": round(pred_h, 2),
            })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "object_id", "object_type",
            "gt_width_mm", "gt_height_mm",
            "pred_width_mm", "pred_height_mm",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {OUT_CSV}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
