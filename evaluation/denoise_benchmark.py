# denoise_benchmark.py
# 노이즈 제거(modules/denoise.py)의 synthetic 성능 평가.
#
# 평가 방법
#   1) "깨끗한" 입력 이미지 N장을 GT(ground truth)로 가정한다.
#      (images/input/train_*.jpg 일부를 사용 — 일반 촬영본이라 완벽히 노이즈
#       프리는 아니지만, 인공 노이즈를 강하게 주입하기 때문에 비교 기준으로 충분)
#
#   2) 각 GT 이미지에 정해진 강도로 인공 노이즈를 주입한다.
#        - gauss_sigma10 : 가우시안 노이즈, σ=10/255
#        - gauss_sigma25 : 가우시안 노이즈, σ=25/255
#        - sp_p02        : Salt & Pepper, p=2%
#        - sp_p05        : Salt & Pepper, p=5%
#
#   3) 각 노이즈 이미지에 5가지 denoise 메서드를 적용한다.
#        median / wiener / gaussian / bilateral / nlm
#      그리고 'auto' 모드도 함께 측정(어떤 메서드를 자동 선택했는지 확인용).
#
#   4) 복원된 이미지와 원본 GT 사이의 PSNR, SSIM을 측정한다.
#        - PSNR: 클수록 좋음 (단위 dB, 일반적으로 30+ 양호, 35+ 우수)
#        - SSIM: 0~1, 클수록 좋음 (구조 유사도)
#
#   5) (scenario × method) 평균 표를 콘솔에 출력하고
#      evaluation/denoise_benchmark_results.csv 로 저장한다.
#
# 사용법:
#   python evaluation/denoise_benchmark.py
#
# 옵션:
#   --n        평가에 사용할 이미지 수 (기본 10)
#   --plot     scenario별 막대그래프 figure 저장

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

from modules.denoise import denoise


IMAGE_DIR = "images/input"
OUT_CSV = "evaluation/denoise_benchmark_results.csv"
OUT_PNG = "evaluation/denoise_benchmark.png"

METHODS = ("median", "wiener", "gaussian", "bilateral", "nlm", "auto")


# =====================================================
# 노이즈 주입
# =====================================================
def add_gaussian_noise(image, sigma):
    """sigma: 0~255 스케일."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    out = image.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_pepper(image, p):
    """p: 0~1, salt와 pepper 합쳐서 p 비율 (각각 p/2)."""
    out = image.copy()
    mask = np.random.rand(*image.shape[:2])
    out[mask < p / 2] = 0           # pepper
    out[mask > 1 - p / 2] = 255     # salt
    return out


SCENARIOS = {
    "gauss_sigma10": lambda img: add_gaussian_noise(img, 10),
    "gauss_sigma25": lambda img: add_gaussian_noise(img, 25),
    "sp_p02":        lambda img: add_salt_pepper(img, 0.02),
    "sp_p05":        lambda img: add_salt_pepper(img, 0.05),
}


# =====================================================
# 지표 측정
# =====================================================
def measure(gt, pred):
    """PSNR, SSIM 측정. 컬러 이미지는 채널 평균."""
    p = psnr(gt, pred, data_range=255)
    if gt.ndim == 3:
        s = ssim(gt, pred, channel_axis=2, data_range=255)
    else:
        s = ssim(gt, pred, data_range=255)
    return float(p), float(s)


# =====================================================
# 벤치마크
# =====================================================
def run_benchmark(image_paths, methods=METHODS):
    rows = []
    for image_path in image_paths:
        gt = cv2.imread(image_path)
        if gt is None:
            print(f"  skip (load fail): {image_path}")
            continue

        # 동일한 노이즈 시드를 보장하려면 시나리오별 seed 고정
        for scen_name, noiser in SCENARIOS.items():
            np.random.seed(hash(image_path + scen_name) & 0xFFFFFFFF)
            noisy = noiser(gt)

            # 노이즈 자체의 품질 (baseline = 'noisy' 그대로)
            p_n, s_n = measure(gt, noisy)
            rows.append({
                "image": os.path.basename(image_path),
                "scenario": scen_name,
                "method": "noisy",
                "psnr": p_n,
                "ssim": s_n,
                "elapsed_s": 0.0,
                "auto_chosen": "",
            })

            for method in methods:
                t0 = time.perf_counter()
                restored, info = denoise(noisy, method=method)
                dt = time.perf_counter() - t0

                p, s = measure(gt, restored)
                rows.append({
                    "image": os.path.basename(image_path),
                    "scenario": scen_name,
                    "method": method,
                    "psnr": p,
                    "ssim": s,
                    "elapsed_s": dt,
                    "auto_chosen": info.get("auto_chosen", ""),
                })
    return rows


def summarize(rows):
    """(scenario, method) → mean(psnr), mean(ssim), mean(elapsed)"""
    bucket = {}
    for r in rows:
        key = (r["scenario"], r["method"])
        bucket.setdefault(key, []).append(r)

    summary = {}
    for key, items in bucket.items():
        psnr_mean = np.mean([x["psnr"] for x in items])
        ssim_mean = np.mean([x["ssim"] for x in items])
        time_mean = np.mean([x["elapsed_s"] for x in items])
        summary[key] = (psnr_mean, ssim_mean, time_mean)
    return summary


def print_summary(summary, methods):
    scenarios = sorted({k[0] for k in summary})
    label_order = ["noisy"] + list(methods)
    print("\n=== PSNR (dB, 클수록 좋음) ===")
    header = "scenario        " + "".join(f"{m:>11}" for m in label_order)
    print(header)
    for scen in scenarios:
        row = f"{scen:<15} "
        for m in label_order:
            v = summary.get((scen, m), (np.nan,) * 3)[0]
            row += f"{v:>11.2f}"
        print(row)

    print("\n=== SSIM (0~1, 클수록 좋음) ===")
    print(header)
    for scen in scenarios:
        row = f"{scen:<15} "
        for m in label_order:
            v = summary.get((scen, m), (np.nan,) * 3)[1]
            row += f"{v:>11.4f}"
        print(row)

    print("\n=== 평균 처리 시간 (초/이미지) ===")
    print(header)
    for scen in scenarios:
        row = f"{scen:<15} "
        for m in label_order:
            v = summary.get((scen, m), (np.nan,) * 3)[2]
            row += f"{v:>11.3f}"
        print(row)


def save_csv(rows, path):
    fieldnames = ["image", "scenario", "method",
                  "psnr", "ssim", "elapsed_s", "auto_chosen"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV 저장: {path}  ({len(rows)} rows)")


def save_plot(summary, methods, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 미설치 — plot 생략")
        return

    scenarios = sorted({k[0] for k in summary})
    label_order = ["noisy"] + list(methods)
    x = np.arange(len(scenarios))
    width = 0.11

    fig, (ax_p, ax_s) = plt.subplots(2, 1, figsize=(11, 8))
    for i, m in enumerate(label_order):
        psnrs = [summary.get((s, m), (0,))[0] for s in scenarios]
        ssims = [summary.get((s, m), (0, 0))[1] for s in scenarios]
        ax_p.bar(x + (i - len(label_order) / 2) * width, psnrs, width, label=m)
        ax_s.bar(x + (i - len(label_order) / 2) * width, ssims, width, label=m)

    ax_p.set_xticks(x); ax_p.set_xticklabels(scenarios)
    ax_p.set_ylabel("PSNR (dB)")
    ax_p.set_title("PSNR by scenario × method")
    ax_p.legend(ncol=4, fontsize=8)
    ax_p.grid(True, alpha=0.3)

    ax_s.set_xticks(x); ax_s.set_xticklabels(scenarios)
    ax_s.set_ylabel("SSIM")
    ax_s.set_title("SSIM by scenario × method")
    ax_s.legend(ncol=4, fontsize=8)
    ax_s.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"PNG 저장: {path}")


# =====================================================
# main
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10,
                        help="사용할 이미지 수 (기본 10)")
    parser.add_argument("--plot", action="store_true",
                        help="결과 그래프(PNG) 저장")
    args = parser.parse_args()

    files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])[:args.n]
    if not files:
        print(f"이미지 없음: {IMAGE_DIR}")
        return

    image_paths = [os.path.join(IMAGE_DIR, f) for f in files]
    print(f"이미지 {len(image_paths)}장, 시나리오 {len(SCENARIOS)}개, "
          f"메서드 {len(METHODS)}개 → 총 평가 조합 "
          f"{len(image_paths) * len(SCENARIOS) * (len(METHODS) + 1)}건")

    rows = run_benchmark(image_paths)
    summary = summarize(rows)
    print_summary(summary, METHODS)
    save_csv(rows, OUT_CSV)
    if args.plot:
        save_plot(summary, METHODS, OUT_PNG)


if __name__ == "__main__":
    main()
