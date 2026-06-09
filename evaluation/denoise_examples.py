# denoise_examples.py
# 보고서용 노이즈 제거 예시 이미지 생성.
#
# 출력 3종:
#   1) noise_types.png       - 원본 + 4가지 노이즈 비교 (한 장)
#   2) methods_<scen>.png    - 시나리오별 모든 메서드 비교 (4장)
#   3) overview_grid.png     - 4×6 전체 오버뷰 (시나리오 × 메서드)
#
# 사용법:
#   python evaluation/denoise_examples.py
#   python evaluation/denoise_examples.py --image images/input/train_005.jpg
#   python evaluation/denoise_examples.py --crop 400 400 800 800  # 디테일 확대용

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

from modules.denoise import denoise


DEFAULT_IMAGE = "images/input/train_001.jpg"
OUT_DIR = "evaluation/denoise_examples"

METHODS = ("median", "wiener", "gaussian", "bilateral", "nlm")
SCENARIOS = ["gauss_sigma10", "gauss_sigma25", "sp_p02", "sp_p05"]


# =====================================================
# 노이즈 주입
# =====================================================
def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper(image, p):
    out = image.copy()
    mask = np.random.rand(*image.shape[:2])
    out[mask < p / 2] = 0
    out[mask > 1 - p / 2] = 255
    return out


def make_noisy(image, scenario):
    np.random.seed(42)  # 재현성
    if scenario == "gauss_sigma10":
        return add_gaussian_noise(image, 10)
    if scenario == "gauss_sigma25":
        return add_gaussian_noise(image, 25)
    if scenario == "sp_p02":
        return add_salt_pepper(image, 0.02)
    if scenario == "sp_p05":
        return add_salt_pepper(image, 0.05)
    raise ValueError(scenario)


SCENARIO_LABELS = {
    "gauss_sigma10": "Gaussian (σ=10)",
    "gauss_sigma25": "Gaussian (σ=25)",
    "sp_p02":        "Salt & Pepper (p=2%)",
    "sp_p05":        "Salt & Pepper (p=5%)",
}


# =====================================================
# 지표
# =====================================================
def metrics(gt, pred):
    p = psnr(gt, pred, data_range=255)
    if gt.ndim == 3:
        s = ssim(gt, pred, channel_axis=2, data_range=255)
    else:
        s = ssim(gt, pred, data_range=255)
    return p, s


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# =====================================================
# Figure 1: 노이즈 종류 비교 (원본 + 4 noise)
# =====================================================
def figure_noise_types(gt, out_path):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(to_rgb(gt)); axes[0].set_title("Original (GT)")
    axes[0].axis("off")
    for i, scen in enumerate(SCENARIOS):
        noisy = make_noisy(gt, scen)
        p, s = metrics(gt, noisy)
        axes[i + 1].imshow(to_rgb(noisy))
        axes[i + 1].set_title(
            f"{SCENARIO_LABELS[scen]}\nPSNR={p:.2f}dB  SSIM={s:.3f}"
        )
        axes[i + 1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


# =====================================================
# Figure 2: 시나리오별 메서드 비교 (각 시나리오마다 1장)
# =====================================================
def figure_methods_per_scenario(gt, scenario, out_path):
    noisy = make_noisy(gt, scenario)
    p_n, s_n = metrics(gt, noisy)

    panels = [("Original (GT)", gt, None, None),
              (f"Noisy\n{SCENARIO_LABELS[scenario]}", noisy, p_n, s_n)]
    for m in METHODS:
        out, _ = denoise(noisy, m)
        p, s = metrics(gt, out)
        panels.append((m, out, p, s))

    n = len(panels)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = np.array(axes).reshape(-1)

    for ax, (label, img, p, s) in zip(axes, panels):
        ax.imshow(to_rgb(img))
        if p is not None:
            ax.set_title(f"{label}\nPSNR={p:.2f}dB  SSIM={s:.3f}", fontsize=11)
        else:
            ax.set_title(label, fontsize=11)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Noise removal — {SCENARIO_LABELS[scenario]}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


# =====================================================
# Figure 3: 전체 오버뷰 (시나리오 행 × 메서드 열)
# =====================================================
def figure_overview(gt, out_path):
    cols = ["Noisy"] + list(METHODS)
    fig, axes = plt.subplots(
        len(SCENARIOS), len(cols),
        figsize=(len(cols) * 3, len(SCENARIOS) * 3),
    )

    for i, scen in enumerate(SCENARIOS):
        noisy = make_noisy(gt, scen)
        p_n, s_n = metrics(gt, noisy)
        axes[i, 0].imshow(to_rgb(noisy))
        axes[i, 0].set_title(f"{p_n:.1f}/{s_n:.2f}", fontsize=9)
        axes[i, 0].set_ylabel(
            SCENARIO_LABELS[scen], fontsize=11, rotation=90, labelpad=8)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

        for j, m in enumerate(METHODS):
            out, _ = denoise(noisy, m)
            p, s = metrics(gt, out)
            ax = axes[i, j + 1]
            ax.imshow(to_rgb(out))
            ax.set_title(f"{p:.1f}/{s:.2f}", fontsize=9)
            ax.axis("off")

    # 첫 행에 column header
    for j, c in enumerate(cols):
        axes[0, j].set_title(
            f"{c}\n" + axes[0, j].get_title(), fontsize=10)

    fig.suptitle(
        "Denoise overview (cell title: PSNR/SSIM)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


# =====================================================
# main
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE,
                        help="입력 이미지 (기본: images/input/train_001.jpg)")
    parser.add_argument("--out_dir", default=OUT_DIR,
                        help="출력 폴더")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                        help="이미지 일부만 사용 (디테일 확대용)")
    parser.add_argument("--max_width", type=int, default=900,
                        help="가로 폭 최대 (기본 900px, NLM 속도 때문에 권장)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        print(f"이미지 로드 실패: {args.image}")
        return

    if args.crop:
        x0, y0, x1, y1 = args.crop
        img = img[y0:y1, x0:x1]

    if img.shape[1] > args.max_width:
        scale = args.max_width / img.shape[1]
        new_w = args.max_width
        new_h = int(img.shape[0] * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    print(f"입력: {args.image}  크기: {img.shape[1]}x{img.shape[0]}")
    print(f"출력 폴더: {args.out_dir}\n")

    # 1. 노이즈 종류 비교
    figure_noise_types(
        img, os.path.join(args.out_dir, "noise_types.png"))

    # 2. 시나리오별 메서드 비교
    for scen in SCENARIOS:
        figure_methods_per_scenario(
            img, scen,
            os.path.join(args.out_dir, f"methods_{scen}.png"))

    # 3. 전체 오버뷰
    figure_overview(
        img, os.path.join(args.out_dir, "overview_grid.png"))

    print(f"\n완료. 총 6개 PNG가 {args.out_dir}/ 에 저장됨.")


if __name__ == "__main__":
    main()
