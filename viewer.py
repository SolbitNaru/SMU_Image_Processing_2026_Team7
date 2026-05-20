# viewer.py
# Tkinter 기반 파이프라인 단계 뷰어
# 한 창에서 사이드바 버튼으로 각 단계(Original ~ Result)를 전환해서 본다.

import os
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk

from modules.preprocess import load_image, preprocess
from modules.noise_filter import remove_small_noise
from modules.edge_detection import detect_edges, filter_edge_components
from modules.contour_detection import find_contours, get_boxes
from modules.visualization import draw_boxes
from modules.measurement_2d import (
    sort_dimensions,
    calculate_ratio,
    measure_object,
)


REFERENCE_WIDTH_MM = 85.6
CARD_RATIO = 1.586
RATIO_TOLERANCE = 0.25

STAGES = [
    "Original",
    "Adaptive",
    "Filtered",
    "Edges",
    "Filtered Edges",
    "Result",
]


def find_reference_box(boxes, target_ratio=CARD_RATIO, tolerance=RATIO_TOLERANCE):
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
    """이미지 경로 → 단계별 이미지 dict + 요약문자열"""
    image, _ = load_image(image_path)
    if image is None:
        return None

    adaptive = preprocess(image)
    filtered = remove_small_noise(adaptive, min_area=500)
    edges = detect_edges(filtered)
    filtered_edges = filter_edge_components(
        edges,
        min_area=1500, min_width=25, min_height=25,
        dilate_kernel=2, dilate_iterations=1, close_iterations=0,
    )
    contours = find_contours(filtered_edges)
    boxes = get_boxes(
        contours,
        min_area=1500, min_width=25, min_height=25,
        min_solidity=0.7,
    )

    labels = []
    summary = "객체 없음"
    if boxes:
        ref_index = find_reference_box(boxes)
        if ref_index != -1:
            ref_w_px, _ = sort_dimensions(
                boxes[ref_index][1], boxes[ref_index][2]
            )
            ratio = calculate_ratio(REFERENCE_WIDTH_MM, ref_w_px)
            for i, (_, w, h, _, _) in enumerate(boxes):
                w_px, h_px = sort_dimensions(w, h)
                w_mm = measure_object(w_px, ratio)
                h_mm = measure_object(h_px, ratio)
                if i == ref_index:
                    labels.append(f"REF {w_mm:.1f}x{h_mm:.1f}mm")
                else:
                    labels.append(f"{w_mm:.1f}x{h_mm:.1f}mm")
            summary = f"{len(boxes)}개 객체 검출 / 기준 카드 #{ref_index}"
        else:
            labels = ["OBJECT"] * len(boxes)
            summary = f"{len(boxes)}개 검출, 기준 카드 미발견"

    result = draw_boxes(image.copy(), boxes, labels=labels)

    return {
        "Original": image,
        "Adaptive": adaptive,
        "Filtered": filtered,
        "Edges": edges,
        "Filtered Edges": filtered_edges,
        "Result": result,
        "_summary": summary,
    }


class ViewerApp:

    SIDEBAR_BG = "#2b2b2b"
    BTN_BG     = "#3c3c3c"
    BTN_ACTIVE = "#0066cc"
    CANVAS_BG  = "#1e1e1e"

    def __init__(self, root):
        self.root = root
        self.root.title("Image Measurement Viewer")
        self.root.geometry("1100x720")
        self.root.configure(bg=self.CANVAS_BG)

        self.stages = None
        self.current_stage = "Result"
        self.tk_image = None
        self.current_path = None

        self._build_ui()

        # 기본 이미지가 있으면 자동 로드
        default_path = os.path.join("images", "input", "test.jpg")
        if os.path.exists(default_path):
            self._load(default_path)

    def _build_ui(self):
        # 사이드바
        sidebar = tk.Frame(self.root, width=200, bg=self.SIDEBAR_BG)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar, text="Pipeline Stages",
            bg=self.SIDEBAR_BG, fg="white",
            font=("Segoe UI", 12, "bold"),
        ).pack(pady=(20, 12))

        self.buttons = {}
        for stage in STAGES:
            btn = tk.Button(
                sidebar, text=stage,
                relief=tk.FLAT, anchor="w", padx=14, pady=8,
                bg=self.BTN_BG, fg="white",
                activebackground="#505050", activeforeground="white",
                font=("Segoe UI", 10),
                command=lambda s=stage: self.show_stage(s),
            )
            btn.pack(fill=tk.X, padx=12, pady=3)
            self.buttons[stage] = btn

        ttk.Separator(sidebar, orient="horizontal").pack(
            fill=tk.X, padx=12, pady=18
        )

        tk.Button(
            sidebar, text="이미지 열기...",
            bg=self.BTN_ACTIVE, fg="white",
            relief=tk.FLAT, padx=12, pady=10,
            font=("Segoe UI", 10, "bold"),
            command=self.open_image,
        ).pack(fill=tk.X, padx=12, pady=4)

        # 메인 영역
        main = tk.Frame(self.root, bg=self.CANVAS_BG)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Label(
            main, bg=self.CANVAS_BG,
            text="이미지를 열어주세요",
            fg="#888", font=("Segoe UI", 14),
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 0))

        self.status = tk.Label(
            main, text="",
            bg=self.CANVAS_BG, fg="#aaa",
            font=("Consolas", 10), anchor="w",
        )
        self.status.pack(fill=tk.X, padx=12, pady=10)

        self.canvas.bind("<Configure>", self._on_resize)

    def open_image(self):
        path = filedialog.askopenfilename(
            initialdir="images/input",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load(path)

    def _load(self, path):
        self.status.config(text=f"처리 중: {os.path.basename(path)}")
        self.root.update_idletasks()

        stages = run_pipeline(path)
        if stages is None:
            self.status.config(text=f"이미지 로드 실패: {path}")
            return

        self.stages = stages
        self.current_path = path
        self.show_stage(self.current_stage)

    def show_stage(self, stage_name):
        if self.stages is None:
            return
        self.current_stage = stage_name

        for name, btn in self.buttons.items():
            btn.config(bg=self.BTN_ACTIVE if name == stage_name else self.BTN_BG)

        self._render(self.stages[stage_name])

        file_name = os.path.basename(self.current_path) if self.current_path else ""
        self.status.config(
            text=f"[{stage_name}]  {file_name}   |   {self.stages['_summary']}"
        )

    def _render(self, cv_image):
        if cv_image.ndim == 2:
            rgb = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(rgb)

        cw = max(self.canvas.winfo_width(),  100)
        ch = max(self.canvas.winfo_height(), 100)
        pil.thumbnail((cw, ch), Image.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(pil)
        self.canvas.config(image=self.tk_image, text="")

    def _on_resize(self, _event):
        if self.stages is not None:
            self._render(self.stages[self.current_stage])


if __name__ == "__main__":
    root = tk.Tk()
    ViewerApp(root)
    root.mainloop()
