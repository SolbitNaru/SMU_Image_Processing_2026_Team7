# viewer.py
# Tkinter 기반 파이프라인 단계 뷰어

import os
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np

from PIL import Image, ImageTk

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
    measure_object,
)

from modules.measurement_3d import (
    load_camera_matrix,
    measure_object_3d,
)

from modules.depth_estimation import (

    estimate_depth,

    box_disparity,

    disparity_to_distance_mm,

    is_available as depth_available,

    load_error as depth_load_error,
)

from modules.yolo_detector import (
    detect_objects as yolo_detect_objects
)


# -------------------------------------------------
# 기준 객체
# -------------------------------------------------
CARD_RATIO = 1.586

RATIO_TOLERANCE = 0.25

REFERENCE_OBJECTS = {

    "card": {

        "width_mm": 85.6,

        "ratio": CARD_RATIO
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


STAGES = [

    "Original",

    "Adaptive",

    "Filtered",

    "Edges",

    "Filtered Edges",

    "Depth",

    "Result",
]


# -------------------------------------------------
# ratio-only fallback
# -------------------------------------------------
def find_reference_box(

    boxes,
    tolerance=RATIO_TOLERANCE
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
# YOLO + ratio
# -------------------------------------------------
def find_reference_box_with_yolo(

    boxes,
    detections,
    tolerance=RATIO_TOLERANCE
):

    if not detections:

        return find_reference_box(
            boxes,
            tolerance
        )

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

        x, y, bw, bh = cv2.boundingRect(box)

        contour_area = bw * bh

        for name, info in REFERENCE_OBJECTS.items():

            ratio_score = abs(
                ratio - info["ratio"]
            )

            if ratio_score > tolerance:

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

                    contour_area +
                    detection_area -
                    intersection
                )

                if union <= 0:

                    continue

                iou = intersection / union

                score = (
                    ratio_score -
                    (iou * 0.5)
                )

                if score < best_score:

                    best_score = score

                    best_index = index

                    best_name = name

    if best_index == -1:

        return find_reference_box(
            boxes,
            tolerance
        )

    return best_index, best_name


# -------------------------------------------------
# OpenCV pipeline
# -------------------------------------------------
def run_pipeline(image_path):

    image, scale = load_image(
        image_path
    )

    if image is None:

        return None

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

        dilate_kernel=5,

        dilate_iterations=1,

        close_iterations=2,
    )

    contours = find_contours(
        filtered_edges
    )

    boxes = get_boxes(

        contours,

        min_area=1500,

        min_width=25,

        min_height=25,

        min_solidity=0.7,
    )

    yolo_detections = yolo_detect_objects(
        image
    )

    return {

        "image": image,

        "scale": scale,

        "boxes": boxes,

        "yolo_detections": yolo_detections,

        "stage_images": {

            "Original": image,

            "Adaptive": adaptive,

            "Filtered": filtered,

            "Edges": edges,

            "Filtered Edges": filtered_edges,
        },
    }


# -------------------------------------------------
# 카드 기준 측정
# -------------------------------------------------
def measure_with_card(

    boxes,

    detections=None,

    preferred=None
):

    if not boxes:

        return [], "객체 없음"

    # 자동 탐지
    if preferred is None or preferred == "auto":

        ref_index, ref_name = (

            find_reference_box_with_yolo(

                boxes,

                detections
            )
        )

    # 강제 선택
    else:

        ref_name = preferred

        best_index = -1

        best_score = float("inf")

        target_ratio = (

            REFERENCE_OBJECTS.get(

                ref_name,

                REFERENCE_OBJECTS["card"]
            )["ratio"]
        )

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

            score = abs(
                ratio - target_ratio
            )

            if score < best_score:

                best_score = score

                best_index = index

        ref_index = best_index

    if ref_index == -1:

        return (

            ["OBJECT"] * len(boxes),

            f"{len(boxes)}개 검출 "
            f"/ 기준 객체 없음"
        )

    ref_info = REFERENCE_OBJECTS.get(

        ref_name,

        REFERENCE_OBJECTS["card"]
    )

    ref_w_px, _ = sort_dimensions(

        boxes[ref_index][1],

        boxes[ref_index][2]
    )

    ratio = calculate_ratio(

        ref_info["width_mm"],

        ref_w_px
    )

    labels = []

    for i, (
        box,
        w,
        h,
        area,
        angle
    ) in enumerate(boxes):

        w_px, h_px = sort_dimensions(
            w,
            h
        )

        w_mm = measure_object(
            w_px,
            ratio
        )

        h_mm = measure_object(
            h_px,
            ratio
        )

        prefix = ""

        if i == ref_index:

            prefix = f"REF({ref_name}) "

        labels.append(

            f"{prefix}"
            f"{w_mm:.1f}x"
            f"{h_mm:.1f}mm"
        )

    return (

        labels,

        f"{len(boxes)}개 객체 "
        f"/ 기준 {ref_name} #{ref_index}"
    )


# -------------------------------------------------
# 거리 기반 측정
# -------------------------------------------------
def measure_with_distance(

    boxes,

    fx,

    fy,

    distance_mm
):

    if not boxes:

        return [], "객체 없음"

    if (

        fx is None or
        fy is None or
        distance_mm is None
    ):

        return (

            ["OBJECT"] * len(boxes),

            "거리/초점거리 정보 부족"
        )

    labels = []

    for (
        box,
        w,
        h,
        area,
        angle
    ) in boxes:

        w_px, h_px = sort_dimensions(
            w,
            h
        )

        w_mm, h_mm = measure_object_3d(

            w_px,

            h_px,

            fx,

            fy,

            distance_mm
        )

        labels.append(
            f"{w_mm:.1f}x{h_mm:.1f}mm"
        )

    return (

        labels,

        f"{len(boxes)}개 객체 "
        f"(거리 {distance_mm:.0f}mm)"
    )


# -------------------------------------------------
# 깊이 기반 측정
# -------------------------------------------------
def measure_with_depth(

    image,

    boxes,

    fx,

    fy,

    anchor_distance_mm
):

    if not boxes:

        return [], "객체 없음", None

    if (

        fx is None or
        fy is None or
        anchor_distance_mm is None
    ):

        return (

            ["OBJECT"] * len(boxes),

            "거리/초점거리 정보 부족",

            None
        )

    if not depth_available():

        return (

            ["OBJECT"] * len(boxes),

            "torch 미설치",

            None
        )

    depth_map = estimate_depth(image)

    if depth_map is None:

        err = (
            depth_load_error() or
            "원인 불명"
        )

        return (

            ["OBJECT"] * len(boxes),

            f"MiDaS 실패: {err}",

            None
        )

    disps = [

        box_disparity(
            depth_map,
            b[0]
        )

        for b in boxes
    ]

    valid = [

        d for d in disps

        if np.isfinite(d) and d > 1e-6
    ]

    if not valid:

        return (

            ["OBJECT"] * len(boxes),

            "깊이 추출 실패",

            depth_map
        )

    anchor_disp = float(
        np.median(valid)
    )

    labels = []

    for (
        box_data,
        disp
    ) in zip(boxes, disps):

        _, w, h, _, _ = box_data

        z_obj = disparity_to_distance_mm(

            disp,

            anchor_disp,

            anchor_distance_mm
        )

        w_px, h_px = sort_dimensions(
            w,
            h
        )

        if np.isnan(z_obj):

            labels.append("?x?mm")

            continue

        w_mm = w_px * z_obj / fx

        h_mm = h_px * z_obj / fy

        labels.append(

            f"{w_mm:.1f}x"
            f"{h_mm:.1f}mm"
        )

    summary = (

        f"{len(boxes)}개 객체 "
        f"(깊이 자동)"
    )

    return labels, summary, depth_map


# -------------------------------------------------
# GUI
# -------------------------------------------------
class ViewerApp:

    SIDEBAR_BG = "#2b2b2b"

    BTN_BG = "#3c3c3c"

    BTN_ACTIVE = "#0066cc"

    CANVAS_BG = "#1e1e1e"

    INPUT_BG = "#3c3c3c"

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Image Measurement Viewer"
        )

        self.root.geometry("1180x780")

        self.root.configure(
            bg=self.CANVAS_BG
        )

        self.detection = None

        self.stage_images = None

        self.current_stage = "Result"

        self.tk_image = None

        self.current_path = None

        self.mode_var = tk.StringVar(
            value="card"
        )

        self.distance_var = tk.StringVar(
            value="300"
        )

        self.focal_var = tk.StringVar(
            value="2400"
        )

        self.ref_var = tk.StringVar(
            value="auto"
        )

        self.K = load_camera_matrix(
            "camparams.mat"
        )

        if self.K is not None:

            self.focal_var.set(
                f"{self.K[0, 0]:.1f}"
            )

        self._build_ui()

        default_path = os.path.join(

            "images",

            "input",

            "test.jpg"
        )

        if os.path.exists(default_path):

            self._load(default_path)

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self):

        sidebar = tk.Frame(

            self.root,

            width=240,

            bg=self.SIDEBAR_BG
        )

        sidebar.pack(

            side=tk.LEFT,

            fill=tk.Y
        )

        sidebar.pack_propagate(False)

        tk.Label(

            sidebar,

            text="Pipeline Stages",

            bg=self.SIDEBAR_BG,

            fg="white",

            font=(
                "Segoe UI",
                12,
                "bold"
            ),
        ).pack(
            pady=(18, 10)
        )

        self.buttons = {}

        for stage in STAGES:

            btn = tk.Button(

                sidebar,

                text=stage,

                relief=tk.FLAT,

                anchor="w",

                padx=14,

                pady=7,

                bg=self.BTN_BG,

                fg="white",

                activebackground="#505050",

                activeforeground="white",

                font=("Segoe UI", 10),

                command=lambda s=stage:
                self.show_stage(s),
            )

            btn.pack(

                fill=tk.X,

                padx=12,

                pady=2
            )

            self.buttons[stage] = btn

        ttk.Separator(

            sidebar,

            orient="horizontal"
        ).pack(

            fill=tk.X,

            padx=12,

            pady=12
        )

        # 모드
        tk.Label(

            sidebar,

            text="Measurement Mode",

            bg=self.SIDEBAR_BG,

            fg="white",

            font=(
                "Segoe UI",
                11,
                "bold"
            ),
        ).pack(
            pady=(2, 6)
        )

        for label, value in [

            ("카드 기준", "card"),

            ("거리 기반", "distance"),

            ("깊이 자동 (MiDaS)", "depth"),
        ]:

            tk.Radiobutton(

                sidebar,

                text=label,

                value=value,

                variable=self.mode_var,

                bg=self.SIDEBAR_BG,

                fg="white",

                selectcolor=self.SIDEBAR_BG,

                activebackground=self.SIDEBAR_BG,

                activeforeground="white",

                font=("Segoe UI", 10),

                command=self._remeasure,
            ).pack(

                anchor="w",

                padx=24
            )

        # 거리
        tk.Label(

            sidebar,

            text="촬영 거리 Z (mm)",

            bg=self.SIDEBAR_BG,

            fg="#cccccc",

            font=("Segoe UI", 9),
        ).pack(

            anchor="w",

            padx=14,

            pady=(10, 0)
        )

        self.distance_entry = tk.Entry(

            sidebar,

            textvariable=self.distance_var,

            bg=self.INPUT_BG,

            fg="white",

            insertbackground="white",

            relief=tk.FLAT,

            font=("Consolas", 10),
        )

        self.distance_entry.pack(

            fill=tk.X,

            padx=14,

            pady=(2, 6)
        )

        self.distance_entry.bind(

            "<Return>",

            lambda _e:
            self._remeasure()
        )

        # 초점거리
        tk.Label(

            sidebar,

            text="초점거리 fx",

            bg=self.SIDEBAR_BG,

            fg="#cccccc",

            font=("Segoe UI", 9),
        ).pack(

            anchor="w",

            padx=14,

            pady=(2, 0)
        )

        self.focal_entry = tk.Entry(

            sidebar,

            textvariable=self.focal_var,

            bg=self.INPUT_BG,

            fg="white",

            insertbackground="white",

            relief=tk.FLAT,

            font=("Consolas", 10),
        )

        self.focal_entry.pack(

            fill=tk.X,

            padx=14,

            pady=(2, 6)
        )

        self.focal_entry.bind(

            "<Return>",

            lambda _e:
            self._remeasure()
        )

        # 기준 객체
        tk.Label(

            sidebar,

            text="Reference Object",

            bg=self.SIDEBAR_BG,

            fg="#cccccc",

            font=("Segoe UI", 9),
        ).pack(

            anchor="w",

            padx=14,

            pady=(6, 0)
        )

        tk.OptionMenu(

            sidebar,

            self.ref_var,

            "auto",

            "card",

            "battery_aa",

            "bill_10000",

            command=lambda _v:
            self._remeasure()
        ).pack(

            fill=tk.X,

            padx=14,

            pady=(2, 8)
        )

        tk.Button(

            sidebar,

            text="다시 측정",

            bg="#444444",

            fg="white",

            relief=tk.FLAT,

            padx=10,

            pady=6,

            font=("Segoe UI", 10),

            command=self._remeasure,
        ).pack(

            fill=tk.X,

            padx=14,

            pady=(4, 8)
        )

        ttk.Separator(

            sidebar,

            orient="horizontal"
        ).pack(

            fill=tk.X,

            padx=12,

            pady=8
        )

        tk.Button(

            sidebar,

            text="이미지 열기...",

            bg=self.BTN_ACTIVE,

            fg="white",

            relief=tk.FLAT,

            padx=12,

            pady=10,

            font=(
                "Segoe UI",
                10,
                "bold"
            ),

            command=self.open_image,
        ).pack(

            fill=tk.X,

            padx=14,

            pady=(4, 14)
        )

        # 메인
        main = tk.Frame(

            self.root,

            bg=self.CANVAS_BG
        )

        main.pack(

            side=tk.LEFT,

            fill=tk.BOTH,

            expand=True
        )

        self.canvas = tk.Label(

            main,

            bg=self.CANVAS_BG,

            text="이미지를 열어주세요",

            fg="#888",

            font=("Segoe UI", 14),
        )

        self.canvas.pack(

            fill=tk.BOTH,

            expand=True,

            padx=12,

            pady=(12, 0)
        )

        self.status = tk.Label(

            main,

            text="",

            bg=self.CANVAS_BG,

            fg="#aaa",

            font=("Consolas", 10),

            anchor="w",

            justify="left",
        )

        self.status.pack(

            fill=tk.X,

            padx=12,

            pady=10
        )

        self.canvas.bind(

            "<Configure>",

            self._on_resize
        )

    # -------------------------------------------------
    # 동작
    # -------------------------------------------------
    def open_image(self):

        path = filedialog.askopenfilename(

            initialdir="images/input",

            filetypes=[

                (
                    "Images",

                    "*.jpg *.jpeg *.png *.bmp"
                ),

                ("All files", "*.*"),
            ],
        )

        if path:

            self._load(path)

    def _load(self, path):

        self.status.config(

            text=f"처리 중: {os.path.basename(path)}"
        )

        self.root.update_idletasks()

        det = run_pipeline(path)

        if det is None:

            self.status.config(

                text=f"로드 실패: {path}"
            )

            return

        self.detection = det

        self.current_path = path

        self._remeasure()

    def _parse_float(self, var):

        try:

            return float(var.get())

        except ValueError:

            return None

    def _remeasure(self):

        if self.detection is None:

            return

        boxes = self.detection["boxes"]

        detections = self.detection.get(

            "yolo_detections",

            []
        )

        scale = self.detection["scale"]

        image = self.detection["image"]

        mode = self.mode_var.get()

        depth_map = None

        if mode == "card":

            preferred = self.ref_var.get()

            labels, summary = (

                measure_with_card(

                    boxes,

                    detections=detections,

                    preferred=preferred
                )
            )

        else:

            distance_mm = self._parse_float(
                self.distance_var
            )

            focal_orig = self._parse_float(
                self.focal_var
            )

            if (

                focal_orig is None or
                distance_mm is None
            ):

                labels = (
                    ["OBJECT"] * len(boxes)
                )

                summary = "입력값 오류"

            else:

                fx = focal_orig * scale

                fy = fx

                if self.K is not None:

                    fy = (
                        self.K[1, 1] *
                        scale
                    )

                if mode == "depth":

                    self.status.config(
                        text="MiDaS 추론 중..."
                    )

                    self.root.update_idletasks()

                    labels, summary, depth_map = (

                        measure_with_depth(

                            image,

                            boxes,

                            fx,

                            fy,

                            distance_mm
                        )
                    )

                else:

                    labels, summary = (

                        measure_with_distance(

                            boxes,

                            fx,

                            fy,

                            distance_mm
                        )
                    )

        result = draw_boxes(

            image.copy(),

            boxes,

            labels=labels
        )

        self.stage_images = dict(
            self.detection["stage_images"]
        )

        self.stage_images["Result"] = result

        if depth_map is not None:

            from modules.depth_estimation import (
                disparity_to_visual
            )

            vis = disparity_to_visual(
                depth_map
            )

            self.stage_images["Depth"] = vis

        else:

            self.stage_images["Depth"] = (
                self._depth_placeholder(image)
            )

        self.summary = summary

        self.show_stage(
            self.current_stage
        )

    def _depth_placeholder(self, image):

        ph = np.zeros(

            image.shape[:2],

            dtype=np.uint8
        )

        cv2.putText(

            ph,

            "Depth Mode Only",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            200,

            1,

            cv2.LINE_AA,
        )

        return ph

    def show_stage(self, stage_name):

        if self.stage_images is None:

            return

        self.current_stage = stage_name

        for name, btn in self.buttons.items():

            btn.config(

                bg=(
                    self.BTN_ACTIVE
                    if name == stage_name
                    else self.BTN_BG
                )
            )

        self._render(

            self.stage_images[stage_name]
        )

        file_name = ""

        if self.current_path is not None:

            file_name = os.path.basename(
                self.current_path
            )

        mode_text = {

            "card": "카드 기준",

            "distance": "거리 기반",

            "depth": "깊이 자동",

        }.get(

            self.mode_var.get(),

            self.mode_var.get()
        )

        self.status.config(

            text=
            f"[{stage_name}] "
            f"{file_name} | "
            f"모드: {mode_text} | "
            f"{self.summary}"
        )

    def _render(self, cv_image):

        if cv_image.ndim == 2:

            rgb = cv2.cvtColor(

                cv_image,

                cv2.COLOR_GRAY2RGB
            )

        else:

            rgb = cv2.cvtColor(

                cv_image,

                cv2.COLOR_BGR2RGB
            )

        pil = Image.fromarray(rgb)

        cw = max(
            self.canvas.winfo_width(),
            100
        )

        ch = max(
            self.canvas.winfo_height(),
            100
        )

        pil.thumbnail(

            (cw, ch),

            Image.LANCZOS
        )

        self.tk_image = ImageTk.PhotoImage(
            pil
        )

        self.canvas.config(

            image=self.tk_image,

            text=""
        )

    def _on_resize(self, _event):

        if self.stage_images is not None:

            self._render(

                self.stage_images[
                    self.current_stage
                ]
            )


# -------------------------------------------------
# 실행
# -------------------------------------------------
if __name__ == "__main__":

    root = tk.Tk()

    ViewerApp(root)

    root.mainloop()
