# preprocess.py
# GrayScale + Blur , 노이즈 감소 엣지 안정화 목적 > Blur → Canny 순서에서 blur 먼저
import cv2


def load_image(path):

    image = cv2.imread(path)

    if image is None:
        return None, None

    max_width = 1000

    scale = max_width / image.shape[1]

    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)

    resized = cv2.resize(image, (width, height))

    return resized, scale


def preprocess(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    contrast = clahe.apply(gray)

    blur = cv2.GaussianBlur(
        contrast,
        (5, 5),
        0
    )

    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    return adaptive