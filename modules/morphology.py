# morphology.py
import cv2
import numpy as np

def apply_morphology(image):

    kernel = np.ones((3,3), np.uint8)

    opened = cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    return opened