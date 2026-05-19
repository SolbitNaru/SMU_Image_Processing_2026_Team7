# noise_filter.py
import cv2
import numpy as np


def remove_small_noise(image, min_area=500):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        image,
        connectivity=8
    )

    output = np.zeros_like(image)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            output[labels == i] = 255

    return output