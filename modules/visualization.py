# visualization.py
# 시각화 모듈

import cv2
import numpy as np


def draw_boxes(
    image,
    boxes,
    labels=None
):

    for index, item in enumerate(boxes):

        box = item[0]

        cv2.drawContours(
            image,
            [box],
            0,
            (0, 255, 0),
            2
        )

        if labels is not None and index < len(labels):

            label = labels[index]

            center = np.mean(
                box,
                axis=0
            ).astype(int)

            cv2.putText(
                image,
                label,
                (center[0], center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

    return image