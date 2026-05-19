# measurement_2d.py
# 핵심 계산(실제 크기 계산) ,  측정된 픽셀 너비를 실제 mm/cm로 변환하는 함수

def sort_dimensions(w, h):

    width = max(w, h)

    height = min(w, h)

    return width, height


def calculate_ratio(
    reference_width_mm,
    reference_width_pixel
):

    ratio = (
        reference_width_mm /
        reference_width_pixel
    )

    return ratio


def measure_object(
    pixel_size,
    ratio
):

    return pixel_size * ratio