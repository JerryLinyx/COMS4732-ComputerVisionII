import numpy as np


def compute_K(
    img_width_px: int,
    img_height_px: int,
    optical_focal_length_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
) -> np.ndarray:
    """
    Compute camera intrinsic matrix K from physical camera specifications.

    For square pixels (fx == fy), sensor aspect ratio must match image aspect ratio.

    Args:
        img_width_px: image width in pixels (W_px)
        img_height_px: image height in pixels (H_px)
        optical_focal_length_mm: physical focal length in mm (f_mm)
        sensor_width_mm: physical sensor width in mm (w_mm)
        sensor_height_mm: physical sensor height in mm (h_mm)

    Returns:
        K: (3, 3) camera intrinsic matrix [[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]]
    """
    # HW spec: f_px = f_mm * (W_px / w_mm)
    f_px = optical_focal_length_mm * (img_width_px / sensor_width_mm)

    cx = img_width_px / 2.0
    cy = img_height_px / 2.0

    return np.array([
        [f_px, 0.0, cx],
        [0.0, f_px, cy],
        [0.0, 0.0, 1.0],
    ], dtype=float)
