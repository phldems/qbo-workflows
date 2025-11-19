"""
Image utility helpers shared across preprocessing and tests.
"""

from __future__ import annotations

import numpy as np


def ensure_portrait(image: np.ndarray) -> np.ndarray:
    """
    Rotate landscape-oriented images so height >= width.

    Args:
        image: NumPy array (H x W x C or H x W)

    Returns:
        Portrait-oriented image (rotated clockwise if needed)
    """
    if image.shape[1] > image.shape[0]:
        return np.rot90(image, k=3)
    return image
