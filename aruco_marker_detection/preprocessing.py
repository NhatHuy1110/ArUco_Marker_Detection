"""Image preprocessing functions used by the ArUco detector."""

from functools import lru_cache

import cv2
import numpy as np


def no_preprocess(gray):
    """Return the input image unchanged."""
    return gray


def clahe_default(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def clahe_strong(gray):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def sharpen_mild(gray):
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def sharpen_strong(gray):
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 2.5, blurred, -1.5, 0)


def sharpen_kernel(gray):
    kernel = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        dtype=np.float32,
    )
    return cv2.filter2D(gray, -1, kernel)


@lru_cache(maxsize=None)
def gamma_lut(gamma):
    """Build a power-law gamma LUT matching the experiments."""
    return np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
        dtype=np.uint8,
    )


def gamma_correction(gray, gamma):
    """Apply inverse-gamma power-law correction.

    With this convention, gamma > 1 lifts mid-tone intensity and gamma < 1
    suppresses mid-tones.
    """
    return cv2.LUT(gray, gamma_lut(float(gamma)))


def gamma_0_6(gray):
    return gamma_correction(gray, 0.6)


def gamma_1_5(gray):
    return gamma_correction(gray, 1.5)


def gamma_2_0(gray):
    return gamma_correction(gray, 2.0)


def gamma_2_5(gray):
    return gamma_correction(gray, 2.5)


def contrast_stretch(gray):
    min_val = np.percentile(gray, 2)
    max_val = np.percentile(gray, 98)
    if max_val - min_val < 10:
        return gray
    stretched = (gray.astype(np.float32) - min_val) / (max_val - min_val) * 255
    return np.clip(stretched, 0, 255).astype(np.uint8)


def bilateral_denoise(gray):
    return cv2.bilateralFilter(gray, 7, 50, 50)


def histogram_equalize(gray):
    return cv2.equalizeHist(gray)


def clahe_then_sharpen(gray):
    return sharpen_mild(clahe_default(gray))


def morphological_tophat(gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return cv2.add(gray, tophat)


ALL_PREPROCESS = {
    "none": no_preprocess,
    "clahe_2.0": clahe_default,
    "clahe_4.0": clahe_strong,
    "sharpen_mild": sharpen_mild,
    "sharpen_strong": sharpen_strong,
    "sharpen_kernel": sharpen_kernel,
    "gamma_0.6": gamma_0_6,
    "gamma_1.5": gamma_1_5,
    "gamma_2.0": gamma_2_0,
    "gamma_2.5": gamma_2_5,
    "contrast_stretch": contrast_stretch,
    "bilateral": bilateral_denoise,
    "hist_equalize": histogram_equalize,
    "clahe+sharpen": clahe_then_sharpen,
    "tophat": morphological_tophat,
}
