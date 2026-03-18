"""
Enhances blueprint images before sending to VLM.
Key goal: make room labels and dimension lines readable.
"""

import cv2
import numpy as np
from PIL import Image


def preprocess_blueprint(img: Image.Image,
                         target_long_side: int = 2048) -> Image.Image:
    """
    Full preprocessing pipeline:
    resize → grayscale → denoise → CLAHE → sharpen → deskew → back to RGB
    """
    img = _resize_for_vlm(img, target_long_side)
    arr = np.array(img)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10,
                                         templateWindowSize=7,
                                         searchWindowSize=21)
    enhanced = _clahe(denoised)
    sharpened = _sharpen(enhanced)
    deskewed = _deskew(sharpened)
    rgb = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def _resize_for_vlm(img: Image.Image, max_side: int) -> Image.Image:
    """Resize so longest side = max_side. VLMs work best ~2048px."""
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)),
                         Image.LANCZOS)
    return img


def _clahe(gray: np.ndarray) -> np.ndarray:
    """CLAHE — boosts local contrast without washing out."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _sharpen(gray: np.ndarray) -> np.ndarray:
    """Unsharp mask sharpening — crisp dimension text."""
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return sharpened


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Detects dominant line angle via Hough transform,
    corrects rotation. Handles blueprints scanned at an angle.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)

    if lines is None or len(lines) < 5:
        return gray   # not enough lines to detect angle safely

    angles = []
    for line in lines[:30]:
        theta = line[0][1]
        if theta < np.pi / 4 or theta > 3 * np.pi / 4:
            angles.append(theta)

    if not angles:
        return gray

    median_theta = np.median(angles)
    angle_deg = np.degrees(median_theta) - 90.0

    if abs(angle_deg) < 0.5:
        return gray

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
