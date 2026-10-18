"""Shared preprocessing and scoring helpers for the RGB anomaly baseline."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


DEFAULT_IMAGE_SIZE = 128
NUM_CHANNELS = 3


def pil_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def preprocess_image(image: Image.Image, size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Resize and normalize an image consistently for training and inference."""
    image = pil_to_rgb(image).resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def mahalanobis_map(features: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """Per-pixel Mahalanobis distance map.

    features: (H, W, C)
    mean: (H, W, C)
    inv_cov: (H, W, C, C)
    """
    centered = features - mean
    tmp = np.einsum("hwc,hwcd->hwd", centered, inv_cov, optimize=True)
    squared = np.einsum("hwd,hwd->hw", tmp, centered, optimize=True)
    return np.sqrt(np.clip(squared, 0.0, None))


def image_anomaly_score(score_map: np.ndarray, percentile: float = 99.0) -> float:
    return float(np.percentile(score_map, percentile))


def standardize_features(
    features: np.ndarray, feature_mean: np.ndarray, feature_std: np.ndarray
) -> np.ndarray:
    return (features - feature_mean) / np.clip(feature_std, 1e-6, None)
