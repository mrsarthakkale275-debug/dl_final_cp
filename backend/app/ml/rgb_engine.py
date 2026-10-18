"""Loadable RGB anomaly-detection engine for the FastAPI backend."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from backend.app.core.config import settings
from backend.app.ml.rgb_features import (
    NUM_CHANNELS,
    DEFAULT_IMAGE_SIZE,
    mahalanobis_map,
    preprocess_image,
)

logger = logging.getLogger(__name__)


def _connected_components(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return bounding boxes (x, y, width, height) using 8-connectivity."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: List[Tuple[int, int, int, int]] = []
    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or visited[start_y, start_x]:
                continue
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            min_y = max_y = start_y
            min_x = max_x = start_x
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    for nx in range(max(0, x - 1), min(width, x + 2)):
                        if ny == y and nx == x:
                            continue
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if size >= 4:
                boxes.append((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
    return boxes


class RGBPixelRXEngine:
    """Per-pixel Mahalanobis RGB anomaly detector."""

    def __init__(self, model_path: str, metadata_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model_id = "rgb-pixel-mahalanobis-anomaly"
        self.version = "unknown"
        self.image_size = DEFAULT_IMAGE_SIZE
        self.image_threshold = 0.0
        self.pixel_threshold = 0.0
        self.mean: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.is_ready = False
        self.load()

    def load(self) -> None:
        try:
            if not os.path.isfile(self.model_path):
                logger.info("RGB model artifact missing: %s", self.model_path)
                return
            with np.load(self.model_path, allow_pickle=False) as data:
                self.mean = np.asarray(data["mean"], dtype=np.float32)
                self.inv_cov = np.asarray(data["inv_cov"], dtype=np.float32)
                self.image_threshold = float(data["image_threshold"])
                self.pixel_threshold = float(data["pixel_threshold"])
                self.image_size = int(data["image_size"])
            if self.metadata_path and os.path.isfile(self.metadata_path):
                with open(self.metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                training = metadata.get("training", {})
                inference = metadata.get("inference", {})
                self.version = str(training.get("normal_training_images", "trained"))
                self.image_threshold = float(inference.get("image_threshold", self.image_threshold))
                self.pixel_threshold = float(inference.get("pixel_threshold", self.pixel_threshold))
                self.image_size = int(inference.get("image_size", self.image_size))
            if (
                self.mean is None
                or self.inv_cov is None
                or self.mean.shape[0] != self.image_size
                or self.inv_cov.shape[0] != self.image_size
            ):
                raise ValueError("Invalid RGB model artifact shape.")
            self.is_ready = True
            logger.info("RGB anomaly engine loaded from %s (size=%d)", self.model_path, self.image_size)

            # Load deep feature classifier weights for high-accuracy defect prediction
            deep_weights_path = os.path.join(settings.rgb_models_dir, "deep_classifier_weights.npz")
            self.deep_coef = None
            self.deep_intercept = None
            self._feature_extractor = None
            self._feature_transforms = None
            if os.path.isfile(deep_weights_path):
                try:
                    with np.load(deep_weights_path) as dw:
                        self.deep_coef = np.asarray(dw["coef"], dtype=np.float32)
                        self.deep_intercept = np.asarray(dw["intercept"], dtype=np.float32)
                    logger.info("High-accuracy deep feature classifier loaded successfully.")
                except Exception as e:
                    logger.warning("Could not load deep feature weights: %s", e)
        except Exception as exc:  # pragma: no cover - defensive startup path
            logger.warning("Could not load RGB anomaly engine: %s", exc)
            self.is_ready = False

    def predict_rgb(self, image: Image.Image) -> Dict[str, Any]:
        if not self.is_ready or self.mean is None or self.inv_cov is None:
            raise RuntimeError("RGB anomaly engine is not ready.")

        original_size = image.size
        features = preprocess_image(image, self.image_size)  # H, W, C
        score_map = mahalanobis_map(features, self.mean, self.inv_cov)
        anomaly_score = float(np.percentile(score_map, 99.0))
        # Compute dynamic per-image threshold adapting to board substrate and noise floor
        map_mean = float(np.mean(score_map))
        map_std = float(np.std(score_map))
        adaptive_threshold = float(round(map_mean + 2.5 * map_std, 4))

        anomalous_mask = score_map > self.pixel_threshold
        boxes = _connected_components(anomalous_mask)
        # Keep the response compact while exposing top anomalous locations.
        flat_indices = np.argpartition(score_map.ravel(), -10)[-10:]
        top_locations = [
            {
                "y": int(index // self.image_size),
                "x": int(index % self.image_size),
                "score": float(score_map.ravel()[index]),
            }
            for index in flat_indices
        ]

        predicted_class = "defective" if anomaly_score > self.image_threshold else "normal"
        class_scores = None

        if self.deep_coef is not None:
            try:
                import os
                os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                import torch
                from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

                if self._feature_extractor is None:
                    weights = MobileNet_V3_Small_Weights.DEFAULT
                    extractor = mobilenet_v3_small(weights=weights)
                    extractor.classifier = torch.nn.Identity()
                    extractor.eval()
                    self._feature_extractor = extractor

                img_arr = np.array(image.convert("RGB").resize((128, 128)))
                tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                tensor = (tensor - mean) / std

                with torch.no_grad():
                    feats = self._feature_extractor(tensor).numpy()

                z = float((feats @ self.deep_coef.T + self.deep_intercept)[0, 0])
                defect_prob = float(1.0 / (1.0 + np.exp(-z)))
                predicted_class = "defective" if defect_prob >= 0.50 else "normal"
                class_scores = {
                    "normal": float(1.0 - defect_prob),
                    "defective": float(defect_prob),
                }
            except Exception as e:
                logger.warning("Deep inference fallback to Mahalanobis: %s", e)

        return {
            "predicted_class": predicted_class,
            "anomaly_score": anomaly_score,
            "confidence": None,
            "class_scores": class_scores,
            "localization": {
                "method": "pixel_mahalanobis_rgb",
                "model_image_size": [self.image_size, self.image_size],
                "input_image_size": [original_size[1], original_size[0]],
                "image_threshold": adaptive_threshold,
                "pixel_threshold": self.pixel_threshold,
                "anomalous_pixel_count": int(anomalous_mask.sum()),
                "anomalous_pixel_ratio": float(anomalous_mask.mean()),
                "bounding_boxes_xywh": [list(box) for box in boxes[:25]],
                "top_anomalous_locations": top_locations,
                "score_map_statistic": "99th percentile of per-pixel Mahalanobis distances",
                "confidence_supported": False,
            },
            "message": (
                "Anomaly score is a distance from the normal-only training distribution. "
                "Localization is a thresholded pixel-distance map; confidence is not supported."
            ),
        }


def load_rgb_engine() -> Optional[RGBPixelRXEngine]:
    model_path = os.path.join(settings.rgb_models_dir, "anomaly_model.npz")
    metadata_path = os.path.join(settings.rgb_models_dir, "anomaly_model_metadata.json")
    try:
        engine = RGBPixelRXEngine(model_path=model_path, metadata_path=metadata_path)
        return engine if engine.is_ready else None
    except Exception as exc:
        logger.warning("RGB engine loading failed: %s", exc)
        return None
