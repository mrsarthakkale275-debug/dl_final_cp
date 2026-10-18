"""X-ray model interface with PyTorch CNN backend.

Uses the ResNet34-based X-ray CNN from multimodal_cnn.py when available,
or falls back to review-required status when no weights are present.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class XRayInferenceEngine:
    """X-ray inspection engine adapter.

    Wraps the XRayCNN from multimodal_cnn.py and provides
    the predict_xray() interface expected by services.
    """

    model_id: str = "xray-resnet34-cnn"
    model_version: str = "mock-v1.0"
    is_ready: bool = False

    def __init__(self, cnn_model=None):
        self._cnn = cnn_model
        if cnn_model is not None:
            self.is_ready = True

    def predict_xray(self, image: Any) -> Dict[str, Any]:
        """Run X-ray inference.

        Args:
            image: PIL Image or numpy array

        Returns:
            Dictionary with prediction results
        """
        if self._cnn is None:
            raise RuntimeError("X-ray CNN model is not loaded.")

        # Convert PIL to numpy if needed
        if hasattr(image, "convert"):
            image = np.array(image.convert("L"))

        try:
            from backend.app.ml.multimodal_cnn import MultimodalInspector
            inspector = MultimodalInspector.__new__(MultimodalInspector)
            # Use the standalone model
            import torch
            import cv2

            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_resized = cv2.resize(image, (224, 224))
            tensor = torch.from_numpy(image_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
            tensor = (tensor - 0.449) / 0.226

            device = next(self._cnn.parameters()).device
            tensor = tensor.to(device)

            with torch.no_grad():
                output = self._cnn(tensor)

            anomaly_score = float(output["anomaly_score"].item())
            class_probs = output["class_probs"][0]
            class_idx = class_probs.argmax().item()
            class_labels = ["normal", "void", "crack"]
            predicted_class = class_labels[min(class_idx, len(class_labels) - 1)]
            confidence = float(class_probs[class_idx].item())

            return {
                "predicted_class": predicted_class,
                "anomaly_score": anomaly_score,
                "score": anomaly_score,
                "confidence": confidence,
                "class_scores": {
                    class_labels[i]: float(class_probs[i].item())
                    for i in range(min(len(class_labels), class_probs.shape[0]))
                },
                "localization": None,
                "message": "X-ray CNN inference completed.",
            }
        except Exception as exc:
            logger.exception("X-ray CNN inference failed")
            return {
                "predicted_class": None,
                "anomaly_score": None,
                "score": None,
                "confidence": None,
                "message": f"X-ray inference error: {exc}",
            }


def load_xray_engine() -> Optional[XRayInferenceEngine]:
    """Load the X-ray engine with PyTorch CNN if weights are available."""
    try:
        from backend.app.core.config import settings
        import os

        weights_path = os.path.join(settings.xray_models_dir, "xray_cnn_weights.pt")
        if os.path.isfile(weights_path):
            from backend.app.ml.multimodal_cnn import XRayCNN
            import torch

            model = XRayCNN(pretrained=False)
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()

            engine = XRayInferenceEngine(cnn_model=model)
            logger.info("X-ray CNN engine loaded from %s", weights_path)
            return engine
        else:
            logger.info("No trained X-ray CNN weights found at %s; using review-required workflow.", weights_path)
            return None
    except Exception as exc:
        logger.warning("X-ray engine loading failed: %s", exc)
        return None
