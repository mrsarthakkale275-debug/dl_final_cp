"""Model loading, discovery and status registry.

The backend never assumes that trained weights exist. This module inspects the
configured model directories and reports availability honestly. Inference
engines are replaceable: services ask the registry for an engine and handle the
unavailable case explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from backend.app.core.config import settings
from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)

WEIGHT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".h5", ".keras", ".pkl", ".joblib", ".npz"}


class RGBPredictionPayload(dict):
    """Container for RGB model output."""


class InferenceEngine(Protocol):
    """Replaceable inference engine interface."""

    model_id: str
    version: str
    is_ready: bool

    def predict_rgb(self, image: Any) -> Dict[str, Any]:
        ...

    def predict_xray(self, image: Any) -> Dict[str, Any]:
        ...


@dataclass
class ModelHandle:
    name: str
    kind: str
    directory: Path
    weights: List[str] = field(default_factory=list)
    status: str = "unavailable"
    ready: bool = False
    model_id: Optional[str] = None
    version: Optional[str] = None
    message: str = "No trained weights found."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "available": len(self.weights) > 0,
            "ready": self.ready,
            "status": self.status,
            "model_id": self.model_id,
            "version": self.version,
            "weights": self.weights,
            "message": self.message,
        }


def _scan_weights(directory: Path) -> List[str]:
    if not directory.exists():
        return []
    found = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in WEIGHT_EXTENSIONS:
            found.append(path.name)
    return found


def _version_from_files(files: List[str], directory: Path) -> str:
    if not files:
        return "untrained"
    # Use newest mtime among weights as a crude version stamp.
    try:
        newest = max((directory / f for f in files), key=lambda p: p.stat().st_mtime)
        return f"w{int(newest.stat().st_mtime)}"
    except OSError:
        return "unknown"


class ModelRegistry:
    """Holds discovered model metadata and optional in-memory engines."""

    def __init__(self) -> None:
        self.rgb = ModelHandle(name="rgb", kind="rgb", directory=settings.rgb_models_dir)
        self.xray = ModelHandle(name="xray", kind="xray", directory=settings.xray_models_dir)
        self.rgb_engine: Optional[Any] = None
        self.xray_engine: Optional[Any] = None
        self.refresh()

    def refresh(self) -> None:
        self.rgb.weights = _scan_weights(self.rgb.directory)
        self.xray.weights = _scan_weights(self.xray.directory)

        if self.rgb.weights:
            self.rgb.model_id = "rgb-inspection-model"
            self.rgb.version = _version_from_files(self.rgb.weights, self.rgb.directory)
            self.rgb.status = "weights_found_not_loaded"
            self.rgb.ready = False
            self.rgb.message = (
                "RGB model weights were found, but no compatible runtime/loader is "
                "configured in this CPU-only deployment. Register a runtime engine "
                "to enable inference."
            )
        else:
            self.rgb.model_id = None
            self.rgb.version = None
            self.rgb.status = "untrained"
            self.rgb.ready = False
            self.rgb.message = (
                "No RGB model weights found in models/rgb/. "
                "Train and place weights there before RGB inference is available."
            )

        if self.xray.weights:
            self.xray.model_id = "xray-inspection-model"
            self.xray.version = _version_from_files(self.xray.weights, self.xray.directory)
            self.xray.status = "weights_found_not_loaded"
            self.xray.ready = False
            self.xray.message = (
                "X-ray model weights were found, but no compatible runtime/loader is "
                "configured in this CPU-only deployment. Register a runtime engine "
                "to enable inference."
            )
        else:
            self.xray.model_id = None
            self.xray.version = None
            self.xray.status = "untrained"
            self.xray.ready = False
            self.xray.message = (
                "No X-ray model weights found in models/xray/. "
                "Train and place weights there before X-ray inference is available."
            )

        logger.info(
            "Model registry refreshed: rgb=%s (%d weights), xray=%s (%d weights)",
            self.rgb.status,
            len(self.rgb.weights),
            self.xray.status,
            len(self.xray.weights),
        )

    def register_rgb_engine(self, engine: Any) -> None:
        self.rgb_engine = engine
        if engine is not None and getattr(engine, "is_ready", False):
            self.rgb.ready = True
            self.rgb.status = "ready"
            self.rgb.model_id = getattr(engine, "model_id", self.rgb.model_id)
            self.rgb.version = getattr(engine, "version", self.rgb.version)
            self.rgb.message = "RGB inference engine registered and ready."
        else:
            self.refresh()

    def register_xray_engine(self, engine: Any) -> None:
        self.xray_engine = engine
        if engine is not None and getattr(engine, "is_ready", False):
            self.xray.ready = True
            self.xray.status = "ready"
            self.xray.model_id = getattr(engine, "model_id", self.xray.model_id)
            self.xray.version = getattr(engine, "model_version", self.xray.version)
            self.xray.message = "X-ray inference engine registered and ready."
        else:
            self.refresh()

    def get_rgb_engine(self) -> Optional[Any]:
        return self.rgb_engine if self.rgb.ready else None

    def get_xray_engine(self) -> Optional[Any]:
        return self.xray_engine if self.xray.ready else None


# Default application-wide registry instance.
registry = ModelRegistry()
