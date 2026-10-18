"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings:
    def __init__(self) -> None:
        self.project_root = _project_root()
        self.backend_root = _backend_root()

        try:
            from dotenv import load_dotenv

            load_dotenv(self.backend_root / ".env", override=False)
        except Exception:
            pass

        self.environment = os.getenv("APP_ENV", "development")
        self.debug = os.getenv("APP_DEBUG", "false").lower() in {"1", "true", "yes"}

        # --- Directory paths ---
        self.data_dir = Path(os.getenv("DATA_DIR", str(self.project_root / "data")))
        self.models_dir = Path(os.getenv("MODELS_DIR", str(self.project_root / "models")))
        self.rgb_models_dir = Path(os.getenv("RGB_MODELS_DIR", str(self.models_dir / "rgb")))
        self.xray_models_dir = Path(os.getenv("XRAY_MODELS_DIR", str(self.models_dir / "xray")))
        self.storage_dir = Path(os.getenv("STORAGE_DIR", str(self.backend_root / "storage")))
        self.image_storage_dir = Path(
            os.getenv("IMAGE_STORAGE_DIR", str(self.backend_root / "storage" / "inspection_images"))
        )
        self.report_storage_dir = Path(
            os.getenv("REPORT_STORAGE_DIR", str(self.backend_root / "storage" / "reports"))
        )

        # --- Database ---
        default_db = self.backend_root / "adaptive_inspect.db"
        self.database_url = os.getenv("DATABASE_URL", f"sqlite:///{default_db}")
        self.direct_url = os.getenv("DIRECT_URL", self.database_url)

        # --- CORS ---
        cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173")
        self.cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
        self.cors_allow_all = os.getenv("CORS_ALLOW_ALL", "false").lower() in {"1", "true", "yes"}

        # --- Upload limits ---
        self.max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
        self.allowed_image_types = {
            "image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp",
        }

        # --- Legacy RGB thresholds ---
        self.rgb_confidence_threshold = float(os.getenv("RGB_CONFIDENCE_THRESHOLD", "0.85"))
        self.rgb_uncertainty_threshold = float(os.getenv("RGB_UNCERTAINTY_THRESHOLD", "0.55"))
        self.anomaly_threshold = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))

        # --- Adaptive Gating Thresholds ---
        self.tau_low = float(os.getenv("TAU_LOW", "0.30"))
        self.tau_high = float(os.getenv("TAU_HIGH", "0.65"))
        self.entropy_threshold = float(os.getenv("ENTROPY_THRESHOLD", "0.50"))
        self.catastrophic_area_ratio = float(os.getenv("CATASTROPHIC_AREA_RATIO", "0.25"))

        # --- Cost Engine Defaults ---
        self.base_material_cost = float(os.getenv("BASE_MATERIAL_COST_USD", "15.0"))
        self.rework_rate = float(os.getenv("REWORK_RATE_PER_AREA", "50.0"))
        self.labor_cost = float(os.getenv("LABOR_COST_PER_HOUR_USD", "35.0"))
        self.xray_scan_cost = float(os.getenv("XRAY_SCAN_COST_USD", "8.50"))
        self.replacement_cost = float(os.getenv("REPLACEMENT_COST_USD", "45.0"))
        self.px_to_mm_ratio = float(os.getenv("PX_TO_MM_RATIO", "0.1"))

        # --- CNN Model Settings ---
        self.cnn_input_size = int(os.getenv("CNN_INPUT_SIZE", "224"))
        self.use_gpu = os.getenv("USE_GPU", "auto").lower()
        self.mock_mode = os.getenv("MOCK_MODE", "true").lower() in {"1", "true", "yes"}

        # --- Dataset names ---
        self.rgb_dataset_name = "VisA PCB subsets (PCB1-PCB4)"
        self.xray_dataset_name = "dezoe/pcb-defect-multi-modal-dataset (X-ray records)"

    def ensure_directories(self) -> None:
        for p in (self.storage_dir, self.image_storage_dir, self.report_storage_dir):
            p.mkdir(parents=True, exist_ok=True)
        self.rgb_models_dir.mkdir(parents=True, exist_ok=True)
        self.xray_models_dir.mkdir(parents=True, exist_ok=True)

    def is_allowed_origin(self, origin: str) -> bool:
        if self.cors_allow_all:
            return True
        return origin in self.cors_origins


settings = Settings()
settings.ensure_directories()
