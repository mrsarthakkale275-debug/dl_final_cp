"""Dataset and provenance status service."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.parquet as pq

from backend.app.core.config import settings
from backend.app.core.logging_config import get_logger
from backend.app.schemas.models import DatasetStatusResponse, VisaDatasetStatus, XRayDatasetStatus
from backend.app.services.model_registry import registry as model_registry

logger = get_logger(__name__)


class DatasetService:
    def visa_status(self) -> VisaDatasetStatus:
        visa_dir = settings.data_dir / "raw" / "visa_parquet" / "data"
        files = sorted(visa_dir.glob("*.parquet")) if visa_dir.exists() else []
        basenames = [p.name for p in files]

        train_records = 0
        test_records = 0
        normal_train_records = 0
        for path in files:
            try:
                pf = pq.ParquetFile(path)
                num_rows = pf.metadata.num_rows
            except Exception as exc:
                logger.warning("Could not read parquet metadata %s: %s", path, exc)
                continue
            name = path.name.lower()
            if "train" in name:
                train_records += num_rows
                # Training contains normal images; count via default known structure.
                # We avoid loading images; train splits in VisA PCB are normal-only.
                normal_train_records += num_rows
            elif "test" in name:
                test_records += num_rows

        return VisaDatasetStatus(
            name=settings.rgb_dataset_name,
            parquet_files=basenames,
            train_records=train_records,
            test_records=test_records,
            normal_train_records=normal_train_records,
            total_records=train_records + test_records,
        )

    def xray_status(self) -> XRayDatasetStatus:
        manifest_path = settings.data_dir / "processed" / "xray" / "xray_manifest.csv"
        images_dir = settings.data_dir / "processed" / "xray" / "images"
        prepared_images = len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0

        xray_records = 0
        labels: Dict[str, int] = {}
        rights: Dict[str, int] = {}
        excluded = 0
        cleared = 0
        source_parquet_files = 0
        raw_xray_dir = settings.data_dir / "raw" / "xray" / "hf_data"
        if raw_xray_dir.exists():
            source_parquet_files = len(list(raw_xray_dir.glob("*.parquet")))

        if manifest_path.exists():
            try:
                with manifest_path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        xray_records += 1
                        label = (row.get("label") or "unknown").strip()
                        labels[label] = labels.get(label, 0) + 1
                        status = (row.get("rights_status") or "unknown").strip()
                        rights[status] = rights.get(status, 0) + 1
                        if status == "permission_not_established":
                            excluded += 1
                        else:
                            cleared += 1
            except Exception as exc:
                logger.warning("Could not read xray manifest: %s", exc)

        # If manifest missing, fall back to counting raw parquet xray records via metadata only.
        if xray_records == 0 and source_parquet_files:
            try:
                import pyarrow.parquet as pq2

                for path in sorted(raw_xray_dir.glob("*.parquet")):
                    pf = pq2.ParquetFile(path)
                    table = pf.read_table(columns=["modality", "primary_label", "rights_status"])
                    rows = table.to_pylist()
                    for row in rows:
                        if row.get("modality") != "xray":
                            continue
                        xray_records += 1
                        label = row.get("primary_label") or "unknown"
                        labels[label] = labels.get(label, 0) + 1
                        status = row.get("rights_status") or "unknown"
                        rights[status] = rights.get(status, 0) + 1
                        if status == "permission_not_established":
                            excluded += 1
                        else:
                            cleared += 1
            except Exception as exc:
                logger.warning("Could not read xray parquet metadata: %s", exc)

        return XRayDatasetStatus(
            name=settings.xray_dataset_name,
            source_parquet_files=source_parquet_files,
            xray_records=xray_records,
            prepared_images=prepared_images,
            labels=labels,
            rights_status=rights,
            permission_cleared_records=cleared,
            excluded_records=excluded,
        )

    def status(self) -> DatasetStatusResponse:
        rgb = self.visa_status()
        xray = self.xray_status()
        notes = [
            "RGB and X-ray datasets are not paired images of the same physical PCB.",
            "Records with permission_not_established are excluded from model training and distribution.",
            "Dataset status is derived from Parquet metadata and manifests without loading image bytes.",
            "X-ray prepared records contain 601 void and 1 solder_ball label; no defensible X-ray model is trained.",
            "X-ray API requests therefore use a review-required workflow instead of fabricated predictions.",
        ]
        rgb_trained = len(model_registry.rgb.weights) > 0
        xray_trained = len(model_registry.xray.weights) > 0
        return DatasetStatusResponse(
            rgb=rgb,
            xray=xray,
            model_trained_from_rgb=rgb_trained,
            model_trained_from_xray=xray_trained,
            notes=notes,
        )
