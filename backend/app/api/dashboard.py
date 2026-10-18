"""Dashboard endpoint providing live inspection metrics and categories."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models.inspection import Inspection
from backend.app.services.model_registry import registry

router = APIRouter(tags=["dashboard"])


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


NORMAL_LABELS = {"normal", "ok", "good", "pass"}
DEFECTIVE_LABELS = {"defective", "defect", "anomaly", "abnormal"}


@router.get("/dashboard/summary", summary="Inspection dashboard summary")
def dashboard_summary(db: Session = Depends(get_db)) -> Dict[str, Any]:
    total = db.query(func.count(Inspection.id)).scalar() or 0
    rgb_count = db.query(func.count(Inspection.id)).filter(Inspection.inspection_type == "rgb").scalar() or 0
    xray_count = db.query(func.count(Inspection.id)).filter(Inspection.inspection_type == "xray").scalar() or 0
    adaptive_count = db.query(func.count(Inspection.id)).filter(Inspection.inspection_type == "adaptive").scalar() or 0

    from sqlalchemy.orm import joinedload
    inspections_with_results = (
        db.query(Inspection)
        .options(joinedload(Inspection.results))
        .order_by(Inspection.created_at.desc())
        .limit(100)
        .all()
    )

    normal_count = 0
    defective_count = 0
    pcb_count = 0
    pcb_normal = 0
    pcb_defective = 0
    phone_count = 0
    phone_normal = 0
    phone_defective = 0
    living_count = 0
    other_count = 0
    other_normal = 0
    other_defective = 0

    recent_items = []

    for idx, insp in enumerate(inspections_with_results):
        detected_obj = None
        cond_label = None
        is_def = None
        thresh = None

        for r in insp.results:
            if isinstance(r.payload, dict):
                p = r.payload
                if not detected_obj and p.get("detected_object"):
                    detected_obj = p.get("detected_object")
                if not cond_label and p.get("condition_label"):
                    cond_label = p.get("condition_label")
                if is_def is None and "is_defective" in p:
                    is_def = p.get("is_defective")
                if thresh is None:
                    thresh = p.get("threshold") or (p.get("localization") or {}).get("image_threshold")

        pred_norm = _norm(insp.predicted_class)
        if is_def is True or pred_norm in DEFECTIVE_LABELS:
            is_def_final = True
            defective_count += 1
        else:
            is_def_final = False
            normal_count += 1

        obj_lower = (detected_obj or "").lower()
        if "pcb" in obj_lower or "circuit" in obj_lower or (not detected_obj and insp.inspection_type in {"rgb", "xray", "adaptive"}):
            pcb_count += 1
            if is_def_final:
                pcb_defective += 1
            else:
                pcb_normal += 1
            if not detected_obj:
                detected_obj = "Printed Circuit Board (PCB)"
        elif "phone" in obj_lower or "cellular" in obj_lower:
            phone_count += 1
            if is_def_final:
                phone_defective += 1
            else:
                phone_normal += 1
        elif "human" in obj_lower or "person" in obj_lower or "animal" in obj_lower or "dog" in obj_lower or "cat" in obj_lower:
            living_count += 1
        else:
            other_count += 1
            if is_def_final:
                other_defective += 1
            else:
                other_normal += 1

        if idx < 15:
            recent_items.append({
                "id": insp.id,
                "inspection_type": insp.inspection_type,
                "original_filename": insp.original_filename or f"Inspection-{insp.id[:8]}",
                "predicted_class": "defective" if is_def_final else "normal",
                "status": insp.status,
                "score": insp.score,
                "threshold": thresh or 3.3498,
                "detected_object": detected_obj or "Inspected Entity",
                "condition_label": cond_label or ("Defective" if is_def_final else "Normal"),
                "is_defective": is_def_final,
                "stored_image_reference": insp.stored_image_path,
                "created_at": insp.created_at.isoformat() if insp.created_at else None,
            })

    category_chart = [
        {"name": "PCBs", "total": pcb_count, "pass": pcb_normal, "defective": pcb_defective},
        {"name": "Phones & Devices", "total": phone_count, "pass": phone_normal, "defective": phone_defective},
        {"name": "Living Subjects", "total": living_count, "pass": living_count, "defective": 0},
        {"name": "Other Objects", "total": other_count, "pass": other_normal, "defective": other_defective},
    ]

    return {
        "total_inspections": total,
        "rgb_inspections": rgb_count,
        "xray_inspections": xray_count,
        "adaptive_inspections": adaptive_count,
        "normal_count": normal_count,
        "defective_count": defective_count,
        "pcb_count": pcb_count,
        "phone_count": phone_count,
        "living_count": living_count,
        "other_count": other_count,
        "category_chart": category_chart,
        "model_availability": {
            "rgb": registry.rgb.status,
            "xray": registry.xray.status,
            "rgb_ready": registry.rgb.ready,
            "xray_ready": registry.xray.ready,
        },
        "recent_inspections": recent_items,
    }
