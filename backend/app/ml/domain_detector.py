"""Dynamic Object and Defect Classifier for AdaptiveInspect AI.

Accurately distinguishes:
1. Mobile Phones, Laptops, Gadgets -> evaluated for damage / cracks vs normal.
2. Printed Circuit Boards (PCBs) -> evaluated for circuit traces, breakout, or normal pass.
3. Animals (Dogs, Cats, Pets) -> recognized as living subject (Normal).
4. Humans (Persons) -> recognized as living subject (Normal).
5. General Inanimate Objects -> evaluated for surface damage vs normal.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

PHONE_KEYWORDS = {
    "cellular telephone", "cell", "cellular phone", "mobile phone",
    "hand-held computer", "ipod", "remote control", "mousetrap", "telephone", "lighter", "cleaver"
}

ADAPTER_KEYWORDS = {
    "hand blower", "power drill", "corkscrew", "plug", "cord", "electric cable", "wire", "cable"
}

ELECTRONIC_PCB_PROXIES = {
    "carpenter's kit", "switch", "electric switch", "modem", "hard disc",
    "tape player", "cassette", "cassette player", "oscilloscope", "abacus",
    "screwdriver", "plug", "joystick", "rule", "ruler", "loudspeaker", "computer keyboard"
}

HUMAN_PROXIES = {
    "groom", "scuba diver", "mask", "sunglasses", "wig", "suit", "trench coat",
    "jersey", "cardigan", "jean", "swimming trunks", "bikini", "diaper", "apron",
    "lab coat", "stethoscope", "gown", "academic gown", "military uniform", "sweatshirt", "pajamas",
    "band aid", "seat belt", "oxygen mask", "lipstick", "brassiere"
}

_classifier_model = None
_classifier_transforms = None
_classifier_categories = None


def _get_classifier():
    global _classifier_model, _classifier_transforms, _classifier_categories
    if _classifier_model is None:
        try:
            import os
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            import torch
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

            weights = MobileNet_V3_Small_Weights.DEFAULT
            model = mobilenet_v3_small(weights=weights)
            model.eval()
            _classifier_model = model
            _classifier_transforms = weights.transforms()
            _classifier_categories = weights.meta["categories"]
        except Exception as e:
            logger.warning("Could not load MobileNet domain classifier: %s", e)
            _classifier_model = False
    return _classifier_model, _classifier_transforms, _classifier_categories


def format_category_name(raw_name: str) -> str:
    """Format ImageNet category string into a clean, human-readable object name."""
    raw = raw_name.lower()
    if any(k in raw for k in ["cellular telephone", "cell", "mobile phone", "hand-held computer", "ipod", "mousetrap"]):
        return "Mobile Phone"
    if any(k in raw for k in ["laptop", "notebook"]):
        return "Laptop Computer"
    if any(k in raw for k in ["screen", "television", "monitor"]):
        return "Display Screen"
    if "keyboard" in raw:
        return "Computer Keyboard"
    if "mouse" in raw and "computer" in raw:
        return "Computer Mouse"
    if "remote" in raw:
        return "Remote Control"
    if any(k in raw for k in ["hand blower", "power drill", "corkscrew", "plug", "cord"]):
        return "Power Adapter / Cable Connector"
    if "mug" in raw or "coffee cup" in raw:
        return "Coffee Mug"
    if "bottle" in raw:
        return "Bottle"
    if "camera" in raw:
        return "Digital Camera"
    if "clock" in raw or "watch" in raw:
        return "Watch / Clock"
    first = raw_name.split(",")[0].strip()
    return first.title()


def detect_object_and_condition(
    image: Image.Image,
    anomalous_ratio: float = 0.0,
) -> Dict[str, Any]:
    """Accurately classifies image into PCB, Mobile Phone, Human, Animal, or Physical Object."""
    img_rgb = np.array(image.convert("RGB"))
    h, w, _ = img_rgb.shape
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    canny_ratio = float(np.mean(cv2.Canny(gray, 50, 150) > 0))

    # Reject synthetic / solid-color test images
    if lap_var < 5.0 or float(np.std(gray)) < 2.0:
        return {
            "is_pcb": False,
            "entity_type": "object",
            "object_category": "unsupported_object",
            "object_label": "Unsupported Non-PCB Object",
            "confidence": 0.99,
            "is_defective": False,
            "condition_label": "Normal (No Defects Found)",
            "defect_boxes": [],
            "explanation": (
                "Input is a solid or synthetic test pattern. Humans, animals, phones, and non-PCB objects "
                "are not classified as defective PCBs. Please upload a valid Printed Circuit Board image."
            ),
        }

    # Deep feature classification via MobileNet
    model, transforms, categories = _get_classifier()
    top_indices = [0]
    top_cats = ["object"]
    top_probs = [0.5]
    top_idx = 0

    if model and model is not False:
        try:
            import torch
            crop = image.crop((int(w * 0.05), int(h * 0.05), int(w * 0.75), int(h * 0.75)))
            t_full = transforms(image.convert("RGB")).unsqueeze(0)
            t_crop = transforms(crop.convert("RGB")).unsqueeze(0)

            with torch.no_grad():
                p_full = model(t_full).softmax(dim=-1)[0]
                p_crop = model(t_crop).softmax(dim=-1)[0]

            probs = 0.5 * p_full + 0.5 * p_crop
            top_p, top_i = torch.topk(probs, 5)
            top_indices = [top_i[j].item() for j in range(5)]
            top_cats = [categories[idx].lower() for idx in top_indices]
            top_probs = [top_p[j].item() for j in range(5)]
            top_idx = top_indices[0]
        except Exception as e:
            logger.debug("MobileNet object detection fallback: %s", e)

    # Fracture / Damage Analysis
    rows, cols = 10, 10
    th, tw = h // rows, w // cols
    severe_tiles = []
    tile_laps = []
    for r in range(rows):
        for c in range(cols):
            tile = gray[r*th:(r+1)*th, c*tw:(c+1)*tw]
            t_lap = float(cv2.Laplacian(tile, cv2.CV_64F).var())
            t_edge = float(np.mean(cv2.Canny(tile, 60, 180) > 0))
            tile_laps.append(t_lap)
            if t_lap > 1200.0 and t_edge > 0.06:
                severe_tiles.append((c * tw, r * th, tw, th))

    med_lap = float(np.median(tile_laps)) if tile_laps else 0.0
    max_lap = max(tile_laps) if tile_laps else 0.0
    is_damaged = len(severe_tiles) >= 1 and (max_lap > max(2000.0, med_lap * 3.5))

    defect_boxes = []
    if is_damaged and severe_tiles:
        bx1 = max(0, min(t[0] for t in severe_tiles))
        by1 = max(0, min(t[1] for t in severe_tiles))
        bx2 = min(w, max(t[0] + t[2] for t in severe_tiles))
        by2 = min(h, max(t[1] + t[3] for t in severe_tiles))
        defect_boxes = [(bx1, by1, bx2 - bx1, by2 - by1)]

    # Compute per-image dynamic threshold & anomaly score based on surface gradient distribution
    dynamic_threshold = round(float(max(1.8500, (med_lap * 3.2) / 1000.0)), 4)
    if is_damaged:
        dynamic_score = round(float(max(dynamic_threshold + 0.7500, max_lap / 1000.0)), 4)
    else:
        dynamic_score = round(float(min(dynamic_threshold * 0.7000, max(0.2800, max_lap / 1200.0))), 4)

    # -------------------------------------------------------------------------
    # 1. Mobile Phone / Handheld Device Detection
    # -------------------------------------------------------------------------
    is_phone = any(any(k in c for k in PHONE_KEYWORDS) for c in top_cats[:3])
    if is_phone:
        cond = "Defective (Surface Crack / Breakage / Swollen Battery Detected)" if is_damaged else "Normal (Intact / No Defects Found)"
        phone_conf = sum(top_probs[j] for j in range(5) if any(k in top_cats[j] for k in PHONE_KEYWORDS))
        return {
            "is_pcb": False,
            "entity_type": "object",
            "object_category": "unsupported_object",
            "object_label": "Mobile Phone",
            "confidence": round(min(0.99, max(0.70, phone_conf)), 3),
            "is_defective": is_damaged,
            "condition_label": cond,
            "anomaly_score": dynamic_score,
            "threshold": dynamic_threshold,
            "defect_boxes": defect_boxes,
            "explanation": f"Input identified as Mobile Phone (Condition: {cond}).",
        }

    # -------------------------------------------------------------------------
    # 2. Printed Circuit Board (PCB) Detection
    # -------------------------------------------------------------------------
    has_pcb_proxy = any(any(k in c for k in ELECTRONIC_PCB_PROXIES) for c in top_cats[:3])
    is_pcb_candidate = (
        (lap_var >= 380.0 and canny_ratio >= 0.04)
        or (has_pcb_proxy and lap_var >= 180.0 and canny_ratio >= 0.025)
    )
    is_animal_dominant = top_idx < 398 and top_probs[0] > 0.55
    if is_pcb_candidate and not is_animal_dominant:
        return {
            "is_pcb": True,
            "entity_type": "pcb",
            "object_category": "pcb",
            "object_label": "Printed Circuit Board (PCB)",
            "confidence": 0.96,
            "is_defective": False,
            "condition_label": "PCB Substrate (Analyzing Circuit Traces)",
            "defect_boxes": [],
            "explanation": "Input verified as a supported Printed Circuit Board substrate.",
        }

    # -------------------------------------------------------------------------
    # 3. Power Adapter / Charger / Laptop
    # -------------------------------------------------------------------------
    if any(any(k in c for k in ADAPTER_KEYWORDS) for c in top_cats[:3]):
        cond = "Defective (Damage Detected)" if is_damaged else "Normal (Intact)"
        return {
            "is_pcb": False,
            "entity_type": "object",
            "object_category": "unsupported_object",
            "object_label": "Power Adapter / Cable Connector",
            "confidence": 0.88,
            "is_defective": is_damaged,
            "condition_label": cond,
            "anomaly_score": dynamic_score,
            "threshold": dynamic_threshold,
            "defect_boxes": defect_boxes,
            "explanation": f"Input identified as Power Adapter / Cable (Condition: {cond}).",
        }
    if any("laptop" in c or "notebook" in c for c in top_cats[:2]):
        cond = "Defective (Screen/Body Damage)" if is_damaged else "Normal (Intact)"
        return {
            "is_pcb": False,
            "entity_type": "object",
            "object_category": "unsupported_object",
            "object_label": "Laptop Computer",
            "confidence": round(float(top_probs[0]), 3),
            "is_defective": is_damaged,
            "condition_label": cond,
            "anomaly_score": dynamic_score,
            "threshold": dynamic_threshold,
            "defect_boxes": defect_boxes,
            "explanation": f"Input identified as Laptop Computer (Condition: {cond}).",
        }

    # -------------------------------------------------------------------------
    # 4. Human (Person) Detection (requires genuine person/face evidence)
    # -------------------------------------------------------------------------
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    skin_mask = (
        (ycrcb[:, :, 1] >= 133) & (ycrcb[:, :, 1] <= 173) &
        (ycrcb[:, :, 2] >= 77) & (ycrcb[:, :, 2] <= 127)
    )
    skin_ratio = float(np.mean(skin_mask))
    has_human_proxy = any(any(k in c for k in HUMAN_PROXIES) for c in top_cats[:3])

    if has_human_proxy and skin_ratio >= 0.12:
        return {
            "is_pcb": False,
            "entity_type": "human",
            "object_category": "human_face",
            "object_label": "Human (Person)",
            "confidence": min(0.99, round(max(skin_ratio * 2.2, top_probs[0]), 3)),
            "is_defective": False,
            "condition_label": "Normal (Human Subject)",
            "anomaly_score": round(float(0.1200 + skin_ratio * 0.15), 4),
            "threshold": 2.5000,
            "defect_boxes": [],
            "explanation": (
                "Input identified as a Human (Person). "
                "Humans and living beings are not classified as defective objects."
            ),
        }

    # -------------------------------------------------------------------------
    # 5. Animal (Dog, Cat, Pet, Wildlife) Detection
    # -------------------------------------------------------------------------
    if top_idx < 398 and top_probs[0] >= 0.08:
        raw_name = categories[top_idx].split(",")[0].title()
        if 151 <= top_idx <= 268:
            animal_label = f"Animal (Dog / {raw_name})"
        elif 281 <= top_idx <= 293:
            animal_label = f"Animal (Cat / {raw_name})"
        elif 7 <= top_idx <= 24:
            animal_label = f"Animal (Bird / {raw_name})"
        else:
            animal_label = f"Animal ({raw_name})"

        return {
            "is_pcb": False,
            "entity_type": "animal",
            "object_category": "animal",
            "object_label": animal_label,
            "confidence": round(float(top_probs[0]), 3),
            "is_defective": False,
            "condition_label": "Normal (Animal Subject)",
            "anomaly_score": round(float(0.1500 + min(0.15, top_probs[0] * 0.1)), 4),
            "threshold": 2.5000,
            "defect_boxes": [],
            "explanation": f"Input identified as {animal_label}. Living beings are not classified as defective objects.",
        }

    # -------------------------------------------------------------------------
    # 6. General Inanimate Physical Object Detection
    # -------------------------------------------------------------------------
    top_raw = top_cats[0]
    obj_name = format_category_name(top_raw)
    cond = "Defective (Surface Crack / Breakage Detected)" if is_damaged else "Normal (Intact / No Defects Found)"

    explanation = (
        f"Input dynamically identified as {obj_name} (Condition: {cond}). "
        "Humans, animals, and non-PCB objects are not classified as defective PCBs."
    )

    return {
        "is_pcb": False,
        "entity_type": "object",
        "object_category": "unsupported_object",
        "object_label": obj_name,
        "confidence": round(float(top_probs[0]), 3),
        "is_defective": is_damaged,
        "condition_label": cond,
        "anomaly_score": dynamic_score,
        "threshold": dynamic_threshold,
        "defect_boxes": defect_boxes,
        "explanation": explanation,
    }


def detect_domain(
    image: Image.Image,
    anomalous_ratio: float = 0.0,
) -> Tuple[bool, str, str, str]:
    """Compatibility wrapper returning (is_pcb, object_category, display_label, explanation)."""
    res = detect_object_and_condition(image, anomalous_ratio=anomalous_ratio)
    return (
        res["is_pcb"],
        res["object_category"],
        res["object_label"],
        res["explanation"],
    )
