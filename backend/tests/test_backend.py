import io
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def tiny_png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (180, 120, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


def _pcb_png_bytes(label: int) -> bytes:
    """Return a downscaled real VisA PCB image (label 0 = normal, 1 = defective) as PNG bytes."""
    images = np.load(PROJECT_ROOT / "data" / "processed" / "rgb" / "test_images.npy")
    labels = np.load(PROJECT_ROOT / "data" / "processed" / "rgb" / "test_labels.npy")
    idx = int(np.where(labels == label)[0][0])
    buffer = io.BytesIO()
    Image.fromarray(images[idx]).resize((16, 16), Image.BILINEAR).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_and_model_status(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["api"] == "available"
    assert body["database"] == "available"
    assert body["status"] == "healthy"

    status = client.get("/api/model-status").json()
    assert status["xray"]["ready"] is False
    assert status["xray"]["status"] == "untrained"
    assert status["rgb"]["weights"] == [] or "anomaly_model.npz" in status["rgb"]["weights"]


def test_dataset_status_exposes_counts_and_provenance(client):
    response = client.get("/api/datasets/status")
    assert response.status_code == 200
    body = response.json()
    assert body["rgb"]["train_records"] == 3614
    assert body["rgb"]["test_records"] == 802
    assert body["xray"]["xray_records"] == 602
    assert body["xray"]["prepared_images"] == 602
    assert body["xray"]["excluded_records"] == 2
    assert any("paired" in note.lower() for note in body["notes"])


def test_invalid_empty_and_corrupt_uploads(client):
    empty = client.post("/api/inspect/rgb", files={"file": ("empty.png", b"", "image/png")})
    assert empty.status_code == 422
    assert "empty" in empty.json()["error"]["message"].lower()

    text = client.post("/api/inspect/rgb", files={"file": ("note.txt", b"hello", "text/plain")})
    assert text.status_code == 422

    corrupt = client.post("/api/inspect/rgb", files={"file": ("bad.png", b"not-an-image", "image/png")})
    assert corrupt.status_code == 422

    xray = client.post("/api/inspect/xray", files={"file": ("bad.png", b"not-an-image", "image/png")})
    assert xray.status_code == 422


def test_upload_size_limit(client):
    oversize = b"a" * 2048
    response = client.post("/api/inspect/xray", files={"file": ("big.png", oversize, "image/png")})
    assert response.status_code == 413


def test_xray_unavailable_backend_behavior(client):
    response = client.post("/api/inspect/xray", files={"file": ("tiny.png", tiny_png_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["inspection_type"] == "xray"
    assert body["status"] == "model_unavailable"
    assert body["predicted_class"] is None
    assert body["score"] is None
    assert body["xray"]["status"] == "model_unavailable"
    assert "x-ray" in body["error_details"].lower() or "xray" in body["error_details"].lower()


def test_rgb_integration_with_existing_model(rgb_client):
    # A real supported normal PCB completes as a normal inspection with localization.
    response = rgb_client.post(
        "/api/inspect/rgb", files={"file": ("pcb.png", _pcb_png_bytes(0), "image/png")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["inspection_type"] == "rgb"
    assert body["model_id"] == "rgb-pixel-mahalanobis-anomaly"
    assert body["predicted_class"] == "normal"
    assert body["score"] is not None
    assert body["confidence"] is None
    assert body["rgb"]["localization"] is not None
    assert body["rgb"]["localization"]["confidence_supported"] is False
    assert body["rgb"]["eligibility"] == "supported"
    assert body["rgb"]["decision_outcome"] == "pass"

    # A non-PCB solid-color image must not be classified as a defective PCB.
    response2 = rgb_client.post("/api/inspect/rgb", files={"file": ("tiny.png", tiny_png_bytes(), "image/png")})
    assert response2.status_code == 200
    body2 = response2.json()
    assert body2["status"] == "unsupported_object"
    assert body2["predicted_class"] is None
    assert body2["rgb"]["eligibility"] == "unsupported_object"
    assert body2["rgb"]["decision_outcome"] == "review_required"
    assert "not classified as defective" in body2["rgb"]["decision_reason"]


def test_inspection_history_detail_and_delete(client):
    submit = client.post("/api/inspect/xray", files={"file": ("tiny.png", tiny_png_bytes(), "image/png")})
    assert submit.status_code == 200
    inspection_id = submit.json()["id"]

    listed = client.get("/api/inspections").json()
    assert listed["total"] == 1
    assert listed["items"][0]["id"] == inspection_id
    assert listed["items"][0]["inspection_type"] == "xray"

    detail = client.get(f"/api/inspections/{inspection_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == inspection_id

    deleted = client.delete(f"/api/inspections/{inspection_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/inspections/{inspection_id}").status_code == 404


def test_dashboard_and_report(client):
    submit = client.post(
        "/api/inspect/adaptive", files={"file": ("pcb.png", _pcb_png_bytes(0), "image/png")}
    )
    assert submit.status_code == 200
    inspection_id = submit.json()["id"]

    dashboard = client.get("/api/dashboard/summary").json()
    assert dashboard["total_inspections"] == 1
    assert dashboard["adaptive_inspections"] == 1
    assert dashboard["rgb_inspections"] == 0
    assert dashboard["xray_inspections"] == 0

    report_json = client.get(f"/api/inspections/{inspection_id}/report")
    assert report_json.status_code == 200
    report = report_json.json()["report"]
    assert report["inspection_id"] == inspection_id
    assert report["inspection_type"] == "adaptive"
    assert report["ground_truth"]["available"] is False
    assert report["prediction"]["predicted_class"] is not None

    report_md = client.get(f"/api/inspections/{inspection_id}/report?format=markdown")
    assert report_md.status_code == 200
    assert report_md.headers["content-type"].startswith("text/markdown")
    assert report_md.text.startswith("# Inspection Report")


def test_adaptive_policy_edge_cases():
    from backend.app.services.adaptive_decision import AdaptiveDecisionService

    service = AdaptiveDecisionService(rgb_confidence_threshold=0.8, rgb_uncertainty_threshold=0.5)

    unavailable = service.evaluate(None, None)
    assert unavailable.recommendation == "human_review_required"
    assert unavailable.requires_human_review is True

    reliable_rgb = service.evaluate(
        {
            "predicted_class": "normal",
            "anomaly_score": 1.0,
            "confidence": 0.9,
            "localization": {"image_threshold": 3.0},
            "status": "completed",
        },
        None,
    )
    assert reliable_rgb.recommendation == "rgb_only_result"
    assert reliable_rgb.requires_human_review is False

    rgb_defect = service.evaluate({"predicted_class": "defective", "anomaly_score": 4.0, "status": "completed"}, None)
    assert rgb_defect.recommendation == "request_xray_inspection"


def test_missing_model_weights_startup_resilience(client, tmp_path, monkeypatch):
    from pathlib import Path

    import backend.app.core.config as config_module
    import backend.app.services.model_registry as registry_module

    empty_rgb = tmp_path / "empty_rgb"
    empty_rgb.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("RGB_MODELS_DIR", str(empty_rgb))
    config_module.settings.rgb_models_dir = empty_rgb
    config_module.settings.rgb_models_dir.mkdir(parents=True, exist_ok=True)
    registry_module.registry.rgb.weights = []
    registry_module.registry.rgb.ready = False
    registry_module.registry.rgb.status = "untrained"
    registry_module.registry.rgb.model_id = None
    registry_module.registry.rgb_engine = None

    health = client.get("/api/health")
    assert health.status_code == 200

    status = client.get("/api/model-status").json()
    assert status["rgb"]["ready"] is False
    assert status["rgb"]["status"] == "untrained"

    submit = client.post("/api/inspect/rgb", files={"file": ("tiny.png", tiny_png_bytes(), "image/png")})
    assert submit.status_code == 200
    assert submit.json()["status"] == "model_unavailable"

