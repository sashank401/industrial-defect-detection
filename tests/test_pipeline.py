"""
Comprehensive Test Suite for Real-Time Industrial Defect Detection System.
Tests preprocessing, augmentation, evaluation metrics, ONNX inference,
PLC client dispatches, Prometheus telemetry, and FastAPI endpoints.
"""

import sys
import io
from pathlib import Path
import pytest
import numpy as np
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_image
from src.augmentation import get_industrial_augmentation_pipeline, apply_augmentation
from src.evaluator import compute_iou, compute_ap, MetricEvaluator
from src.inference_engine import DefectInferenceEngine
from src.plc_client import PLCControllerClient
from src.metrics import MetricsManager
from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture
def sample_metal_image():
    """Generates a synthetic metallic strip image for test fixtures."""
    img = np.full((200, 200, 3), 160, dtype=np.uint8)
    # Add scratch defect
    cv2.line(img, (30, 30), (170, 70), (40, 40, 40), 2)
    return img


def test_preprocessing(sample_metal_image):
    """Verifies classical CV preprocessing pipeline."""
    gray, blurred, enhanced = preprocess_image(sample_metal_image)

    assert gray.shape == (200, 200)
    assert blurred.shape == (200, 200)
    assert enhanced.shape == (200, 200)
    assert enhanced.dtype == np.uint8
    assert np.max(enhanced) > np.min(enhanced)


def test_augmentation(sample_metal_image):
    """Verifies Albumentations preserves bounding boxes within [0, 1]."""
    bboxes = [[0.5, 0.5, 0.2, 0.3]]
    labels = [0]
    pipeline = get_industrial_augmentation_pipeline()

    aug_img, aug_boxes, aug_labels = apply_augmentation(sample_metal_image, bboxes, labels, pipeline)

    assert aug_img.shape == sample_metal_image.shape
    if len(aug_boxes) > 0:
        for box in aug_boxes:
            xc, yc, w, h = box
            assert 0.0 <= xc <= 1.0
            assert 0.0 <= yc <= 1.0
            assert 0.0 < w <= 1.0
            assert 0.0 < h <= 1.0


def test_iou_and_evaluator():
    """Validates IoU geometric calculation and mAP evaluator."""
    box_a = np.array([10, 10, 50, 50])
    box_b = np.array([10, 10, 50, 50])
    box_c = np.array([100, 100, 150, 150])

    assert compute_iou(box_a, box_b) == pytest.approx(1.0)
    assert compute_iou(box_a, box_c) == pytest.approx(0.0)

    # Evaluator verification
    evaluator = MetricEvaluator()
    gts = [{"image_id": "1", "boxes": [box_a], "labels": [0]}]
    preds = [{"image_id": "1", "boxes": [box_a], "scores": [0.95], "labels": [0]}]
    summary = evaluator.evaluate_detections(gts, preds)

    assert "mAP@0.5" in summary
    assert "confusion_matrix" in summary
    assert summary["mAP@0.5"] > 0.0


def test_inference_engine(sample_metal_image):
    """Verifies edge inference engine output schema and quality decision."""
    engine = DefectInferenceEngine()
    result = engine.predict(sample_metal_image)

    assert "is_defective" in result
    assert "action" in result
    assert result["action"] in ["ACCEPT", "REJECT"]
    assert "latency_ms" in result
    assert "fps" in result
    assert "backend" in result
    assert isinstance(result["detections"], list)


def test_plc_client():
    """Verifies PLC dispatch packet structure and logging."""
    client = PLCControllerClient(enabled=True)
    res = client.send_reject_signal(
        defect_class="crazing",
        coordinates={"x": 50, "y": 50, "width": 30, "height": 30},
        severity=0.85,
        confidence=0.92,
    )

    assert res["signal"] == "REJECT_ACTUATOR_PULSE"
    assert res["defect_class"] == "crazing"
    assert res["severity"] == 0.85
    assert "part_id" in res
    assert "timestamp" in res


def test_prometheus_metrics():
    """Verifies telemetry metrics recording and Prometheus format exposition."""
    mgr = MetricsManager()
    mgr.record_inference(0.015, 66.6)
    mgr.record_part_scan(True, True, [{"class_name": "inclusion"}])

    raw = mgr.export_metrics().decode("utf-8")
    assert "defect_detection_fps" in raw
    assert "defect_detection_camera_uptime_seconds" in raw
    assert "defect_detection_parts_inspected_total" in raw


def test_fastapi_endpoints(sample_metal_image):
    """Verifies REST API endpoints via TestClient."""
    with TestClient(app) as client:
        # 1. Health check
        res_health = client.get("/api/v1/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "healthy"

        # 2. Prometheus metrics
        res_metrics = client.get("/metrics")
        assert res_metrics.status_code == 200
        assert "defect_detection" in res_metrics.text

        # 3. Image inspection endpoint
        _, encoded = cv2.imencode(".jpg", sample_metal_image)
        files = {"file": ("test.jpg", io.BytesIO(encoded.tobytes()), "image/jpeg")}
        res_detect = client.post("/api/v1/detect", files=files)
        assert res_detect.status_code == 200
        data = res_detect.json()
        assert "action" in data
        assert "latency_ms" in data

        # 4. Manual PLC trigger endpoint
        res_plc = client.post("/api/v1/plc/trigger", json={
            "defect_class": "test_defect",
            "severity": 0.88,
            "confidence": 0.95
        })
        assert res_plc.status_code == 200
        assert res_plc.json()["status"] == "SUCCESS"
