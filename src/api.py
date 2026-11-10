"""
Production FastAPI Edge Service for Industrial Defect Detection.
Provides REST APIs for real-time defect inspection, batch quality control,
PLC actuator triggers, and Prometheus telemetry metrics.
(No automatic webcam capture - zero hardware camera interference).
"""

import sys
import time
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import model_cfg, plc_cfg, STATIC_DIR, DATA_DIR, NEU_CLASSES
from src.inference_engine import DefectInferenceEngine
from src.plc_client import plc_client
from src.metrics import metrics_manager


# Global inference engine instance
engine = DefectInferenceEngine()

app = FastAPI(
    title="Real-Time Industrial Defect Detection Edge Service",
    description="Quality control CV inference service interfacing with assembly line PLCs and Prometheus.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PLCTriggerRequest(BaseModel):
    defect_class: str = "manual_inspection_override"
    severity: float = 0.90
    confidence: float = 1.00
    part_id: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the interactive industrial operator dashboard."""
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        with open(html_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Industrial Defect Detection Dashboard Loading...</h2>")


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus telemetry exposition endpoint for Grafana scraping."""
    return Response(
        content=metrics_manager.export_metrics(),
        media_type="text/plain; version=0.0.4",
    )


@app.get("/api/v1/health")
async def health_check():
    """Returns edge system health, model status, and runtime performance stats."""
    return {
        "status": "healthy",
        "system_online": True,
        "backend": engine.backend,
        "fps": metrics_manager.last_fps,
        "latency_ms": metrics_manager.last_latency_ms,
        "parts_scanned": metrics_manager.parts_scanned,
        "parts_rejected": metrics_manager.parts_rejected,
        "plc_enabled": plc_client.enabled,
    }


@app.get("/api/v1/samples")
async def list_benchmark_samples():
    """Returns available NEU defect benchmark sample names."""
    samples_dir = DATA_DIR / "test_samples"
    available = []
    for cls in NEU_CLASSES:
        target = samples_dir / f"sample_{cls}.jpg"
        if target.exists():
            available.append({"defect_class": cls, "filename": target.name})
    return {"classes": NEU_CLASSES, "samples": available}


@app.post("/api/v1/inspect-sample")
async def inspect_benchmark_sample(defect_class: str = Query(...)):
    """Runs immediate inspection on a standard NEU benchmark sample."""
    sample_file = DATA_DIR / "test_samples" / f"sample_{defect_class}.jpg"
    if not sample_file.exists():
        # Fallback to test image
        sample_file = DATA_DIR / "test_image.png.png"

    image = cv2.imread(str(sample_file))
    if image is None:
        raise HTTPException(status_code=404, detail=f"Sample for {defect_class} not found.")

    result = engine.predict(image)
    annotated = engine.annotate(image, result)

    _, encoded_jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_str = base64.b64encode(encoded_jpg.tobytes()).decode("utf-8")
    result["annotated_image_base64"] = b64_str
    result["source"] = f"benchmark_sample_{defect_class}"

    # Telemetry
    rejected = result["action"] == "REJECT"
    metrics_manager.record_inference(result["latency_ms"] / 1000.0, result["fps"])
    metrics_manager.record_part_scan(result["is_defective"], rejected, result["detections"])

    # PLC trigger
    if rejected and plc_client.enabled and len(result["detections"]) > 0:
        top_defect = result["detections"][0]
        plc_res = plc_client.send_reject_signal(
            defect_class=top_defect["class_name"],
            coordinates={
                "x": top_defect["bbox"][0],
                "y": top_defect["bbox"][1],
                "width": top_defect["width"],
                "height": top_defect["height"],
            },
            severity=top_defect["severity"],
            confidence=top_defect["confidence"],
        )
        result["plc_dispatch"] = plc_res

    return result


@app.post("/api/v1/detect")
async def detect_single_image(file: UploadFile = File(...)):
    """
    Inspects an uploaded image file for microscopic surface defects,
    computes sorting verdict (ACCEPT vs REJECT), and broadcasts to PLC if defective.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file or format.")

    # Run inference
    result = engine.predict(image)
    annotated = engine.annotate(image, result)

    # Encode annotated image to base64 JPEG for frontend display
    _, encoded_jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_str = base64.b64encode(encoded_jpg.tobytes()).decode("utf-8")
    result["annotated_image_base64"] = b64_str

    # Update telemetry
    rejected = result["action"] == "REJECT"
    metrics_manager.record_inference(result["latency_ms"] / 1000.0, result["fps"])
    metrics_manager.record_part_scan(result["is_defective"], rejected, result["detections"])

    # Broadcast to PLC if reject triggered
    if rejected and plc_client.enabled and len(result["detections"]) > 0:
        top_defect = result["detections"][0]
        plc_res = plc_client.send_reject_signal(
            defect_class=top_defect["class_name"],
            coordinates={
                "x": top_defect["bbox"][0],
                "y": top_defect["bbox"][1],
                "width": top_defect["width"],
                "height": top_defect["height"],
            },
            severity=top_defect["severity"],
            confidence=top_defect["confidence"],
        )
        result["plc_dispatch"] = plc_res

    return result


@app.post("/api/v1/detect/batch")
async def detect_batch_images(files: List[UploadFile] = File(...)):
    """Performs batch quality inspection over multiple uploaded images."""
    batch_results = []
    total_parts = len(files)
    total_rejects = 0

    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            continue

        res = engine.predict(image)
        if res["action"] == "REJECT":
            total_rejects += 1

        batch_results.append({
            "filename": file.filename,
            "action": res["action"],
            "is_defective": res["is_defective"],
            "detections_count": res["detections_count"],
            "detections": res["detections"],
            "latency_ms": res["latency_ms"],
        })

    return {
        "batch_size": total_parts,
        "total_rejected": total_rejects,
        "rejection_rate_percent": round((total_rejects / total_parts) * 100, 2) if total_parts > 0 else 0.0,
        "results": batch_results,
    }


@app.post("/api/v1/plc/trigger")
async def trigger_manual_plc(req: PLCTriggerRequest):
    """Manually fires a PLC diverter actuator test pulse."""
    dispatch = plc_client.send_reject_signal(
        defect_class=req.defect_class,
        coordinates={"x": 0, "y": 0, "width": 0, "height": 0},
        severity=req.severity,
        confidence=req.confidence,
        part_id=req.part_id,
        force=True,
    )
    return {"status": "SUCCESS", "dispatch": dispatch}


@app.get("/api/v1/plc/status")
async def get_plc_status():
    """Returns connection and history log from the PLC sorting controller."""
    return plc_client.get_status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
