"""
Prometheus Metrics Collector for Industrial Quality Control.
Tracks model inference latency, edge FPS throughput, camera uptime,
defect counts by class, and PLC actuator trigger statistics.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import NEU_CLASSES

# 1. Latency Histogram (Buckets configured for sub-100ms real-time edge CV)
INFERENCE_LATENCY = Histogram(
    "defect_detection_inference_latency_seconds",
    "Model inference latency per frame in seconds",
    buckets=[0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.075, 0.100, 0.200, 0.500],
)

# 2. Frame Throughput Gauge
FPS_GAUGE = Gauge(
    "defect_detection_fps",
    "Real-time processing throughput in frames per second (FPS)",
)

# 3. Camera Status & Uptime Gauges
CAMERA_STATUS = Gauge(
    "defect_detection_camera_status",
    "Assembly line camera health status (1 = Active / Online, 0 = Disconnected)",
)

CAMERA_UPTIME = Gauge(
    "defect_detection_camera_uptime_seconds",
    "Cumulative continuous uptime of industrial camera feed in seconds",
)

# 4. Total Parts Inspected & Rejected Counters
PARTS_INSPECTED = Counter(
    "defect_detection_parts_inspected_total",
    "Total manufactured parts scanned by the vision system",
)

PARTS_REJECTED = Counter(
    "defect_detection_parts_rejected_total",
    "Total defective parts flagged for automated rejection",
)

# 5. Defect Breakdown Counter by Class
DEFECTS_DETECTED = Counter(
    "defect_detection_defects_total",
    "Total microscopic surface defects detected",
    ["defect_type"],
)

# 6. PLC Actuator Triggers Counter
PLC_TRIGGERS = Counter(
    "defect_detection_plc_triggers_total",
    "Total diverter actuator pulses transmitted to external PLC",
)


class MetricsManager:
    """Manages telemetry recording and Prometheus export."""

    def __init__(self):
        self.start_time = time.time()
        self.system_online = True
        self.parts_scanned = 0
        self.parts_rejected = 0
        self.last_latency_ms = 0.0
        self.last_fps = 0.0
        CAMERA_STATUS.set(1)

    def record_inference(self, latency_seconds: float, fps: float):
        """Records inference latency and current FPS."""
        self.last_latency_ms = round(latency_seconds * 1000.0, 2)
        self.last_fps = round(fps, 1)
        INFERENCE_LATENCY.observe(latency_seconds)
        FPS_GAUGE.set(fps)
        CAMERA_UPTIME.set(time.time() - self.start_time)

    def record_part_scan(self, is_defective: bool, rejected: bool, detections: list):
        """Updates inspection counters, defect classes, and rejection stats."""
        self.parts_scanned += 1
        PARTS_INSPECTED.inc()

        if rejected:
            self.parts_rejected += 1
            PARTS_REJECTED.inc()
            PLC_TRIGGERS.inc()

        for det in detections:
            cname = det.get("class_name", "unknown")
            DEFECTS_DETECTED.labels(defect_type=cname).inc()

    def set_camera_status(self, is_online: bool):
        """Updates camera connectivity gauge."""
        self.camera_online = is_online
        CAMERA_STATUS.set(1 if is_online else 0)

    def export_metrics(self) -> bytes:
        """Returns Prometheus exposition formatted text."""
        CAMERA_UPTIME.set(time.time() - self.start_time)
        return generate_latest(REGISTRY)


# Global singleton instance
metrics_manager = MetricsManager()


if __name__ == "__main__":
    metrics_manager.record_inference(0.018, 55.5)
    metrics_manager.record_part_scan(True, True, [{"class_name": "crazing"}])
    raw_metrics = metrics_manager.export_metrics().decode("utf-8")
    print("Prometheus Metrics Sample:")
    for line in raw_metrics.splitlines()[:20]:
        print(line)
