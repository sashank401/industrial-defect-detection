"""
Real-Time Industrial Camera Ingestion & Video Pipeline.
Features threaded frame capture, assembly line metallic strip simulation,
high-FPS ONNX inference, live HUD overlay, and automated PLC reject dispatch.
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional, Generator, Dict, Any, Tuple
import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import video_cfg, plc_cfg, OUTPUT_DIR
from src.inference_engine import DefectInferenceEngine
from src.plc_client import plc_client
from src.metrics import metrics_manager
from data.neu_dataset import NEUDatasetManager


class AssemblyLineSimulator:
    """Simulates a continuous high-speed metallic sheet conveyor feed with periodic defects."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.frame_idx = 0
        self.neu_manager = NEUDatasetManager()
        self.classes = self.neu_manager.classes

    def get_frame(self) -> Tuple[np.ndarray, bool]:
        """Generates the next simulated industrial frame with moving steel strip."""
        self.frame_idx += 1

        # Background metal strip with rolling grain
        base = np.full((self.height, self.width), 160, dtype=np.uint8)
        # Conveyor movement phase
        offset = (self.frame_idx * 12) % self.height
        grain = np.sin(np.linspace(0, 15 * np.pi, self.height) + offset * 0.05) * 15.0
        grain_map = np.tile(grain[:, None], (1, self.width)).astype(np.float32)
        noise = np.random.normal(0, 7, (self.height, self.width))
        strip = np.clip(base + grain_map + noise, 35, 230).astype(np.uint8)

        # Draw steel coil margins / conveyor guide rails
        cv2.line(strip, (60, 0), (60, self.height), 70, 4)
        cv2.line(strip, (self.width - 60, 0), (self.width - 60, self.height), 70, 4)

        has_defect = False
        # Inject defect every ~40 frames
        if (self.frame_idx % 45) in [15, 16, 17, 18, 19]:
            has_defect = True
            defect_type = self.classes[(self.frame_idx // 45) % len(self.classes)]
            # Generate small patch
            pw, ph = 180, 180
            patch, _ = self.neu_manager.generate_synthetic_neu_sample(defect_type, pw, ph)
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

            # Location on strip
            px = int(self.width * 0.45)
            py = int((self.height * 0.25) + ((self.frame_idx % 45) - 15) * 40)
            py = min(py, self.height - ph - 10)

            # Blend defect into metallic strip
            alpha = 0.85
            strip[py:py + ph, px:px + pw] = cv2.addWeighted(
                strip[py:py + ph, px:px + pw], 1 - alpha, patch_gray, alpha, 0
            )

        bgr = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
        return bgr, has_defect


class VideoPipeline:
    """Thread-safe high-throughput video ingestion and defect detection pipeline."""

    def __init__(
        self,
        source: Optional[Any] = None,
        inference_engine: Optional[DefectInferenceEngine] = None,
        enable_plc: bool = True,
    ):
        # Default strictly to simulator unless a valid video file is provided
        if source is not None and (isinstance(source, (str, Path)) and Path(str(source)).is_file()):
            self.source = source
        else:
            self.source = "simulator"

        self.engine = inference_engine or DefectInferenceEngine()
        self.enable_plc = enable_plc

        self.cap = None
        self.simulator = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_annotated: Optional[np.ndarray] = None
        self.latest_result: Dict[str, Any] = {}
        self.fps = 0.0
        self.parts_scanned = 0
        self.defects_found = 0
        self.part_in_inspection_zone = False

        self._init_source()

    def _init_source(self):
        """Initializes simulated industrial conveyor or designated video file. Personal webcams are never touched."""
        # Check if source is a pre-recorded test video file on disk
        if isinstance(self.source, (str, Path)) and Path(str(self.source)).is_file():
            try:
                self.cap = cv2.VideoCapture(str(self.source))
                if self.cap.isOpened():
                    print(f"[VideoPipeline] Loaded pre-recorded industrial inspection video: {self.source}")
                    metrics_manager.set_camera_status(True)
                    return
            except Exception:
                pass

        # Default: Pure synthetic metallic strip conveyor simulation
        print("[VideoPipeline] Active Source: Industrial Conveyor Simulator (Personal webcam disabled).")
        self.simulator = AssemblyLineSimulator()
        metrics_manager.set_camera_status(True)

    def _read_raw_frame(self) -> Optional[np.ndarray]:
        """Reads frame from camera or simulator."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame

        if self.simulator:
            frame, _ = self.simulator.get_frame()
            return frame

        return None

    def start(self):
        """Starts background worker thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()
            print("[VideoPipeline] Pipeline thread started.")

    def stop(self):
        """Stops background worker thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("[VideoPipeline] Pipeline stopped.")

    def _worker_loop(self):
        """Continuous high-speed capture and inference loop."""
        prev_time = time.perf_counter()

        while self.running:
            raw_frame = self._read_raw_frame()
            if raw_frame is None:
                time.sleep(0.01)
                continue

            # Run inference
            t0 = time.perf_counter()
            result = self.engine.predict(raw_frame)
            t_infer = time.perf_counter() - t0

            # Calculate FPS
            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            prev_time = curr_time
            current_fps = 1.0 / dt if dt > 0 else 0.0

            result["fps"] = round(current_fps, 1)

            # Annotate
            annotated = self.engine.annotate(raw_frame, result)

            # Record metrics
            metrics_manager.record_inference(t_infer, current_fps)
            is_defective = result["is_defective"]
            rejected = result["action"] == "REJECT"
            metrics_manager.record_part_scan(is_defective, rejected, result["detections"])

            # Dispatch PLC reject signal once per passing defective part
            if rejected and self.enable_plc:
                if not self.part_in_inspection_zone:
                    self.part_in_inspection_zone = True
                    top_defect = result["detections"][0]
                    plc_client.send_reject_signal(
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
            elif not is_defective:
                # Part has cleared the inspection zone; reset tracking state for next part
                self.part_in_inspection_zone = False

            with self.lock:
                self.latest_frame = raw_frame
                self.latest_annotated = annotated
                self.latest_result = result
                self.fps = current_fps
                self.parts_scanned += 1
                if is_defective:
                    self.defects_found += 1

            # Throttle to max 35 FPS to conserve CPU if running simulator
            time.sleep(0.02)

    def get_latest_annotated(self) -> Optional[np.ndarray]:
        """Thread-safe access to latest annotated frame."""
        with self.lock:
            if self.latest_annotated is not None:
                return self.latest_annotated.copy()
            return None

    def get_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """Yields multipart MJPEG frame chunks for HTTP live streaming."""
        while self.running:
            frame = self.get_latest_annotated()
            if frame is None:
                time.sleep(0.03)
                continue

            success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not success:
                continue

            chunk = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            )
            yield chunk
            time.sleep(0.03)


# Global singleton instance
video_pipeline = VideoPipeline()


if __name__ == "__main__":
    print("Testing VideoPipeline with local display (press 'q' to exit)...")
    pipe = VideoPipeline()
    pipe.start()

    try:
        for _ in range(50):
            frame = pipe.get_latest_annotated()
            if frame is not None:
                cv2.imshow("Industrial Defect Detection - Edge Feed", frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            time.sleep(0.03)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
