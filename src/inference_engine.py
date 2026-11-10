"""
Unified High-Performance Inference Engine for Edge Deployment.
Supports ONNX Runtime with hardware execution providers (DirectML/CUDA/CPU)
and PyTorch fallback, with letterbox preprocessing and fast vector NMS.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    NEU_CLASSES,
    CLASS_COLORS,
    DEFECT_SEVERITY,
    model_cfg,
    plc_cfg,
)

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


def letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resizes and pads image to new_shape while preserving aspect ratio.
    Returns: (padded_image, scale_ratio, (pad_left, pad_top))
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2.0  # divide padding into 2 sides
    dh /= 2.0

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


class DefectInferenceEngine:
    """Production Edge Inference Engine with ONNX Runtime acceleration."""

    def __init__(
        self,
        onnx_model_path: Optional[Path] = None,
        pytorch_model_path: Optional[Path] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        preferred_backend: str = "onnx",  # 'onnx' or 'pytorch'
    ):
        self.classes = NEU_CLASSES
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.backend = preferred_backend
        self.session = None
        self.pt_model = None

        onnx_path = onnx_model_path or model_cfg.onnx_path
        pt_path = pytorch_model_path or model_cfg.weights_path

        # Attempt initializing ONNX Runtime
        if self.backend == "onnx" and HAS_ORT and onnx_path.exists():
            try:
                available = ort.get_available_providers()
                providers = ["CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available:
                    providers.insert(0, "CUDAExecutionProvider")
                elif "DmlExecutionProvider" in available:
                    providers.insert(0, "DmlExecutionProvider")

                self.session = ort.InferenceSession(str(onnx_path), providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                self.active_providers = self.session.get_providers()
                print(f"[InferenceEngine] Initialized ONNX Runtime engine with providers: {self.active_providers}")
                return
            except Exception as e:
                print(f"[InferenceEngine] Warning: Failed to initialize ONNX engine ({e}). Falling back to PyTorch.")

        # Fallback to PyTorch YOLO
        if HAS_YOLO:
            model_target = pt_path if pt_path.exists() else "yolov8n.pt"
            self.pt_model = YOLO(str(model_target))
            self.backend = "pytorch"
            print(f"[InferenceEngine] Initialized PyTorch YOLO engine: {model_target}")
        else:
            raise RuntimeError("Neither ONNX Runtime nor Ultralytics YOLO could be initialized.")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Prepares image for model inference: letterbox, BGR->RGB, float32, NCHW."""
        padded_img, ratio, (pad_w, pad_h) = letterbox(image, (self.imgsz, self.imgsz))
        rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # HWC to CHW
        tensor = np.expand_dims(tensor, axis=0)   # Add batch dimension -> 1x3xHxW
        return tensor, ratio, (pad_w, pad_h)

    def postprocess(
        self,
        output_tensor: np.ndarray,
        orig_shape: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float],
    ) -> List[Dict[str, Any]]:
        """
        Processes raw YOLOv8 output tensor (1, 4+num_classes, 8400) via Non-Maximum Suppression (NMS)
        and projects bounding boxes back onto original image dimensions.
        """
        # Shape check: (1, 10, 8400) -> transpose to (8400, 10)
        predictions = np.squeeze(output_tensor)
        if predictions.shape[0] < predictions.shape[1]:
            predictions = np.transpose(predictions)

        orig_h, orig_w = orig_shape
        pad_w, pad_h = pad

        boxes_list = []
        confidences = []
        class_ids = []

        num_classes = len(self.classes)

        for row in predictions:
            classes_scores = row[4:4 + num_classes]
            max_score = float(np.max(classes_scores))
            if max_score >= self.conf_threshold:
                class_id = int(np.argmax(classes_scores))

                # Box format: cx, cy, w, h
                cx, cy, w, h = row[0], row[1], row[2], row[3]

                # Convert from padded model space to original image space
                x1 = (cx - w / 2.0 - pad_w) / ratio
                y1 = (cy - h / 2.0 - pad_h) / ratio
                x2 = (cx + w / 2.0 - pad_w) / ratio
                y2 = (cy + h / 2.0 - pad_h) / ratio

                # Clip coordinates
                x1 = max(0, min(orig_w - 1, int(round(x1))))
                y1 = max(0, min(orig_h - 1, int(round(y1))))
                x2 = max(0, min(orig_w - 1, int(round(x2))))
                y2 = max(0, min(orig_h - 1, int(round(y2))))

                box_w = x2 - x1
                box_h = y2 - y1

                if box_w > 2 and box_h > 2:
                    boxes_list.append([x1, y1, box_w, box_h])
                    confidences.append(max_score)
                    class_ids.append(class_id)

        # Apply OpenCV NMS
        detections = []
        if boxes_list:
            indices = cv2.dnn.NMSBoxes(
                boxes_list,
                confidences,
                score_threshold=self.conf_threshold,
                nms_threshold=self.iou_threshold,
            )

            if len(indices) > 0:
                for idx in indices.flatten():
                    x, y, w, h = boxes_list[idx]
                    cid = class_ids[idx]
                    cname = self.classes[cid] if cid < len(self.classes) else f"defect_{cid}"
                    conf = confidences[idx]
                    severity = DEFECT_SEVERITY.get(cname, 0.70) * conf

                    detections.append({
                        "class_id": cid,
                        "class_name": cname,
                        "confidence": round(float(conf), 4),
                        "severity": round(float(severity), 4),
                        "bbox": [x, y, x + w, y + h],
                        "width": w,
                        "height": h,
                        "area": w * h,
                    })

        return detections

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Runs real-time inference on an image and calculates manufacturing quality decision.
        """
        start_time = time.perf_counter()
        orig_shape = image.shape[:2]

        if self.backend == "onnx" and self.session is not None:
            tensor, ratio, pad = self.preprocess(image)
            outputs = self.session.run(None, {self.input_name: tensor})
            detections = self.postprocess(outputs[0], orig_shape, ratio, pad)
        else:
            # PyTorch YOLO inference
            results = self.pt_model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                verbose=False
            )[0]
            detections = []
            if results.boxes is not None:
                for box in results.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    conf = float(box.conf[0].cpu().numpy())
                    cid = int(box.cls[0].cpu().numpy())
                    cname = self.classes[cid] if cid < len(self.classes) else f"defect_{cid}"
                    severity = DEFECT_SEVERITY.get(cname, 0.70) * conf
                    w = x2 - x1
                    h = y2 - y1

                    detections.append({
                        "class_id": cid,
                        "class_name": cname,
                        "confidence": round(conf, 4),
                        "severity": round(severity, 4),
                        "bbox": [x1, y1, x2, y2],
                        "width": w,
                        "height": h,
                        "area": w * h,
                    })

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        # Industrial quality sorting logic
        is_defective = len(detections) > 0
        max_severity = max([d["severity"] for d in detections], default=0.0)
        action = "REJECT" if (is_defective and max_severity >= plc_cfg.rejection_threshold) else "ACCEPT"

        return {
            "is_defective": is_defective,
            "action": action,
            "max_severity": round(max_severity, 4),
            "detections_count": len(detections),
            "detections": detections,
            "latency_ms": round(latency_ms, 2),
            "fps": round(fps, 1),
            "backend": self.backend,
        }

    def annotate(self, image: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Draws bounding boxes, defect labels, and quality status on frame."""
        annotated = image.copy()
        detections = result.get("detections", [])

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cname = det["class_name"]
            conf = det["confidence"]
            color = CLASS_COLORS.get(cname, (0, 0, 255))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label banner
            label = f"{cname} {conf * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(y1 - 20, 0)), (x1 + tw + 6, max(y1, 20)), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Quality Banner
        action = result.get("action", "ACCEPT")
        banner_color = (0, 0, 220) if action == "REJECT" else (0, 180, 0)
        cv2.rectangle(annotated, (10, 10), (220, 50), banner_color, -1)
        status_text = f"QC: {action}"
        cv2.putText(annotated, status_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # FPS & Latency
        fps = result.get("fps", 0.0)
        lat = result.get("latency_ms", 0.0)
        cv2.putText(
            annotated,
            f"{fps:.1f} FPS | {lat:.1f}ms",
            (15, annotated.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        return annotated


if __name__ == "__main__":
    engine = DefectInferenceEngine()
    test_img = np.random.randint(100, 200, (640, 640, 3), dtype=np.uint8)
    res = engine.predict(test_img)
    print("Test Prediction Output:")
    print(res)
