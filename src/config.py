"""
Central Configuration for Industrial Defect Detection System.
Specifies paths, NEU defect classes, model hyperparameters, PLC triggers, and metrics.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
STATIC_DIR = PROJECT_ROOT / "static"

for directory in [DATA_DIR, DATASET_DIR, OUTPUT_DIR, MODELS_DIR, STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Official NEU Metal Surface Defect Classes
NEU_CLASSES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Color palette for defect classes (BGR format for OpenCV)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "crazing": (34, 139, 34),          # Forest Green
    "inclusion": (0, 165, 255),        # Orange
    "patches": (255, 0, 0),            # Blue
    "pitted_surface": (238, 130, 238), # Violet
    "rolled-in_scale": (0, 215, 255),  # Yellow-Gold
    "scratches": (0, 0, 255),          # Bright Red
}

# Hex colors for web UI & frontend display
CLASS_COLORS_HEX: Dict[str, str] = {
    "crazing": "#228B22",
    "inclusion": "#FFA500",
    "patches": "#007BFF",
    "pitted_surface": "#EE82EE",
    "rolled-in_scale": "#FFD700",
    "scratches": "#FF2D55",
}

# Defect severity weights (used for automated PLC reject sorting decision)
DEFECT_SEVERITY: Dict[str, float] = {
    "crazing": 0.85,
    "inclusion": 0.90,
    "patches": 0.60,
    "pitted_surface": 0.95,
    "rolled-in_scale": 0.75,
    "scratches": 0.70,
}


@dataclass
class ModelConfig:
    """YOLOv8 & ONNX Runtime Model Configuration"""
    base_model: str = "yolov8n.pt"
    input_size: int = 640
    batch_size: int = 16
    epochs: int = 25
    learning_rate: float = 0.01
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "cpu"  # 'cpu', '0' for CUDA GPU
    weights_path: Path = MODELS_DIR / "best_neu_yolov8.pt"
    onnx_path: Path = MODELS_DIR / "best_neu_yolov8.onnx"
    tensorrt_path: Path = MODELS_DIR / "best_neu_yolov8.engine"


@dataclass
class VideoConfig:
    """Industrial Assembly Line Camera / Video Stream Settings"""
    allow_webcam: bool = False  # Strictly False: Personal webcams are completely disabled
    source_type: str = "simulator"  # Default: Simulated industrial steel conveyor feed
    camera_index: int = -1  # Disabled by default
    rtsp_url: str = ""
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 30
    save_defective_frames: bool = True
    defects_save_dir: Path = OUTPUT_DIR / "defects_detected"


@dataclass
class PLCConfig:
    """Industrial Programmable Logic Controller (PLC) Interface Settings"""
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 5020  # Industrial Modbus / Socket port
    protocol: str = "TCP_SOCKET"  # 'TCP_SOCKET', 'MODBUS_TCP', 'MQTT'
    rejection_threshold: float = 0.65  # Confidence / severity to trigger reject piston
    actuator_pulse_ms: int = 150  # Diverter solenoid activation duration (ms)


@dataclass
class MetricsConfig:
    """Prometheus & Monitoring Configuration"""
    prometheus_port: int = 8000
    export_interval_sec: float = 1.0


# Default instances
model_cfg = ModelConfig()
video_cfg = VideoConfig()
plc_cfg = PLCConfig()
metrics_cfg = MetricsConfig()
