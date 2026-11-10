"""
YOLOv8 Training Pipeline for Real-Time Industrial Defect Detection.
Fine-tunes YOLOv8n on NEU metal surface defects, tracks convergence,
and packages weights for edge deployment.
"""

import sys
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional
from ultralytics import YOLO

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import model_cfg, DATASET_DIR, MODELS_DIR, OUTPUT_DIR


def train_yolov8(
    epochs: int = 15,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "cpu",
    data_yaml: Optional[Path] = None,
    pretrained_weights: str = "yolov8n.pt",
) -> Dict[str, Any]:
    """
    Fine-tunes YOLOv8 on NEU Surface Defect dataset.
    """
    if data_yaml is None:
        data_yaml = DATASET_DIR / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}. Run neu_dataset.py first.")

    print("=" * 65)
    print("      INDUSTRIAL DEFECT DETECTION - YOLOv8 TRAINING PIPELINE")
    print("=" * 65)
    print(f"Data config:        {data_yaml}")
    print(f"Base model:         {pretrained_weights}")
    print(f"Epochs:             {epochs}")
    print(f"Batch size:         {batch_size}")
    print(f"Image resolution:   {imgsz}x{imgsz}")
    print(f"Device:             {device}")
    print("-" * 65)

    # Initialize model
    model = YOLO(pretrained_weights)

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=0,  # Avoid multiprocessing issues on Windows
        project=str(OUTPUT_DIR / "runs"),
        name="neu_yolov8_train",
        exist_ok=True,
        verbose=True,
        plots=True,
        save=True,
        val=True,
    )

    # Locate best.pt
    run_dir = OUTPUT_DIR / "runs" / "neu_yolov8_train"
    trained_best = run_dir / "weights" / "best.pt"
    if not trained_best.exists():
        trained_best = run_dir / "weights" / "last.pt"

    dest_weights = model_cfg.weights_path
    if trained_best.exists():
        shutil.copy(trained_best, dest_weights)
        print(f"\n[Train] Copied best weights to edge target: {dest_weights}")

    # Validate model on test set
    print("\n[Train] Running validation on test set...")
    val_results = model.val(data=str(data_yaml), split="test", imgsz=imgsz, device=device)

    # Extract metrics summary
    metrics_summary = {
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "device": device,
        "weights_saved": str(dest_weights),
        "mAP50": float(val_results.box.map50) if hasattr(val_results.box, "map50") else 0.0,
        "mAP50_95": float(val_results.box.map) if hasattr(val_results.box, "map") else 0.0,
        "precision": float(val_results.box.mp) if hasattr(val_results.box, "mp") else 0.0,
        "recall": float(val_results.box.mr) if hasattr(val_results.box, "mr") else 0.0,
    }

    summary_file = OUTPUT_DIR / "training_summary.json"
    with open(summary_file, "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print("\n" + "=" * 65)
    print("                 TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f"Validation mAP@0.5:      {metrics_summary['mAP50']:.4f}")
    print(f"Validation mAP@0.5:0.95: {metrics_summary['mAP50_95']:.4f}")
    print(f"Mean Precision:          {metrics_summary['precision']:.4f}")
    print(f"Mean Recall:             {metrics_summary['recall']:.4f}")
    print(f"Summary saved:           {summary_file}")
    print("=" * 65)

    return metrics_summary


if __name__ == "__main__":
    train_yolov8(epochs=10, imgsz=640, batch_size=16)
