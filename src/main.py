"""
Industrial Defect Detection System - Unified Master Orchestrator.
Supports end-to-end operation across traditional CV preprocessing,
deep learning training, hyperparameter tuning, ONNX export,
live assembly line video streaming, and FastAPI microservice hosting.
"""

import sys
import argparse
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.config import model_cfg, video_cfg, OUTPUT_DIR, DATA_DIR
from src.preprocessing import preprocess_image
from src.inference_engine import DefectInferenceEngine
from src.plc_client import plc_client
from src.metrics import metrics_manager


def run_single_image_demo(image_path: Path):
    """
    Executes combined traditional computer vision analysis and deep learning edge inference.
    """
    print("\n" + "=" * 65)
    print("      REAL-TIME INDUSTRIAL DEFECT DETECTION PIPELINE")
    print("=" * 65)

    if not image_path.exists():
        # Fallback to test samples
        test_samples = list((DATA_DIR / "test_samples").glob("*.jpg"))
        if test_samples:
            image_path = test_samples[0]
        else:
            image_path = DATA_DIR / "test_image.png.png"

    print(f"Inspecting input sample: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # 1. Traditional CV Preprocessing Stage
    print("\n[Stage 1] Classical Image Preprocessing & Contrast Enhancement...")
    gray, blurred, enhanced = preprocess_image(image)

    # Canny Edge analysis
    edges = cv2.Canny(enhanced, 100, 200)
    edge_pixels = int(cv2.countNonZero(edges))
    print(f"  - Grayscale & Gaussian blur completed.")
    print(f"  - CLAHE contrast enhancement completed.")
    print(f"  - High-frequency edge pixels: {edge_pixels}")

    # 2. Deep Learning / Edge Inference Stage
    print("\n[Stage 2] Neural Edge Defect Inference (YOLOv8 + ONNX)...")
    engine = DefectInferenceEngine()
    result = engine.predict(image)

    print(f"  - Backend:        {result['backend'].upper()}")
    print(f"  - Latency:        {result['latency_ms']:.2f} ms ({result['fps']:.1f} FPS)")
    print(f"  - Detections:     {result['detections_count']}")
    print(f"  - QC Decision:    {result['action']} (Max Severity: {result['max_severity']:.2f})")

    for idx, det in enumerate(result["detections"], start=1):
        print(f"    [{idx}] Defect: {det['class_name']} | Conf: {det['confidence']*100:.1f}% | Box: {det['bbox']}")

    # 3. PLC Sorting Actuator Dispatch
    print("\n[Stage 3] Industrial PLC Sorting Trigger...")
    if result["action"] == "REJECT" and len(result["detections"]) > 0:
        top_defect = result["detections"][0]
        plc_res = plc_client.send_reject_signal(
            defect_class=top_defect["class_name"],
            coordinates={"x": top_defect["bbox"][0], "y": top_defect["bbox"][1], "width": top_defect["width"], "height": top_defect["height"]},
            severity=top_defect["severity"],
            confidence=top_defect["confidence"],
        )
        print(f"  - Reject Piston Actuated: {plc_res['network_status']}")
    else:
        print("  - Quality Passed: Part routed along primary acceptance conveyor.")

    # 4. Save Annotated Artifacts
    print("\n[Stage 4] Saving Inspection Artifacts...")
    annotated = engine.annotate(image, result)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_orig = OUTPUT_DIR / "demo_original.png"
    out_edge = OUTPUT_DIR / "demo_edges.png"
    out_annot = OUTPUT_DIR / "demo_detected_defects.png"

    cv2.imwrite(str(out_orig), image)
    cv2.imwrite(str(out_edge), edges)
    cv2.imwrite(str(out_annot), annotated)

    print(f"  - Saved edge mask:        {out_edge}")
    print(f"  - Saved detection overlay: {out_annot}")
    print("\n" + "=" * 65)
    print("                 INSPECTION RUN COMPLETE")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Industrial Defect Detection System")
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "train", "tune", "export", "eval", "video", "serve", "data"],
        help="Pipeline execution mode",
    )
    parser.add_argument("--image", type=str, default="", help="Path to custom inspection image")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI server port")

    args = parser.parse_args()

    if args.mode == "data":
        from data.neu_dataset import NEUDatasetManager
        mgr = NEUDatasetManager()
        mgr.build_curated_dataset(samples_per_class=35)
        report = mgr.verify_dataset()
        print("Dataset Ready:", report)

    elif args.mode == "demo":
        target = Path(args.image) if args.image else (DATA_DIR / "test_image.png.png")
        run_single_image_demo(target)

    elif args.mode == "train":
        from src.train import train_yolov8
        train_yolov8(epochs=args.epochs)

    elif args.mode == "tune":
        from src.tune import run_hyperparameter_tuning
        run_hyperparameter_tuning()

    elif args.mode == "export":
        from src.export_onnx import export_to_onnx, generate_tensorrt_build_script
        onnx_path = export_to_onnx()
        generate_tensorrt_build_script(onnx_path, model_cfg.tensorrt_path)

    elif args.mode == "eval":
        from src.evaluator import run_test_set_evaluation
        run_test_set_evaluation()

    elif args.mode == "video":
        from src.video_stream import VideoPipeline
        pipe = VideoPipeline()
        pipe.start()
        print("Video stream running. Press Ctrl+C to terminate.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pipe.stop()

    elif args.mode == "serve":
        import uvicorn
        print(f"Launching FastAPI Web Dashboard & Inspection Service at http://localhost:{args.port}...")
        uvicorn.run("src.api:app", host="0.0.0.0", port=args.port, reload=False)


if __name__ == "__main__":
    main()
