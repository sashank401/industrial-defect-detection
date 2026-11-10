"""
Edge Optimization & ONNX / TensorRT Export Pipeline.
Converts PyTorch YOLOv8 weights to ONNX format, verifies tensor graphs,
benchmarks runtime latency, and provides TensorRT engine compilation scripts.
"""

import sys
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import model_cfg, MODELS_DIR, OUTPUT_DIR


def export_to_onnx(
    weights_path: Optional[Path] = None,
    output_onnx_path: Optional[Path] = None,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
    dynamic: bool = False,
) -> Path:
    """
    Exports YOLOv8 PyTorch model to ONNX format.
    """
    if weights_path is None:
        weights_path = model_cfg.weights_path
        if not weights_path.exists():
            weights_path = Path("yolov8n.pt")

    if output_onnx_path is None:
        output_onnx_path = model_cfg.onnx_path

    print("=" * 65)
    print("      EDGE OPTIMIZATION - ONNX MODEL EXPORT PIPELINE")
    print("=" * 65)
    print(f"Source PyTorch Weights: {weights_path}")
    print(f"Target ONNX Model:      {output_onnx_path}")
    print(f"Input Shape:            (1, 3, {imgsz}, {imgsz})")
    print(f"Opset Version:          {opset}")
    print(f"Dynamic Batching:       {dynamic}")
    print("-" * 65)

    model = YOLO(str(weights_path))

    # Export using Ultralytics exporter
    exported_file = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
    )

    exported_path = Path(exported_file)
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != output_onnx_path.resolve():
        shutil.copy(exported_path, output_onnx_path)

    # Validate ONNX graph
    print("\n[Export] Validating ONNX graph integrity with onnx.checker...")
    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[Export] ONNX model graph structure is valid and verified!")

    # Print input / output metadata
    session = ort.InferenceSession(str(output_onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print("\n[ONNX Tensor Architecture]")
    for inp in inputs:
        print(f"  Input:  '{inp.name}' | Shape: {inp.shape} | Type: {inp.type}")
    for out in outputs:
        print(f"  Output: '{out.name}' | Shape: {out.shape} | Type: {out.type}")

    return output_onnx_path


def benchmark_pytorch_vs_onnx(
    pt_path: Path,
    onnx_path: Path,
    imgsz: int = 640,
    iterations: int = 30,
) -> Dict[str, Any]:
    """
    Benchmarks runtime latency between PyTorch and ONNX Runtime.
    """
    print("\n" + "=" * 65)
    print("      PYTORCH VS. ONNX RUNTIME LATENCY BENCHMARK")
    print("=" * 65)

    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

    # 1. PyTorch Benchmark
    pt_model = YOLO(str(pt_path))
    # Warmup
    for _ in range(5):
        _ = pt_model(dummy_input, imgsz=imgsz, verbose=False)

    pt_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = pt_model(dummy_input, imgsz=imgsz, verbose=False)
        pt_times.append((time.perf_counter() - t0) * 1000.0)

    avg_pt_ms = float(np.mean(pt_times))
    pt_fps = 1000.0 / avg_pt_ms if avg_pt_ms > 0 else 0.0

    # 2. ONNX Runtime Benchmark
    available_providers = ort.get_available_providers()
    selected_providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available_providers:
        selected_providers.insert(0, "CUDAExecutionProvider")
    elif "DmlExecutionProvider" in available_providers:
        selected_providers.insert(0, "DmlExecutionProvider")

    ort_session = ort.InferenceSession(str(onnx_path), providers=selected_providers)
    input_name = ort_session.get_inputs()[0].name

    # Warmup
    for _ in range(5):
        _ = ort_session.run(None, {input_name: dummy_input})

    ort_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = ort_session.run(None, {input_name: dummy_input})
        ort_times.append((time.perf_counter() - t0) * 1000.0)

    avg_ort_ms = float(np.mean(ort_times))
    ort_fps = 1000.0 / avg_ort_ms if avg_ort_ms > 0 else 0.0
    speedup = avg_pt_ms / avg_ort_ms if avg_ort_ms > 0 else 1.0

    print(f"PyTorch Runtime:       {avg_pt_ms:6.2f} ms ({pt_fps:5.1f} FPS)")
    print(f"ONNX Runtime:          {avg_ort_ms:6.2f} ms ({ort_fps:5.1f} FPS) [Providers: {selected_providers}]")
    print(f"Edge Acceleration:     {speedup:6.2f}x speedup")
    print("=" * 65)

    return {
        "pytorch_latency_ms": round(avg_pt_ms, 2),
        "pytorch_fps": round(pt_fps, 1),
        "onnx_latency_ms": round(avg_ort_ms, 2),
        "onnx_fps": round(ort_fps, 1),
        "speedup": round(speedup, 2),
        "providers": selected_providers,
    }


def generate_tensorrt_build_script(onnx_path: Path, engine_path: Path) -> Path:
    """
    Generates a shell/batch script to compile the ONNX model into an optimized
    NVIDIA TensorRT engine on edge platforms (e.g. Jetson Orin / Xavier / RTX).
    """
    script_path = MODELS_DIR / "build_tensorrt_engine.bat"
    content = f"""@echo off
echo =========================================================
echo      NVIDIA TensorRT Edge Engine Builder
echo =========================================================

REM Path to trtexec (ensure TensorRT bin is in PATH)
SET ONNX_PATH={onnx_path}
SET ENGINE_PATH={engine_path}

echo Compiling ONNX model to TensorRT engine with FP16 precision...
trtexec --onnx="%ONNX_PATH%" --saveEngine="%ENGINE_PATH%" --fp16 --memPoolSize=workspace:2048 --verbose

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] TensorRT Engine successfully built: %ENGINE_PATH%
) else (
    echo [ERROR] TensorRT compilation failed. Ensure TensorRT and CUDA are installed.
)
pause
"""
    with open(script_path, "w") as f:
        f.write(content)

    print(f"[TensorRT] Generated engine compilation script: {script_path}")
    return script_path


if __name__ == "__main__":
    onnx_file = export_to_onnx()
    generate_tensorrt_build_script(onnx_file, model_cfg.tensorrt_path)
