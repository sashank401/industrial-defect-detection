@echo off
echo =========================================================
echo      NVIDIA TensorRT Edge Engine Builder
echo =========================================================

REM Path to trtexec (ensure TensorRT bin is in PATH)
SET ONNX_PATH=C:\Users\SASHANK VARMA K\OneDrive\Desktop\P2-zaalima\industrial-defect-detection\models\best_neu_yolov8.onnx
SET ENGINE_PATH=C:\Users\SASHANK VARMA K\OneDrive\Desktop\P2-zaalima\industrial-defect-detection\models\best_neu_yolov8.engine

echo Compiling ONNX model to TensorRT engine with FP16 precision...
trtexec --onnx="%ONNX_PATH%" --saveEngine="%ENGINE_PATH%" --fp16 --memPoolSize=workspace:2048 --verbose

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] TensorRT Engine successfully built: %ENGINE_PATH%
) else (
    echo [ERROR] TensorRT compilation failed. Ensure TensorRT and CUDA are installed.
)
pause
