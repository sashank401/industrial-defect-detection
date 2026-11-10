# C++ Edge Defect Detection Engine (ONNX Runtime)

This C++ component provides an edge-optimized runtime pipeline for low-latency defect detection on industrial IPCs, NVIDIA Jetson Orin/Xavier, and embedded Linux devices.

## Features
- **Zero Python Overhead**: Direct C++ execution using the official ONNX Runtime C++ API.
- **Hardware Acceleration**: Out-of-the-box support for CUDA, TensorRT, and OpenVINO Execution Providers.
- **High FPS Throughput**: Optimized memory allocations and vector operations.
- **OpenCV Integration**: Native letterbox scaling and Non-Maximum Suppression (NMS).

## Prerequisites
- CMake >= 3.16
- C++17 compatible compiler (GCC >= 9.0, Clang >= 10.0, or MSVC 2019+)
- OpenCV >= 4.5
- ONNX Runtime C++ library

## Build Instructions (Linux / Embedded Jetson)

```bash
cd edge_cpp
mkdir build && cd build
cmake -DONNXRUNTIME_DIR=/usr/local/onnxruntime ..
make -j4
```

## Running Inference

```bash
./edge_defect_detector ../models/best_neu_yolov8.onnx ../data/test_samples/sample_crazing.jpg
```
