/**
 * Real-Time Edge Industrial Defect Detection in C++
 * Accelerated with ONNX Runtime C++ API & OpenCV.
 * Designed for assembly-line edge embedded devices (NVIDIA Jetson, Intel IPC).
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// NEU Defect Class Labels
const std::vector<std::string> NEU_CLASSES = {
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
};

struct Detection {
    int class_id;
    std::string class_name;
    float confidence;
    cv::Rect box;
};

// Letterbox pre-processing preserving aspect ratio
cv::Mat letterbox(const cv::Mat& src, int target_w, int target_h, float& scale, int& pad_w, int& pad_h) {
    float r = std::min((float)target_w / src.cols, (float)target_h / src.rows);
    int new_unpad_w = std::round(src.cols * r);
    int new_unpad_h = std::round(src.rows * r);

    pad_w = (target_w - new_unpad_w) / 2;
    pad_h = (target_h - new_unpad_h) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat out;
    cv::copyMakeBorder(resized, out, pad_h, target_h - new_unpad_h - pad_h,
                       pad_w, target_w - new_unpad_w - pad_w,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    scale = r;
    return out;
}

int main(int argc, char* argv[]) {
    std::string model_path = (argc > 1) ? argv[1] : "../models/best_neu_yolov8.onnx";
    std::string image_path = (argc > 2) ? argv[2] : "../data/test_samples/sample_crazing.jpg";

    std::cout << "========================================================\n";
    std::cout << "   C++ Edge Defect Detection Engine (ONNX Runtime)     \n";
    std::cout << "========================================================\n";
    std::cout << "Loading model: " << model_path << "\n";

    // 1. Initialize ONNX Runtime Environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "IndustrialDefectDetector");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Optional: Enable CUDA / TensorRT Execution Provider if hardware available
    // OrtCUDAProviderOptions cuda_options;
    // session_options.AppendExecutionProvider_CUDA(cuda_options);

    Ort::Session session(env, model_path.c_str(), session_options);
    std::cout << "ONNX Runtime session initialized successfully!\n";

    // 2. Load Input Frame
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        std::cerr << "Failed to load image from: " << image_path << "\n";
        return -1;
    }

    const int input_w = 640;
    const int input_h = 640;
    float scale = 1.0f;
    int pad_w = 0, pad_h = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    // 3. Preprocessing (Letterbox -> RGB -> Normalize -> CHW Flatten)
    cv::Mat letterboxed = letterbox(frame, input_w, input_h, scale, pad_w, pad_h);
    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);

    std::vector<float> input_tensor_values(input_w * input_h * 3);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_h; ++h) {
            for (int w = 0; w < input_w; ++w) {
                input_tensor_values[c * input_h * input_w + h * input_w + w] =
                    rgb.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }

    std::vector<int64_t> input_shape = {1, 3, input_h, input_w};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    // 4. Inference Execution
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // 5. Postprocessing (NMS & Bounding Boxes)
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    // Shape: [1, 10, 8400]
    int num_channels = output_shape[1]; // 4 + 6 classes = 10
    int num_anchors = output_shape[2];  // 8400

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> class_indices;
    const float conf_threshold = 0.35f;
    const float nms_threshold = 0.45f;

    for (int i = 0; i < num_anchors; ++i) {
        float max_class_score = 0.0f;
        int best_class_id = -1;

        for (int c = 0; c < (num_channels - 4); ++c) {
            float s = output_data[(4 + c) * num_anchors + i];
            if (s > max_class_score) {
                max_class_score = s;
                best_class_id = c;
            }
        }

        if (max_class_score >= conf_threshold) {
            float cx = output_data[0 * num_anchors + i];
            float cy = output_data[1 * num_anchors + i];
            float w  = output_data[2 * num_anchors + i];
            float h  = output_data[3 * num_anchors + i];

            int x1 = std::round((cx - w / 2.0f - pad_w) / scale);
            int y1 = std::round((cy - h / 2.0f - pad_h) / scale);
            int box_w = std::round(w / scale);
            int box_h = std::round(h / scale);

            bboxes.push_back(cv::Rect(x1, y1, box_w, box_h));
            scores.push_back(max_class_score);
            class_indices.push_back(best_class_id);
        }
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(bboxes, scores, conf_threshold, nms_threshold, nms_indices);

    auto end_time = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    std::cout << "Inference completed in: " << latency_ms << " ms ("
              << (1000.0 / latency_ms) << " FPS)\n";
    std::cout << "Detections found: " << nms_indices.size() << "\n";

    // Draw Detections
    for (int idx : nms_indices) {
        const auto& box = bboxes[idx];
        float score = scores[idx];
        int cid = class_indices[idx];
        std::string name = (cid < NEU_CLASSES.size()) ? NEU_CLASSES[cid] : "defect";

        cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2);
        std::string label = name + " " + std::to_string(int(score * 100)) + "%";
        cv::putText(frame, label, cv::Point(box.x, std::max(box.y - 5, 20)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        std::cout << " - Defect: " << name << " | Confidence: " << score << "\n";
    }

    cv::imwrite("cpp_detection_output.png", frame);
    std::cout << "Saved detection result to cpp_detection_output.png\n";

    return 0;
}
