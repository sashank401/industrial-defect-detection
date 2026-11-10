cmake_minimum_required(VERSION 3.16)
project(IndustrialDefectDetector VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime dependency (set ONNXRUNTIME_DIR if not in system path)
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime installation directory")
if(ONNXRUNTIME_DIR)
    include_directories(${ONNXRUNTIME_DIR}/include)
    link_directories(${ONNXRUNTIME_DIR}/lib)
endif()

add_executable(edge_defect_detector main.cpp)

target_link_libraries(edge_defect_detector
    PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
)

target_compile_definitions(edge_defect_detector PRIVATE
    -DPROJECT_VERSION="${PROJECT_VERSION}"
)
