# Industrial Defect Detection System

## Overview
An image-processing-based system designed to identify potential surface defects in industrial images using Python and OpenCV.

## Features
- Industrial image loading and validation
- Grayscale image conversion
- Gaussian blur and CLAHE enhancement
- Canny edge detection
- Morphological image processing
- Candidate anomaly region detection
- Bounding-box visualization
- Region statistics and size-based classification
- CSV analysis report generation

## Technologies Used
- Python
- OpenCV
- NumPy
- Git and GitHub

## Project Workflow
1. Load the input image.
2. Preprocess and enhance the image.
3. Detect edges and potential anomaly regions.
4. Extract candidate regions using contours.
5. Calculate region statistics.
6. Save processed images and analysis reports.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python src/main.py
```

## Current Status
This project currently uses traditional computer vision techniques to identify candidate defect regions. These detections are preliminary and are not yet validated by a trained machine-learning model.

## Future Improvements
- Integrate the NEU-DET dataset.
- Train a YOLO-based defect detection model.
- Classify different defect types.
- Evaluate model performance using precision, recall and mAP.
- Develop a web-based interface.
