
import cv2
import numpy as np
import csv
import time
from pathlib import Path
from datetime import datetime


# ==============================================
# INDUSTRIAL DEFECT DETECTION SYSTEM
# Computer Vision Prototype - Version 3.0
# ==============================================

IMAGE_PATH = Path(
    r"C:\Users\SASHANK VARMA K\OneDrive\Desktop\P2-zaalima\industrial-defect-detection\data\test_image.png.png"
)

OUTPUT_DIR = Path("outputs")


# ==============================================
# 1. IMAGE LOADING
# ==============================================

def load_image():

    print("\n[1] IMAGE LOADING")

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print("ERROR: Unable to load image.")
        print("Check the image path.")
        return None

    height, width, channels = image.shape

    print("Image loaded successfully!")
    print("Width:", width)
    print("Height:", height)
    print("Channels:", channels)
    print("Total pixels:", height * width)

    return image


# ==============================================
# 2. IMAGE PREPROCESSING
# ==============================================

def preprocess_image(image):

    print("\n[2] IMAGE PREPROCESSING")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(blurred)

    print("Grayscale conversion completed.")
    print("Gaussian noise reduction completed.")
    print("CLAHE contrast enhancement completed.")

    return gray, blurred, enhanced


# ==============================================
# 3. EDGE DETECTION
# ==============================================

def detect_edges(enhanced):

    print("\n[3] EDGE DETECTION")

    edges = cv2.Canny(
        enhanced,
        100,
        200
    )

    edge_pixels = cv2.countNonZero(edges)

    print("Canny edge detection completed.")
    print("Edge pixels:", edge_pixels)

    return edges, edge_pixels


# ==============================================
# 4. SURFACE ANOMALY ANALYSIS
# ==============================================

def detect_anomalies(enhanced):

    print("\n[4] SURFACE ANOMALY ANALYSIS")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 15)
    )

    dark_features = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    bright_features = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_TOPHAT,
        kernel
    )

    combined = cv2.add(
        dark_features,
        bright_features
    )

    _, binary = cv2.threshold(
        combined,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    clean_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        clean_kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        clean_kernel
    )

    print("Dark feature extraction completed.")
    print("Bright feature extraction completed.")
    print("Binary thresholding completed.")

    return dark_features, bright_features, binary


# ==============================================
# 5. CANDIDATE REGION ANALYSIS
# ==============================================

def analyze_regions(binary, original):

    print("\n[5] CANDIDATE REGION ANALYSIS")

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    annotated = original.copy()

    detected_regions = []

    image_area = original.shape[0] * original.shape[1]

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 80 or area > image_area * 0.10:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        detected_regions.append({
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "area": round(area, 2)
        })

        cv2.rectangle(
            annotated,
            (x, y),
            (x + width, y + height),
            (0, 0, 255),
            2
        )

        cv2.putText(
            annotated,
            "Candidate",
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

    print("Total contours:", len(contours))
    print("Candidate regions:", len(detected_regions))

    return annotated, detected_regions


# ==============================================
# 6. REGION STATISTICS
# ==============================================

def calculate_statistics(regions):

    print("\n[6] REGION STATISTICS")

    total_regions = len(regions)

    if total_regions == 0:

        total_area = 0
        average_area = 0
        largest_area = 0

    else:

        areas = [
            region["area"]
            for region in regions
        ]

        total_area = sum(areas)

        average_area = (
            total_area / total_regions
        )

        largest_area = max(areas)

    statistics = {
        "total_regions": total_regions,
        "total_area": round(total_area, 2),
        "average_area": round(average_area, 2),
        "largest_area": round(largest_area, 2)
    }

    print("Total candidate regions:", total_regions)
    print("Combined region area:", statistics["total_area"])
    print("Average region area:", statistics["average_area"])
    print("Largest region area:", statistics["largest_area"])

    return statistics


# ==============================================
# 7. REGION SIZE LABELING
# ==============================================

def classify_region_size(regions):

    print("\n[7] REGION SIZE CLASSIFICATION")

    for region in regions:

        area = region["area"]

        if area < 500:
            category = "Small"

        elif area < 2000:
            category = "Medium"

        else:
            category = "Large"

        region["size_category"] = category

        print(
            "Region area:",
            area,
            "| Size category:",
            category
        )

    return regions


# ==============================================
# 8. CSV REPORT GENERATION
# ==============================================

def save_report(regions, statistics, edge_pixels, processing_time):

    print("\n[8] REPORT GENERATION")

    report_path = OUTPUT_DIR / "defect_report.csv"

    with open(
        report_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Region ID",
            "X Coordinate",
            "Y Coordinate",
            "Width",
            "Height",
            "Area",
            "Size Category"
        ])

        for index, region in enumerate(regions, start=1):

            writer.writerow([
                index,
                region["x"],
                region["y"],
                region["width"],
                region["height"],
                region["area"],
                region["size_category"]
            ])

        writer.writerow([])
        writer.writerow(["ANALYSIS SUMMARY"])

        writer.writerow([
            "Total Candidate Regions",
            statistics["total_regions"]
        ])

        writer.writerow([
            "Total Candidate Area",
            statistics["total_area"]
        ])

        writer.writerow([
            "Average Candidate Area",
            statistics["average_area"]
        ])

        writer.writerow([
            "Largest Candidate Area",
            statistics["largest_area"]
        ])

        writer.writerow([
            "Edge Pixels",
            edge_pixels
        ])

        writer.writerow([
            "Processing Time (seconds)",
            round(processing_time, 4)
        ])

        writer.writerow([
            "Analysis Date",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])

    print("CSV report generated:", report_path)


# ==============================================
# 9. SAVE PROCESSED IMAGES
# ==============================================

def save_outputs(images):

    print("\n[9] SAVING PROCESSED IMAGES")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for name, image in images.items():

        output_path = OUTPUT_DIR / f"{name}.png"

        success = cv2.imwrite(
            str(output_path),
            image
        )

        if success:
            print("Saved:", output_path)

        else:
            print("Failed:", output_path)


# ==============================================
# 10. DISPLAY RESULTS
# ==============================================

def display_results(images):

    print("\n[10] DISPLAYING RESULTS")

    for name, image in images.items():

        cv2.imshow(name, image)

    print("Press any key in an image window to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==============================================
# 11. MAIN EXECUTION
# ==============================================

def main():

    overall_start = time.perf_counter()

    print("=" * 55)
    print("      INDUSTRIAL DEFECT DETECTION SYSTEM")
    print("           COMPUTER VISION PROTOTYPE")
    print("=" * 55)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load image
    image = load_image()

    if image is None:
        return

    # Preprocessing
    gray, blurred, enhanced = preprocess_image(image)

    # Edge detection
    edges, edge_pixels = detect_edges(enhanced)

    # Anomaly extraction
    dark_features, bright_features, binary = (
        detect_anomalies(enhanced)
    )

    # Candidate region detection
    annotated, regions = analyze_regions(
        binary,
        image
    )

    # Statistics
    statistics = calculate_statistics(regions)

    # Region size labeling
    regions = classify_region_size(regions)

    # Prepare output images
    processed_images = {

        "01_original": image,
        "02_grayscale": gray,
        "03_blurred": blurred,
        "04_enhanced": enhanced,
        "05_edges": edges,
        "06_dark_features": dark_features,
        "07_bright_features": bright_features,
        "08_binary_mask": binary,
        "09_detected_regions": annotated
    }

    # Save images
    save_outputs(processed_images)

    # Calculate total processing time
    processing_time = (
        time.perf_counter() - overall_start
    )

    print("\n" + "=" * 55)
    print("                 FINAL SUMMARY")
    print("=" * 55)

    print("Input image:", IMAGE_PATH.name)
    print("Resolution:", image.shape[1], "x", image.shape[0])

    print(
        "Candidate regions:",
        statistics["total_regions"]
    )

    print(
        "Total candidate area:",
        statistics["total_area"]
    )

    print(
        "Average candidate area:",
        statistics["average_area"]
    )

    print(
        "Largest candidate area:",
        statistics["largest_area"]
    )

    print("Edge pixels:", edge_pixels)

    print(
        "Processing time:",
        round(processing_time, 4),
        "seconds"
    )

    # Save detailed CSV report
    save_report(
        regions,
        statistics,
        edge_pixels,
        processing_time
    )

    print("=" * 55)
    print("        IMAGE ANALYSIS COMPLETED!")
    print("=" * 55)

    # Display all results
    display_results(processed_images)


if __name__ == "__main__":
    main()
