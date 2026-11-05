
import cv2
import numpy as np
import csv
from pathlib import Path


# ==========================================
# INDUSTRIAL DEFECT DETECTION SYSTEM
# Computer Vision Prototype - Version 2.0
# ==========================================

IMAGE_PATH = Path(
    r"C:\Users\SASHANK VARMA K\OneDrive\Desktop\P2-zaalima\industrial-defect-detection\data\test_image.png.png"
)

OUTPUT_DIR = Path("outputs")


def load_image():
    """Load and validate the input image."""

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print("ERROR: Image could not be loaded.")
        return None

    print("\nIMAGE LOADED SUCCESSFULLY")
    print("Dimensions:", image.shape)
    print("Height:", image.shape[0])
    print("Width:", image.shape[1])
    print("Channels:", image.shape[2])

    return image


def preprocess_image(image):
    """Prepare image for defect analysis."""

    print("\nSTEP 1: IMAGE PREPROCESSING")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(blurred)

    print("Grayscale conversion completed.")
    print("Noise reduction completed.")
    print("Contrast enhancement completed.")

    return gray, blurred, enhanced


def detect_edges(enhanced):
    """Detect image boundaries using Canny."""

    print("\nSTEP 2: EDGE DETECTION")

    edges = cv2.Canny(enhanced, 100, 200)

    edge_pixels = cv2.countNonZero(edges)

    print("Canny edge detection completed.")
    print("Edge pixels detected:", edge_pixels)

    return edges


def detect_anomalies(enhanced):
    """
    Identify possible dark and bright surface anomalies.

    This is a traditional image-processing method,
    not a trained machine-learning model.
    """

    print("\nSTEP 3: SURFACE ANOMALY ANALYSIS")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (15, 15)
    )

    # Highlight dark regions
    dark_features = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    # Highlight bright regions
    bright_features = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_TOPHAT,
        kernel
    )

    # Combine both feature maps
    combined = cv2.add(
        dark_features,
        bright_features
    )

    # Automatic thresholding
    _, binary = cv2.threshold(
        combined,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Remove small isolated noise
    clean_kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        clean_kernel
    )

    # Connect nearby candidate pixels
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        clean_kernel
    )

    print("Dark-region analysis completed.")
    print("Bright-region analysis completed.")
    print("Thresholding completed.")

    return dark_features, bright_features, binary


def analyze_regions(binary, original):
    """Find and annotate candidate anomaly regions."""

    print("\nSTEP 4: REGION ANALYSIS")

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

        # Ignore tiny noise and excessively large regions
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

        # Draw candidate bounding box
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

    print("Total contours found:", len(contours))
    print("Candidate regions:", len(detected_regions))

    return annotated, detected_regions


def save_report(regions):
    """Save detected region information as CSV."""

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
            "Area"
        ])

        for index, region in enumerate(regions, start=1):

            writer.writerow([
                index,
                region["x"],
                region["y"],
                region["width"],
                region["height"],
                region["area"]
            ])

    print("CSV report saved:", report_path)


def save_outputs(images):
    """Save all processed images."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    for name, image in images.items():

        output_path = OUTPUT_DIR / f"{name}.png"

        success = cv2.imwrite(
            str(output_path),
            image
        )

        if success:
            print("Saved:", output_path)
        else:
            print("Failed to save:", output_path)


def display_results(images):
    """Display the processing results."""

    print("\nSTEP 5: DISPLAYING RESULTS")

    for name, image in images.items():

        cv2.imshow(name, image)

    print("Press any key to close all windows.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():

    print("=" * 55)
    print("   INDUSTRIAL DEFECT DETECTION SYSTEM")
    print("   COMPUTER VISION PROTOTYPE")
    print("=" * 55)

    # Create output folder
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load image
    image = load_image()

    if image is None:
        return

    # Preprocessing
    gray, blurred, enhanced = preprocess_image(image)

    # Edge detection
    edges = detect_edges(enhanced)

    # Surface anomaly detection
    dark_features, bright_features, binary = (
        detect_anomalies(enhanced)
    )

    # Identify candidate regions
    annotated, regions = analyze_regions(
        binary,
        image
    )

    # Collect outputs
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
    print("\nSTEP 6: SAVING OUTPUTS")

    save_outputs(processed_images)

    # Generate report
    save_report(regions)

    # Final summary
    print("\n" + "=" * 55)
    print("             ANALYSIS SUMMARY")
    print("=" * 55)

    print("Input image:", IMAGE_PATH.name)
    print("Image resolution:", image.shape[1], "x", image.shape[0])
    print("Candidate regions:", len(regions))
    print("Processed images:", len(processed_images))
    print("Output directory:", OUTPUT_DIR)

    print("=" * 55)
    print("       IMAGE ANALYSIS COMPLETED!")
    print("=" * 55)

    # Display images
    display_results(processed_images)


if __name__ == "__main__":
    main()
