
import cv2
from pathlib import Path


def main():
    image_path = Path(
        r"C:\Users\SASHANK VARMA K\OneDrive\Desktop\P2-zaalima\industrial-defect-detection\data\test_image.png.png"
    )

    image = cv2.imread(str(image_path))

    if image is None:
        print("Error: Could not load image.")
        print("Check the image path and file format.")
        return

    print("Image loaded successfully!")
    print("Image dimensions:", image.shape)
    print("Image height:", image.shape[0])
    print("Image width:", image.shape[1])

    # Step 1: Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Detect edges
    edges = cv2.Canny(gray_image, 100, 200)

    # Step 3: Save processed images
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(str(output_dir / "grayscale.png"), gray_image)
    cv2.imwrite(str(output_dir / "edges.png"), edges)

    print("Grayscale image saved to outputs/grayscale.png")
    print("Edge image saved to outputs/edges.png")

    # Step 4: Display results
    cv2.imshow("Original Image", image)
    cv2.imshow("Grayscale Image", gray_image)
    cv2.imshow("Detected Edges", edges)

    print("Press any key while an image window is selected to close the windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
