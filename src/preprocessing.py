import cv2


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def enhance_contrast(image):
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    return clahe.apply(image)


def preprocess_image(image):
    gray = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(gray)
    enhanced = enhance_contrast(blurred)

    return gray, blurred, enhanced