"""
Data Augmentation Pipeline for Industrial Surface Defect Detection using Albumentations.
Applies geometric transformations, lighting/contrast variations, sensor noise,
and coarse dropout while preserving normalized YOLO bounding boxes.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import albumentations as A

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import NEU_CLASSES, CLASS_COLORS, OUTPUT_DIR, DATASET_DIR


def get_industrial_augmentation_pipeline() -> A.Compose:
    """
    Constructs an industrial-grade Albumentations augmentation pipeline tailored for
    metallic surface defect inspection under fluctuating assembly line conditions.
    """
    return A.Compose(
        [
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.4),

            # Industrial lighting and contrast fluctuations
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.6
            ),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.4),

            # Sensor noise and vibration motion blur
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=5, p=1.0),
                    A.GaussNoise(p=1.0),
                ],
                p=0.4,
            ),

            # Simulated sensor occlusion / defect cutout
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(10, 25),
                hole_width_range=(10, 25),
                p=0.3,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.25,
        ),
    )


def apply_augmentation(
    image: np.ndarray,
    bboxes: List[List[float]],
    class_labels: List[int],
    pipeline: Optional[A.Compose] = None,
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """Applies augmentation pipeline to an image and its YOLO bounding boxes."""
    if pipeline is None:
        pipeline = get_industrial_augmentation_pipeline()

    # Ensure bounding boxes are valid YOLO coordinates within [0, 1]
    valid_boxes = []
    valid_labels = []
    for box, label in zip(bboxes, class_labels):
        x, y, w, h = box
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        w = min(max(w, 0.001), 1.0)
        h = min(max(h, 0.001), 1.0)
        valid_boxes.append([x, y, w, h])
        valid_labels.append(label)

    try:
        transformed = pipeline(image=image, bboxes=valid_boxes, class_labels=valid_labels)
        return transformed["image"], transformed["bboxes"], transformed["class_labels"]
    except Exception as e:
        # Fallback to original if bounding box clipping fails
        return image, valid_boxes, valid_labels


def draw_yolo_bboxes(
    image: np.ndarray,
    bboxes: List[List[float]],
    class_labels: List[int],
    line_thickness: int = 2,
) -> np.ndarray:
    """Draws YOLO normalized bounding boxes on an image canvas."""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    for box, label_idx in zip(bboxes, class_labels):
        label_idx = int(label_idx)
        xc, yc, bw, bh = box
        xmin = int((xc - bw / 2.0) * w)
        ymin = int((yc - bh / 2.0) * h)
        xmax = int((xc + bw / 2.0) * w)
        ymax = int((yc + bh / 2.0) * h)

        class_name = NEU_CLASSES[label_idx] if label_idx < len(NEU_CLASSES) else f"class_{label_idx}"
        color = CLASS_COLORS.get(class_name, (0, 255, 0))

        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, line_thickness)

        label_text = f"{class_name}"
        cv2.putText(
            annotated,
            label_text,
            (xmin, max(ymin - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def generate_augmentation_showcase(output_path: Optional[Path] = None) -> Path:
    """
    Creates a visual multi-panel showcase comparing original sample with various
    industrial augmentations (rotation, lighting shift, sensor noise, cutout).
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "augmentation_showcase.png"

    # Find a sample image with defects
    train_img_dir = DATASET_DIR / "images" / "train"
    train_lbl_dir = DATASET_DIR / "labels" / "train"

    sample_imgs = list(train_img_dir.glob("*.jpg"))
    if not sample_imgs:
        raise FileNotFoundError("No training images found to demonstrate augmentation.")

    sample_img_path = sample_imgs[0]
    sample_lbl_path = train_lbl_dir / (sample_img_path.stem + ".txt")

    img = cv2.imread(str(sample_img_path))
    bboxes = []
    labels = []

    if sample_lbl_path.exists():
        with open(sample_lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append(int(parts[0]))
                    bboxes.append([float(p) for p in parts[1:]])

    # 1. Original annotated
    orig_annotated = draw_yolo_bboxes(img, bboxes, labels)
    cv2.putText(orig_annotated, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 2. Rotation & Flip
    rot_pipe = A.Compose([
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=1.0),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
    t1 = rot_pipe(image=img, bboxes=bboxes, class_labels=labels)
    panel1 = draw_yolo_bboxes(t1["image"], t1["bboxes"], t1["class_labels"])
    cv2.putText(panel1, "Rotate + Flip", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 3. Lighting & Contrast Variation
    light_pipe = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.CLAHE(clip_limit=4.0, p=1.0),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
    t2 = light_pipe(image=img, bboxes=bboxes, class_labels=labels)
    panel2 = draw_yolo_bboxes(t2["image"], t2["bboxes"], t2["class_labels"])
    cv2.putText(panel2, "Lighting & CLAHE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 4. Sensor Noise & Motion Blur
    noise_pipe = A.Compose([
        A.GaussNoise(p=1.0),
        A.MotionBlur(blur_limit=7, p=1.0),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
    t3 = noise_pipe(image=img, bboxes=bboxes, class_labels=labels)
    panel3 = draw_yolo_bboxes(t3["image"], t3["bboxes"], t3["class_labels"])
    cv2.putText(panel3, "Noise + Motion Blur", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 5. Full Pipeline with Cutout
    full_pipe = get_industrial_augmentation_pipeline()
    t4 = full_pipe(image=img, bboxes=bboxes, class_labels=labels)
    panel4 = draw_yolo_bboxes(t4["image"], t4["bboxes"], t4["class_labels"])
    cv2.putText(panel4, "Full Augmentation", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Assemble 1x5 horizontal strip (or 2x3 grid)
    # Let's make a 2x3 grid
    blank = np.zeros_like(orig_annotated)
    row1 = np.hstack([orig_annotated, panel1, panel2])
    row2 = np.hstack([panel3, panel4, blank])
    collage = np.vstack([row1, row2])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), collage)
    print(f"[Augmentation] Generated augmentation showcase: {output_path}")
    return output_path


if __name__ == "__main__":
    showcase = generate_augmentation_showcase()
    print("Augmentation showcase saved to:", showcase)
