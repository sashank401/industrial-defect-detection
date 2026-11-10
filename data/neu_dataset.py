"""
NEU Surface Defect Database Ingestion and Organization Pipeline.
Handles VOC XML annotations, YOLO text labels, train/val/test splitting,
dataset validation, and high-fidelity synthetic benchmark generation.
"""

import os
import sys
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import NEU_CLASSES, DATA_DIR, DATASET_DIR


class NEUDatasetManager:
    """Manages ingestion, VOC-to-YOLO conversion, and splitting for NEU-DET."""

    def __init__(self, dataset_root: Optional[Path] = None):
        self.root = dataset_root or DATASET_DIR
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.classes = NEU_CLASSES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

    def setup_directories(self):
        """Creates the YOLO standard directory structure."""
        for split in ["train", "val", "test"]:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)

    def voc_xml_to_yolo(
        self, xml_path: Path, img_width: int, img_height: int
    ) -> List[Tuple[int, float, float, float, float]]:
        """Parses a Pascal VOC XML file and converts bounding boxes to YOLO normalized coordinates."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        yolo_boxes = []

        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            name = name.replace(" ", "_").replace("-", "_")
            if name not in self.class_to_idx:
                continue

            class_idx = self.class_to_idx[name]
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Clamp to image boundaries
            xmin = max(0.0, min(xmin, float(img_width)))
            ymin = max(0.0, min(ymin, float(img_height)))
            xmax = max(0.0, min(xmax, float(img_width)))
            ymax = max(0.0, min(ymax, float(img_height)))

            box_w = xmax - xmin
            box_h = ymax - ymin
            if box_w <= 1 or box_h <= 1:
                continue

            x_center = (xmin + box_w / 2.0) / img_width
            y_center = (ymin + box_h / 2.0) / img_height
            norm_w = box_w / img_width
            norm_h = box_h / img_height

            yolo_boxes.append((class_idx, x_center, y_center, norm_w, norm_h))

        return yolo_boxes

    def create_data_yaml(self, output_path: Optional[Path] = None) -> Path:
        """Generates the data.yaml file required by Ultralytics YOLOv8."""
        if output_path is None:
            output_path = self.root / "data.yaml"

        yaml_content = {
            "path": str(self.root.resolve()).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.classes),
            "names": self.classes,
        }

        with open(output_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        print(f"[NEUDataset] Generated YOLO dataset descriptor: {output_path}")
        return output_path

    def generate_synthetic_neu_sample(
        self, defect_type: str, width: int = 200, height: int = 200
    ) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        """
        Synthesizes high-fidelity metallic surface textures and realistic defect morphologies
        representative of hot-rolled steel strip defects in NEU-DET.
        """
        # Base metallic texture with rolling direction grain
        base_intensity = np.random.randint(130, 185)
        metal = np.full((height, width), base_intensity, dtype=np.uint8)

        # Steel rolling grain noise (horizontal bands)
        grain = np.random.normal(0, 12, (height, 1)).astype(np.float32)
        grain_map = np.tile(grain, (1, width))
        noise = np.random.normal(0, 8, (height, width))
        metal = np.clip(metal.astype(np.float32) + grain_map + noise, 30, 235).astype(np.uint8)

        # Defect class idx
        class_idx = self.class_to_idx[defect_type]
        boxes = []

        if defect_type == "crazing":
            # Network of fine, interconnected micro-cracks
            num_cracks = random.randint(3, 7)
            for _ in range(num_cracks):
                cx, cy = random.randint(30, width - 40), random.randint(30, height - 40)
                bw, bh = random.randint(35, 75), random.randint(35, 75)
                xmin, ymin = max(2, cx - bw // 2), max(2, cy - bh // 2)
                xmax, ymax = min(width - 2, cx + bw // 2), min(height - 2, cy + bh // 2)

                # Draw branching network
                pts = [(random.randint(xmin, xmax), random.randint(ymin, ymax)) for _ in range(5)]
                for k in range(len(pts) - 1):
                    cv2.line(metal, pts[k], pts[k + 1], color=random.randint(40, 75), thickness=1)

                boxes.append((
                    class_idx,
                    (xmin + xmax) / (2.0 * width),
                    (ymin + ymax) / (2.0 * height),
                    (xmax - xmin) / float(width),
                    (ymax - ymin) / float(height)
                ))

        elif defect_type == "inclusion":
            # Non-metallic oxide inclusion dots/clusters
            num_inc = random.randint(2, 5)
            for _ in range(num_inc):
                cx, cy = random.randint(25, width - 25), random.randint(25, height - 25)
                radius = random.randint(4, 12)
                cv2.circle(metal, (cx, cy), radius, color=random.randint(20, 50), thickness=-1)
                cv2.circle(metal, (cx, cy), radius + 2, color=random.randint(200, 240), thickness=1)

                xmin, ymin = max(2, cx - radius - 3), max(2, cy - radius - 3)
                xmax, ymax = min(width - 2, cx + radius + 3), min(height - 2, cy + radius + 3)
                boxes.append((
                    class_idx,
                    (xmin + xmax) / (2.0 * width),
                    (ymin + ymax) / (2.0 * height),
                    (xmax - xmin) / float(width),
                    (ymax - ymin) / float(height)
                ))

        elif defect_type == "patches":
            # Irregular oxidized surface patches / discoloration
            num_patches = random.randint(1, 3)
            for _ in range(num_patches):
                cx, cy = random.randint(35, width - 35), random.randint(35, height - 35)
                bw, bh = random.randint(40, 90), random.randint(30, 80)
                xmin, ymin = max(2, cx - bw // 2), max(2, cy - bh // 2)
                xmax, ymax = min(width - 2, cx + bw // 2), min(height - 2, cy + bh // 2)

                patch = metal[ymin:ymax, xmin:xmax]
                if patch.size > 0:
                    delta = random.choice([-55, 45])
                    metal[ymin:ymax, xmin:xmax] = np.clip(patch.astype(np.int16) + delta, 20, 245).astype(np.uint8)

                boxes.append((
                    class_idx,
                    (xmin + xmax) / (2.0 * width),
                    (ymin + ymax) / (2.0 * height),
                    (xmax - xmin) / float(width),
                    (ymax - ymin) / float(height)
                ))

        elif defect_type == "pitted_surface":
            # Small cavity depressions / pitting corrosion
            num_pits = random.randint(6, 14)
            cluster_cx, cluster_cy = random.randint(40, width - 40), random.randint(40, height - 40)
            c_xmin, c_ymin = cluster_cx, cluster_cy
            c_xmax, c_ymax = cluster_cx, cluster_cy

            for _ in range(num_pits):
                px = int(np.clip(cluster_cx + random.gauss(0, 18), 5, width - 5))
                py = int(np.clip(cluster_cy + random.gauss(0, 18), 5, height - 5))
                r = random.randint(2, 6)
                cv2.circle(metal, (px, py), r, color=random.randint(15, 60), thickness=-1)
                c_xmin = min(c_xmin, px - r)
                c_ymin = min(c_ymin, py - r)
                c_xmax = max(c_xmax, px + r)
                c_ymax = max(c_ymax, py + r)

            c_xmin, c_ymin = max(2, c_xmin - 3), max(2, c_ymin - 3)
            c_xmax, c_ymax = min(width - 2, c_xmax + 3), min(height - 2, c_ymax + 3)
            boxes.append((
                class_idx,
                (c_xmin + c_xmax) / (2.0 * width),
                (c_ymin + c_ymax) / (2.0 * height),
                (c_xmax - c_xmin) / float(width),
                (c_ymax - c_ymin) / float(height)
            ))

        elif defect_type == "rolled-in_scale":
            # Dark rolled-in iron oxide streaks parallel to rolling direction
            num_streaks = random.randint(2, 4)
            for _ in range(num_streaks):
                sx = random.randint(20, width - 80)
                sy = random.randint(15, height - 25)
                sw = random.randint(45, 100)
                sh = random.randint(6, 18)
                ex, ey = min(width - 5, sx + sw), min(height - 5, sy + sh)

                pts = np.array([
                    [sx, sy + sh // 2],
                    [sx + sw // 3, sy],
                    [ex, sy + sh // 3],
                    [sx + sw // 2, ey]
                ], np.int32)
                cv2.fillPoly(metal, [pts], color=random.randint(20, 50))

                boxes.append((
                    class_idx,
                    (sx + ex) / (2.0 * width),
                    (sy + ey) / (2.0 * height),
                    (ex - sx) / float(width),
                    (ey - sy) / float(height)
                ))

        elif defect_type == "scratches":
            # Linear scratches, deep groove marks
            num_scratches = random.randint(1, 3)
            for _ in range(num_scratches):
                pt1 = (random.randint(15, width - 40), random.randint(15, height - 40))
                length = random.randint(50, 110)
                angle = random.uniform(-math.pi / 4, math.pi / 4)
                pt2 = (
                    int(np.clip(pt1[0] + length * math.cos(angle), 5, width - 5)),
                    int(np.clip(pt1[1] + length * math.sin(angle), 5, height - 5))
                )

                # Shadow line and highlight line for 3D metallic groove appearance
                cv2.line(metal, pt1, pt2, color=random.randint(25, 55), thickness=2)
                cv2.line(metal, (pt1[0] + 1, pt1[1] + 1), (pt2[0] + 1, pt2[1] + 1), color=230, thickness=1)

                xmin = max(2, min(pt1[0], pt2[0]) - 3)
                ymin = max(2, min(pt1[1], pt2[1]) - 3)
                xmax = min(width - 2, max(pt1[0], pt2[0]) + 3)
                ymax = min(height - 2, max(pt1[1], pt2[1]) + 3)

                boxes.append((
                    class_idx,
                    (xmin + xmax) / (2.0 * width),
                    (ymin + ymax) / (2.0 * height),
                    (xmax - xmin) / float(width),
                    (ymax - ymin) / float(height)
                ))

        # Convert grayscale to 3-channel BGR for model input compatibility
        bgr = cv2.cvtColor(metal, cv2.COLOR_GRAY2BGR)
        return bgr, boxes

    def build_curated_dataset(
        self, samples_per_class: int = 40, split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1)
    ) -> Dict[str, int]:
        """
        Builds and stratifies a complete NEU defect dataset split across train, val, test splits.
        """
        self.setup_directories()
        random.seed(42)
        np.random.seed(42)

        counts = {"train": 0, "val": 0, "test": 0}
        test_samples_dir = DATA_DIR / "test_samples"
        test_samples_dir.mkdir(parents=True, exist_ok=True)

        for defect_name in self.classes:
            samples = []
            for i in range(samples_per_class):
                img, boxes = self.generate_synthetic_neu_sample(defect_name)
                samples.append((img, boxes))

            # Shuffle and partition
            random.shuffle(samples)
            n_total = len(samples)
            n_train = int(n_total * split_ratios[0])
            n_val = int(n_total * split_ratios[1])

            train_split = samples[:n_train]
            val_split = samples[n_train:n_train + n_val]
            test_split = samples[n_train + n_val:]

            partitions = [("train", train_split), ("val", val_split), ("test", test_split)]

            for split_name, split_data in partitions:
                for idx, (img, boxes) in enumerate(split_data):
                    img_name = f"{defect_name}_{idx + 1:04d}.jpg"
                    lbl_name = f"{defect_name}_{idx + 1:04d}.txt"

                    img_path = self.images_dir / split_name / img_name
                    lbl_path = self.labels_dir / split_name / lbl_name

                    cv2.imwrite(str(img_path), img)

                    # Write YOLO format txt
                    with open(lbl_path, "w") as f:
                        for box in boxes:
                            cid, xc, yc, w, h = box
                            f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

                    counts[split_name] += 1

                    # Save first sample of each class in test_samples for quick inspection
                    if split_name == "test" and idx == 0:
                        cv2.imwrite(str(test_samples_dir / f"sample_{defect_name}.jpg"), img)

        self.create_data_yaml()

        print("[NEUDataset] Dataset generation and organization completed successfully!")
        for split, count in counts.items():
            print(f"  - {split}: {count} images")

        return counts

    def verify_dataset(self) -> Dict[str, any]:
        """Audits dataset integrity, bounding box coordinates, and class distribution."""
        report = {
            "valid": True,
            "splits": {},
            "total_images": 0,
            "total_labels": 0,
            "class_distribution": {c: 0 for c in self.classes},
            "errors": [],
        }

        for split in ["train", "val", "test"]:
            img_dir = self.images_dir / split
            lbl_dir = self.labels_dir / split

            images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            labels = list(lbl_dir.glob("*.txt"))

            split_stats = {"images": len(images), "labels": len(labels), "defects_count": 0}

            for lbl in labels:
                with open(lbl, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if len(parts) != 5:
                            report["errors"].append(f"Invalid label line in {lbl.name}: {line}")
                            report["valid"] = False
                            continue

                        cid, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            report["errors"].append(f"Out of bounds box in {lbl.name}: {line}")
                            report["valid"] = False

                        class_name = self.classes[cid]
                        report["class_distribution"][class_name] += 1
                        split_stats["defects_count"] += 1

            report["splits"][split] = split_stats
            report["total_images"] += len(images)
            report["total_labels"] += len(labels)

        return report


if __name__ == "__main__":
    manager = NEUDatasetManager()
    counts = manager.build_curated_dataset(samples_per_class=35)
    verification = manager.verify_dataset()
    print("\nDataset Verification Report:")
    print(verification)
