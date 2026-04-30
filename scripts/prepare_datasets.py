#!/usr/bin/env python3
"""
Prepare datasets for BAMF-Eco ultralytics training.

Converts existing COCO JSON annotations to YOLO TXT format and creates
ultralytics-compatible dataset YAML files.

Datasets ():
  - COCO 2017 val: /path/to/datasets/coco2017
  - NightOwls:     /path/to/datasets/nightowls
  - ECP Night:     /path/to/datasets/ecp/night/ECP/night

Usage:
    python scripts/prepare_datasets.py
"""

import os
import json
import yaml
from pathlib import Path
from collections import defaultdict

# ============================================================
# Paths
# ============================================================
DATASETS_ROOT = Path("/path/to/datasets")
BAMF_DATASETS = Path("/path/to/bamf-eco/datasets")
BAMF_DATASETS.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. COCO 2017 val — person class (class 0 in COCO = person)
# ============================================================
def prepare_coco_person():
    """Convert COCO val2017 annotations for person class to YOLO format."""
    coco_root = DATASETS_ROOT / "coco2017"
    images_dir = coco_root / "val2017"
    ann_file = coco_root / "annotations" / "instances_val2017.json"

    # Output directories
    out_root = BAMF_DATASETS / "coco_person"
    out_images = out_root / "images" / "val"
    out_labels = out_root / "labels" / "val"
    out_labels.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    print(f"Loading COCO annotations from {ann_file}...")
    with open(ann_file) as f:
        coco = json.load(f)

    # Build image ID → info mapping
    img_info = {img["id"]: img for img in coco["images"]}

    # Find person category ID (typically 1 in COCO)
    person_cat_ids = set()
    for cat in coco["categories"]:
        if cat["name"] == "person":
            person_cat_ids.add(cat["id"])
    print(f"Person category IDs in COCO: {person_cat_ids}")

    # Group annotations by image
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in person_cat_ids and not ann.get("iscrowd", 0):
            img_anns[ann["image_id"]].append(ann)

    # Convert to YOLO format
    n_images = 0
    n_labels = 0
    for img_id, anns in img_anns.items():
        info = img_info[img_id]
        w, h = info["width"], info["height"]
        fname = info["file_name"]

        # Symlink image (avoid copying 5GB of images)
        src = images_dir / fname
        dst = out_images / fname
        if not dst.exists() and src.exists():
            os.symlink(str(src), str(dst))

        # Write YOLO label
        label_file = out_labels / fname.replace(".jpg", ".txt")
        with open(label_file, "w") as f:
            for ann in anns:
                x, y, bw, bh = ann["bbox"]  # COCO: [x, y, width, height]
                # Convert to YOLO: [cx, cy, w, h] normalized
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                # Clip to [0, 1]
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = max(0, min(1, nw))
                nh = max(0, min(1, nh))
                if nw > 0.001 and nh > 0.001:  # Skip tiny boxes
                    f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                    n_labels += 1
        n_images += 1

    # Create dataset YAML
    yaml_path = out_root / "coco_person.yaml"
    dataset_yaml = {
        "path": str(out_root),
        "train": "images/val",  # Using val for both (no training set download)
        "val": "images/val",
        "test": "images/val",
        "nc": 1,
        "names": ["person"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"COCO person: {n_images} images, {n_labels} labels → {yaml_path}")
    return str(yaml_path)


# ============================================================
# 2. NightOwls — pedestrian class
# ============================================================
def prepare_nightowls():
    """Convert NightOwls validation annotations to YOLO format."""
    nightowls_root = DATASETS_ROOT / "nightowls"
    images_dir = nightowls_root / "nightowls_validation"
    ann_file = nightowls_root / "nightowls_validation.json"

    out_root = BAMF_DATASETS / "nightowls"
    out_images = out_root / "images" / "val"
    out_labels = out_root / "labels" / "val"
    out_labels.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    print(f"Loading NightOwls annotations from {ann_file}...")
    with open(ann_file) as f:
        data = json.load(f)

    img_info = {img["id"]: img for img in data["images"]}

    # Find pedestrian category IDs
    pedestrian_cat_ids = set()
    for cat in data["categories"]:
        name = cat["name"].lower()
        if "pedestrian" in name or "person" in name:
            pedestrian_cat_ids.add(cat["id"])
    print(f"Pedestrian category IDs: {pedestrian_cat_ids}")

    # If no categories found, use all
    if not pedestrian_cat_ids:
        pedestrian_cat_ids = {cat["id"] for cat in data["categories"]}
        print(f"Using all categories: {pedestrian_cat_ids}")

    img_anns = defaultdict(list)
    for ann in data["annotations"]:
        if ann["category_id"] in pedestrian_cat_ids and not ann.get("iscrowd", 0):
            img_anns[ann["image_id"]].append(ann)

    n_images = 0
    n_labels = 0
    for img_id, anns in img_anns.items():
        info = img_info[img_id]
        w, h = info["width"], info["height"]
        fname = info["file_name"]

        # Handle nested paths in file_name
        src = images_dir / Path(fname).name
        if not src.exists():
            src = nightowls_root / fname
        dst = out_images / Path(fname).name
        if not dst.exists() and src.exists():
            os.symlink(str(src), str(dst))

        label_file = out_labels / Path(fname).name.replace(".png", ".txt").replace(".jpg", ".txt")
        with open(label_file, "w") as f:
            for ann in anns:
                x, y, bw, bh = ann["bbox"]
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = max(0, min(1, nw))
                nh = max(0, min(1, nh))
                if nw > 0.001 and nh > 0.001:
                    f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                    n_labels += 1
        n_images += 1

    yaml_path = out_root / "nightowls.yaml"
    dataset_yaml = {
        "path": str(out_root),
        "train": "images/val",
        "val": "images/val",
        "nc": 1,
        "names": ["pedestrian"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"NightOwls: {n_images} images, {n_labels} labels → {yaml_path}")
    return str(yaml_path)


# ============================================================
# 3. ECP Night — pedestrian class (per-image JSON labels)
# ============================================================
def prepare_ecp_night():
    """Convert ECP Night per-image JSON labels to YOLO format."""
    ecp_root = DATASETS_ROOT / "ecp" / "night" / "ECP" / "night"
    images_base = ecp_root / "img" / "val"
    labels_base = ecp_root / "labels" / "val"

    out_root = BAMF_DATASETS / "ecp_night"
    out_images = out_root / "images" / "val"
    out_labels = out_root / "labels" / "val"
    out_labels.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    print(f"Processing ECP Night from {ecp_root}...")

    n_images = 0
    n_labels = 0

    # ECP Night has city subdirectories
    for city_dir in sorted(images_base.iterdir()):
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        label_dir = labels_base / city

        for img_file in sorted(city_dir.iterdir()):
            if not img_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                continue

            # Flat name for output
            flat_name = f"{city}_{img_file.name}"

            # Symlink image
            dst = out_images / flat_name
            if not dst.exists():
                os.symlink(str(img_file), str(dst))

            # Find corresponding label JSON
            label_file = label_dir / img_file.name.replace(img_file.suffix, ".json")
            if not label_file.exists():
                continue

            try:
                with open(label_file) as f:
                    label_data = json.load(f)
            except Exception:
                continue

            # Parse ECP format
            txt_name = flat_name.replace(img_file.suffix, ".txt")
            txt_path = out_labels / txt_name

            # Get image dimensions from the label or use standard ECP size
            img_w = label_data.get("imagewidth", 1920)
            img_h = label_data.get("imageheight", 1024)

            with open(txt_path, "w") as f:
                children = label_data.get("children", [])
                for obj in children:
                    identity = obj.get("identity", "").lower()
                    if "pedestrian" not in identity and "person" not in identity:
                        continue

                    x0 = obj.get("x0", 0)
                    y0 = obj.get("y0", 0)
                    x1 = obj.get("x1", 0)
                    y1 = obj.get("y1", 0)

                    bw = x1 - x0
                    bh = y1 - y0
                    cx = (x0 + x1) / 2 / img_w
                    cy = (y0 + y1) / 2 / img_h
                    nw = bw / img_w
                    nh = bh / img_h

                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    nw = max(0, min(1, nw))
                    nh = max(0, min(1, nh))

                    if nw > 0.001 and nh > 0.001:
                        f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                        n_labels += 1

            n_images += 1

    yaml_path = out_root / "ecp_night.yaml"
    dataset_yaml = {
        "path": str(out_root),
        "train": "images/val",
        "val": "images/val",
        "nc": 1,
        "names": ["pedestrian"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"ECP Night: {n_images} images, {n_labels} labels → {yaml_path}")
    return str(yaml_path)


if __name__ == "__main__":
    print("=" * 60)
    print("BAMF-Eco Dataset Preparation")
    print("=" * 60)

    paths = {}
    paths["coco_person"] = prepare_coco_person()
    print()
    paths["nightowls"] = prepare_nightowls()
    print()
    paths["ecp_night"] = prepare_ecp_night()

    print()
    print("=" * 60)
    print("Dataset YAMLs created:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("=" * 60)
