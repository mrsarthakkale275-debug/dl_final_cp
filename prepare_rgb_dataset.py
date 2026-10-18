#!/usr/bin/env python3
"""Prepare compact RGB thumbnails for the VisA anomaly-detection baseline."""

from __future__ import annotations

import glob
import io
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

IMAGE_SIZE = 128
OUTPUT_DIR = Path("data/processed/rgb")


def parquet_files(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))


def resize_rgb(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def resize_mask(mask_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(mask_bytes)).convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    return (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)


def prepare() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_arrays = []
    train_count = 0
    train_files = parquet_files("data/raw/visa_parquet/data/*.train-*.parquet")
    for filename in train_files:
        for row in pq.read_table(filename).to_pylist():
            if int(row.get("label", 0)) != 0:
                continue
            train_arrays.append(resize_rgb(row["image"]["bytes"]))
            train_count += 1
            if train_count % 500 == 0:
                print(f"prepared {train_count} normal RGB training images", flush=True)

    if not train_arrays:
        raise RuntimeError("No normal VisA training images found.")
    normal_train = np.stack(train_arrays, axis=0)
    np.save(OUTPUT_DIR / "normal_train.npy", normal_train)

    test_images = []
    test_labels = []
    test_masks = []
    test_count = 0
    test_files = parquet_files("data/raw/visa_parquet/data/*.test-*.parquet")
    for filename in test_files:
        for row in pq.read_table(filename).to_pylist():
            test_images.append(resize_rgb(row["image"]["bytes"]))
            test_labels.append(int(row.get("label", 0)))
            if row.get("mask", {}).get("bytes"):
                test_masks.append(resize_mask(row["mask"]["bytes"]))
            else:
                test_masks.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
            test_count += 1
            if test_count % 200 == 0:
                print(f"prepared {test_count} RGB test images", flush=True)

    test_images_arr = np.stack(test_images, axis=0)
    test_labels_arr = np.asarray(test_labels, dtype=np.int64)
    test_masks_arr = np.stack(test_masks, axis=0)
    np.save(OUTPUT_DIR / "test_images.npy", test_images_arr)
    np.save(OUTPUT_DIR / "test_labels.npy", test_labels_arr)
    np.save(OUTPUT_DIR / "test_masks.npy", test_masks_arr)

    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "image_size": IMAGE_SIZE,
                "normal_training_images": int(normal_train.shape[0]),
                "test_images": int(test_images_arr.shape[0]),
                "test_normal_images": int((test_labels_arr == 0).sum()),
                "test_defective_images": int((test_labels_arr == 1).sum()),
                "source_train_files": [os.path.basename(name) for name in train_files],
                "source_test_files": [os.path.basename(name) for name in test_files],
                "normal_train_shape": list(normal_train.shape),
                "test_images_shape": list(test_images_arr.shape),
                "test_masks_shape": list(test_masks_arr.shape),
            },
            handle,
            indent=2,
        )

    print(f"Prepared {train_count} normal RGB training images: {OUTPUT_DIR / 'normal_train.npy'}")
    print(f"Prepared {test_count} RGB test images: {OUTPUT_DIR}")
    print(f"Manifest: {OUTPUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    prepare()
