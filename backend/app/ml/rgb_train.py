#!/usr/bin/env python3
"""Train a CPU-friendly RGB anomaly baseline on the VisA normal-only split."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.app.core.config import settings
from backend.app.ml.rgb_features import (
    NUM_CHANNELS,
    mahalanobis_map,
)


MODEL_FILENAME = "anomaly_model.npz"
METADATA_FILENAME = "anomaly_model_metadata.json"
EVALUATION_FILENAME = "anomaly_evaluation.json"


def prepared_dir() -> Path:
    return settings.project_root / "data" / "processed" / "rgb"


def load_prepared_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directory = prepared_dir()
    required = [
        directory / "normal_train.npy",
        directory / "test_images.npy",
        directory / "test_labels.npy",
        directory / "test_masks.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared RGB data is missing. Run `python prepare_rgb_dataset.py` first. Missing: "
            + ", ".join(missing)
        )
    normal = np.load(required[0], allow_pickle=False)
    test_images = np.load(required[1], allow_pickle=False)
    test_labels = np.load(required[2], allow_pickle=False)
    test_masks = np.load(required[3], allow_pickle=False)
    return normal, test_images, test_labels, test_masks


def train_model(regularization: float) -> tuple[np.ndarray, np.ndarray, int, dict]:
    print("Loading prepared RGB data...")
    normal_uint8, _, _, _ = load_prepared_data()
    normal = normal_uint8.astype(np.float32) / 255.0
    size = int(normal.shape[1])
    print(f"Normal training shape: {normal.shape}")

    print("Computing per-pixel mean...")
    mean = normal.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = normal.astype(np.float64) - mean.astype(np.float64)

    print("Computing per-pixel covariance...")
    covariance = np.einsum("nijc,nijd->ijcd", centered, centered, optimize=True)
    covariance /= max(normal.shape[0] - 1, 1)
    covariance += np.eye(NUM_CHANNELS, dtype=np.float64)[None, None, :, :] * regularization

    print("Inverting per-pixel covariance...")
    flat = covariance.reshape(-1, NUM_CHANNELS, NUM_CHANNELS)
    inv_flat = np.linalg.inv(flat)
    inv_cov = inv_flat.reshape(size, size, NUM_CHANNELS, NUM_CHANNELS).astype(np.float32)

    print("Computing normal training image scores...")
    training_image_scores = []
    batch_size = 128
    for start in range(0, normal.shape[0], batch_size):
        batch = normal[start : start + batch_size]
        for image in batch:
            score_map = mahalanobis_map(image, mean, inv_cov)
            training_image_scores.append(float(np.percentile(score_map, 99.0)))
    training_image_scores = np.asarray(training_image_scores, dtype=np.float64)
    image_threshold = float(np.percentile(training_image_scores, 99.5))

    # Sample pixel scores from all training images in batches; use a bounded sample.
    print("Computing sampled normal pixel scores...")
    sampled_pixels: List[np.ndarray] = []
    for start in range(0, normal.shape[0], batch_size):
        batch = normal[start : start + batch_size]
        for image in batch:
            score_map = mahalanobis_map(image, mean, inv_cov)
            sampled_pixels.append(score_map.reshape(-1)[::8])
    sampled_pixels = np.concatenate(sampled_pixels)
    pixel_threshold = float(np.percentile(sampled_pixels, 99.9))

    metadata = {
        "model_type": "pixel_mahalanobis_rgb_anomaly",
        "image_size": size,
        "num_channels": NUM_CHANNELS,
        "normal_training_images": int(normal.shape[0]),
        "regularization": regularization,
        "aggregation": "99th percentile of per-pixel Mahalanobis distances",
        "image_threshold_99_5_percentile": image_threshold,
        "pixel_threshold_99_9_percentile_sampled": pixel_threshold,
        "training_time_note": "Trained on compact 128x128 VisA normal thumbnails.",
    }
    print(f"Trained on {normal.shape[0]} normal images.")
    print(f"Image anomaly threshold (99.5th percentile): {image_threshold:.6f}")
    print(f"Pixel anomaly threshold (sampled 99.9th percentile): {pixel_threshold:.6f}")
    return mean, inv_cov, int(normal.shape[0]), {
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "image_size": size,
        "metadata": metadata,
    }


def manual_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    count = 0.0
    ties = 0.0
    for value in positives:
        count += float(np.sum(value > negatives))
        ties += float(np.sum(value == negatives))
    return (count + 0.5 * ties) / (positives.size * negatives.size)


def evaluate_model(mean: np.ndarray, inv_cov: np.ndarray, image_threshold: float, pixel_threshold: float, image_size: int) -> dict:
    normal_arrays, test_images_uint8, test_labels, test_masks_uint8 = load_prepared_data()
    del normal_arrays
    test_images = test_images_uint8.astype(np.float32) / 255.0
    print("Evaluating on prepared VisA test images...")
    image_scores: List[float] = []
    image_predictions: List[int] = []
    labels = [int(value) for value in test_labels.tolist()]
    localization_ious: List[float] = []
    intersection = 0
    union = 0

    for index, image in enumerate(test_images):
        score_map = mahalanobis_map(image, mean, inv_cov)
        score = float(np.percentile(score_map, 99.0))
        image_scores.append(score)
        image_predictions.append(int(score > image_threshold))

        if int(labels[index]) == 1:
            ground_truth = test_masks_uint8[index].astype(bool)
            predicted = score_map > pixel_threshold
            inter = int(np.logical_and(predicted, ground_truth).sum())
            union_local = int(np.logical_or(predicted, ground_truth).sum())
            intersection += inter
            union += union_local
            if union_local > 0:
                localization_ious.append(inter / union_local)

    image_score_arr = np.asarray(image_scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    auc = manual_auc(labels_arr, image_score_arr)

    tp = sum(1 for pred, label in zip(image_predictions, labels) if pred == 1 and label == 1)
    tn = sum(1 for pred, label in zip(image_predictions, labels) if pred == 0 and label == 0)
    fp = sum(1 for pred, label in zip(image_predictions, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(image_predictions, labels) if pred == 0 and label == 1)
    total = len(labels) or 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "image_size": image_size,
        "test_images": len(labels),
        "test_normal_images": int((labels_arr == 0).sum()),
        "test_defective_images": int((labels_arr == 1).sum()),
        "image_level": {
            "auroc": float(auc),
            "threshold": float(image_threshold),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": float((tp + tn) / total),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        "localization": {
            "pixel_threshold": float(pixel_threshold),
            "mean_iou": float(np.mean(localization_ious)) if localization_ious else 0.0,
            "global_iou": float(intersection / union) if union > 0 else 0.0,
            "defective_images_evaluated": len(localization_ious),
        },
        "notes": [
            "Training used only the normal VisA training images.",
            "Test labels and masks come from the VisA test split.",
            "Evaluation used compact 128x128 prepared thumbnails; masks were resized with nearest-neighbour resampling.",
            "This baseline returns an anomaly distance, not a calibrated confidence.",
        ],
    }


def save_artifacts(output_dir: str, mean: np.ndarray, inv_cov: np.ndarray, image_threshold: float, pixel_threshold: float, image_size: int, metadata: dict, evaluation: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, MODEL_FILENAME)
    np.savez_compressed(
        model_path,
        mean=mean.astype(np.float32),
        inv_cov=inv_cov.astype(np.float32),
        image_threshold=np.asarray(image_threshold, dtype=np.float64),
        pixel_threshold=np.asarray(pixel_threshold, dtype=np.float64),
        image_size=np.asarray(image_size, dtype=np.int64),
        num_channels=np.asarray(NUM_CHANNELS, dtype=np.int64),
    )
    with open(os.path.join(output_dir, METADATA_FILENAME), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "training": metadata,
                "inference": {
                    "model_type": metadata["model_type"],
                    "image_size": image_size,
                    "num_channels": NUM_CHANNELS,
                    "aggregation": metadata["aggregation"],
                    "image_threshold": image_threshold,
                    "pixel_threshold": pixel_threshold,
                },
            },
            handle,
            indent=2,
        )
    with open(os.path.join(output_dir, EVALUATION_FILENAME), "w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2)
    print(f"Saved model artifact: {model_path}")
    print(f"Saved metadata: {os.path.join(output_dir, METADATA_FILENAME)}")
    print(f"Saved evaluation: {os.path.join(output_dir, EVALUATION_FILENAME)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train compact RGB VisA anomaly baseline")
    parser.add_argument("--regularization", type=float, default=1e-2)
    parser.add_argument("--output-dir", default=os.path.join("models", "rgb"))
    args = parser.parse_args()

    started = time.perf_counter()
    mean, inv_cov, count, trained = train_model(args.regularization)
    # Backfill timing into metadata after training is complete.
    trained["metadata"]["training_time_seconds"] = round(time.perf_counter() - started, 3)
    image_threshold = trained["image_threshold"]
    pixel_threshold = trained["pixel_threshold"]
    image_size = trained["image_size"]
    metadata = trained["metadata"]
    evaluation = evaluate_model(mean, inv_cov, image_threshold, pixel_threshold, image_size)
    save_artifacts(args.output_dir, mean, inv_cov, image_threshold, pixel_threshold, image_size, metadata, evaluation)


if __name__ == "__main__":
    main()
