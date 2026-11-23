"""Build reproducible labeled feature datasets from local or synthetic records."""

from pathlib import Path

import numpy as np

from data_sim.generator import CLASS_NAMES, SensorRecord, generate_recording
from data_sim.loader import load_local_records
from features.pipeline import record_to_features


def build_feature_dataset(
    dataset: str = "auto",
    data_dir: str | Path = "data/real",
    recordings_per_class: int = 60,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return 18-feature rows, five-state labels, and the selected source name."""
    if dataset not in ("auto", "synthetic", "real"):
        raise ValueError("dataset must be 'auto', 'synthetic', or 'real'")
    if recordings_per_class < 1:
        raise ValueError("recordings_per_class must be positive")
    records = [] if dataset == "synthetic" else load_local_records(data_dir)
    if dataset == "real" and not records:
        raise FileNotFoundError(f"No supported labeled CSV/MAT records found in {data_dir}")
    source = "real"
    if dataset == "synthetic" or (dataset == "auto" and not records):
        source = "synthetic"
        records = [
            generate_recording(label, sample_count=512, seed=seed + label * 10000 + index)
            for label in range(len(CLASS_NAMES))
            for index in range(recordings_per_class)
        ]
    if not records:
        raise ValueError("The selected dataset contains no records")

    feature_rows = []
    labels = []
    for record in records:
        features = record_to_features(record)
        feature_rows.extend(features)
        labels.extend([record.label] * len(features))
    if source == "real":
        sparse_labels = {
            label
            for label in range(len(CLASS_NAMES))
            if labels.count(label) < 2
        }
        if sparse_labels:
            source = "real+synthetic"
            for label in sorted(sparse_labels):
                for index in range(recordings_per_class):
                    record = generate_recording(
                        label,
                        sample_count=512,
                        seed=seed + label * 10000 + index,
                    )
                    features = record_to_features(record)
                    feature_rows.extend(features)
                    labels.extend([label] * len(features))
    return np.asarray(feature_rows, dtype=np.float32), np.asarray(labels, dtype=np.int64), source
