"""Train and evaluate the reproducible Keras 1D-CNN."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_sim.dataset import build_feature_dataset
from models.metrics import evaluate_predictions, mean_latency_ms, print_report
from models.network import build_classifier

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("auto", "synthetic", "real"), default="auto")
    parser.add_argument("--data-dir", default="data/real")
    parser.add_argument("--recordings-per-class", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args()
    if args.recordings_per_class < 2 or args.epochs < 1:
        parser.error("recordings-per-class must be >= 2 and epochs must be >= 1")

    seed = 42
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    features, labels, source = build_feature_dataset(
        args.dataset, args.data_dir, args.recordings_per_class, seed
    )
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x).astype(np.float32)
    test_x = scaler.transform(test_x).astype(np.float32)
    train_x = train_x[..., np.newaxis]
    test_x = test_x[..., np.newaxis]

    model = build_classifier()
    model.fit(
        train_x,
        train_y,
        epochs=args.epochs,
        batch_size=32,
        validation_split=0.15,
        verbose=2,
        shuffle=True,
    )
    probabilities = model.predict(test_x, verbose=0)
    metrics = evaluate_predictions(test_y, probabilities)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "fault_classifier.keras"
    scaler_path = ARTIFACT_DIR / "feature_scaler.npz"
    metrics_path = ARTIFACT_DIR / "float_metrics.json"
    model.save(model_path)
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)
    metrics["dataset_source"] = source
    metrics["data_dir"] = str(Path(args.data_dir).resolve())
    metrics["recordings_per_class"] = args.recordings_per_class
    metrics["seed"] = seed
    metrics["samples"] = int(len(labels))
    latency = mean_latency_ms(lambda sample: model(sample[np.newaxis, ...], training=False), test_x)
    metrics["latency_ms"] = latency
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Dataset source: {source} ({len(labels)} feature windows)")
    print_report("Float32 Keras evaluation", metrics, latency, model_path)
    print(f"Saved scaler: {scaler_path}")


if __name__ == "__main__":
    main()
