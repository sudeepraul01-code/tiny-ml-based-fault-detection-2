"""Shared classifier metrics, latency, and model footprint reporting."""

import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=range(5)).tolist(),
    }


def mean_latency_ms(predict_one, samples: np.ndarray, repetitions: int = 30) -> float:
    for sample in samples[: min(3, len(samples))]:
        predict_one(sample)
    count = min(repetitions, len(samples))
    if count == 0:
        return 0.0
    start = time.perf_counter()
    for sample in samples[:count]:
        predict_one(sample)
    return (time.perf_counter() - start) * 1000.0 / count


def print_report(title: str, metrics: dict, latency_ms: float, model_path: Path) -> None:
    print(f"\n{title}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print("  Confusion matrix (rows=true, columns=predicted):")
    for row in metrics["confusion_matrix"]:
        print("   ", row)
    print(f"  Per-window inference latency: {latency_ms:.3f} ms")
    print(f"  Model size: {model_path.stat().st_size / 1024:.2f} KB")
