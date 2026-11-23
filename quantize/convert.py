"""Calibrate, convert, evaluate, and export the model as a C byte array."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_sim.dataset import build_feature_dataset
from models.metrics import evaluate_predictions, mean_latency_ms, print_report
from quantize.export_header import write_c_header

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "models" / "artifacts"


def convert_to_int8(model: tf.keras.Model, representative_features: np.ndarray) -> bytes:
    """Convert a Keras model using representative calibration data and INT8 I/O."""
    if len(representative_features) == 0:
        raise ValueError("representative_features must not be empty")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        [sample[np.newaxis, ...].astype(np.float32)]
        for sample in representative_features[: min(200, len(representative_features))]
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def convert_to_float32(model: tf.keras.Model) -> bytes:
    """Convert the Keras model to an unoptimized float32 TFLite baseline."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def _float32_interpreter_predict(
    interpreter: tf.lite.Interpreter, sample: np.ndarray
) -> np.ndarray:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_detail["index"], sample[np.newaxis, ...].astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_detail["index"])[0]


def _interpreter_predict(interpreter: tf.lite.Interpreter, sample: np.ndarray) -> np.ndarray:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    scale, zero_point = input_detail["quantization"]
    quantized_input = np.clip(np.round(sample / scale + zero_point), -128, 127).astype(np.int8)
    interpreter.set_tensor(input_detail["index"], quantized_input[np.newaxis, ...])
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])[0].astype(np.float32)
    output_scale, output_zero_point = output_detail["quantization"]
    return (output - output_zero_point) * output_scale


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "fault_classifier.keras"
    scaler_path = ARTIFACT_DIR / "feature_scaler.npz"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Train first with `python -m models.train`")

    model = tf.keras.models.load_model(model_path)
    scaler_values = np.load(scaler_path)
    scaler = StandardScaler()
    scaler.mean_ = scaler_values["mean"]
    scaler.scale_ = scaler_values["scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    float_metrics_path = ARTIFACT_DIR / "float_metrics.json"
    if not float_metrics_path.exists():
        raise FileNotFoundError("Float-model evaluation metadata is missing; retrain the model")
    float_metrics = json.loads(float_metrics_path.read_text(encoding="utf-8"))
    data_mode = float_metrics["dataset_source"]
    data_dir = float_metrics.get("data_dir", "data/real")
    recordings_per_class = int(float_metrics.get("recordings_per_class", 60))
    dataset_mode = "real" if data_mode.startswith("real") else "synthetic"
    features, labels, _ = build_feature_dataset(
        dataset_mode, data_dir, recordings_per_class=recordings_per_class, seed=42
    )
    train_x, test_x, _, test_y = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    standardized = scaler.transform(test_x).astype(np.float32)[..., np.newaxis]
    representative = scaler.transform(train_x).astype(np.float32)[..., np.newaxis]

    # Rebuild the same deterministic split used by training for comparable INT8 metrics.
    tflite_bytes = convert_to_int8(model, representative)
    tflite_path = ARTIFACT_DIR / "fault_model_int8.tflite"
    float_tflite_path = ARTIFACT_DIR / "fault_model_float32.tflite"
    header_path = ARTIFACT_DIR / "model_data.h"
    float_tflite_path.write_bytes(convert_to_float32(model))
    tflite_path.write_bytes(tflite_bytes)
    write_c_header(tflite_bytes, header_path)

    float_interpreter = tf.lite.Interpreter(model_path=str(float_tflite_path), num_threads=1)
    float_interpreter.allocate_tensors()
    float_probabilities = np.stack(
        [_float32_interpreter_predict(float_interpreter, sample) for sample in standardized]
    )
    float_metrics = evaluate_predictions(test_y, float_probabilities)
    float_latency = mean_latency_ms(
        lambda sample: _float32_interpreter_predict(float_interpreter, sample), standardized
    )
    float_metrics["latency_ms"] = float_latency
    (ARTIFACT_DIR / "float_tflite_metrics.json").write_text(
        json.dumps(float_metrics, indent=2), encoding="utf-8"
    )

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=1)
    interpreter.allocate_tensors()
    probabilities = np.stack([_interpreter_predict(interpreter, sample) for sample in standardized])
    metrics = evaluate_predictions(test_y, probabilities)
    latency = mean_latency_ms(lambda sample: _interpreter_predict(interpreter, sample), standardized)
    metrics["latency_ms"] = latency
    metrics["input_dtype"] = str(interpreter.get_input_details()[0]["dtype"])
    metrics["output_dtype"] = str(interpreter.get_output_details()[0]["dtype"])
    metrics_path = ARTIFACT_DIR / "int8_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_report("Float32 TFLite evaluation", float_metrics, float_latency, float_tflite_path)
    print_report("Full INT8 TFLite evaluation", metrics, latency, tflite_path)
    print(f"Float32 TFLite model size: {float_tflite_path.stat().st_size / 1024:.2f} KB")
    print(f"INT8 TFLite model size: {tflite_path.stat().st_size / 1024:.2f} KB")
    print(f"Exported C header: {header_path}")


if __name__ == "__main__":
    main()
