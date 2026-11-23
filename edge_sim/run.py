"""Run a real-time-style local sensor stream through the INT8 classifier."""

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_sim.generator import CLASS_NAMES, generate_recording
from edge_sim.alert_logger import AlertLogger
from edge_sim.protection import FaultProtection
from features.extract import extract_window_features
from features.windowing import WINDOW_SIZE, WINDOW_STEP
from quantize.convert import _interpreter_predict

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "models" / "artifacts"
SCENARIOS = {
    "normal": 0,
    "bearing_wear": 1,
    "imbalance": 2,
    "misalignment": 3,
    "overheating": 4,
}


def _stream_states(scenario: str, windows: int) -> list[int]:
    if scenario != "cycle":
        return [SCENARIOS[scenario]] * windows
    states = [0, 1, 1, 0, 0, 0, 2, 2, 0, 0, 0, 3, 3, 4, 4]
    return [states[index % len(states)] for index in range(windows)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("cycle", *SCENARIOS), default="cycle")
    parser.add_argument("--windows", type=int, default=30)
    parser.add_argument("--interval", type=float, default=0.25, help="seconds between simulated windows")
    args = parser.parse_args()
    if args.windows < 1 or args.interval < 0:
        parser.error("windows must be positive and interval must be non-negative")

    model_path = ARTIFACT_DIR / "fault_model_int8.tflite"
    scaler_path = ARTIFACT_DIR / "feature_scaler.npz"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Run `python -m models.train` and `python -m quantize.convert` first")
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    scaling = np.load(scaler_path)
    protection = FaultProtection(threshold=0.85, trip_windows=2, recovery_windows=3)
    logger = AlertLogger()
    states = _stream_states(args.scenario, args.windows)
    initial_record = generate_recording(states[0], sample_count=WINDOW_SIZE, seed=1000)
    window_buffer = initial_record.vibration
    temperature_buffer = initial_record.temperature
    current_buffer = initial_record.current

    print("TinyML industrial fault-detection edge simulation")
    print("window | simulated sensors (x RMS / temp / current) | prediction | confidence | latency | relay/buzzer/LED")
    for index, state in enumerate(states):
        features = extract_window_features(window_buffer, temperature_buffer, current_buffer)
        normalized = ((features - scaling["mean"]) / scaling["scale"]).astype(np.float32)[:, np.newaxis]

        inference_start = time.perf_counter()
        probabilities = _interpreter_predict(interpreter, normalized)
        latency_ms = (time.perf_counter() - inference_start) * 1000.0
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])
        event = protection.update(prediction, confidence, CLASS_NAMES[prediction])
        if event is not None:
            logger.write(event)
            print(f"  ALERT: {event.event.upper()} — {event.state_name}, confidence {event.confidence:.3f}")

        actuator = "ON" if protection.active else "off"
        print(
            f"{index + 1:>6} | {np.sqrt(np.mean(window_buffer[:, 0] ** 2)):.3f} / "
            f"{temperature_buffer.mean():.1f} C / {current_buffer.mean():.2f} A | "
            f"{CLASS_NAMES[prediction]:<13} | {confidence:>6.1%} | "
            f"{latency_ms:>6.2f} ms | relay={actuator}, buzzer={actuator}, LED={actuator}"
        )
        if index + 1 < len(states):
            next_state = states[index + 1]
            next_record = generate_recording(
                next_state, sample_count=WINDOW_STEP, seed=1000 + index + 1
            )
            window_buffer = np.concatenate((window_buffer[WINDOW_STEP:], next_record.vibration))
            temperature_buffer = np.concatenate(
                (temperature_buffer[WINDOW_STEP:], next_record.temperature)
            )
            current_buffer = np.concatenate((current_buffer[WINDOW_STEP:], next_record.current))
        if args.interval:
            time.sleep(args.interval)
    print(f"Alert logs: {logger.json_path} and {logger.csv_path}")


if __name__ == "__main__":
    main()
