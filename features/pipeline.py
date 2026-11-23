"""Convert synchronized records into labeled overlapping feature windows."""

import numpy as np

from data_sim.generator import SAMPLE_RATE_HZ, SensorRecord
from features.extract import FEATURE_COUNT, extract_window_features
from features.windowing import WINDOW_SIZE, WINDOW_STEP, window_signal


def record_to_features(record: SensorRecord) -> np.ndarray:
    """Extract a feature matrix from one sensor record."""
    vibration_windows = window_signal(record.vibration, WINDOW_SIZE, WINDOW_STEP)
    temperature_windows = window_signal(record.temperature, WINDOW_SIZE, WINDOW_STEP)
    current_windows = window_signal(record.current, WINDOW_SIZE, WINDOW_STEP)
    if not (len(vibration_windows) == len(temperature_windows) == len(current_windows)):
        raise ValueError("sensor streams must have equal sample counts")
    if not len(vibration_windows):
        return np.empty((0, FEATURE_COUNT), dtype=np.float32)
    return np.stack(
        [
            extract_window_features(vib, temp, amps, SAMPLE_RATE_HZ)
            for vib, temp, amps in zip(vibration_windows, temperature_windows, current_windows)
        ]
    )
