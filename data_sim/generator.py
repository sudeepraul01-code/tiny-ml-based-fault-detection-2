"""Deterministic software sensor generator for industrial machine states."""

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE_HZ = 1000
CLASS_NAMES = ("Normal", "Bearing Wear", "Imbalance", "Misalignment", "Overheating")


@dataclass(frozen=True)
class SensorRecord:
    """A synchronized multichannel sensor recording and its health-state label."""

    vibration: np.ndarray
    temperature: np.ndarray
    current: np.ndarray
    label: int


def generate_recording(
    label: int,
    sample_count: int = 1024,
    seed: int = 42,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> SensorRecord:
    """Generate an independent three-axis vibration, temperature, and current record."""
    if label not in range(len(CLASS_NAMES)):
        raise ValueError(f"label must be in 0..{len(CLASS_NAMES) - 1}, got {label}")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if sample_rate_hz < 1:
        raise ValueError("sample_rate_hz must be positive")

    rng = np.random.default_rng(seed)
    time = np.arange(sample_count, dtype=np.float32) / sample_rate_hz
    phase = rng.uniform(-np.pi, np.pi, size=3)
    base_frequency = rng.uniform(27.0, 34.0)
    vibration = np.empty((sample_count, 3), dtype=np.float32)

    for axis in range(3):
        fundamental = np.sin(2 * np.pi * base_frequency * time + phase[axis])
        signal = 0.22 * fundamental + 0.025 * rng.normal(size=sample_count)
        if label == 1:  # Bearing wear: periodic impacts excite a damped high-frequency ring.
            impact_period = max(1, int(sample_rate_hz / rng.uniform(10.0, 15.0)))
            for impact in range(axis * 7, sample_count, impact_period):
                length = min(sample_count - impact, 28)
                ring_time = np.arange(length) / sample_rate_hz
                signal[impact : impact + length] += (
                    0.8
                    * np.exp(-ring_time * 95)
                    * np.sin(2 * np.pi * 185 * ring_time)
                )
        elif label == 2:  # Imbalance: strong once-per-revolution radial vibration.
            signal += (0.85 + 0.08 * axis) * fundamental
        elif label == 3:  # Misalignment: strong second and third harmonics.
            signal += (
                0.62 * np.sin(4 * np.pi * base_frequency * time + phase[axis])
                + 0.24 * np.sin(6 * np.pi * base_frequency * time + phase[axis] / 2)
            )
        elif label == 4:  # Overheating also increases the motor-load current.
            signal += 0.15 * np.sin(4 * np.pi * base_frequency * time + phase[axis])
        vibration[:, axis] = signal

    temperature_mean = 25.0 if label == 0 else (27.0 if label in (1, 2, 3) else 88.0)
    current_mean = 2.0 if label == 0 else (2.35 if label in (1, 2, 3) else 3.9)
    temperature = (temperature_mean + rng.normal(0, 0.35, sample_count)).astype(np.float32)
    current = (current_mean + rng.normal(0, 0.06, sample_count)).astype(np.float32)
    return SensorRecord(vibration, temperature, current, label)
