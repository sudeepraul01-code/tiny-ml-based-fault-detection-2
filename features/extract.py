"""Exactly 18 time/frequency features from synchronized sensor windows."""

import numpy as np

FEATURE_NAMES = tuple(
    f"vibration_{axis}_{name}"
    for axis in ("x", "y", "z")
    for name in ("rms", "kurtosis", "peak_to_peak", "dominant_hz", "spectral_centroid_hz")
) + ("temperature_mean", "temperature_max", "current_mean")
FEATURE_COUNT = 18


def _axis_features(signal: np.ndarray, sample_rate_hz: int) -> list[float]:
    centered = np.asarray(signal, dtype=np.float64) - np.mean(signal)
    rms = float(np.sqrt(np.mean(np.square(centered))))
    variance = float(np.mean(np.square(centered)))
    kurtosis = float(np.mean(np.power(centered, 4)) / (variance * variance) - 3.0) if variance > 1e-12 else 0.0
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate_hz)
    spectrum = np.abs(np.fft.rfft(centered))
    if spectrum.size:
        spectrum[0] = 0.0
    total = float(np.sum(spectrum))
    if total > 1e-12:
        dominant_hz = float(frequencies[int(np.argmax(spectrum))])
        centroid_hz = float(np.sum(frequencies * spectrum) / total)
    else:
        dominant_hz = 0.0
        centroid_hz = 0.0
    return [rms, kurtosis, float(np.ptp(signal)), dominant_hz, centroid_hz]


def extract_window_features(
    vibration: np.ndarray,
    temperature: np.ndarray,
    current: np.ndarray,
    sample_rate_hz: int = 1000,
) -> np.ndarray:
    """Extract 18 ordered features from one three-axis, temperature, and current window."""
    vibration = np.asarray(vibration)
    temperature = np.asarray(temperature).reshape(-1)
    current = np.asarray(current).reshape(-1)
    if vibration.ndim != 2 or vibration.shape[1] != 3:
        raise ValueError("vibration must have shape (samples, 3)")
    if vibration.shape[0] == 0 or temperature.size != vibration.shape[0] or current.size != vibration.shape[0]:
        raise ValueError("vibration, temperature, and current must have equal non-zero lengths")
    if sample_rate_hz < 1:
        raise ValueError("sample_rate_hz must be positive")

    features = []
    for axis in range(3):
        features.extend(_axis_features(vibration[:, axis], sample_rate_hz))
    features.extend(
        [
            float(np.mean(temperature)),
            float(np.max(temperature)),
            float(np.mean(current)),
        ]
    )
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (FEATURE_COUNT,):
        raise RuntimeError(f"feature extraction produced {result.size} values, expected {FEATURE_COUNT}")
    return result
