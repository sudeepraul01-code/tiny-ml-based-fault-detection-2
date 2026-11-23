"""Overlapping fixed-size windows for synchronized sensor streams."""

import numpy as np

WINDOW_SIZE = 128
WINDOW_STEP = 64


def window_signal(
    signal: np.ndarray,
    window_size: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
) -> np.ndarray:
    """Return all complete windows along the first axis (no padding is applied)."""
    values = np.asarray(signal)
    if values.ndim == 0:
        raise ValueError("signal must have at least one dimension")
    if window_size < 1 or step < 1:
        raise ValueError("window_size and step must be positive")
    if values.shape[0] < window_size:
        return np.empty((0, window_size) + values.shape[1:], dtype=values.dtype)
    starts = range(0, values.shape[0] - window_size + 1, step)
    return np.stack([values[start : start + window_size] for start in starts])
