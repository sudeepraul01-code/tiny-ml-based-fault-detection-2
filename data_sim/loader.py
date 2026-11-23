"""Load supported local CSV and CWRU-style MAT recordings."""

import csv
import logging
from pathlib import Path

import numpy as np

from data_sim.generator import CLASS_NAMES, SensorRecord

LABEL_ALIASES = {
    "normal": 0,
    "healthy": 0,
    "no fault": 0,
    "bearing": 1,
    "inner": 1,
    "outer": 1,
    "ball": 1,
    "defect": 1,
    "imbalance": 2,
    "unbalance": 2,
    "misalignment": 3,
    "overheat": 4,
    "thermal": 4,
    "overheating": 4,
    "fault": 1,
}


def _infer_label(value: str) -> int:
    normalized = value.strip().lower().replace("_", " ").replace("-", " ")
    if normalized.isdigit() and int(normalized) in range(len(CLASS_NAMES)):
        return int(normalized)
    for keyword, label in LABEL_ALIASES.items():
        if keyword in normalized:
            return label
    raise ValueError(f"Cannot infer one of the five health states from {value!r}")


def _column(row: dict, aliases: tuple, default: float = 0.0) -> float:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for alias in aliases:
        value = lowered.get(alias)
        if value not in (None, ""):
            return float(value)
    return default


def _load_csv(path: Path) -> SensorRecord:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if not rows or not reader.fieldnames:
        raise ValueError(f"{path} has no header or sample rows")

    label_column = next(
        (name for name in reader.fieldnames if name.strip().lower() in ("label", "state", "fault")),
        None,
    )
    label = _infer_label(str(rows[0][label_column])) if label_column else _infer_label(str(path))
    vibration = np.asarray(
        [
            [
                _column(row, ("x", "acc_x", "accel_x", "vibration_x", "de_time", "sensor")),
                _column(row, ("y", "acc_y", "accel_y", "vibration_y")),
                _column(row, ("z", "acc_z", "accel_z", "vibration_z")),
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    temperature = np.asarray(
        [_column(row, ("temperature", "temp", "lm35"), 25.0) for row in rows],
        dtype=np.float32,
    )
    current = np.asarray(
        [_column(row, ("current", "amps", "acs712"), 2.0) for row in rows],
        dtype=np.float32,
    )
    return SensorRecord(vibration, temperature, current, label)


def _load_numeric_table(path: Path) -> SensorRecord:
    """Load headerless, whitespace/tab-delimited bearing test data by state folder."""
    label = _infer_label(str(path))
    samples = np.genfromtxt(path, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError(f"{path} does not contain a numeric sample table")
    vibration = np.zeros((samples.shape[0], 3), dtype=np.float32)
    channel_count = min(samples.shape[1], 3)
    vibration[:, :channel_count] = samples[:, :channel_count]
    count = samples.shape[0]
    return SensorRecord(
        vibration,
        np.full(count, 25.0, dtype=np.float32),
        np.full(count, 2.0, dtype=np.float32),
        label,
    )


def _load_mat(path: Path) -> SensorRecord:
    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise RuntimeError("SciPy is required to read CWRU .mat files") from error

    data = loadmat(path)
    signal_key = next(
        (
            key
            for key in data
            if not key.startswith("__")
            and any(token in key.lower() for token in ("de_time", "fe_time", "ba_time", "time"))
        ),
        None,
    )
    if signal_key is None:
        raise ValueError(f"No recognized vibration channel found in {path}")
    vibration = np.asarray(data[signal_key], dtype=np.float32).reshape(-1, 1)
    vibration = np.pad(vibration, ((0, 0), (0, 2)))
    label = _infer_label(str(path))
    count = vibration.shape[0]
    return SensorRecord(
        vibration,
        np.full(count, 25.0, dtype=np.float32),
        np.full(count, 2.0, dtype=np.float32),
        label,
    )


def load_local_records(data_dir: str | Path) -> list[SensorRecord]:
    """Load labeled CSV, MAT, or numeric text files; report malformed supported files."""
    root = Path(data_dir)
    if not root.exists():
        return []
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".csv":
                record = _load_csv(path)
            elif path.suffix.lower() in (".txt", ".dat"):
                record = _load_numeric_table(path)
            elif path.suffix.lower() == ".mat":
                record = _load_mat(path)
            else:
                continue
        except (ValueError, OSError, RuntimeError, csv.Error) as error:
            logging.warning("Skipping local dataset file %s: %s", path, error)
            continue
        if record.vibration.shape[0] >= 128:
            records.append(record)
    return records
