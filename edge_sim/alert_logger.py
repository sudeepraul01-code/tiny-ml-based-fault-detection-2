"""Append alert events to local JSON Lines and CSV logs."""

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from edge_sim.protection import ProtectionEvent


class AlertLogger:
    def __init__(self, log_dir: str | Path = "edge_sim/logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.log_dir / "alerts.jsonl"
        self.csv_path = self.log_dir / "alerts.csv"

    def write(self, event: ProtectionEvent) -> None:
        row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **asdict(event)}
        with self.json_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        fields = list(row)
        needs_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
