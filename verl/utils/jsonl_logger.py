"""
JSONL file logger for veRL training.

Writes metrics to a JSONL file at each step, enabling live dashboard
visualization via the companion plot_training.py script.
"""
import json
import os
import numbers
from pathlib import Path


class JSONLLogger:
    """Logger that appends metrics as JSON lines to a file."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # Truncate if starting fresh
        with open(self.log_path, 'w') as f:
            pass  # create empty file

    def log(self, data: dict, step: int):
        record = {'step': step}
        for k, v in data.items():
            if isinstance(v, numbers.Number):
                record[k] = round(float(v), 6)
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def flush(self):
        pass
