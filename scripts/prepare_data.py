"""Download, validate, de-duplicate, and split the UCI review dataset."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reviewradar_data import prepare_dataset


if __name__ == "__main__":
    report = prepare_dataset(PROJECT_ROOT)
    print(f"Prepared {report.kept_rows} rows from {report.raw_rows} raw lines.")
    print(f"Removed {report.duplicate_rows_removed} duplicate/conflicting rows.")
