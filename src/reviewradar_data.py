"""Deterministic UCI data download, validation, de-duplication, and splitting."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
SOURCE_FILES = {
    "amazon_cells_labelled.txt": "amazon",
    "imdb_labelled.txt": "imdb",
    "yelp_labelled.txt": "yelp",
}
SPLIT_SEED = 42


@dataclass(frozen=True)
class QualityReport:
    raw_rows: int
    kept_rows: int
    malformed_rows: int
    empty_rows: int
    duplicate_rows_removed: int
    conflicting_text_groups_removed: int
    labels: dict[str, int]
    sources: dict[str, int]


def normalize_for_duplicate_check(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def download_raw_dataset(raw_dir: Path) -> Path:
    """Download the UCI archive once and return the extracted source directory."""
    extracted = raw_dir / "sentiment labelled sentences"
    if all((extracted / filename).exists() for filename in SOURCE_FILES):
        return extracted

    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "sentiment-labelled-sentences.zip"
    if not archive.exists():
        urlretrieve(DATASET_URL, archive)
    with zipfile.ZipFile(archive) as package:
        package.extractall(raw_dir)

    if not all((extracted / filename).exists() for filename in SOURCE_FILES):
        raise RuntimeError("The UCI archive did not contain the expected three labelled text files.")
    return extracted


def read_raw_rows(source_dir: Path) -> tuple[pd.DataFrame, int, int]:
    rows: list[dict[str, object]] = []
    malformed_rows = 0
    empty_rows = 0

    for filename, source_domain in SOURCE_FILES.items():
        with (source_dir / filename).open("r", encoding="utf-8", errors="replace") as source_file:
            for line_number, raw_line in enumerate(source_file, start=1):
                record = raw_line.rstrip("\r\n")
                if not record.strip():
                    empty_rows += 1
                    continue
                if "\t" not in record:
                    malformed_rows += 1
                    continue
                review, label_text = record.rsplit("\t", 1)
                review = review.strip()
                if not review or label_text not in {"0", "1"}:
                    malformed_rows += 1
                    continue
                rows.append(
                    {
                        "row_id": f"{source_domain}-{line_number:04d}",
                        "review": review,
                        "label": int(label_text),
                        "source": source_domain,
                        "normalized_review": normalize_for_duplicate_check(review),
                    }
                )

    return pd.DataFrame(rows), malformed_rows, empty_rows


def remove_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    grouped_labels = frame.groupby("normalized_review")["label"].nunique()
    conflicting_keys = set(grouped_labels[grouped_labels > 1].index)
    conflicting_groups = len(conflicting_keys)
    without_conflicts = frame.loc[~frame["normalized_review"].isin(conflicting_keys)].copy()
    deduplicated = without_conflicts.drop_duplicates(subset="normalized_review", keep="first").copy()
    # Report every excluded repeat or conflicting row in one transparent count.
    duplicates_removed = len(frame) - len(deduplicated)
    return deduplicated, duplicates_removed, conflicting_groups


def create_splits(frame: pd.DataFrame) -> dict[str, list[str]]:
    strata = frame["source"].astype(str) + "_" + frame["label"].astype(str)
    train, held_out = train_test_split(frame, test_size=0.4, random_state=SPLIT_SEED, stratify=strata)
    held_strata = held_out["source"].astype(str) + "_" + held_out["label"].astype(str)
    validation, test = train_test_split(held_out, test_size=0.5, random_state=SPLIT_SEED, stratify=held_strata)
    return {
        "train": train["row_id"].tolist(),
        "validation": validation["row_id"].tolist(),
        "test": test["row_id"].tolist(),
    }


def prepare_dataset(project_root: Path) -> QualityReport:
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    source_dir = download_raw_dataset(raw_dir)
    raw_frame, malformed_rows, empty_rows = read_raw_rows(source_dir)
    cleaned, duplicates_removed, conflicting_groups = remove_duplicates(raw_frame)
    if set(cleaned["label"].unique()) != {0, 1}:
        raise RuntimeError("Expected both binary sentiment labels after cleaning.")
    if cleaned.empty:
        raise RuntimeError("No usable UCI rows remain after validation.")

    splits = create_splits(cleaned)
    split_sets = {name: set(ids) for name, ids in splits.items()}
    if any(split_sets[first] & split_sets[second] for first in split_sets for second in split_sets if first != second):
        raise RuntimeError("Split IDs overlap after deterministic splitting.")
    if len(set().union(*split_sets.values())) != len(cleaned):
        raise RuntimeError("Split IDs do not cover every cleaned row.")

    processed_dir.mkdir(parents=True, exist_ok=True)
    cleaned.drop(columns="normalized_review").to_csv(processed_dir / "reviews.csv", index=False)
    (processed_dir / "split_ids.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    report = QualityReport(
        raw_rows=len(raw_frame) + malformed_rows + empty_rows,
        kept_rows=len(cleaned),
        malformed_rows=malformed_rows,
        empty_rows=empty_rows,
        duplicate_rows_removed=duplicates_removed,
        conflicting_text_groups_removed=conflicting_groups,
        labels={str(key): int(value) for key, value in cleaned["label"].value_counts().sort_index().items()},
        sources={str(key): int(value) for key, value in cleaned["source"].value_counts().sort_index().items()},
    )
    (processed_dir / "data_quality.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report
