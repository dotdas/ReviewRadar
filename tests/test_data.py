from __future__ import annotations

import pandas as pd

from reviewradar_data import create_splits, remove_duplicates


def test_duplicate_and_conflicting_texts_are_excluded() -> None:
    frame = pd.DataFrame(
        [
            {"row_id": "amazon-0001", "review": "Good", "label": 1, "source": "amazon", "normalized_review": "good"},
            {"row_id": "amazon-0002", "review": " good ", "label": 1, "source": "amazon", "normalized_review": "good"},
            {"row_id": "imdb-0001", "review": "Mixed", "label": 0, "source": "imdb", "normalized_review": "mixed"},
            {"row_id": "yelp-0001", "review": "MIXED", "label": 1, "source": "yelp", "normalized_review": "mixed"},
        ]
    )

    cleaned, removed, conflicts = remove_duplicates(frame)

    assert cleaned["row_id"].tolist() == ["amazon-0001"]
    assert removed == 3
    assert conflicts == 1


def test_splits_are_disjoint_and_cover_rows() -> None:
    frame = pd.DataFrame(
        [
            {"row_id": f"{source}-{label}-{index}", "source": source, "label": label}
            for source in ("amazon", "imdb", "yelp")
            for label in (0, 1)
            for index in range(10)
        ]
    )

    splits = create_splits(frame)
    split_sets = {name: set(ids) for name, ids in splits.items()}

    assert not (split_sets["train"] & split_sets["validation"])
    assert not (split_sets["train"] & split_sets["test"])
    assert not (split_sets["validation"] & split_sets["test"])
    assert set().union(*split_sets.values()) == set(frame["row_id"])
