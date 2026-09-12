from __future__ import annotations

from pathlib import Path

import joblib

from reviewradar_model import predict_one


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_oov_text_returns_insufficient_state() -> None:
    model = joblib.load(PROJECT_ROOT / "artifacts" / "reviewradar_pipeline.joblib")
    result = predict_one(model, "qzxvplm nonlexicaltoken", top_n=5)

    assert result["status"] == "insufficient_known_text"
    assert result["sentiment"] is None
    assert result["positive_probability"] is None


def test_saved_model_explanation_is_directional_and_bounded() -> None:
    model = joblib.load(PROJECT_ROOT / "artifacts" / "reviewradar_pipeline.joblib")
    result = predict_one(model, "The movie was excellent and enjoyable", top_n=5)

    assert result["status"] == "ok"
    assert 0 <= result["positive_probability"] <= 1
    assert result["contributions"]
    assert all("term" in item and "contribution" in item for item in result["contributions"])
