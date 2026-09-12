"""Select on validation data, test exactly once, and save the evaluated pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reviewradar_model import choose_model, evaluate, evaluate_dummy, file_sha256, make_pipeline, predict_one


def subset(frame: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    return frame.set_index("row_id").loc[ids].reset_index()


def main() -> None:
    processed = PROJECT_ROOT / "data" / "processed"
    artifacts = PROJECT_ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    frame = pd.read_csv(processed / "reviews.csv")
    splits = json.loads((processed / "split_ids.json").read_text(encoding="utf-8"))
    train, validation, test = (subset(frame, splits[name]) for name in ("train", "validation", "test"))
    dummy = evaluate_dummy(train, validation)
    name, value, _, validation_metrics = choose_model(train, validation)
    selected = make_pipeline(name, value).fit(pd.concat([train, validation])["review"], pd.concat([train, validation])["label"])
    test_metrics = evaluate(selected, test)
    model_path = artifacts / "reviewradar_pipeline.joblib"
    joblib.dump(selected, model_path)
    reloaded = joblib.load(model_path)
    if list(selected.predict(test["review"])) != list(reloaded.predict(test["review"])):
        raise RuntimeError("Saved pipeline predictions differ from the evaluated pipeline.")

    domain_metrics = {domain: {"sample_count": int(len(group)), **evaluate(selected, group)} for domain, group in test.groupby("source")}
    cross_domain = {}
    for held_domain in sorted(frame["source"].unique()):
        domain_train = pd.concat([train, validation]).query("source != @held_domain")
        domain_test = test.query("source == @held_domain")
        cross_model = make_pipeline(name, value).fit(domain_train["review"], domain_train["label"])
        cross_domain[held_domain] = {"train_domains": sorted(domain_train["source"].unique().tolist()), "sample_count": int(len(domain_test)), **evaluate(cross_model, domain_test)}

    errors = test.assign(prediction=selected.predict(test["review"]))
    errors = errors.loc[errors["label"] != errors["prediction"]].sample(frac=1, random_state=42).head(20)
    errors[["row_id", "source", "review", "label", "prediction"]].to_csv(artifacts / "test_errors.csv", index=False)
    sample = test.groupby(["source", "label"], group_keys=False).apply(lambda group: group.sample(n=5, random_state=42), include_groups=False)
    sample.to_csv(artifacts / "sample_reviews.csv", index=False)

    metrics = {"dataset_rows": int(len(frame)), "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)}, "dummy_validation": dummy, "selected_model": {"name": name, "parameter": value, "validation": validation_metrics}, "test": test_metrics, "by_source_domain": domain_metrics, "cross_domain_robustness_experiment": cross_domain, "artifact_sha256": file_sha256(model_path), "explanation_note": "Word contributions are model-score contributions, not proof of human preference."}
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"selected_model": name, "parameter": value, "test_macro_f1": test_metrics["macro_f1"], "test_accuracy": test_metrics["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
