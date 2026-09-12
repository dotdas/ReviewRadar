"""Training, evaluation, prediction, and transparent feature contributions."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def make_pipeline(model_name: str, value: float) -> Pipeline:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000, sublinear_tf=True)
    if model_name == "logistic_regression":
        classifier = LogisticRegression(C=value, max_iter=2000, random_state=42)
    elif model_name == "multinomial_nb":
        classifier = MultinomialNB(alpha=value)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def evaluate(model: Pipeline, frame: pd.DataFrame) -> dict:
    predictions = model.predict(frame["review"])
    report = classification_report(frame["label"], predictions, labels=[0, 1], target_names=["negative", "positive"], output_dict=True, zero_division=0)
    return {
        "macro_f1": float(f1_score(frame["label"], predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(frame["label"], predictions)),
        "negative_recall": float(recall_score(frame["label"], predictions, pos_label=0, zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(frame["label"], predictions, labels=[0, 1]).tolist(),
    }


def choose_model(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[str, float, Pipeline, dict]:
    candidates = [("logistic_regression", value) for value in (0.5, 1, 2)] + [("multinomial_nb", value) for value in (0.5, 1)]
    scored = []
    for name, value in candidates:
        model = make_pipeline(name, value).fit(train["review"], train["label"])
        metrics = evaluate(model, validation)
        scored.append((name, value, model, metrics))
    scored.sort(key=lambda item: (item[3]["macro_f1"], item[3]["negative_recall"], item[0] == "logistic_regression"), reverse=True)
    return scored[0]


def evaluate_dummy(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    model = DummyClassifier(strategy="most_frequent").fit(train[["label"]], train["label"])
    predictions = model.predict(validation[["label"]])
    return {"macro_f1": float(f1_score(validation["label"], predictions, average="macro", zero_division=0)), "accuracy": float(accuracy_score(validation["label"], predictions))}


def predict_one(model: Pipeline, review: str, top_n: int = 12) -> dict:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    vector = vectorizer.transform([review])
    if vector.nnz == 0:
        return {"status": "insufficient_known_text", "sentiment": None, "positive_probability": None, "contributions": [], "intercept_or_prior": None}

    feature_names = vectorizer.get_feature_names_out()
    indices = vector.indices
    values = vector.data
    if isinstance(classifier, LogisticRegression):
        weights = classifier.coef_[0]
        contributions = values * weights[indices]
        intercept_or_prior = float(classifier.intercept_[0])
    elif isinstance(classifier, MultinomialNB):
        weights = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
        contributions = values * weights[indices]
        intercept_or_prior = float(classifier.class_log_prior_[1] - classifier.class_log_prior_[0])
    else:
        raise ValueError("Only deployable ReviewRadar classifiers support explanations.")

    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    explanation = [{"term": str(feature_names[indices[index]]), "contribution": float(contributions[index])} for index in order]
    probability = float(model.predict_proba([review])[0, list(classifier.classes_).index(1)])
    return {
        "status": "ok",
        "sentiment": "positive" if probability >= 0.5 else "negative",
        "positive_probability": probability,
        "contributions": explanation,
        "intercept_or_prior": intercept_or_prior,
    }


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
