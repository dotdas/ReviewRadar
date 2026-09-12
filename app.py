from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.reviewradar_model import predict_one


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MAX_BYTES = 2 * 1024 * 1024
MAX_ROWS = 1_000
MAX_CHARS = 5_000

st.set_page_config(page_title="ReviewRadar", page_icon="R", layout="wide")
st.markdown("""<style>
    .stApp { background: #101314; color: #edf0ed; }
    [data-testid="stMetric"] { background: #171c1c; border: 1px solid #2a3432; padding: .65rem; }
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    path = ARTIFACTS / "reviewradar_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError("Saved model artifact is missing. Run scripts/train_model.py locally; the app will not retrain itself.")
    return joblib.load(path)


@st.cache_data
def load_static_artifacts():
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    samples = pd.read_csv(ARTIFACTS / "sample_reviews.csv")
    errors = pd.read_csv(ARTIFACTS / "test_errors.csv")
    return metrics, samples, errors


def safe_cell(value: object) -> object:
    if isinstance(value, str) and value[:1] in {"=", "+", "-", "@"}:
        return "'" + value
    return value


def export_csv(frame: pd.DataFrame) -> bytes:
    return frame.map(safe_cell).to_csv(index=False).encode("utf-8-sig")


def explain(result: dict) -> None:
    if result["status"] != "ok":
        st.info("Insufficient known text: none of these terms are in the model vocabulary, so no sentiment label was assigned.")
        return
    st.metric("Predicted sentiment", result["sentiment"].title())
    st.caption(f"Positive-class probability: {result['positive_probability']:.1%}. This is a model score, not calibrated confidence.")
    contribution = pd.DataFrame(result["contributions"])
    contribution["direction"] = contribution["contribution"].map(lambda value: "toward positive" if value > 0 else "toward negative")
    st.dataframe(contribution[["term", "contribution", "direction"]], hide_index=True, use_container_width=True)
    st.caption("Terms are TF-IDF-weighted contributions to the model score. They do not prove why a person liked or disliked a review.")


def predict_frame(model, input_frame: pd.DataFrame, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid, skipped = [], []
    for row_number, value in input_frame[column].items():
        text = "" if pd.isna(value) else str(value).strip()
        if not text:
            skipped.append({"row_id": row_number + 1, "review": "", "reason": "Empty review"})
        elif len(text) > MAX_CHARS:
            skipped.append({"row_id": row_number + 1, "review": text, "reason": f"Review exceeds {MAX_CHARS:,} characters"})
        else:
            outcome = predict_one(model, text)
            valid.append({"row_id": row_number + 1, "review": text, "predicted_sentiment": outcome["sentiment"], "positive_probability": outcome["positive_probability"], "status": outcome["status"]})
    return pd.DataFrame(valid), pd.DataFrame(skipped)


st.title("ReviewRadar")
st.write("Inspect English review batches with a saved binary sentiment model and transparent word-level contributions.")
st.caption("Uploads stay in this server session only. They are not written to disk, logs, analytics, or persistent storage.")
if st.button("Clear session"):
    st.session_state.clear()
    st.rerun()

try:
    model = load_model()
    metrics, samples, errors = load_static_artifacts()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(str(error))
    st.stop()

batch_tab, single_tab, evaluation_tab = st.tabs(["Batch Reviews", "Single Review", "Model & Evaluation"])

with batch_tab:
    st.subheader("Batch reviews")
    st.caption(f"UTF-8 CSV only; maximum {MAX_BYTES // 1024 // 1024} MB, {MAX_ROWS:,} rows, and {MAX_CHARS:,} characters per review.")
    template = pd.DataFrame({"review": ["The service was excellent.", "I would not recommend this."]})
    st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8"), "reviewradar_template.csv", "text/csv")
    use_sample = st.button("Try sample reviews")
    uploaded = st.file_uploader("Upload a CSV", type="csv")
    source_frame = None
    if use_sample:
        source_frame = samples[["review"]].copy()
        st.info("Using 30 held-out evaluation examples, balanced across source domains and labels. This is not independent evidence.")
    elif uploaded is not None:
        if uploaded.size > MAX_BYTES:
            st.error("This file exceeds the 2 MB limit.")
        else:
            try:
                source_frame = pd.read_csv(uploaded, encoding="utf-8-sig")
            except Exception as error:
                st.error(f"Could not parse this UTF-8 CSV: {error}")
    if source_frame is not None:
        if len(source_frame) > MAX_ROWS:
            st.error(f"This file contains {len(source_frame):,} rows; the maximum is {MAX_ROWS:,}.")
        elif source_frame.empty:
            st.error("The CSV has no rows.")
        else:
            default = list(source_frame.columns).index("review") if "review" in source_frame.columns else 0
            column = st.selectbox("Review column", source_frame.columns, index=default)
            if st.button("Analyze reviews"):
                results, skipped = predict_frame(model, source_frame, column)
                st.session_state["batch_results"] = results
                st.session_state["batch_skipped"] = skipped
    results = st.session_state.get("batch_results")
    skipped = st.session_state.get("batch_skipped")
    if isinstance(results, pd.DataFrame):
        labeled = results.loc[results["status"] == "ok"].copy()
        columns = st.columns(4)
        columns[0].metric("Analyzed", len(labeled)); columns[1].metric("Positive", int((labeled["predicted_sentiment"] == "positive").sum())); columns[2].metric("Negative", int((labeled["predicted_sentiment"] == "negative").sum())); columns[3].metric("Skipped", len(skipped) + int((results["status"] != "ok").sum()))
        if not labeled.empty:
            st.bar_chart(labeled["predicted_sentiment"].value_counts())
        negative_only = st.checkbox("Show negative reviews only")
        displayed = labeled.loc[labeled["predicted_sentiment"].eq("negative")] if negative_only else labeled
        st.dataframe(displayed, hide_index=True, use_container_width=True)
        st.download_button("Download displayed results", export_csv(displayed), "reviewradar_results.csv", "text/csv")
        if not displayed.empty:
            selected_id = st.selectbox("Explain a displayed review", displayed["row_id"].tolist())
            selected = displayed.loc[displayed["row_id"] == selected_id, "review"].iloc[0]
            explain(predict_one(model, selected))
        oov = results.loc[results["status"] != "ok"]
        if not oov.empty:
            st.warning("Reviews with insufficient known text")
            st.dataframe(oov[["row_id", "review"]], hide_index=True)
        if isinstance(skipped, pd.DataFrame) and not skipped.empty:
            st.warning("Skipped input rows")
            st.dataframe(skipped, hide_index=True)

with single_tab:
    st.subheader("Single review")
    review = st.text_area("Paste an English review", max_chars=MAX_CHARS, placeholder="The service was excellent, but the wait was long.")
    if st.button("Predict review"):
        if not review.strip():
            st.warning("Enter a review before predicting.")
        else:
            explain(predict_one(model, review.strip()))

with evaluation_tab:
    st.subheader("Model and evaluation")
    st.write("Dataset: UCI Sentiment Labelled Sentences (3,000 originally labelled Amazon, Yelp, and IMDb sentences; no neutral class), CC BY 4.0.")
    st.write(f"Selected model: **{metrics['selected_model']['name']}** with parameter `{metrics['selected_model']['parameter']}` selected by validation macro-F1.")
    first, second, third = st.columns(3)
    first.metric("Test macro-F1", f"{metrics['test']['macro_f1']:.3f}"); second.metric("Test accuracy", f"{metrics['test']['accuracy']:.3f}"); third.metric("Dummy validation macro-F1", f"{metrics['dummy_validation']['macro_f1']:.3f}")
    st.json(metrics["by_source_domain"])
    st.caption("Limitations: English positive/negative only. Mixed, neutral, sarcastic, and unfamiliar-domain reviews can be misclassified. Probabilities are not calibrated confidence or business urgency.")
    st.dataframe(errors, hide_index=True, use_container_width=True)
