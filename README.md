# ReviewRadar

ReviewRadar is an English-language Streamlit prototype for inspecting batches of reviews with positive/negative sentiment predictions and transparent word-level model contributions.

## Status

Scaffold only. The reproducible data preparation, model training, evaluation, Streamlit application, and public demo will be added in recoverable milestones.

## Project principles

- Python 3.11, scikit-learn, and Streamlit.
- UCI Sentiment Labelled Sentences with attribution and CC BY 4.0 licensing information.
- Offline training only; the app loads a saved evaluated pipeline and never retrains on startup.
- User uploads remain in the current server session and are not written to disk or committed.
- English binary sentiment only; mixed, neutral, sarcastic, and unfamiliar-domain reviews can be misclassified.
