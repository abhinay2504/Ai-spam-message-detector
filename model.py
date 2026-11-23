"""Training and prediction helpers for the spam message classifier."""
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "messages.csv"


def train_model(data_path: str | Path = DATA_PATH) -> Pipeline:
    """Train a TF-IDF + Logistic Regression text classifier."""
    data = pd.read_csv(data_path)
    required = {"text", "label"}
    if not required.issubset(data.columns):
        raise ValueError("Training CSV must contain 'text' and 'label' columns.")
    if data.empty or data["text"].isna().any() or data["label"].isna().any():
        raise ValueError("Training data must not be empty or contain missing values.")
    labels = set(data["label"].str.lower())
    if labels != {"spam", "ham"}:
        raise ValueError("Training data must contain both 'spam' and 'ham' labels.")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(data["text"].astype(str), data["label"].str.lower())
    return pipeline


def predict_message(model: Pipeline, message: str) -> dict[str, float | str]:
    """Return the predicted label and model probability for a message."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please enter a message containing at least one character.")
    probabilities = model.predict_proba([message])[0]
    classes = list(model.classes_)
    best_index = int(probabilities.argmax())
    label = str(classes[best_index])
    return {"label": label, "confidence": float(probabilities[best_index])}