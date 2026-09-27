"""
predict.py — Interactive source-style classifier with word-level explanations.

Usage:
    python predict.py --text "Government announces new education policy"
    python predict.py                    # interactive mode
    python predict.py --file article.txt # read from file
"""

import argparse
import os
import re
import sys

import joblib
import numpy as np

# ---------- Paths (relative to this file) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
LR_PATH    = os.path.join(MODELS_DIR, "logistic_regression.pkl")
RF_PATH    = os.path.join(MODELS_DIR, "random_forest.pkl")
BERT_DIR   = os.path.join(MODELS_DIR, "distilbert_fake_news")

# The dataset uses 0 for Fake.csv and 1 for True.csv.  Keep this mapping in
# one place, but always obtain the winning value from model.classes_ (rather
# than assuming the probability array is ordered as [0, 1]).
LABELS = {0: "FAKE", 1: "REAL"}


# ---------- Text cleaning (must match training) ----------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------- Load models ----------
def load_models():
    print("[1/4] Loading TF-IDF vectorizer...")
    if not os.path.exists(TFIDF_PATH):
        sys.exit(f"ERROR: {TFIDF_PATH} not found. Run the training notebook first.")
    tfidf = joblib.load(TFIDF_PATH)

    print("[2/4] Loading Logistic Regression...")
    if not os.path.exists(LR_PATH):
        sys.exit(f"ERROR: {LR_PATH} not found.")
    lr_model = joblib.load(LR_PATH)

    print("[3/4] Loading Random Forest...")
    rf_model = joblib.load(RF_PATH) if os.path.exists(RF_PATH) else None
    if rf_model is None:
        print("      (skipped — random_forest.pkl not found)")

    print("[4/4] Loading DistilBERT...")
    bert_pipeline = None
    if os.path.isdir(BERT_DIR):
        try:
            import torch
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            bert_pipeline = pipeline(
                "text-classification",
                model=BERT_DIR,
                tokenizer=BERT_DIR,
                device=device,
            )
            where = "GPU" if device == 0 else "CPU"
            print(f"      DistilBERT loaded on {where}")
        except Exception as e:
            print(f"      DistilBERT not loaded: {e}")
    else:
        print("      (skipped — distilbert_fake_news/ not found)")

    print("Models ready.\n")
    return tfidf, lr_model, rf_model, bert_pipeline


def prediction_from_probabilities(model, probabilities):
    """Return the dataset label and confidence for a fitted sklearn model."""
    winner = int(np.argmax(probabilities))
    class_value = int(model.classes_[winner])
    try:
        label = LABELS[class_value]
    except KeyError as error:
        raise ValueError(
            f"Unsupported model class {class_value}; expected one of "
            f"{sorted(LABELS)}. Retrain the model with the project dataset."
        ) from error
    return label, float(probabilities[winner])


# ---------- Explanations ----------
def explain_lr(text_clean, tfidf, lr_model, top_k=10):
    """
    Exact per-feature contribution to the LR log-odds.
    For a linear model this equals the SHAP value
    (Shapley values for LR with independent features).

    Positive value -> pushes toward REAL (class 1)
    Negative value -> pushes toward FAKE (class 0)
    """
    vec = tfidf.transform([text_clean])
    contribs = vec.toarray()[0] * lr_model.coef_[0]
    nonzero = np.where(contribs != 0)[0]
    if len(nonzero) == 0:
        return []
    order = nonzero[np.argsort(np.abs(contribs[nonzero]))[::-1]]
    top = order[:top_k]
    feature_names = tfidf.get_feature_names_out()
    return [(feature_names[i], float(contribs[i])) for i in top]


def explain_rf(text_clean, tfidf, rf_model, top_k=10):
    """
    Stable per-article RF explanation.

    True SHAP on a sparse-trained RF overflows numerically, so we use
    global feature importance weighted by the article's TF-IDF value.
    This gives a monotonic, bounded, interpretable ranking.
    """
    if rf_model is None:
        return []

    vec = tfidf.transform([text_clean]).toarray()[0]
    importances = rf_model.feature_importances_
    contribs = vec * importances              # bounded by importances

    nonzero = np.where(contribs > 0)[0]
    if len(nonzero) == 0:
        return []

    order = nonzero[np.argsort(contribs[nonzero])[::-1]]
    top = order[:top_k]
    feature_names = tfidf.get_feature_names_out()
    return [(feature_names[i], float(contribs[i])) for i in top]

# ---------- Prediction ----------
def predict(text, tfidf, lr_model, rf_model, bert_pipeline):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])

    results = {}

    # Logistic Regression
    lr_proba = lr_model.predict_proba(vec)[0]
    lr_label, lr_confidence = prediction_from_probabilities(lr_model, lr_proba)
    results["Logistic Regression"] = {
        "label": lr_label,
        "prob": lr_confidence,
    }

    # Random Forest
    if rf_model is not None:
        rf_proba = rf_model.predict_proba(vec)[0]
        rf_label, rf_confidence = prediction_from_probabilities(rf_model, rf_proba)
        results["Random Forest"] = {
            "label": rf_label,
            "prob": rf_confidence,
        }

    # DistilBERT
    if bert_pipeline is not None:
        try:
            out = bert_pipeline(text[:2000], truncation=True, max_length=256)
            # pipeline returns list-of-dicts when top_k=None, else dict
            if isinstance(out, list) and isinstance(out[0], list):
                out = out[0]
            if isinstance(out, list):
                out = out[0]
            label_str = out["label"]
            prob = float(out["score"])
            if label_str == "LABEL_0":
                label = "FAKE"
            elif label_str == "LABEL_1":
                label = "REAL"
            else:
                label = label_str
            results["DistilBERT"] = {"label": label, "prob": prob}
        except Exception as e:
            print(f"  DistilBERT prediction failed: {e}")

    lr_expl = explain_lr(cleaned, tfidf, lr_model)
    rf_expl = explain_rf(cleaned, tfidf, rf_model)

    return results, lr_expl, rf_expl


# ---------- Display ----------
def display(text, results, lr_expl, rf_expl):
    line = "=" * 62

    print(line)
    print("INPUT")
    print("-" * 62)
    preview = text if len(text) <= 300 else text[:300] + "..."
    print(preview)
    print()

    print("PREDICTIONS")
    print("-" * 62)
    for name, r in results.items():
        print(f"  {name:22s}  {r['label']:5s}   ({r['prob']*100:5.1f}%)")
    print("  Note: this estimates patterns from the training sources; it does")
    print("  not independently verify whether a claim in the article is true.")
    print()

    print("IMPORTANT WORDS / PHRASES")
    print("-" * 62)
    if not lr_expl:
        print("  No informative words found in this article.")
    else:
        print("  Top features (from Logistic Regression):")
        print(f"  {'word/phrase':30s} {'contribution':>14s}   direction")
        for word, c in lr_expl:
            direction = "→ REAL" if c > 0 else "→ FAKE"
            print(f"  {word:30s} {c:+14.4f}   {direction}")
    print()

    if rf_expl:
        print("  Top features (from Random Forest importance × TF-IDF):")
        print(f"  {'word/phrase':30s} {'score':>14s}")
        for word, c in rf_expl:
            print(f"  {word:30s} {c:14.6f}")
        print()

    print(line)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Fake News Classifier with explanations."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", type=str, help="Text to classify")
    group.add_argument("--file", type=str, help="Path to a text file")
    args = parser.parse_args()

    tfidf, lr_model, rf_model, bert_pipeline = load_models()

    # Get input
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        print("Enter a news article (finish with an empty line):\n")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "":
                break
            lines.append(line)
        text = " ".join(lines).strip()

    if not text:
        print("No input provided.")
        return

    results, lr_expl, rf_expl = predict(
        text, tfidf, lr_model, rf_model, bert_pipeline
    )
    display(text, results, lr_expl, rf_expl)


if __name__ == "__main__":
    main()
