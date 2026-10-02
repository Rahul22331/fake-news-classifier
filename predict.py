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

# Optional imports for DistilBERT gradient saliency
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

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
    bert_model = None
    bert_tokenizer = None
    if os.path.isdir(BERT_DIR) and _TORCH_AVAILABLE:
        try:
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            bert_pipeline = pipeline(
                "text-classification",
                model=BERT_DIR,
                tokenizer=BERT_DIR,
                device=device,
            )

            # Also load raw model + tokenizer for gradient saliency
            bert_tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
            bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
            bert_model.eval()
            if torch.cuda.is_available():
                bert_model.to("cuda")

            where = "GPU" if device == 0 else "CPU"
            print(f"      DistilBERT loaded on {where}")
        except Exception as e:
            print(f"      DistilBERT not loaded: {e}")
    elif not os.path.isdir(BERT_DIR):
        print("      (skipped — distilbert_fake_news/ not found)")
    elif not _TORCH_AVAILABLE:
        print("      (skipped — torch/transformers not installed)")

    print("Models ready.\n")
    return tfidf, lr_model, rf_model, bert_pipeline, bert_model, bert_tokenizer


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


def explain_distilbert(text, bert_model, bert_tokenizer, top_k=10):
    """
    Gradient saliency explanation for DistilBERT.

    Computes the gradient of the predicted class logit with respect to the
    input token embeddings.  The L2 norm of each token's gradient vector
    indicates how sensitive the prediction is to that token — higher norm
    means the token had more influence on the decision.

    Positive contribution → pushes toward the predicted class.
    Returns a list of (token, saliency_score) sorted by importance.
    """
    if bert_model is None or bert_tokenizer is None:
        return [], None

    device = next(bert_model.parameters()).device

    # Tokenize
    inputs = bert_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256, padding=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Get the embedding layer
    embedding_layer = bert_model.distilbert.embeddings.word_embeddings

    # Get embeddings and enable gradient tracking
    embeddings = embedding_layer(input_ids)
    embeddings.retain_grad()
    embeddings.requires_grad_(True)

    # Forward pass using embeddings directly
    outputs = bert_model.distilbert(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
    )
    # Pass through the classifier head
    hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
    logits = bert_model.classifier(bert_model.pre_classifier(hidden_state[:, 0]))

    # Get predicted class and compute gradient of that logit
    probs = F.softmax(logits, dim=-1)
    pred_class = int(torch.argmax(probs, dim=-1).item())
    pred_label = "REAL" if pred_class == 1 else "FAKE"

    # Backpropagate from the predicted class logit
    target_logit = logits[0, pred_class]
    target_logit.backward()

    # Gradient saliency = L2 norm of gradient at each token position
    # Shape: embeddings.grad → (1, seq_len, hidden_dim)
    grad = embeddings.grad[0]  # (seq_len, hidden_dim)
    saliency = torch.norm(grad, dim=-1).detach().cpu().numpy()  # (seq_len,)

    # Map back to tokens
    tokens = bert_tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
    mask = attention_mask[0].cpu().numpy()

    # Collect (token, score) — skip [CLS], [SEP], [PAD]
    token_scores = []
    for i, (tok, score, m) in enumerate(zip(tokens, saliency, mask)):
        if m == 0 or tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        token_scores.append((tok, float(score)))

    if not token_scores:
        return [], pred_label

    # Merge sub-word tokens (e.g., "govern", "##ment" → "government")
    merged = []
    for tok, score in token_scores:
        if tok.startswith("##") and merged:
            prev_tok, prev_score = merged[-1]
            merged[-1] = (prev_tok + tok[2:], max(prev_score, score))
        else:
            merged.append((tok, score))

    # Sort by saliency (descending) and return top_k
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k], pred_label

# ---------- Prediction ----------
def predict(text, tfidf, lr_model, rf_model, bert_pipeline,
            bert_model=None, bert_tokenizer=None):
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
    bert_expl, bert_expl_label = explain_distilbert(text, bert_model, bert_tokenizer)

    return results, lr_expl, rf_expl, bert_expl, bert_expl_label


# ---------- Display ----------
def display(text, results, lr_expl, rf_expl, bert_expl=None, bert_expl_label=None):
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

    if bert_expl:
        direction_label = bert_expl_label or "PREDICTED"
        print(f"  Top features (from DistilBERT Gradient Saliency → {direction_label}):")
        print(f"  {'token':30s} {'saliency':>14s}   influence")
        # Normalize scores for visual bars
        max_score = bert_expl[0][1] if bert_expl else 1.0
        for word, score in bert_expl:
            bar_len = int(20 * score / max_score) if max_score > 0 else 0
            bar = "█" * bar_len
            print(f"  {word:30s} {score:14.4f}   {bar}")
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

    tfidf, lr_model, rf_model, bert_pipeline, bert_model, bert_tokenizer = load_models()

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

    results, lr_expl, rf_expl, bert_expl, bert_expl_label = predict(
        text, tfidf, lr_model, rf_model, bert_pipeline, bert_model, bert_tokenizer
    )
    display(text, results, lr_expl, rf_expl, bert_expl, bert_expl_label)


if __name__ == "__main__":
    main()
