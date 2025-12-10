# app.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from nrclex import NRCLex
import re, html, emoji
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import mlflow
import nltk
from nltk.corpus import stopwords

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Paths & Base
# -----------------------------
BASE = Path(__file__).resolve().parent.parent
ARTIFACTS_PATH = BASE / "data" / "features"

# -----------------------------
# Load MLflow Model (Staging)
# -----------------------------
MLFLOW_TRACKING_URI = "http://ec2-13-49-76-135.eu-north-1.compute.amazonaws.com:5000/"
MODEL_NAME = "yt_chrome_plugin_model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()
versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
if not versions:
    raise ValueError(f"No model found in Staging for {MODEL_NAME}")

model_uri = f"models:/{MODEL_NAME}/Staging"
model = mlflow.lightgbm.load_model(model_uri)

# -----------------------------
# Load preprocessing artifacts
# -----------------------------
sbert = joblib.load(ARTIFACTS_PATH / "sbert_model.pkl")
scaler = joblib.load(ARTIFACTS_PATH / "scaler.pkl")
ohe = joblib.load(ARTIFACTS_PATH / "ohe.pkl")
numeric_cols = json.load(open(ARTIFACTS_PATH / "numeric_cols.json"))

# -----------------------------
# Load Sentiment Model
# -----------------------------
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
labels_sentiment = ["negative", "neutral", "positive"]

# -----------------------------
# NLTK Stopwords
# -----------------------------
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# -----------------------------
# Regex for text cleaning
# -----------------------------
url_re = re.compile(r'https?://\S+|www\.\S+')
mention_re = re.compile(r'@\w+')
html_tag_re = re.compile(r'<.*?>')
multispace_re = re.compile(r'\s+')

def clean_text(text, remove_emojis=True):
    s = str(text) if pd.notna(text) else ""
    s = html.unescape(s)
    s = url_re.sub(" ", s)
    s = mention_re.sub(" ", s)
    s = html_tag_re.sub(" ", s)
    if remove_emojis:
        try:
            s = emoji.replace_emoji(s, replace='')
        except:
            s = s.encode('ascii', errors='ignore').decode()
    s = s.lower()
    s = multispace_re.sub(" ", s).strip()
    return s

def predict_sentiment(text):
    if not text.strip():
        return "unknown"
    try:
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        outputs = sentiment_model(**tokens)
        scores = torch.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(scores, dim=1).item()
        return labels_sentiment[predicted]
    except:
        return "error"

# -----------------------------
# Feature Engineering
# -----------------------------
def add_numeric_features(text_clean):
    num_stop_words = len([w for w in text_clean.split() if w in stop_words])
    num_chars = len(text_clean)
    num_punctuation_chars = sum([1 for c in text_clean if c in '.,!?;:"\'()[]{}-'])
    return [num_stop_words, num_chars, num_punctuation_chars]

def get_emotion_scores(text):
    emotions = NRCLex(text)
    raw_scores = emotions.raw_emotion_scores
    total = sum(raw_scores.values())
    all_emotions = {'fear':0,'anger':0,'anticipation':0,'trust':0,'surprise':0,
                    'positive':0,'negative':0,'sadness':0,'disgust':0,'joy':0}
    if total > 0:
        probs = {emo: cnt/total for emo, cnt in raw_scores.items()}
        all_emotions.update(probs)
    return all_emotions

def prepare_features(text, category=None, published_at=None):
    text_clean = clean_text(text)
    sentiment = predict_sentiment(text_clean)
    sentiment_numeric = {"positive": 1, "neutral": 0, "negative": -1}.get(sentiment, 0)

    df = pd.DataFrame({
        "text_clean": [text_clean],
        "category": [category if category else "unknown"],
    })

    df[['num_stop_words', 'num_chars', 'num_punctuation_chars']] = pd.DataFrame(
        [add_numeric_features(text_clean)], index=df.index
    )

    emotions = get_emotion_scores(text_clean)
    for k, v in emotions.items():
        df[k] = v

    df['sentiment_numeric'] = sentiment_numeric

    # Drop non-feature columns
    df = df.drop(columns=[c for c in ['text_clean', 'published_at'] if c in df.columns])

    # OHE category
    cat_encoded = ohe.transform(df[['category']])
    cat_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(['category']))
    df = pd.concat([df.drop(columns=['category']), cat_df], axis=1)

    # Ensure numeric columns order
    df = df.reindex(columns=numeric_cols, fill_value=0)

    # Scale numeric
    scaled_numeric = scaler.transform(df).flatten().tolist()

    # SBERT embedding
    embedding = sbert.encode([text_clean], convert_to_numpy=True)[0].tolist()

    # Combine embeddings + numeric
    final_features = embedding + scaled_numeric
    return np.array(final_features).reshape(1, -1)

# -----------------------------
# API Endpoints
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.json
        texts = data.get("texts")
        text = data.get("text")
        category = data.get("category", None)

        if texts:  # Batch predictions
            results = []
            for t in texts:
                features = prepare_features(t, category)
                pred_numeric = model.predict(features)[0]
                pred_label = {1:"positive",0:"neutral",-1:"negative"}.get(pred_numeric,"unknown")
                results.append({"text": t, "prediction_numeric": int(pred_numeric), "prediction_label": pred_label})
            return jsonify({"results": results}), 200

        if not text:
            return jsonify({"error": "No text provided"}), 400

        features = prepare_features(text, category)
        pred_numeric = model.predict(features)[0]
        pred_label = {1:"positive",0:"neutral",-1:"negative"}.get(pred_numeric,"unknown")

        return jsonify({"prediction_numeric": int(pred_numeric), "prediction_label": pred_label}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

# -----------------------------
# Run App  
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
