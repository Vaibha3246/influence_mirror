
# app.py
print("🔥 CORRECT flask_app/app.py IS RUNNING")

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from pathlib import Path
from flask_cors import CORS
import torch
import nltk
import re, html, emoji
from nltk.corpus import stopwords
from collections import Counter
from collections import Counter, defaultdict

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nrclex import NRCLex
import mlflow
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(" GROQ_API_KEY not loaded")

client = Groq(api_key=GROQ_API_KEY)


# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__, static_folder="static")

CORS(app, resources={r"/*": {"origins": "*"}})


# -----------------------------
# Base Paths
# -----------------------------
BASE = Path(__file__).resolve().parent.parent
ARTIFACTS_PATH = BASE / "data" / "features"

# -----------------------------
# Load MLflow Model (Staging)
# -----------------------------
MLFLOW_TRACKING_URI = "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000/"
MODEL_NAME = "yt_chrome_plugin_model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}/Staging")

# -----------------------------
# Load Training Artifacts
# -----------------------------
sbert = joblib.load(ARTIFACTS_PATH / "sbert_model.pkl")
scaler = joblib.load(ARTIFACTS_PATH / "scaler.pkl")
ohe = joblib.load(ARTIFACTS_PATH / "ohe.pkl")
numeric_cols = json.load(open(ARTIFACTS_PATH / "numeric_cols.json"))

# -----------------------------
# Load RoBERTa (FOR PROBS ONLY)
#  FIX: same model as training
# -----------------------------
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
roberta = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL)
roberta.eval()

# -----------------------------
# NLTK
# -----------------------------
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# -----------------------------
# Regex
# -----------------------------
url_re = re.compile(r'https?://\S+|www\.\S+')
mention_re = re.compile(r'@\w+')
html_tag_re = re.compile(r'<.*?>')
multispace_re = re.compile(r'\s+')

# -----------------------------
# Text Cleaning (MATCH TRAINING)
# -----------------------------
def clean_text(text):
    s = str(text) if pd.notna(text) else ""
    s = html.unescape(s)
    s = url_re.sub(" ", s)
    s = mention_re.sub(" ", s)
    s = html_tag_re.sub(" ", s)
    s = emoji.demojize(s)
    s = s.lower()
    s = multispace_re.sub(" ", s).strip()
    return s

# -----------------------------
# Numeric Features (MATCH TRAINING)
# -----------------------------
def add_numeric_features(text):
    return {
        "num_stop_words": len([w for w in text.split() if w in stop_words]),
        "num_chars": len(text),
        "num_words": len(text.split()),
        "num_exclamation": text.count("!"),
        "num_question": text.count("?"),
        "num_emojis": emoji.emoji_count(text),
        "num_punctuation": sum(1 for c in text if c in '.,!?;:"\'()[]{}-')
    }

# -----------------------------
# NRC Emotion Features
# -----------------------------
def get_emotion_scores(text):
    emotions = NRCLex(text)
    raw = emotions.raw_emotion_scores
    total = sum(raw.values())
    base = {
        'fear':0,'anger':0,'anticipation':0,'trust':0,'surprise':0,
        'positive':0,'negative':0,'sadness':0,'disgust':0,'joy':0
    }
    if total > 0:
        for k, v in raw.items():
            base[k] = v / total
    return base

# -----------------------------
# RoBERTa Probabilities
#  FIX: REQUIRED FOR MODEL
# -----------------------------
def get_roberta_probs(text):
    tokens = tokenizer(
        text, return_tensors="pt",
        padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        out = roberta(**tokens)
        probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    return probs.tolist()  # [neg, neu, pos]

# -----------------------------
# Feature Builder (100% MATCH)
# -----------------------------
def prepare_features(text, category=None):

    text_clean = clean_text(text)

    df = pd.DataFrame({
        "text_clean": [text_clean],
        "category": [category if category else "unknown"]
    })

    # Numeric
    for k, v in add_numeric_features(text_clean).items():
        df[k] = v

    # Emotion
    for k, v in get_emotion_scores(text_clean).items():
        df[k] = v

    # One-hot category
    cat_ohe = ohe.transform(df[['category']])
    cat_df = pd.DataFrame(cat_ohe, columns=ohe.get_feature_names_out(['category']))
    df = pd.concat([df.drop(columns=['category']), cat_df], axis=1)

    # Keep EXACT numeric column order
    df = df.reindex(columns=numeric_cols, fill_value=0)

    # Scale numeric
    X_num = scaler.transform(df.to_numpy())

    # SBERT embedding
    embedding = sbert.encode([text_clean], convert_to_numpy=True)

    # RoBERTa probs
    roberta_probs = np.array(get_roberta_probs(text_clean)).reshape(1, -1)

    # FINAL FEATURE VECTOR
    X = np.hstack([embedding, X_num, roberta_probs])
    return X

def build_sentiment_over_time(predictions):
    timeline = defaultdict(lambda: {
        "positive": 0,
        "neutral": 0,
        "negative": 0
    })

    for i, p in enumerate(predictions):
        bucket = i // 10   # har 10 comments = 1 time unit
        timeline[bucket][p["prediction_label"]] += 1

    result = []
    for t in sorted(timeline.keys()):
        result.append({
            "time": t,
            **timeline[t]
        })

    return result

def build_wordcloud(predictions):
    text = " ".join([clean_text(p["text"]) for p in predictions])
    words = text.split()

    freq = Counter(
        w for w in words
        if len(w) > 3 and w not in stop_words
    )

    return freq.most_common(100)


def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def retrieve_chunks(question, chunks, top_k=4):
    chunk_embeddings = sbert.encode(chunks, convert_to_numpy=True)
    q_emb = sbert.encode([question], convert_to_numpy=True)

    sims = cosine_similarity(q_emb, chunk_embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    return [chunks[i] for i in top_idx]


def ask_video_ai(question: str, video_context: str):
    system_prompt = """
You are an intelligent YouTube Video AI Assistant.

STRICT RULES:
- Answer ONLY using the given video content
- If answer is NOT in the video, say:
  "This question is outside the scope of the video."
- Understand Hinglish / broken English
- Be clear, structured, and honest
- Never hallucinate
- No extra assumptions

Answer styles:
- Summary → concise explanation
- Interview → Q&A format
- Resume → professional bullet points
"""

    user_prompt = f"""
Video Content:
{video_context}

User Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

def generate_video_topics(video_context: str, max_topics=6):
    
    # -------- HARD SAFETY LIMIT --------
    MAX_CHARS = 4000  # safe for 6k TPM
    if len(video_context) > MAX_CHARS:
        video_context = video_context[:MAX_CHARS]

    # -------- MINIMAL & TOKEN-EFFICIENT PROMPT --------
    prompt = f"""
Extract learning topics from the video content.

Rules:
- Use only the given content
- Short topic titles (3–6 words)
- No explanations
- Max {max_topics} topics

Content:
{video_context}

Return a numbered list.
""".strip()

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )

        raw = response.choices[0].message.content.strip()

    except Exception as e:
        print(" LLM Topic Generation Failed:", e)
        return []

    # -------- ROBUST PARSING --------
    topics = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Handle: "1. Topic" OR "- Topic" OR "• Topic"
        if line[0].isdigit() and "." in line:
            topics.append(line.split(".", 1)[1].strip())
        elif line.startswith(("-", "•")):
            topics.append(line[1:].strip())

    return topics[:max_topics]


def normalize_question(question: str):
    q = question.lower()

    if any(word in q for word in ["summary", "summarize", "short me", "brief"]):
        return "Give a clear and concise summary of the video."

    if any(word in q for word in ["interview", "question", "viva"]):
        return "Explain this topic as interview questions with answers."

    if any(word in q for word in ["resume", "cv"]):
        return "Explain this topic in a resume-friendly way."

    return question


# -----------------------------
# Batch helper (PERFORMANCE FIX)
# -----------------------------
def batch(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# -----------------------------
# API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    texts = data.get("texts")
    category = data.get("category")

    if isinstance(texts, list):

        BATCH_SIZE = 50   # ⭐ SAFE VALUE

        results = []
        pos = neu = neg = 0

        for chunk in batch(texts, BATCH_SIZE):
            for t in chunk:
                try:
                    X = prepare_features(t, category)
                    pred = model.predict(X)[0]
                    label = {0: "negative", 1: "neutral", 2: "positive"}[pred]

                    if label == "positive":
                        pos += 1
                    elif label == "negative":
                        neg += 1
                    else:
                        neu += 1

                    results.append({
                        "text": t,
                        "prediction_label": label
                    })

                except Exception as e:
                    # Skip bad comment instead of crashing
                    print("Prediction error:", e)

        total = len(results) if results else 1

        summary = {
            "positive": round(pos / total * 100, 2),
            "neutral": round(neu / total * 100, 2),
            "negative": round(neg / total * 100, 2),
        }

        sentiment_trend = build_sentiment_over_time(results)
        wordcloud = build_wordcloud(results)

        return jsonify({
            "sentiment_summary": summary,
            "sentiment_over_time": sentiment_trend,
            "wordcloud": wordcloud,
            "predictions": results,
            "top_comments": results[:5]
        })

    # -------- Single text (unchanged) --------
    if "text" in data:
        X = prepare_features(data["text"], category)
        pred = model.predict(X)[0]
        label = {0: "negative", 1: "neutral", 2: "positive"}[pred]

        return jsonify({
            "sentiment_summary": {
                "positive": 100 if label == "positive" else 0,
                "neutral": 100 if label == "neutral" else 0,
                "negative": 100 if label == "negative" else 0,
            },
            "top_comments": [{
                "text": data["text"],
                "prediction_label": label
            }]
        })

    return jsonify({"error": "Invalid input"}), 400

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/ask-video", methods=["POST"])
def ask_video():
    data = request.json or {}

    question = data.get("question", "").strip()
    video_context = data.get("video_context", "").strip()

    if not question:
        return jsonify({"error": "Question missing"}), 400

    if not video_context:
        return jsonify({"error": "Video context missing"}), 400

    # Normalize user intent (summary / interview / resume)
    clean_question = normalize_question(question)

    # 🔴 FIX 1: RAG CHUNKING + RETRIEVAL
    chunks = chunk_text(video_context)
    top_chunks = retrieve_chunks(clean_question, chunks, top_k=4)

    # Build focused context
    focused_context = "\n\n".join(top_chunks)

    # Ask AI using ONLY relevant chunks
    answer = ask_video_ai(clean_question, focused_context)

    return jsonify({
        "question_understood_as": clean_question,
        "answer": answer
    })


@app.route("/video-topics", methods=["POST"])
def video_topics():
    data = request.json or {}
    video_context = data.get("video_context", "").strip()

    if not video_context:
        return jsonify({"error": "Video context missing"}), 400

    topics = generate_video_topics(video_context)

    return jsonify({
        "topics": topics
    })

@app.route("/suggested-questions", methods=["POST"])
def suggested_questions():
    data = request.json or {}

    video_context = data.get("video_context", "").strip()
    chat_history = data.get("chat_history", [])

    if not video_context:
        return jsonify({"error": "Video context missing"}), 400

    # last 4 messages only (recent context)
    recent_chat = ""
    for item in chat_history[-4:]:
        role = item.get("role")
        content = item.get("content")
        recent_chat += f"{role}: {content}\n"

    prompt = f"""
You are an intelligent AI assistant.

Rules:
- Use ONLY video content and conversation context
- Generate smart FOLLOW-UP questions
- Questions should feel like a real chat continuation
- No answers, ONLY questions
- Max 4 questions

Video Content:
{video_context}

Recent Conversation:
{recent_chat}

Generate next questions user may ask:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=200
    )

    raw = response.choices[0].message.content.strip()

    questions = []
    for line in raw.split("\n"):
        line = line.strip("- ").strip()
        if line:
            questions.append(line)

    return jsonify({
        "suggested_questions": questions[:4]
    })



# Run App # -
if __name__ == "__main__": 
    app.run(host="0.0.0.0", port=8000, debug=True)
