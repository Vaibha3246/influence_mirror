import os
import logging
import yaml
import nltk
from nltk.corpus import stopwords
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from nrclex import NRCLex
import joblib
import json

# -----------------------------
# Logger setup
# -----------------------------
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# -----------------------------
# Base Path and params
# -----------------------------
BASE = Path(__file__).resolve().parents[2]
with open(BASE / "params.yaml") as f:
    params = yaml.safe_load(f)["feature_engineering"]

train_processed_path = BASE / params["train_processed"]
test_processed_path = BASE / params["test_processed"]
output_path = BASE / params["output_path"]

BATCH_SIZE = params["batch_size"]
SBERT_MODEL = "all-MiniLM-L6-v2"

# -----------------------------
# Download stopwords
# -----------------------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# -----------------------------
# Feature functions
# -----------------------------
def add_numeric_features(df):
    df['num_stop_words'] = df['text_clean'].apply(lambda x: len([w for w in x.split() if w in stop_words]))
    df['num_chars'] = df['text_clean'].apply(len)
    df['num_punctuation_chars'] = df['text_clean'].apply(lambda x: sum([1 for c in x if c in '.,!?;:"\'()[]{}-']))
    return df

def add_category_features(df, ohe=None, fit_ohe=True):
    if 'category' not in df.columns:
        return df, ohe

    if ohe is None:
        try:
            ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        except TypeError:
            ohe = OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore")

    if fit_ohe:
        cat_encoded = ohe.fit_transform(df[['category']])
    else:
        cat_encoded = ohe.transform(df[['category']])

    cat_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(['category']))
    df = pd.concat([df.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)
    return df, ohe



def get_emotion_scores(text):
    emotions = NRCLex(text)
    raw_scores = emotions.raw_emotion_scores
    total = sum(raw_scores.values())
    all_emotions = {'fear':0,'anger':0,'anticipation':0,'trust':0,'surprise':0,'positive':0,
                    'negative':0,'sadness':0,'disgust':0,'joy':0}
    if total > 0:
        probs = {emo: cnt/total for emo, cnt in raw_scores.items()}
        all_emotions.update(probs)
    return all_emotions

# -----------------------------
# Feature creation
# -----------------------------
def create_features(df, sbert=None, scaler=None, ohe=None, fit_scaler=True, fit_ohe=True):
    df = df.dropna(subset=['text_clean', 'sentiment_numeric']).reset_index(drop=True)
    logger.info(f"Rows after cleaning: {df.shape[0]}")

    # Numeric features
    df = add_numeric_features(df)

    # Category features
    df, ohe = add_category_features(df, ohe=ohe, fit_ohe=fit_ohe)

    # NRC emotion features
    logger.info("Extracting NRC emotion features...")
    emotion_features = df['text_clean'].apply(get_emotion_scores)
    emotion_df = pd.DataFrame(list(emotion_features))
    df = pd.concat([df, emotion_df], axis=1)

    # Labels
    y = df['sentiment_numeric'].astype(int).to_numpy()

    # Drop non-feature columns
    drop_cols = ['text', 'sentiment', 'category', 'sentiment_numeric']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # SBERT embeddings
    if sbert is None:
        logger.info(f"Loading SBERT model: {SBERT_MODEL}")
        sbert = SentenceTransformer(SBERT_MODEL)

    texts = df['text_clean'].astype(str).tolist()
    df = df.drop(columns=['text_clean'])

    logger.info("Generating SBERT embeddings…")
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i + BATCH_SIZE]
        emb = sbert.encode(batch, convert_to_numpy=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)

    # Scale numeric features
    X_num = df.to_numpy()
    if scaler is None:
        scaler = StandardScaler(with_mean=False)
    if fit_scaler:
        X_num_scaled = scaler.fit_transform(X_num)
    else:
        X_num_scaled = scaler.transform(X_num)

    # Combine embeddings + numeric features
    X = np.hstack([embeddings, X_num_scaled])
    logger.info(f"Final shape → X: {X.shape}, y: {y.shape}")

    # Save numeric column names for API use
    numeric_cols = df.columns.tolist()

    return X, y, sbert, scaler, ohe, numeric_cols

# -----------------------------
# Main
# -----------------------------
def main():
    logger.info("Loading processed train & test files...")
    train_df = pd.read_csv(train_processed_path)
    test_df = pd.read_csv(test_processed_path)

    # Train features
    X_train, y_train, sbert, scaler, ohe, numeric_cols = create_features(train_df)

    # Test features using same SBERT, scaler, OHE
    X_test, y_test, *_ = create_features(test_df, sbert=sbert, scaler=scaler, ohe=ohe,
                                         fit_scaler=False, fit_ohe=False)

    # Save features and artifacts
    os.makedirs(output_path, exist_ok=True)
    np.savez(output_path / "features_train.npz", X=X_train, y=y_train)
    np.savez(output_path / "features_test.npz", X=X_test, y=y_test)

    joblib.dump(sbert, output_path / "sbert_model.pkl")
    joblib.dump(scaler, output_path / "scaler.pkl")
    joblib.dump(ohe, output_path / "ohe.pkl")
    json.dump(numeric_cols, open(output_path / "numeric_cols.json", "w"))

    logger.info(f"Saved features and artifacts to {output_path}")
    logger.info("Feature Engineering Completed Successfully!")

if __name__ == "__main__":
    main()
