import os
import pandas as pd
import numpy as np
import joblib
import logging
import nltk
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nrclex import NRCLex
from scipy import sparse as sp
import yaml

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('feature_engineering_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

nltk.download('vader_lexicon', quiet=True)

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["feature_engineering"]

MAX_FEATURES = params.get("max_features", 10000)
NGRAM_RANGE = tuple(params.get("ngram_range", [1, 3]))


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    if 'category' not in df.columns:
        return df
    try:
        ohe = OneHotEncoder(sparse_output=False, drop='first')
    except TypeError:
        ohe = OneHotEncoder(sparse=False, drop='first')
    encoded = ohe.fit_transform(df[['category']])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['category']))
    df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    return df


def add_vader_features(df: pd.DataFrame) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    df[['sent_neg', 'sent_neu', 'sent_pos', 'sent_compound']] = df['text_clean'].apply(
        lambda x: pd.Series(sia.polarity_scores(str(x)))
    )
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'published_at' not in df.columns:
        return df
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    df['hour'] = df['published_at'].dt.hour
    df['weekday'] = df['published_at'].dt.weekday
    df['month'] = df['published_at'].dt.month
    return df


def get_emotion_probabilities(text: str) -> dict:
    emotions = NRCLex(str(text))
    raw_scores = emotions.raw_emotion_scores
    total = sum(raw_scores.values())
    base_emotions = {
        'fear': 0, 'anger': 0, 'anticipation': 0, 'trust': 0,
        'surprise': 0, 'positive': 0, 'negative': 0,
        'sadness': 0, 'disgust': 0, 'joy': 0
    }
    if total > 0:
        normalized = {k: v / total for k, v in raw_scores.items()}
        base_emotions.update(normalized)
    return base_emotions


def add_emotion_features(df: pd.DataFrame) -> pd.DataFrame:
    emotion_data = df['text_clean'].apply(get_emotion_probabilities)
    emotion_df = pd.DataFrame(list(emotion_data))
    df = pd.concat([df.reset_index(drop=True), emotion_df.reset_index(drop=True)], axis=1)
    return df


def scale_numeric(df: pd.DataFrame, exclude_cols: list):
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler


def process_dataset(df: pd.DataFrame, output_dir: str, name: str):
    logger.debug(f"Processing dataset: {name}")

    df = one_hot_encode(df)
    df = add_vader_features(df)
    df = add_time_features(df)
    df = add_emotion_features(df)
    df = df.dropna(subset=['sentiment_numeric'])

    tfidf = TfidfVectorizer(ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES)
    X_text = tfidf.fit_transform(df['text_clean'])
    joblib.dump(tfidf, os.path.join(output_dir, f"{name}_tfidf_vectorizer.pkl"))

    drop_cols = [
        'category', 'text', 'text_clean', 'sentiment',
        'published_at', 'dominant_emotion', 'sentiment_numeric'
    ]
    X_num = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X_num_scaled, scaler = scale_numeric(X_num, [])
    joblib.dump(scaler, os.path.join(output_dir, f"{name}_scaler.pkl"))

    X_combined = sp.hstack([X_text, sp.csr_matrix(X_num_scaled)])
    y = df['sentiment_numeric']

    joblib.dump({'X': X_combined, 'y': y}, os.path.join(output_dir, f"{name}_features.pkl"))
    logger.debug(f"Saved features for {name}")


def main():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        input_path = os.path.join(base_dir, 'data', 'processed')
        output_path = os.path.join(base_dir, 'data', 'features')
        os.makedirs(output_path, exist_ok=True)

        train_df = pd.read_csv(os.path.join(input_path, 'train_processed.csv'))
        test_df = pd.read_csv(os.path.join(input_path, 'test_processed.csv'))

        process_dataset(train_df, output_path, 'train')
        process_dataset(test_df, output_path, 'test')

        logger.debug("Feature engineering completed successfully.")
    except Exception as e:
        logger.error(f"Feature engineering error: {e}")
        raise


if __name__ == '__main__':
    main()
