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


# LOGGING SETUP
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('feature_engineering_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

nltk.download('vader_lexicon', quiet=True)

# LOAD PARAMS
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["feature_engineering"]

MAX_FEATURES = params.get("max_features", 10000)
NGRAM_RANGE = tuple(params.get("ngram_range", [1, 3]))


# HELPER FUNCTIONS

def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    if 'category' not in df.columns:
        return df
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    encoded = ohe.fit_transform(df[['category']])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['category']))
    return pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)


def add_vader_features(df: pd.DataFrame) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    vader_scores = df['text_clean'].apply(lambda x: pd.Series(sia.polarity_scores(str(x))))
    return pd.concat([df.reset_index(drop=True), vader_scores.reset_index(drop=True)], axis=1)


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
    return pd.concat([df.reset_index(drop=True), emotion_df.reset_index(drop=True)], axis=1)


def scale_numeric(df: pd.DataFrame, exclude_cols: list, scaler=None, fit=True):
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler


def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = one_hot_encode(df)
    df = add_vader_features(df)
    df = add_time_features(df)
    df = add_emotion_features(df)
    df = df.dropna(subset=['sentiment_numeric'])
    return df


# MAIN FUNCTION

def main():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        input_path = os.path.join(base_dir, 'data', 'processed')
        output_path = os.path.join(base_dir, 'data', 'features')
        os.makedirs(output_path, exist_ok=True)

        logger.info("Loading processed train and test data...")
        train_df = pd.read_csv(os.path.join(input_path, 'train_processed.csv'))
        test_df = pd.read_csv(os.path.join(input_path, 'test_processed.csv'))

        # Drop any NaN text before vectorization (safety)
        train_df = train_df.dropna(subset=['text_clean'])
        test_df = test_df.dropna(subset=['text_clean'])

        logger.info("Applying feature transformations...")
        train_df = process_dataset(train_df)
        test_df = process_dataset(test_df)

        # TF-IDF (fit on train only)
        tfidf_path = os.path.join(output_path, "tfidf_vectorizer.pkl")
        tfidf = TfidfVectorizer(ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES)
        X_train_text = tfidf.fit_transform(train_df['text_clean'])
        X_test_text = tfidf.transform(test_df['text_clean'])
        joblib.dump(tfidf, tfidf_path)
        logger.info(f"Saved TF-IDF vectorizer to {tfidf_path}")

        # Prepare numeric features
        drop_cols = ['category', 'text', 'text_clean', 'sentiment', 'published_at', 'sentiment_numeric']
        X_train_num = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors='ignore')
        X_test_num = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors='ignore')

        # Scale numeric (fit on train only)
        scaler_path = os.path.join(output_path, "scaler.pkl")
        X_train_num_scaled, scaler = scale_numeric(X_train_num, [], fit=True)
        X_test_num_scaled, _ = scale_numeric(X_test_num, [], scaler=scaler, fit=False)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

        # Combine TF-IDF + Numeric features
        X_train = sp.hstack([X_train_text, sp.csr_matrix(X_train_num_scaled)])
        X_test = sp.hstack([X_test_text, sp.csr_matrix(X_test_num_scaled)])

        y_train = train_df['sentiment_numeric'].to_numpy()
        y_test = test_df['sentiment_numeric'].to_numpy()

        # Log feature shapes
        logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        logger.info(f"y_train size: {len(y_train)}, y_test size: {len(y_test)}")

        # Save final feature sets
        joblib.dump({'X': X_train, 'y': y_train}, os.path.join(output_path, 'train_features.pkl'))
        joblib.dump({'X': X_test, 'y': y_test}, os.path.join(output_path, 'test_features.pkl'))

        logger.info("Feature engineering completed successfully.")

    except Exception as e:
        logger.exception(f"Feature engineering failed: {e}")
        raise


if __name__ == '__main__':
    main()
