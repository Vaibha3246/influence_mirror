# src/data/data_preprocessing.py

import numpy as np
import pandas as pd
import os
import re, html, emoji
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging


# Logging Configuration

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# NLTK Resources

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')


# Sentiment Analyzer

analyzer = SentimentIntensityAnalyzer()


# Regex Patterns

url_re = re.compile(r'https?://\S+|www\.\S+')
mention_re = re.compile(r'@\w+')
html_tag_re = re.compile(r'<.*?>')
multispace_re = re.compile(r'\s+')


# Base Cleaning Function

def clean_text(text, remove_emojis=False):
    """Clean raw text: remove URLs, mentions, HTML tags, extra spaces, etc."""
    try:
        if pd.isna(text):
            return ""

        s = html.unescape(str(text))
        s = url_re.sub(' ', s)
        s = mention_re.sub(' ', s)
        s = html_tag_re.sub(' ', s)

        # Optionally remove emojis
        if remove_emojis:
            try:
                s = emoji.replace_emoji(s, replace='')
            except Exception:
                s = s.encode('ascii', errors='ignore').decode()

        # Lowercase and remove multiple spaces
        s = s.lower()
        s = multispace_re.sub(' ', s).strip()

        return s
    except Exception as e:
        logger.error(f"Error in clean_text: {e}")
        return str(text)


# Text Preprocessing

def preprocess_text(text):
    """Further preprocess text (remove special chars, lemmatize, remove stopwords)."""
    try:
        if pd.isna(text):
            return ""

        # Remove newline characters
        text = re.sub(r'\n', ' ', str(text))

        # Remove non-alphanumeric except punctuation
        text = re.sub(r'[^A-Za-z0-9\s!?.,]', '', text)

        # Define stopwords but retain negations
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        text = ' '.join([word for word in text.split() if word.lower() not in stop_words])

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

        return text.strip()
    except Exception as e:
        logger.error(f"Error in preprocess_text: {e}")
        return text


# Sentiment Classification

def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))['compound']
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


# Full Text Normalization Pipeline

def normalize_text(df):
    """Apply full preprocessing pipeline on df['text'].""" 
    try:
        logger.debug("Starting text normalization...")

        # Step 1: Clean raw text
        df['text_clean'] = df['text'].apply(lambda x: clean_text(x, remove_emojis=False))

        # Step 2: Further preprocessing (stopwords, lemmatization)
        df['text_clean'] = df['text_clean'].apply(preprocess_text)

        # Step 3: Remove empty rows
        df = df[~(df['text_clean'].str.strip() == '')].reset_index(drop=True)

        # Step 4: Safety cleanup (remove non-English chars)
        df['text_clean'] = df['text_clean'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s!?.,]', '', str(x)))

        # Step 5: Sentiment Analysis
        df['sentiment'] = df['text_clean'].apply(get_sentiment)
        df['sentiment_numeric'] = df['sentiment'].replace({
            'positive': 2,
            'negative': 0,
            'neutral': 1
        })

        # Step 6: Feature Engineering
        stop_words = set(stopwords.words('english'))
        df['word_count'] = df['text_clean'].apply(lambda x: len(x.split()))
        df = df[(df['word_count'] >= 2) & (df['word_count'] <= 100)]
        df['num_chars'] = df['text_clean'].apply(len)
        df['num_stop_words'] = df['text_clean'].apply(
            lambda x: len([word for word in x.split() if word in stop_words])
        )
        df['num_punctuation_chars'] = df['text_clean'].apply(
            lambda x: sum([1 for char in x if char in '.,!?;:"\'()[]{}-'])
        )

        logger.debug("Text normalization completed successfully.")
        return df

    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise


# Save Processed Data
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        processed_data_path = os.path.join(data_path, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(processed_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(processed_data_path, "test_processed.csv"), index=False)

        logger.debug(f"Processed data saved to {processed_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise


# Main Execution

def main():
    try:
        logger.debug("Starting full data preprocessing pipeline...")

        # --- Base project root (same as ingestion)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Load interim train/test
        train_data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'interim', 'train.csv'))
        test_data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'interim', 'test.csv'))
        logger.debug("Data loaded successfully from interim folder.")

        # Apply preprocessing
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save processed data
        save_data(train_processed_data, test_processed_data, data_path=os.path.join(BASE_DIR, 'data'))

        logger.info("Data preprocessing completed successfully.")

    except Exception as e:
        logger.error(f"Failed to complete preprocessing: {e}")
        print(f"Error: {e}")


# Run Main

if __name__ == '__main__':
    main()
