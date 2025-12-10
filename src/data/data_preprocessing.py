import os
import logging
import pandas as pd
import re, html, emoji
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# ----------------------------
# Logger Setup
# ----------------------------
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# ----------------------------
# Paths and Params
# ----------------------------
BASE = Path(__file__).resolve().parents[2]

with open(BASE / "params.yaml") as f:
    params_all = yaml.safe_load(f)
params = params_all["data_preprocessing"]

train_path = BASE / params["train_input"]
test_path = BASE / params["test_input"]
processed_path = BASE / params["output_path"]

# ----------------------------
# Load Sentiment Model
# ----------------------------
logger.info("Loading RoBERTa sentiment model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(params["sentiment_model_name"])
model = AutoModelForSequenceClassification.from_pretrained(params["sentiment_model_name"]).to(device)
model.eval()  # Set model to eval mode
labels = ["negative", "neutral", "positive"]

# ----------------------------
# Cleaning regex
# ----------------------------
url_re = re.compile(r'https?://\S+|www\.\S+')
mention_re = re.compile(r'@\w+')
html_tag_re = re.compile(r'<.*?>')
multispace_re = re.compile(r'\s+')

def clean_text(text, remove_emojis=False):
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

# ----------------------------
# Batch Sentiment Prediction
# ----------------------------
def predict_sentiment_batch(texts, batch_size=32):
    all_sentiments = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Batches"):
        batch_texts = texts[i:i+batch_size].tolist()
        try:
            tokens = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            tokens = {k:v.to(device) for k,v in tokens.items()}
            with torch.no_grad():
                outputs = model(**tokens)
                scores = torch.softmax(outputs.logits, dim=1)
                predicted = torch.argmax(scores, dim=1).cpu().numpy()
                batch_labels = [labels[p] for p in predicted]
                all_sentiments.extend(batch_labels)
        except Exception as e:
            logger.warning(f"Batch prediction error at index {i}-{i+batch_size}: {e}")
            all_sentiments.extend(["error"]*len(batch_texts))
    return all_sentiments

# ----------------------------
# Preprocessing Function
# ----------------------------
def preprocess(df):
    # Drop unnecessary columns to save memory
    df = df.drop(columns=['author', 'video_id', 'likes', 'published_at'], errors='ignore')

    # Remove empty or duplicate texts
    df = df[df["text"].astype(str).str.strip() != ""].reset_index(drop=True)
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)

    logger.info("Cleaning text...")
    df["text_clean"] = df["text"].apply(lambda x: clean_text(x, params["remove_emojis"]))
    df = df[df["text_clean"] != ""].reset_index(drop=True)

    # Word count filter (optional for SBERT)
    df["word_count"] = df["text_clean"].apply(lambda x: len(x.split()))
    df = df[
        (df["word_count"] >= params["min_word_count"]) &
        (df["word_count"] <= params["max_word_count"])
    ].reset_index(drop=True)

    # Predict sentiment in batches
    logger.info("Predicting sentiment...")
    df["sentiment"] = predict_sentiment_batch(df["text_clean"], batch_size=params.get("batch_size", 32))

    # Convert sentiment to numeric
    df["sentiment_numeric"] = df["sentiment"].replace({
        "positive": 1,
        "negative": -1,
        "neutral": 0,
        "error": 0
    })

    return df

# ----------------------------
# Main Function
# ----------------------------
def main():
    logger.info("Loading interim train & test CSVs...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info("Preprocessing train data...")
    train_processed = preprocess(train_df)

    logger.info("Preprocessing test data...")
    test_processed = preprocess(test_df)

    os.makedirs(processed_path, exist_ok=True)
    train_processed.to_csv(processed_path / "train_preprocessed.csv", index=False)
    test_processed.to_csv(processed_path / "test_preprocessed.csv", index=False)

    logger.info(f"Processed data saved in: {processed_path}")
    logger.info("Data Preprocessing Stage Completed Successfully!")

# ----------------------------
if __name__ == "__main__":
    main()
