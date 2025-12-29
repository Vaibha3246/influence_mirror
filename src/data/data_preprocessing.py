import os
import logging
import pandas as pd
import re
import html
import emoji
import json
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
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
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

# Slang dictionary JSON file path
slang_file_path = BASE / params.get("slang_json", "slang_dict.json")
if slang_file_path.exists():
    with open(slang_file_path) as f:
        slang_dict = json.load(f)
else:
    slang_dict = {}  # fallback

# ----------------------------
# Load Sentiment Model
# ----------------------------
logger.info("Loading RoBERTa sentiment model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(params["sentiment_model_name"])
model = AutoModelForSequenceClassification.from_pretrained(
    params["sentiment_model_name"]
).to(device)
model.eval()
LABELS = ["negative", "neutral", "positive"]

# ----------------------------
# Cleaning regex
# ----------------------------
url_re = re.compile(r"https?://\S+|www\.\S+")
mention_re = re.compile(r"@\w+")
html_tag_re = re.compile(r"<.*?>")
multispace_re = re.compile(r"\s+")
repeat_re = re.compile(r"(.)\1{2,}")

def normalize_repeated_letters(text):
    """Reduce repeated letters to max 2 (cooool -> cool)"""
    return repeat_re.sub(r"\1\1", text)

def normalize_slang(text):
    return " ".join([slang_dict.get(w, w) for w in text.split()])

def clean_text(text):
    s = str(text) if pd.notna(text) else ""
    s = html.unescape(s)
    s = url_re.sub(" ", s)
    s = mention_re.sub(" ", s)
    s = html_tag_re.sub(" ", s)
    s = emoji.demojize(s)
    s = normalize_repeated_letters(s)
    s = normalize_slang(s)
    s = s.lower()
    s = multispace_re.sub(" ", s).strip()
    return s

# ----------------------------
# Batch Sentiment Prediction
# ----------------------------
def predict_sentiment_batch(texts, batch_size=32):
    sentiments = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Batches"):
        batch_texts = texts[i:i+batch_size].tolist()
        try:
            tokens = tokenizer(
                batch_texts, return_tensors="pt",
                padding=True, truncation=True, max_length=512
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.no_grad():
                outputs = model(**tokens)
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
            sentiments.extend([LABELS[p] for p in preds])
        except Exception as e:
            logger.warning(f"Prediction error at batch {i}: {e}")
            sentiments.extend(["neutral"] * len(batch_texts))
    return sentiments

# ----------------------------
# Preprocessing Function
# ----------------------------
def preprocess_with_sentiment(df):
    df = df.drop(columns=["author", "video_id", "likes", "published_at"], errors="ignore")
    df = df[df["text"].astype(str).str.strip() != ""].reset_index(drop=True)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    logger.info("Cleaning text...")
    df["text_clean"] = df["text"].apply(clean_text)
    df["word_count"] = df["text_clean"].apply(lambda x: len(x.split()))
    df = df[df["word_count"] <= params["max_word_count"]].reset_index(drop=True)

    logger.info("Generating sentiment labels...")
    df["sentiment"] = predict_sentiment_batch(df["text_clean"], batch_size=params.get("batch_size", 32))
    df["sentiment_numeric"] = df["sentiment"].map({"negative":0,"neutral":1,"positive":2}).fillna(1).astype(int)

    return df

# ----------------------------
# Main
# ----------------------------
def main():
    logger.info("Loading train & test data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info("Preprocessing TRAIN data...")
    train_processed = preprocess_with_sentiment(train_df)

    logger.info("Preprocessing TEST data...")
    test_processed = preprocess_with_sentiment(test_df)

    os.makedirs(processed_path, exist_ok=True)
    train_processed.to_csv(processed_path / "train_preprocessed.csv", index=False)
    test_processed.to_csv(processed_path / "test_preprocessed.csv", index=False)

    logger.info(f"Processed files saved to: {processed_path}")
    logger.info("Data Preprocessing Completed Successfully!")

if __name__ == "__main__":
    main()
