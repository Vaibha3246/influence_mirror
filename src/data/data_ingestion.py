import pandas as pd
import yaml
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

def load_params(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(path):
    df = pd.read_csv(path)
    logger.info(f"Data loaded: {df.shape}")
    return df

def preprocess_data(df):
    if "text" not in df.columns:
        raise KeyError("Missing required column: text")
    df = df.dropna()
    df = df.drop_duplicates(subset=['text'], keep='first')
    df = df[df['text'].str.strip() != ""]
    df = df.reset_index(drop=True)
    logger.info(f"After preprocessing: {df.shape}")
    return df

def save_data(train, test, path: Path):
    os.makedirs(path, exist_ok=True)
    train.to_csv(path / "train.csv", index=False)
    test.to_csv(path / "test.csv", index=False)
    logger.info(f"Train and test data saved to: {path}")

def main():
    BASE = Path(__file__).resolve().parents[2]
    params = load_params(BASE / "params.yaml")

    raw_path = BASE / params["data_ingestion"]["raw_data_path"]
    interim_path = BASE / params["data_ingestion"]["interim_data_path"]
    test_size = params["data_ingestion"]["test_size"]

    df = load_data(raw_path)
    final_df = preprocess_data(df)

    train, test = train_test_split(final_df, test_size=test_size, random_state=42)
    save_data(train, test, interim_path)

    logger.info("Data ingestion stage completed successfully.")

if __name__ == "__main__":
    main()
