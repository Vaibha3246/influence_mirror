import pandas as pd
import yaml
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import boto3


logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def load_params(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_from_s3(bucket, key, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    logger.info(f"Downloaded raw data from S3 to {local_path}")


def load_data(path):
    df = pd.read_csv(path)
    logger.info(f"Data loaded: {df.shape}")
    return df


def preprocess_data(df):
    if "text" not in df.columns:
        raise KeyError("Missing required column: text")

    df = df.dropna()
    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    df = df.reset_index(drop=True)

    logger.info(f"After preprocessing: {df.shape}")
    return df


def save_data(train, test, path: Path):
    os.makedirs(path, exist_ok=True)
    train.to_csv(path / "train.csv", index=False)
    test.to_csv(path / "test.csv", index=False)
    logger.info(f"Train/test saved to {path}")


def main():
    BASE = Path(__file__).resolve().parents[2]
    params = load_params(BASE / "params.yaml")

    raw_path = BASE / params["data_ingestion"]["raw_data_path"]
    interim_path = BASE / params["data_ingestion"]["interim_data_path"]
    test_size = params["data_ingestion"]["test_size"]

    #  
    if not raw_path.exists():
        logger.info("Raw data not found locally. Downloading from S3...")
        download_from_s3(
            bucket="my-s3-bucket-of-store-artifact-youtube-data12",
            key="youtube_bulk_raw.csv",
            local_path=str(raw_path)
        )

    df = load_data(raw_path)
    final_df = preprocess_data(df)

    train, test = train_test_split(final_df, test_size=test_size, random_state=42)
    save_data(train, test, interim_path)

    logger.info("Data ingestion completed successfully")


if __name__ == "__main__":
    main()
