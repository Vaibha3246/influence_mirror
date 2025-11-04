import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# Logging Configuration 
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler('logs/errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# Helper Functions 
def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error(' params.yaml not found at %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error while loading params: %s', e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        if df.empty:
            logger.warning(' CSV file is empty: %s', data_url)
        logger.debug('Data loaded successfully from %s (shape=%s)', data_url, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading data: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, empty strings, and dropping columns."""
    try:
        # Validate expected column
        if 'text' not in df.columns:
            logger.error(" Required column 'text' not found in dataframe.")
            raise KeyError("Missing required column 'text'.")

        # Drop unwanted columns safely
        df = df.drop(columns=['author', 'video_id', 'likes'], errors='ignore')

        # Clean data
        df = df.dropna()
        df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
        df = df[df['text'].str.strip() != ""].reset_index(drop=True)

        logger.debug(f" Data preprocessing done. Remaining columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, creating the interim folder if it doesn't exist."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        os.makedirs(interim_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(interim_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test.csv"), index=False)

        logger.info('Train and test data saved to %s', interim_data_path)
    except Exception as e:
        logger.error('Error occurred while saving data: %s', e)
        raise


# Main Function 
def main():
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        params_path = os.path.join(BASE_DIR, 'params.yaml')
        params = load_params(params_path)
        test_size = params['data_ingestion']['test_size']

        data_url = os.path.join(BASE_DIR, 'data', 'raw', 'youtube_bulk_raw.csv')
        df = load_data(data_url)

        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        save_data(train_data, test_data, data_path=os.path.join(BASE_DIR, 'data'))
        logger.info(' Data ingestion completed successfully.')

    except Exception as e:
        logger.error(' Failed to complete data ingestion: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
