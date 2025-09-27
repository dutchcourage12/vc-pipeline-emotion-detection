
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os
import logging


logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('error.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'rb') as f:
            params = yaml.safe_load(f)
        test_size = params['data_ingestion']['test_size']
        logger.debug('test_size retrieved successfully')
        return test_size
    except FileNotFoundError:
        logger.error('Params file not found')
        raise FileNotFoundError(f"Params file not found: {params_path}")
    except KeyError as e:
        logger.error('Missing key in params file')
        raise KeyError(f"Missing key in params file: {e}")
    except Exception as e:
        logger.error('Error loading parameters')
        raise RuntimeError(f"Error loading parameters: {e}")

def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        if df.empty:
            logger.error('Loaded dataframe is empty.')
            raise ValueError("Loaded dataframe is empty.")
        return df
    except pd.errors.ParserError:
        logger.error('Error parsing CSV. Check file format.')
        raise ValueError("Error parsing CSV. Check file format.")
    except Exception as e:
        logger.error('Error reading data from url')
        raise RuntimeError(f"Error reading data from {url}: {e}")

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if 'tweet_id' in df.columns:
            df.drop(columns=['tweet_id'], inplace=True)

        if 'sentiment' not in df.columns:
            logger.error("Missing 'sentiment' column in dataframe.")
            raise KeyError("Missing 'sentiment' column in dataframe.")

        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)

        if final_df.empty:
            logger.error('Filtered dataframe is empty after processing.')
            raise ValueError("Filtered dataframe is empty after processing.")

        logger.debug("Data processed successfully")
        return final_df
    except Exception:
        logger.exception("Error processing data")
        raise


def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)

        train_fp = os.path.join(data_path, 'train.csv')
        test_fp = os.path.join(data_path, 'test.csv')

        train_data.to_csv(train_fp, index=False)
        test_data.to_csv(test_fp, index=False)

        print(f"Train data saved at: {train_fp}")
        print(f"Test data saved at: {test_fp}")
    except Exception as e:
        logger.error('Error saving data')
        raise RuntimeError(f"Error saving data: {e}")

def main():
    try:
        test_size = load_params('params.yaml')
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = process_data(df)

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        data_path = os.path.join('data', 'raw')

        save_data(data_path, train_data, test_data)
        print("Pipeline executed successfully.")

    except Exception as e:
        logger.error('Pipeline failed')
        print(f"Pipeline failed: {e}")

if __name__ == '__main__':
    main()
