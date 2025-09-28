
import os
import yaml
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
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

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)



def load_params(params_path: str) -> int:
    """Read max_features from a YAML params file."""
    try:
        if not os.path.exists(params_path):
            logger.error('Params file not found')
            raise FileNotFoundError(f"Params file not found: {params_path}")
        with open(params_path, "rb") as f:
            params = yaml.safe_load(f)
        max_features = params["feature_engineering"]["max_features"]
        if not isinstance(max_features, int) or max_features <= 0:
            logger.error('feature_engineering.max_features must be a positive int')
            raise ValueError("feature_engineering.max_features must be a positive int")
        return max_features
    except (KeyError, TypeError) as e:
        logger.error('Invalid params structure')
        raise KeyError(f"Invalid params structure: {e}") from e
    except Exception as e:
        logger.exception('Failed to load params')
        raise RuntimeError(f"Failed to load params: {e}") from e


def load_train_data(train_path: str) -> pd.DataFrame:
    """Load training dataframe."""
    try:
        return pd.read_csv(train_path)
    except FileNotFoundError as e:
        logger.error('Train file not found')
        raise FileNotFoundError(f"Train file not found: {train_path}") from e
    except pd.errors.ParserError as e:
        logger.error('Train CSV parse error')
        raise ValueError(f"Train CSV parse error: {e}") from e
    except Exception as e:
        logger.exception('Failed to load train data')
        raise RuntimeError(f"Failed to load train data: {e}") from e


def load_test_data(test_path: str) -> pd.DataFrame:
    """Load test dataframe."""
    try:
        return pd.read_csv(test_path)
    except FileNotFoundError as e:
        logger.error('Test file not found')
        raise FileNotFoundError(f"Test file not found: {test_path}") from e
    except pd.errors.ParserError as e:
        logger.error('Test CSV parse error')
        raise ValueError(f"Test CSV parse error: {e}") from e
    except Exception as e:
        logger.error('Failed to load test data')
        raise RuntimeError(f"Failed to load test data: {e}") from e


def get_train_content(train_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract X/y from train frame with basic validation."""
    try:
        for col in ("content", "sentiment"):
            if col not in train_data.columns:
                logger.error('Missing column in train data')
                raise KeyError(f"Missing column in train data: '{col}'")
        X_train = train_data["content"].fillna("").astype(str).values
        y_train = train_data["sentiment"].values
        if len(X_train) == 0:
            logger.error('Train features are empty')
            raise ValueError("Train features are empty")
        return X_train, y_train
    except Exception as e:
        logger.exception('Failed to extract train content')
        raise RuntimeError(f"Failed to extract train content: {e}") from e


def get_test_content(test_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract X/y from test frame with basic validation."""
    try:
        for col in ("content", "sentiment"):
            if col not in test_data.columns:
                logger.error('Missing column in test data')
                raise KeyError(f"Missing column in test data: '{col}'")
        X_test = test_data["content"].fillna("").astype(str).values
        y_test = test_data["sentiment"].values
        if len(X_test) == 0:
            logger.error('Test features are empty')
            raise ValueError("Test features are empty")
        return X_test, y_test
    except Exception as e:
        logger.exception('Failed to extract test content')
        raise RuntimeError(f"Failed to extract test content: {e}") from e


def vectorize_tfidf(
    X_train: np.ndarray,
    X_test: np.ndarray,
    max_features: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Fit TfidfVectorizer on train and transform test."""
    try:
        vect = TfidfVectorizer(max_features=max_features)
        X_train_bow = vect.fit_transform(X_train)
        X_test_bow = vect.transform(X_test)
        return X_train_bow, X_test_bow, vect
    except ValueError as e:
        # e.g., if inputs contain non-strings / NaNs slipped through
        logger.error('BOW vectorization failed')
        raise ValueError(f"BOW vectorization failed: {e}") from e
    except Exception as e:
        logger.exception('Unexpected error in vectorize_bow')
        raise RuntimeError(f"Unexpected error in vectorize_bow: {e}") from e


def make_df_from_sparse(
    X_train_bow,
    X_test_bow,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert sparse matrices to DataFrames and attach labels."""
    try:
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df["label"] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df["label"] = y_test

        return train_df, test_df
    except Exception as e:
        logger.exception('Failed to build DataFrames from sparse arrays')
        raise RuntimeError(f"Failed to build DataFrames from sparse arrays: {e}") from e


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str) -> None:
    """Persist train/test feature CSVs."""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_fp = os.path.join(data_path, "train_tfidf.csv")
        test_fp = os.path.join(data_path, "test_tfidf.csv")
        train_df.to_csv(train_fp, index=False)
        test_df.to_csv(test_fp, index=False)
    except Exception as e:
        logger.error('Failed to save features')
        raise RuntimeError(f"Failed to save features: {e}") from e


def main() -> None:
    try:
        params_path = "params.yaml"
        max_features = load_params(params_path)

        train_path = "./data/interim/train_processed_data.csv"
        test_path = "./data/interim/test_processed_data.csv"
        train_data = load_train_data(train_path)
        test_data = load_test_data(test_path)

        X_train, y_train = get_train_content(train_data)
        X_test, y_test = get_test_content(test_data)

        X_train_bow, X_test_bow, _ = vectorize_tfidf(
            X_train, X_test, max_features=max_features
        )

        # NOTE: pass the **BOW** matrices here (not the raw X)
        train_df, test_df = make_df_from_sparse(
            X_train_bow, X_test_bow, y_train, y_test
        )

        data_path = os.path.join("data", "processed")
        save_data(train_df, test_df, data_path)

        logger.info("Feature engineering completed successfully.")

    except Exception as e:
        # single failure surface so DVC/CLI sees one clear message
        logger.exception('Pipeline failed')
        print(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
