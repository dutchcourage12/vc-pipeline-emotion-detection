
import os
import re
import nltk
import numpy as np
import pandas as pd
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------- Logging Setup ---------- #
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("data_preprocessing_error.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ---------- Utility Setup ---------- #
def setup_nltk():
    """Download required NLTK resources."""
    try:
        nltk.download("wordnet", quiet=True)
        nltk.download("stopwords", quiet=True)
        logger.info("NLTK resources downloaded successfully.")
    except Exception as e:
        logger.exception("Failed to download NLTK resources")
        raise


# ---------- Text Cleaning Functions ---------- #
def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    return " ".join([lemmatizer.lemmatize(word) for word in words])


def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    words = [w for w in str(text).split() if w not in stop_words]
    return " ".join(words)


def removing_numbers(text: str) -> str:
    return "".join([ch for ch in text if not ch.isdigit()])


def lower_case(text: str) -> str:
    return " ".join([word.lower() for word in text.split()])


def removing_punctuations(text: str) -> str:
    text = re.sub(
        "[%s]" % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), " ", text
    )
    text = text.replace("؛", "")
    text = re.sub("\s+", " ", text)
    return " ".join(text.split()).strip()


def removing_urls(text: str) -> str:
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub(r"", text)


def remove_small_sentences(df: pd.DataFrame, col: str = "content") -> pd.DataFrame:
    df.loc[df[col].str.split().str.len() < 3, col] = np.nan
    return df


def normalize_text(df: pd.DataFrame, col: str = "content") -> pd.DataFrame:
    try:
        df[col] = df[col].astype(str)
        df[col] = df[col].apply(lower_case)
        df[col] = df[col].apply(remove_stop_words)
        df[col] = df[col].apply(removing_numbers)
        df[col] = df[col].apply(removing_punctuations)
        df[col] = df[col].apply(removing_urls)
        df[col] = df[col].apply(lemmatization)

        logger.debug(f"Normalized text in column '{col}'")
        return df
    except Exception as e:
        logger.exception("Failed during text normalization")
        raise


# ---------- Pipeline Functions ---------- #
def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logger.error("Train or test CSV file not found.")
            raise FileNotFoundError(
                f"Missing file: {train_path} or {test_path}"
            )

        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        if train.empty or test.empty:
            logger.error("One of the datasets is empty.")
            raise ValueError("Train or test dataset is empty")

        logger.info(
            f"Loaded data successfully. Train shape: {train.shape}, Test shape: {test.shape}"
        )
        return train, test
    except Exception as e:
        logger.exception("Failed to load data")
        raise


def save_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = "data/interim"
) -> None:
    try:
        os.makedirs(output_dir, exist_ok=True)

        train_fp = os.path.join(output_dir, "train_processed_data.csv")
        test_fp = os.path.join(output_dir, "test_processed_data.csv")

        train_df.to_csv(train_fp, index=False)
        test_df.to_csv(test_fp, index=False)

        logger.info(f"Saved processed data to {output_dir}")
    except Exception as e:
        logger.exception("Failed to save processed data")
        raise


def preprocess_and_save(
    train_path: str, test_path: str, output_dir: str = "data/interim"
) -> None:
    try:
        setup_nltk()
        train_df, test_df = load_data(train_path, test_path)

        train_processed = normalize_text(train_df, col="content")
        test_processed = normalize_text(test_df, col="content")

        save_data(train_processed, test_processed, output_dir)

        logger.info("Preprocessing pipeline executed successfully.")
    except Exception as e:
        logger.exception("Preprocessing pipeline failed")
        print(f"Pipeline failed: {e}")


def main():
    preprocess_and_save(
        train_path="./data/raw/train.csv",
        test_path="./data/raw/test.csv",
        output_dir="./data/interim",
    )


if __name__ == "__main__":
    main()
