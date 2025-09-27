
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml
import logging

# --- Logging setup ---
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_building_error.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# --- Functions ---
def load_params(params_path: str) -> dict:
    """Load model parameters from params.yaml"""
    try:
        if not os.path.exists(params_path):
            logger.error("Params file not found")
            raise FileNotFoundError(f"Params file not found: {params_path}")

        with open(params_path, "rb") as f:
            params = yaml.safe_load(f)["model_building"]

        if not isinstance(params, dict):
            logger.error("model_building section missing or invalid in params.yaml")
            raise ValueError("model_building section missing or invalid")

        logger.debug("Parameters loaded successfully")
        return params

    except Exception as e:
        logger.exception("Failed to load params")
        raise


def load_data(data_path: str) -> pd.DataFrame:
    """Load training dataset from CSV"""
    try:
        if not os.path.exists(data_path):
            logger.error("Training data file not found")
            raise FileNotFoundError(f"Training data file not found: {data_path}")

        df = pd.read_csv(data_path)
        if df.empty:
            logger.error("Training dataset is empty")
            raise ValueError("Training dataset is empty")

        logger.debug(f"Training data loaded with shape {df.shape}")
        return df

    except Exception as e:
        logger.exception("Failed to load training data")
        raise


def separate_xy(train_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split features (X) and labels (y)"""
    try:
        if train_data.shape[1] < 2:
            logger.error("Dataset must have at least one feature and one label column")
            raise ValueError("Insufficient columns in dataset")

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        logger.debug(f"Separated X shape: {X_train.shape}, y shape: {y_train.shape}")
        return X_train, y_train

    except Exception as e:
        logger.exception("Failed to separate features and labels")
        raise


def model_fit(X_train: np.ndarray, y_train: np.ndarray, params: dict):
    """Fit GradientBoostingClassifier"""
    try:
        clf = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"]
        )
        clf.fit(X_train, y_train)
        logger.info("Model trained successfully")
        return clf
    except Exception as e:
        logger.exception("Failed to train model")
        raise


def save_model(clf, model_dir: str = "models", model_name: str = "model.pkl") -> str:
    """Save trained model into models/ directory as model.pkl."""
    try:
        # ensure models/ directory exists
        os.makedirs(model_dir, exist_ok=True)

        # full path = models/model.pkl
        model_path = os.path.join(model_dir, model_name)

        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        print(f"Model saved successfully at {model_path}")
        return model_path
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {e}")


# --- Main entrypoint ---
def main() -> None:
    try:
        params = load_params("params.yaml")
        train_data = load_data("./data/processed/train_bow.csv")
        X_train, y_train = separate_xy(train_data)

        clf = model_fit(X_train, y_train, params)
        save_model(clf, model_dir="models", model_name="model.pkl")


        logger.info("Pipeline executed successfully.")

    except Exception as e:
        logger.exception("Pipeline failed")
        print(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
