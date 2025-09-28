
import os
import numpy as np
import pandas as pd
import pickle
import json
import logging
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score

# ---------- Logging Setup ---------- #
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_evaluation_error.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ---------- Functions ---------- #
def load_model(model_path: str):
    """Load a trained model from pickle file."""
    try:
        if not os.path.exists(model_path):
            logger.error("Model file not found")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            clf = pickle.load(f)

        logger.info(f"Model loaded from {model_path}")
        return clf
    except Exception as e:
        logger.exception("Failed to load model")
        raise


def load_test_data(test_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load test dataset and separate into X and y."""
    try:
        if not os.path.exists(test_path):
            logger.error("Test dataset not found")
            raise FileNotFoundError(f"Test dataset not found: {test_path}")

        test_data = pd.read_csv(test_path)
        if test_data.empty:
            logger.error("Test dataset is empty")
            raise ValueError("Test dataset is empty")

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        logger.info(f"Test data loaded successfully with shape {test_data.shape}")
        return X_test, y_test
    except Exception as e:
        logger.exception("Failed to load test data")
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model performance using common metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba),
        }

        logger.info("Model evaluation completed successfully")
        return metrics_dict
    except Exception as e:
        logger.exception("Failed to evaluate model")
        raise


def save_metrics(metrics: dict, output_path: str = "reports/metrics.json") -> None:
    try:
        dir_ = os.path.dirname(output_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {output_path}")
    except Exception:
        logger.exception("Failed to save metrics")
        raise


# ---------- Pipeline ---------- #
def main() -> None:
    try:
        logger.info("Starting model evaluation pipeline...")

        # Paths expected by dvc.yaml
        model_path = os.path.join("models", "model.pkl")
        test_path  = os.path.join("data", "processed", "test_tfidf.csv")

        # Load artifacts
        clf = load_model(model_path)
        X_test, y_test = load_test_data(test_path)
        logger.debug(f"Loaded test data: X_test={X_test.shape}, y_test={y_test.shape}")

        # Evaluate
        metrics = evaluate_model(clf, X_test, y_test)

        # Ensure reports/ exists and save metrics there
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        metrics_path = os.path.join(reports_dir, "metrics.json")
        save_metrics(metrics, metrics_path)

        logger.info(f"Evaluation pipeline executed successfully. Metrics saved to {metrics_path}")
        print("Evaluation completed. Metrics saved to reports/metrics.json")

    except Exception as e:
        logger.exception("Evaluation pipeline failed")
        print(f"Pipeline failed: {e}")



if __name__ == "__main__":
    main()
