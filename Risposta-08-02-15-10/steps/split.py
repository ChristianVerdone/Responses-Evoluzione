import pandas as pd
import logging
from typing import Dict, List
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_data(data: pd.DataFrame, split_ratios: List[float]) -> Dict[str, pd.DataFrame]:
    """
    Divide il dataset in training, validation e test set.
    """
    try:
        X = data.drop(columns=["is_red"])
        y = data["is_red"]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - split_ratios[0], random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]),
            random_state=42
        )

        logger.info("Split completato.")
        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test
        }
    except Exception as e:
        logger.error(f"Errore durante lo split: {e}")
        raise