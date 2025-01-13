import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_data(data, target_col, split_ratios):
    """
    Divide i dati in training, validation e test set.
    Args:
        data (pd.DataFrame): Dataset combinato.
        target_col (str): Nome della colonna target.
        split_ratios (list): Proporzioni per train, validation, test.
    Returns:
        dict: Dataset diviso in training, validation e test.
    """
    logger.info("Inizio dello split dei dati...")
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=1 - split_ratios[0], random_state=42)
    
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
