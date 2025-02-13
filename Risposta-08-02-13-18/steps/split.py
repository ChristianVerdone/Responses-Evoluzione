from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def split_data(data, split_ratios, target_col):
    """
    Divide il dataset in training, validation e test.
    """
    logger.info("Divisione dei dati...")
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=1 - split_ratios[0], random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]),
        random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test