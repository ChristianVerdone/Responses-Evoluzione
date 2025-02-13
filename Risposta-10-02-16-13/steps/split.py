from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def split_data(data, split_ratios):
    """
    Divide il dataset in training, validation e test.
    """
    X = data.drop(columns=["is_red"])
    y = data["is_red"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=1 - split_ratios[0], random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]),
        random_state=42
    )

    logger.info("Split completato. Training: %s, Val: %s, Test: %s",
                X_train.shape, X_val.shape, X_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test