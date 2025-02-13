from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_data(data, target_col, split_ratios):
    """Divide i dati in train/val/test."""
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - split_ratios[0], random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                    test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]),
                                                    random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_step(data):
    split_ratios = {{SPLIT_RATIOS}}  # Valore da Jinja2
    return split_data(data, "is_red", split_ratios)