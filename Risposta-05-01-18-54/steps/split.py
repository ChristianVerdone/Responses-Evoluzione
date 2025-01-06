import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def split_data(data, split_ratios):
    """
    Divide i dati nei set di training, validation e test.

    :param data: DataFrame contenente i dati da dividere
    :param split_ratios: Lista di tre valori che rappresentano le proporzioni per training, validation e test
    :return: Tuple contenente i DataFrame per training, validation e test
    """
    target_col = "is_red"
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - split_ratios[0], random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                    test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]),
                                                    random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    data = pd.read_csv("./data/transformed_data.csv")
    split_ratios = [0.80, 0.10, 0.10]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, split_ratios)

    X_train.to_csv("./data/X_train.csv", index=False)
    X_val.to_csv("./data/X_val.csv", index=False)
    X_test.to_csv("./data/X_test.csv", index=False)
    y_train.to_csv("./data/y_train.csv", index=False)
    y_val.to_csv("./data/y_val.csv", index=False)
    y_test.to_csv("./data/y_test.csv", index=False)