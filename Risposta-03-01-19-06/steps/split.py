import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def split_data(data, target_col, split_ratios):
    """
    Divide il dataset nei set di training, validation e test.

    :param data: DataFrame contenente i dati
    :param target_col: Nome della colonna target
    :param split_ratios: Lista di tre valori che rappresentano le proporzioni per training, validation e test
    :return: Tuple contenente i DataFrame per training, validation e test
    """
    logger.info(f"Divisione del dataset con proporzioni: {split_ratios}")

    X = data.drop(columns=[target_col])
    y = data[target_col]

    train_size, val_size = split_ratios[0], split_ratios[1]
    test_size = 1 - train_size - val_size

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_size + test_size), random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_size / (val_size + test_size)),
                                                    random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


# Esempio di utilizzo della funzione
if __name__ == "__main__":
    data = pd.read_csv("/data/ingested_data.csv")
    target_col = "is_red"
    split_ratios = [0.80, 0.10, 0.10]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, target_col, split_ratios)
    X_train.to_csv("/data/X_train.csv", index=False)
    X_val.to_csv("/data/X_val.csv", index=False)
    X_test.to_csv("/data/X_test.csv", index=False)
    y_train.to_csv("/data/y_train.csv", index=False)
    y_val.to_csv("/data/y_val.csv", index=False)
    y_test.to_csv("/data/y_test.csv", index=False)
