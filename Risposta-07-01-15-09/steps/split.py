import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def split_data(data, target_col, split_ratios=[0.80, 0.10, 0.10]):
    """
    Divide il dataset nei set di training, validation e test.

    :param data: DataFrame contenente i dati
    :param target_col: Nome della colonna target
    :param split_ratios: Lista di tre valori che rappresentano le proporzioni per training, validation e test
    :return: Tuple contenente i DataFrame per training, validation e test
    """
    logger.info(f"Dividendo il dataset con i seguenti rapporti: {split_ratios}")

    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - split_ratios[0], random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                    test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]),
                                                    random_state=42)

    logger.info(f"Dimensioni dei set: Training={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# Esempio di utilizzo della funzione
if __name__ == "__main__":
    file_paths = ["./data/winequality-white.csv", "./data/winequality-red.csv"]
    data = pd.read_csv(file_paths[0], delimiter=';')
    target_col = "is_red"
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, target_col)
    print(f"Training set:\n{X_train.head()}")