import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def split_data(data, target_col, split_ratios):
    """
    Suddivide i dati in training, validation e test.

    Parametri:
    - data (pd.DataFrame): DataFrame contenente i dati.
    - target_col (str): Nome della colonna target.
    - split_ratios (list): Lista di proporzioni per la suddivisione dei dati.

    Restituisce:
    - tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Inizio della suddivisione dei dati...")
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - split_ratios[0], random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                    test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]),
                                                    random_state=42)
    logger.info("Suddivisione dei dati completata.")
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # Esempio di DataFrame
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'is_red': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    target_col = "is_red"
    split_ratios = [0.80, 0.10, 0.10]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, target_col, split_ratios)
    print("Training Data:", X_train.shape)
    print("Validation Data:", X_val.shape)
    print("Test Data:", X_test.shape)
