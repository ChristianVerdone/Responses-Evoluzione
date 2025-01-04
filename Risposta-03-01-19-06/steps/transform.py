import logging
import pandas as pd

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def transform_data(data):
    """
    Trasforma i dati rimuovendo i valori nulli.

    :param data: DataFrame contenente i dati
    :return: DataFrame contenente i dati trasformati
    """
    logger.info("Trasformazione dei dati: rimozione dei valori nulli")
    data = data.dropna()
    return data


# Esempio di utilizzo della funzione
if __name__ == "__main__":
    X_train = pd.read_csv("/data/X_train.csv")
    X_val = pd.read_csv("/data/X_val.csv")
    X_test = pd.read_csv("/data/X_test.csv")

    X_train = transform_data(X_train)
    X_val = transform_data(X_val)
    X_test = transform_data(X_test)

    X_train.to_csv("/data/X_train_transformed.csv", index=False)
    X_val.to_csv("/data/X_val_transformed.csv", index=False)
    X_test.to_csv("/data/X_test_transformed.csv", index=False)
