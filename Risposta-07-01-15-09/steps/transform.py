import pandas as pd
import logging

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def transform_data(data):
    """
    Trasforma i dati rimuovendo i valori nulli.

    :param data: DataFrame contenente i dati
    :return: DataFrame trasformato
    """
    logger.info("Trasformazione dei dati")
    data = data.dropna()
    return data


# Esempio di utilizzo della funzione
if __name__ == "__main__":
    file_paths = ["./data/winequality-white.csv", "./data/winequality-red.csv"]
    data = pd.read_csv(file_paths[0], delimiter=';')
    transformed_data = transform_data(data)
    print(transformed_data.head())