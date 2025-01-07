import logging
import pandas as pd

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def transform_data(data):
    """
    Esegue trasformazioni personalizzate sui dati.

    :param data: DataFrame contenente i dati da trasformare
    :return: DataFrame trasformato
    """
    logger.info("Trasformazione dei dati.")

    # Esempio di trasformazioni personalizzate
    data = data.dropna()  # Rimuovi valori nulli
    return data


# Esempio di utilizzo della funzione
data = pd.read_csv("./data/ingested_data.csv")
transformed_data = transform_data(data)
transformed_data.to_csv("./data/transformed_data.csv", index=False)