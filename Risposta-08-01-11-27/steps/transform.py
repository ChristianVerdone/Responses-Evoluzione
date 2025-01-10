import pandas as pd
import logging

logger = logging.getLogger("TransformStep")
logger.setLevel(logging.INFO)

def transform_data(data):
    """
    Trasforma i dati rimuovendo valori nulli.
    """
    logger.info("Inizio trasformazione dei dati.")
    data = data.dropna()
    return data

if __name__ == "__main__":
    # Carica i dati ingettati
    data = pd.read_csv("./data/ingested_data.csv")

    # Trasforma i dati
    data = transform_data(data)
    data.to_csv("./data/transformed_data.csv", index=False)
    logger.info("Dati trasformati e salvati in ./data/transformed_data.csv.")
