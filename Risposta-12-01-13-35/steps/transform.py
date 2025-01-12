import pandas as pd
import logging

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def transform_data(data):
    """
    Applica trasformazioni al dataset.

    Args:
        data (pd.DataFrame): Dataset di input.

    Returns:
        pd.DataFrame: Dataset trasformato.
    """
    logger.info("Trasformazione dei dati...")
    data = data.dropna()
    logger.info("Trasformazione completata.")
    return data

if __name__ == "__main__":
    data = pd.read_csv("data/ingested_data.csv")
    transformed_data = transform_data(data)
    transformed_data.to_csv("data/transformed_data.csv", index=False)
    logger.info("Dati trasformati salvati in 'data/transformed_data.csv'.")
