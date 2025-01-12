import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transform_data(data):
    """
    Applica trasformazioni personalizzate ai dati.
    Args:
        data (pd.DataFrame): Dataset.
    Returns:
        pd.DataFrame: Dataset trasformato.
    """
    logger.info("Trasformazione dei dati in corso...")
    data = data.dropna()
    logger.info("Trasformazione completata.")
    return data
