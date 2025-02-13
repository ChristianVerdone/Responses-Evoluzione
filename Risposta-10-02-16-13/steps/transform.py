import logging

logger = logging.getLogger(__name__)

def transform_data(data, drop_na=True):
    """
    Rimuove i valori nulli se richiesto.
    """
    if drop_na:
        data = data.dropna()
        logger.info("Valori nulli rimossi. Dimensioni: %s", data.shape)
    return data