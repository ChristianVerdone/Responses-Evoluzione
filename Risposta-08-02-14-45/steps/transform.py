import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transform_data(data):
    """Esegue la pulizia dei dati (rimozione valori nulli)."""
    data = data.dropna()
    logger.info(f"Dati trasformati. Dimensioni dopo pulizia: {data.shape}")
    return data