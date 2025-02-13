import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transform_data(data):
    """Pulizia dei dati (esempio semplice)."""
    try:
        logger.info("Trasformazione dati...")
        transformed_data = data.dropna()
        logger.info(f"Dati trasformati. Dimensioni: {transformed_data.shape}")
        return transformed_data
    except Exception as e:
        logger.error(f"Errore durante la trasformazione: {e}")
        raise