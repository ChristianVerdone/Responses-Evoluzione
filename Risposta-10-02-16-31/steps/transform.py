import logging

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transform_data(data):
    """
    Applica trasformazioni ai dati.
    """
    logger.info("Applicando trasformazioni ai dati...")
    data = data.dropna()  # Rimuovi valori nulli
    logger.info("Trasformazioni applicate correttamente.")
    return data