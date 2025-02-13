import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_data(file_paths, delimiter=";"):
    """Carica e combina i dataset dei vini."""
    try:
        logger.info("Caricamento dati...")
        data_white = pd.read_csv(file_paths[0], delimiter=delimiter)
        data_red = pd.read_csv(file_paths[1], delimiter=delimiter)

        data_white["is_red"] = 0
        data_red["is_red"] = 1

        combined_data = pd.concat([data_white, data_red], ignore_index=True)
        logger.info(f"Dati caricati. Dimensioni: {combined_data.shape}")
        return combined_data
    except Exception as e:
        logger.error(f"Errore durante l'ingestione: {e}")
        raise