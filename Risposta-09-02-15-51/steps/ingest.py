import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_paths):
    """
    Carica e unisce i dataset dei vini bianchi e rossi.
    """
    try:
        logger.info("Caricamento dati...")
        data_white = pd.read_csv(file_paths[0], delimiter=';')
        data_red = pd.read_csv(file_paths[1], delimiter=';')

        data_white["is_red"] = 0
        data_red["is_red"] = 1

        data = pd.concat([data_white, data_red], ignore_index=True)
        logger.info("Dati caricati con successo.")
        return data

    except Exception as e:
        logger.error(f"Errore durante l'ingestione: {e}")
        raise