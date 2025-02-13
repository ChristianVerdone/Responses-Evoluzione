import pandas as pd
import logging

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_data(file_paths):
    """
    Carica i dati dai file CSV e li unisce.
    """
    logger.info("Caricamento dei dati...")
    data_white = pd.read_csv(file_paths[0], delimiter=';')
    data_red = pd.read_csv(file_paths[1], delimiter=';')

    # Aggiungi la colonna is_red (0 per bianco, 1 per rosso)
    data_white["is_red"] = 0
    data_red["is_red"] = 1

    # Unisci i due dataset
    data = pd.concat([data_white, data_red], ignore_index=True)
    logger.info("Dati caricati correttamente.")
    return data