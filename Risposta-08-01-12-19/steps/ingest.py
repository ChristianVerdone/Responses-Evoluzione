# steps/ingest.py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_data(file_paths):
    """
    Carica e unisce i dataset dei vini bianchi e rossi.
    
    Args:
        file_paths: Lista contenente i percorsi dei file CSV per vini bianchi e rossi
    Returns:
        DataFrame: Dataset unificato con colonna target
    """
    try:
        data_white = pd.read_csv(file_paths[0], delimiter=';')
        data_red = pd.read_csv(file_paths[1], delimiter=';')
        
        data_white["is_red"] = 0
        data_red["is_red"] = 1
        
        data = pd.concat([data_white, data_red], ignore_index=True)
        return data
    except Exception as e:
        logger.error(f"Errore durante l'ingestione dei dati: {str(e)}")
        raise