import logging
import pandas as pd

logger = logging.getLogger(__name__)

def load_file(file_paths):
    """
    Carica i dati dei vini da file CSV e li unisce in un unico dataset.
    
    Args:
        file_paths: Lista contenente i percorsi dei file CSV per vini bianchi e rossi
    
    Returns:
        pandas.DataFrame: Dataset unificato con colonna target 'is_red'
    """
    try:
        # Carica i dati dei vini bianco e rosso
        data_white = pd.read_csv(file_paths[0], delimiter=';')
        data_red = pd.read_csv(file_paths[1], delimiter=';')
        
        # Aggiungi la colonna target
        data_white["is_red"] = 0
        data_red["is_red"] = 1
        
        # Unisci i dataset
        data = pd.concat([data_white, data_red], ignore_index=True)
        
        return data
        
    except Exception as e:
        logger.error(f"Errore durante il caricamento dei dati: {str(e)}")
        raise