import pandas as pd
import logging

logger = logging.getLogger(__name__)

def ingest_data(file_paths):
    """
    Load and combine wine quality datasets.
    
    Args:
        file_paths: List containing paths to white and red wine datasets
    Returns:
        pandas.DataFrame: Combined dataset with is_red label
    """
    logger.info("Loading wine quality datasets...")
    
    data_white = pd.read_csv(file_paths[0], delimiter=';')
    data_red = pd.read_csv(file_paths[1], delimiter=';')
    
    data_white["is_red"] = 0
    data_red["is_red"] = 1
    
    data = pd.concat([data_white, data_red], ignore_index=True)
    
    logger.info(f"Loaded dataset with {len(data)} samples")
    return data