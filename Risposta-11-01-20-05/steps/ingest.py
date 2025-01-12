import logging
import pandas as pd
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_file_as_dataframe(file_path: str) -> pd.DataFrame:
    """Load a CSV file with wine quality data."""
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path, delimiter=';')

def run(file_paths: List[str]) -> pd.DataFrame:
    """
    Main ingest function that loads and combines wine quality datasets.
    
    Args:
        file_paths: List containing paths to white and red wine datasets
    
    Returns:
        pd.DataFrame: Combined dataset with is_red label
    """
    # Load white and red wine data
    data_white = load_file_as_dataframe(file_paths[0])
    data_red = load_file_as_dataframe(file_paths[1])
    
    # Add is_red label
    data_white["is_red"] = 0
    data_red["is_red"] = 1
    
    # Combine datasets
    data = pd.concat([data_white, data_red], ignore_index=True)
    logger.info(f"Created combined dataset with shape {data.shape}")
    
    return data