# steps/ingest.py
import logging
import pandas as pd
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_file_as_dataframe(file_paths: List[str]) -> pd.DataFrame:
    """
    Load wine quality data from CSV files and combine them.
    
    Args:
        file_paths: List of paths to CSV files
    Returns:
        Combined DataFrame with wine quality data
    """
    try:
        data_white = pd.read_csv(file_paths[0], delimiter=';')
        data_red = pd.read_csv(file_paths[1], delimiter=';')
        
        data_white["is_red"] = 0
        data_red["is_red"] = 1
        
        data = pd.concat([data_white, data_red], ignore_index=True)
        logger.info(f"Successfully loaded {len(data)} records")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise