# steps/ingest.py
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def ingest_data(file_paths, options=None):
    """
    Load wine quality data from CSV files and combine them.
    
    Args:
        file_paths (list): List of paths to CSV files
        options (dict): Additional options including delimiter
    """
    logger.info("Starting data ingestion")
    
    # Set default delimiter if not provided
    delimiter = options.get('delimiter', ';') if options else ';'
    
    # Load white and red wine data
    data_white = pd.read_csv(file_paths[0], delimiter=delimiter)
    data_red = pd.read_csv(file_paths[1], delimiter=delimiter)
    
    # Add is_red column
    data_white["is_red"] = 0
    data_red["is_red"] = 1
    
    # Combine datasets
    data = pd.concat([data_white, data_red], ignore_index=True)
    
    logger.info(f"Loaded {len(data)} rows of data")
    return data