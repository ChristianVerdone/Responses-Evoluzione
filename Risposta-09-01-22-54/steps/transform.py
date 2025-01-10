# steps/transform.py
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def transform_data(data):
    """
    Transform the input data by removing null values.
    
    Args:
        data (pd.DataFrame): Input DataFrame
    Returns:
        pd.DataFrame: Transformed DataFrame
    """
    logger.info("Starting data transformation")
    
    # Remove null values
    data = data.dropna()
    
    logger.info(f"Transformation complete. Remaining rows: {len(data)}")
    return data