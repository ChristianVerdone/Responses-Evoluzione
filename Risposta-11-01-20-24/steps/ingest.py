# ingest.py
import logging
import pandas as pd
from pandas import DataFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_file_as_dataframe(file_path: str) -> DataFrame:
    """
    Load a CSV file and return it as a pandas DataFrame.
    Args:
        file_path: Path to the CSV file
    Returns:
        DataFrame containing the loaded data
    """
    logger.info(f"Loading data from: {file_path}")
    return pd.read_csv(file_path, delimiter=';')

def process_data(df: DataFrame, is_red: int) -> DataFrame:
    """
    Process the wine data by adding the is_red column
    """
    df["is_red"] = is_red
    return df

def run(file_paths):
    """
    Load and combine wine datasets
    """
    data_white = process_data(load_file_as_dataframe(file_paths[0]), 0)
    data_red = process_data(load_file_as_dataframe(file_paths[1]), 1)
    return pd.concat([data_white, data_red], ignore_index=True)