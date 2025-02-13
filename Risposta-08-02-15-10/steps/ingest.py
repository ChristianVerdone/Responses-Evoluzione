import pandas as pd
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_paths: List[str], delimiter: str = ";") -> pd.DataFrame:
    """
    Carica e unisce i dataset dei vini bianchi e rossi.
    """
    try:
        data_white = pd.read_csv(file_paths[0], delimiter=delimiter)
        data_red = pd.read_csv(file_paths[1], delimiter=delimiter)

        data_white["is_red"] = 0
        data_red["is_red"] = 1

        data = pd.concat([data_white, data_red], ignore_index=True)
        logger.info("Dataset caricato correttamente.")
        return data
    except Exception as e:
        logger.error(f"Errore durante l'ingestione: {e}")
        raise