import pandas as pd
import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transformer_fn():
    """
    Definisce la pipeline di trasformazione (rimozione valori nulli).
    """
    try:
        preprocessor = ColumnTransformer(
            transformers=[("drop_na", "passthrough", [])],  # Placeholder per operazioni future
            remainder="passthrough"
        )
        logger.info("Trasformatore definito.")
        return preprocessor
    except Exception as e:
        logger.error(f"Errore nella trasformazione: {e}")
        raise