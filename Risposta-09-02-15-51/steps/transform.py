import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transformer_fn():
    """
    Definisce il trasformatore (esempio base: rimozione valori nulli).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    try:
        logger.info("Creazione del trasformatore...")
        preprocessor = ColumnTransformer(
            transformers=[("drop_na", "drop", "any")]
        )
        logger.info("Trasformatore creato.")
        return preprocessor

    except Exception as e:
        logger.error(f"Errore durante la trasformazione: {e}")
        raise