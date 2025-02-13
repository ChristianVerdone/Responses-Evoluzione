from sklearn.ensemble import RandomForestClassifier
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def estimator_fn(estimator_params: Dict[str, Any] = None):
    """
    Crea un classificatore RandomForest non addestrato.
    """
    try:
        logger.info("Creazione del modello...")
        if estimator_params is None:
            estimator_params = {}

        model = RandomForestClassifier(**estimator_params)
        logger.info("Modello creato.")
        return model

    except Exception as e:
        logger.error(f"Errore durante il training: {e}")
        raise