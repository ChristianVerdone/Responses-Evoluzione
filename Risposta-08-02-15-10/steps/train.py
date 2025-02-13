import logging
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def estimator_fn(estimator_params: dict = None):
    """
    Definisce lo stimatore RandomForest.
    """
    try:
        if estimator_params is None:
            estimator_params = {}

        model = RandomForestClassifier(**estimator_params)
        logger.info("Modello inizializzato.")
        return model
    except Exception as e:
        logger.error(f"Errore durante il training: {e}")
        raise