from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def estimator_fn(estimator_params=None):
    """Crea un'istanza di RandomForestClassifier."""
    if estimator_params is None:
        estimator_params = {}
    model = RandomForestClassifier(**estimator_params)
    logger.info("Modello RandomForest inizializzato")
    return model