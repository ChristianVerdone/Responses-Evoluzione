from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)


def estimator_fn(estimator_params=None):
    """
    Definisce il classificatore RandomForest.
    """
    if estimator_params is None:
        estimator_params = {}

    estimator = RandomForestClassifier(**estimator_params)
    logger.info("Modello inizializzato: %s", estimator)
    return estimator