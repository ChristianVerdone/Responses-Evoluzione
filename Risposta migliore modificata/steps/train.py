from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def custom_estimator(params=None):
    """
    Restituisce un'istanza di un modello compatibile con scikit-learn.

    Il modello deve implementare i metodi `fit()` e `predict()`.

    :param params: Dizionario opzionale con i parametri del modello.
    :return: Modello scikit-learn.
    """
    logger.info("Creazione del modello RandomForestClassifier...")

    # Usa i parametri forniti o imposta valori di default
    n_estimators = params.get("n_estimators", 100) if params else 100
    max_depth = params.get("max_depth", 10) if params else 10
    random_state = params.get("random_state", 42) if params else 42

    # Crea il modello
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    logger.info(f"Modello RandomForestClassifier creato con n_estimators={n_estimators}, "
                f"max_depth={max_depth}, random_state={random_state}.")
    return model
