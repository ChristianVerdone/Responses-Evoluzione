from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(X_train, y_train, estimator_params):
    """Addestra il modello RandomForest."""
    try:
        logger.info("Addestramento modello...")
        model = RandomForestClassifier(**estimator_params)
        model.fit(X_train, y_train)
        logger.info("Modello addestrato con successo.")
        return model
    except Exception as e:
        logger.error(f"Errore durante l'addestramento: {e}")
        raise