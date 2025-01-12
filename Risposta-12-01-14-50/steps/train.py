from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(X_train, y_train):
    """
    Addestra un modello RandomForest.
    Args:
        X_train: Feature del training set.
        y_train: Target del training set.
    Returns:
        RandomForestClassifier: Modello addestrato.
    """
    logger.info("Inizio dell'addestramento del modello...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    logger.info("Modello addestrato.")
    return model
