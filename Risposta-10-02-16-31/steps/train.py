from sklearn.ensemble import RandomForestClassifier
import logging

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(X_train, y_train):
    """
    Addestra un modello RandomForest.
    """
    logger.info("Addestramento del modello...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    logger.info("Modello addestrato correttamente.")
    return model