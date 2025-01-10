# steps/train.py
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)

def train_model(X_train, y_train, **kwargs):
    """
    Train a RandomForestClassifier model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
    Returns:
        sklearn.ensemble.RandomForestClassifier: Trained model
    """
    logger.info("Starting model training")
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    logger.info("Model training completed")
    return model