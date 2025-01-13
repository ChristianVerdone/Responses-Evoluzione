import logging
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def estimator_fn(estimator_params: Dict[str, Any] = None) -> BaseEstimator:
    """
    Create and return the RandomForest classifier.
    
    Args:
        estimator_params: Dictionary of model parameters
        
    Returns:
        RandomForestClassifier: Configured classifier
    """
    params = estimator_params or {}
    logger.info(f"Creating RandomForest classifier with params: {params}")
    
    return RandomForestClassifier(
        random_state=42,
        **params
    )