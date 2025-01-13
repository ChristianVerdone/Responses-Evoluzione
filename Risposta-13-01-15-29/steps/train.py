from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any

def custom_estimator(estimator_params: Dict[str, Any] = None):
    """
    Creates and returns an unfitted RandomForestClassifier.
    
    Args:
        estimator_params: Optional dictionary of parameters for the classifier
    Returns:
        sklearn.ensemble.RandomForestClassifier: Unfitted classifier
    """
    if estimator_params is None:
        estimator_params = {}
        
    return RandomForestClassifier(random_state=42, **estimator_params)