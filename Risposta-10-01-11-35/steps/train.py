# steps/train.py
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier

def train_estimator(estimator_params: Dict[str, Any] = None):
    """
    Creates and returns an unfitted RandomForestClassifier.
    
    Args:
        estimator_params: Optional dictionary of parameters for the classifier
    Returns:
        Unfitted RandomForestClassifier instance
    """
    if estimator_params is None:
        estimator_params = {}
    
    return RandomForestClassifier(
        random_state=42,
        **estimator_params
    )