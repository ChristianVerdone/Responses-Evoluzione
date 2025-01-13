# train.py
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier

def estimator_fn(estimator_params: Dict[str, Any] = None):
    """
    Returns an unfitted estimator that has fit() and predict() methods
    """
    if estimator_params is None:
        estimator_params = {}
    
    return RandomForestClassifier(random_state=42, **estimator_params)