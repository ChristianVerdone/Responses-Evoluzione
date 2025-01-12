from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier


def estimator_fn(estimator_params: Dict[str, Any] = None):
    """
    Restituisce un estimatore RandomForestClassifier non addestrato.
    """
    estimator_params = estimator_params or {}
    return RandomForestClassifier(random_state=42, **estimator_params)


def get_estimator(estimator_params: Dict[str, Any] = None):
    return estimator_fn(estimator_params)
