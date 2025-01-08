# steps/train.py
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any

def train_method(estimator_params: Dict[str, Any] = None):
    """
    Crea e configura il modello RandomForestClassifier.
    
    Args:
        estimator_params: Parametri opzionali per il classificatore
    Returns:
        RandomForestClassifier: Modello non addestrato
    """
    if estimator_params is None:
        estimator_params = {}
    
    return RandomForestClassifier(random_state=42, **estimator_params)