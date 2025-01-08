from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier

def estimator_fn(estimator_params: Dict[str, Any] = None) -> RandomForestClassifier:
    """
    Crea e restituisce un classificatore Random Forest non addestrato.
    
    Args:
        estimator_params: Dizionario opzionale di parametri per il classificatore
    
    Returns:
        RandomForestClassifier non addestrato
    """
    if estimator_params is None:
        estimator_params = {}
    
    return RandomForestClassifier(
        random_state=42,
        **estimator_params
    )