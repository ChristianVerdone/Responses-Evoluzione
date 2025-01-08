from typing import Any, Dict
import pandas as pd

def transformer_fn():
    """
    Restituisce un trasformatore per la preparazione dei dati.
    """
    class CustomTransformer:
        def fit(self, X: pd.DataFrame, y=None):
            return self
            
        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            # Applica le trasformazioni come nello script originale
            X = X.dropna()
            return X
    
    return CustomTransformer()