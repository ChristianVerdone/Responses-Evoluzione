# transform.py
from pandas import DataFrame

def transformer_fn():
    """
    Returns a transformer object with fit and transform methods
    """
    class CustomTransformer:
        def fit(self, X, y=None):
            return self
            
        def transform(self, X):
            return X.dropna()
    
    return CustomTransformer()