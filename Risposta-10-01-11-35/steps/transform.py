# steps/transform.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.dropna()
        return X

def create_transformer():
    """
    Creates and returns the custom transformer.
    """
    return CustomTransformer()