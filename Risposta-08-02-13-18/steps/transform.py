from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    Trasformatore personalizzato per rimuovere valori nulli.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.dropna()
        return X


def transformer_fn():
    return CustomTransformer()