import pandas as pd


class DropNATransformer:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.dropna()


def transformer_fn():
    """
    Restituisce un trasformatore che rimuove le righe con valori nulli.
    """
    return DropNATransformer()


def get_transformer():
    return transformer_fn()
