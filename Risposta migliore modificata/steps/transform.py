from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    Trasformatore personalizzato che rimuove i valori nulli e può normalizzare i dati numerici.
    """
    def __init__(self, normalize=False):
        self.normalize = normalize

    def fit(self, X, y=None):
        """
        Metodo fit richiesto, ma in questo caso non fa nulla perché il trasformatore non è addestrabile.

        :param X: Dati di input (DataFrame).
        :param y: Target (non utilizzato).
        :return: self
        """
        return self

    def transform(self, X):
        """
        Trasforma i dati.

        :param X: DataFrame di input.
        :return: DataFrame trasformato.
        """
        logger.info("Esecuzione della trasformazione...")

        # Rimuovi righe con valori nulli
        X = X.dropna()

        # Normalizza colonne numeriche se richiesto
        if self.normalize:
            for col in X.select_dtypes(include=["float64", "int64"]).columns:
                X[f"{col}_normalized"] = (X[col] - X[col].mean()) / X[col].std()
                logger.info(f"Colonna normalizzata: {col}")

        logger.info("Trasformazione completata.")
        return X


def transformer_fn():
    """
    Restituisce un'istanza del trasformatore personalizzato.

    :return: Un trasformatore non addestrato che definisce i metodi ``fit()`` e ``transform()``.
    """
    # Configura il trasformatore con i parametri desiderati (esempio: normalizzazione)
    return CustomTransformer(normalize=True)
