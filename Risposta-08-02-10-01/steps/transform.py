import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomTransformer:
    def fit(self, data):
        return self  # Nessuna logica di fit necessaria per questa trasformazione

    def transform(self, data):
        data = data.dropna()
        logger.info("Dati trasformati. Dimensioni dopo pulizia: %s", data.shape)
        return data

def transformer_fn():
    return CustomTransformer()