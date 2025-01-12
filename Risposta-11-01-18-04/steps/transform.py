import pandas as pd
import logging

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transform_data(data, dropna=True):
    if dropna:
        data = data.dropna()  # Rimuovi valori nulli
    return data

# Esempio di utilizzo
X_train = transform_data(X_train)
X_val = transform_data(X_val)
X_test = transform_data(X_test)
logger.info(f"Dati trasformati con successo. Numero totale di righe di training: {len(X_train)}")
