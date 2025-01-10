import logging
import pandas as pd

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def transform_data(data):
    """
    Trasforma i dati.

    Parametri:
    - data (pd.DataFrame): DataFrame contenente i dati da trasformare.

    Restituisce:
    - pd.DataFrame: DataFrame contenente i dati trasformati.
    """
    logger.info("Inizio della trasformazione dei dati...")
    data = data.dropna()  # Rimuovi valori nulli
    logger.info("Trasformazione dei dati completata.")
    return data

if __name__ == "__main__":
    # Esempio di DataFrame
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    })
    transformed_data = transform_data(data)
    print(transformed_data.head())
