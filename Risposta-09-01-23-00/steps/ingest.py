import logging
import pandas as pd

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def ingest_data(file_paths):
    """
    Carica i dati dai file specificati.

    Parametri:
    - file_paths (list): Lista di percorsi dei file da caricare.

    Restituisce:
    - pd.DataFrame: DataFrame contenente i dati caricati.
    """
    logger.info("Inizio dell'ingestione dei dati...")
    data_white = pd.read_csv(file_paths[0], delimiter=';')
    data_red = pd.read_csv(file_paths[1], delimiter=';')

    data_white["is_red"] = 0
    data_red["is_red"] = 1

    data = pd.concat([data_white, data_red], ignore_index=True)
    logger.info("Ingestione dei dati completata.")
    return data

if __name__ == "__main__":
    file_paths = ["./data/winequality-white.csv", "./data/winequality-red.csv"]
    data = ingest_data(file_paths)
    print(data.head())
