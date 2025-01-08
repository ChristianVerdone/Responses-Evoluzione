import pandas as pd
import logging

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def ingest_data(file_paths):
    """
    Carica i dati dei vini bianco e rosso con il delimitatore corretto.

    Parametri:
    - file_paths (list): Lista di percorsi dei file CSV.

    Restituisce:
    - DataFrame: DataFrame contenente i dati caricati.
    """
    data_white = pd.read_csv(file_paths[0], delimiter=';')
    data_red = pd.read_csv(file_paths[1], delimiter=';')

    data_white["is_red"] = 0
    data_red["is_red"] = 1

    data = pd.concat([data_white, data_red], ignore_index=True)
    return data

if __name__ == "__main__":
    file_paths = ["./data/winequality-white.csv", "./data/winequality-red.csv"]
    data = ingest_data(file_paths)
    data.to_csv("./data/ingested_data.csv", index=False)
    logger.info("Dati ingeriti e salvati in ./data/ingested_data.csv")
