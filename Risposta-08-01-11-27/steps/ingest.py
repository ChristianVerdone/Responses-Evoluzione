import pandas as pd
import logging

logger = logging.getLogger("IngestStep")
logger.setLevel(logging.INFO)

def ingest_data(file_paths):
    """
    Carica i dati dei vini bianco e rosso e li combina in un unico dataset.
    """
    logger.info("Caricamento dei dati da: %s", file_paths)
    data_white = pd.read_csv(file_paths[0], delimiter=';')
    data_red = pd.read_csv(file_paths[1], delimiter=';')

    # Aggiungi la colonna is_red
    data_white["is_red"] = 0
    data_red["is_red"] = 1

    # Unisci i due dataset
    data = pd.concat([data_white, data_red], ignore_index=True)
    return data

if __name__ == "__main__":
    file_paths = ["./data/winequality-white.csv", "./data/winequality-red.csv"]
    data = ingest_data(file_paths)
    data.to_csv("./data/ingested_data.csv", index=False)
    logger.info("Dati ingettati e salvati in ./data/ingested_data.csv.")
