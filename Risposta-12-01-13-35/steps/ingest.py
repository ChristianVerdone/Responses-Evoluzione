import pandas as pd
import logging

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def ingest_data(file_paths):
    """
    Carica e combina i dati dei vini bianchi e rossi, aggiungendo la colonna `is_red`.

    Args:
        file_paths (list): Lista di percorsi dei file CSV [white, red].

    Returns:
        pd.DataFrame: Dataset combinato.
    """
    logger.info("Caricamento dei dati...")
    data_white = pd.read_csv(file_paths[0], delimiter=';')
    data_red = pd.read_csv(file_paths[1], delimiter=';')

    data_white["is_red"] = 0
    data_red["is_red"] = 1

    data = pd.concat([data_white, data_red], ignore_index=True)
    logger.info("Caricamento completato.")
    return data

if __name__ == "__main__":
    import yaml
    with open("local.yaml", "r") as f:
        config = yaml.safe_load(f)

    file_paths = config["INGEST_CONFIG"]["location"]
    data = ingest_data(file_paths)
    data.to_csv("data/ingested_data.csv", index=False)
    logger.info("Dati salvati in 'data/ingested_data.csv'.")
