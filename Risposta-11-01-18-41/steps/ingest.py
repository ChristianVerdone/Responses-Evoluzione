import logging
import pandas as pd

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_data(file_paths):
    # Carica i dati dei vini bianco e rosso con il delimitatore corretto
    data_white = pd.read_csv(file_paths[0], delimiter=';')
    data_red = pd.read_csv(file_paths[1], delimiter=';')

    # Aggiungi la colonna is_red (0 per bianco, 1 per rosso)
    data_white["is_red"] = 0
    data_red["is_red"] = 1

    # Unisci i due dataset
    data = pd.concat([data_white, data_red], ignore_index=True)
    return data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step di ingest per la recipe di MLflow")
    parser.add_argument("--file_paths", nargs='+', required=True, help="Percorsi dei file da caricare")
    args = parser.parse_args()

    # Carica i dati
    data = ingest_data(args.file_paths)

    # Salva i dati ingestiti in un file CSV (opzionale)
    data.to_csv("ingested_data.csv", index=False)
    logger.info("Dati ingestiti salvati in 'ingested_data.csv'")
