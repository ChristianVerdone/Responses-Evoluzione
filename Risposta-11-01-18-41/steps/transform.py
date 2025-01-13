import logging
import pandas as pd

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transform_data(data):
    # Esempio di trasformazioni personalizzate
    data = data.dropna()  # Rimuovi valori nulli
    return data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step di trasformazione per la recipe di MLflow")
    parser.add_argument("--file_path", required=True, help="Percorso del file da caricare")
    args = parser.parse_args()

    # Carica i dati
    data = pd.read_csv(args.file_path)

    # Trasforma i dati
    transformed_data = transform_data(data)

    # Salva i dati trasformati in un file CSV (opzionale)
    transformed_data.to_csv("transformed_data.csv", index=False)
    logger.info("Dati trasformati salvati in 'transformed_data.csv'")
