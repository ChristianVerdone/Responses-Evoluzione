import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_data(data, split_ratios):
    train_ratio, val_ratio, test_ratio = split_ratios
    X = data.drop(columns=["is_red"])
    y = data["is_red"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

    logger.info(f"Dati suddivisi in training ({len(X_train)} righe), validation ({len(X_val)} righe) e test ({len(X_test)} righe)")
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step di split per la recipe di MLflow")
    parser.add_argument("--file_path", required=True, help="Percorso del file da caricare")
    parser.add_argument("--split_ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1], help="Proporzioni per la suddivisione dei dati")
    args = parser.parse_args()

    # Carica i dati ingestiti
    data = pd.read_csv(args.file_path)

    # Suddividi i dati
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, args.split_ratios)

    # Salva i dati suddivisi in file CSV (opzionale)
    X_train.to_csv("X_train.csv", index=False)
    X_val.to_csv("X_val.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_val.to_csv("y_val.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    logger.info("Dati suddivisi salvati in 'X_train.csv', 'X_val.csv', 'X_test.csv', 'y_train.csv', 'y_val.csv', 'y_test.csv'")
