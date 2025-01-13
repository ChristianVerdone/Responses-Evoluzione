import pandas as pd
import logging

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def ingest_data(file_path, file_format):
    """
    Carica i dati da un file e aggiunge una colonna `is_red` in base al tipo di vino.

    :param file_path: Percorso del file (può essere un oggetto Path).
    :param file_format: Formato del file (ad esempio, "csv").
    :return: DataFrame Pandas contenente i dati caricati.
    """
    if file_format != "csv":
        raise NotImplementedError(f"Formato non supportato: {file_format}")

    # Converti file_path in stringa (se necessario)
    file_path_str = str(file_path)

    # Carica i dati dal file CSV
    logger.info(f"Caricamento del file: {file_path_str} (formato: {file_format})")
    df = pd.read_csv(file_path_str, delimiter=";")

    # Aggiungi colonna per distinguere i vini rossi e bianchi
    df["is_red"] = 1 if "red" in file_path_str.lower() else 0
    return df


if __name__ == "__main__":
    # Esempio di utilizzo per il debug
    file_paths = ["./data/winequality-white.csv", "./data/winequality-red.csv"]

    # Carica i dati da entrambi i file
    data_white = ingest_data(file_paths[0], "csv")
    data_red = ingest_data(file_paths[1], "csv")

    # Combina i dataset
    combined_data = pd.concat([data_white, data_red], ignore_index=True)
    logger.info("Dati combinati con successo.")

    # Salva i dati combinati
    output_path = "./data/ingested_data.csv"
    combined_data.to_csv(output_path, index=False)
    logger.info(f"Dati ingeriti e salvati in {output_path}")
