import pandas as pd
import logging

# Configura il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def ingest_data(file_paths):
    """
    Carica i dati dei vini bianco e rosso con il delimitatore corretto.

    :param file_paths: Lista dei percorsi dei file
    :return: DataFrame contenente i dati uniti
    """
    logger.info("Ingestione dei dati dai file: %s", file_paths)
    data_white = pd.read_csv(file_paths[0], delimiter=';')
    data_red = pd.read_csv(file_paths[1], delimiter=';')

    data_white["is_red"] = 0
    data_red["is_red"] = 1

    data = pd.concat([data_white, data_red], ignore_index=True)
    return data

'''
# Esempio di utilizzo
file_paths = ["./data/winequality-white.csv", "./data/winequality-red.csv"]
data = ingest_data(file_paths)'''