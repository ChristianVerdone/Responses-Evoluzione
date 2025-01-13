import pandas as pd
import logging

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_data(file_paths):
    """
    Carica i dati da file CSV.

    Parametri:
    - file_paths (list): Lista dei percorsi dei file da caricare.

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
    import yaml
    with open('local.yaml', 'r') as file:
        config = yaml.safe_load(file)
    ingest_config = config['INGEST_CONFIG']
    file_paths = ingest_config['location']
    data = ingest_data(file_paths)
    print(data.head())
