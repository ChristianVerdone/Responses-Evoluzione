import pandas as pd
import logging

# Configura il logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transform_data(df, dropna=True):
    """
    Trasforma i dati.

    Parametri:
    - df (DataFrame): DataFrame contenente i dati da trasformare.
    - dropna (bool): Se True, rimuove i valori nulli.

    Restituisce:
    - DataFrame: DataFrame contenente i dati trasformati.
    """
    if dropna:
        df = df.dropna()
    return df

if __name__ == "__main__":
    import yaml
    with open('local.yaml', 'r') as file:
        config = yaml.safe_load(file)
    transform_config = {'transform_params': config['TRANSFORM_PARAMS']}
    dropna = transform_config['transform_params']['dropna']
    file_paths = config['INGEST_CONFIG']['location']
    data = pd.concat([pd.read_csv(file_path, delimiter=';') for file_path in file_paths], ignore_index=True)
    transformed_data = transform_data(data, dropna)
    print(transformed_data.head())
