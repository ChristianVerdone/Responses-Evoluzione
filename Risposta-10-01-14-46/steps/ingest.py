import pandas as pd
import mlflow
import yaml

# Configurazione di MLflow
mlflow.set_tracking_uri("sqlite:///metadata/mlflow/mlruns.db")
mlflow.set_experiment("sklearn_classification_experiment")

# Funzione per l'ingestione dei dati
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

# Carica i parametri dal file local.yaml
with open('local.yaml', 'r') as file:
    params = yaml.safe_load(file)

file_paths = params['FILE_PATHS']
data = ingest_data(file_paths)

# Salva i dati ingestiti
data.to_csv('data_ingested.csv', index=False)
