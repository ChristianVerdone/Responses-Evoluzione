import pandas as pd

def ingest_data(file_paths):
    """
    Carica e unisce i dati di input da file CSV.

    Args:
        file_paths (list): Elenco dei percorsi ai file CSV.

    Returns:
        pd.DataFrame: Dataset combinato con una colonna 'is_red'.
    """
    data_white = pd.read_csv(file_paths[0], delimiter=';')
    data_red = pd.read_csv(file_paths[1], delimiter=';')

    data_white["is_red"] = 0
    data_red["is_red"] = 1

    data = pd.concat([data_white, data_red], ignore_index=True)
    return data
