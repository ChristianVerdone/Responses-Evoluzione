import pandas as pd


def ingest_data(file_paths):
    """Carica i dati dei vini bianco e rosso con il delimitatore corretto.

  Args:
      file_paths (list): Lista di percorsi ai file CSV.

  Returns:
      pandas.DataFrame: DataFrame contenente i dati uniti.
  """
    # Carica i dati dei vini bianco e rosso con delimitatore ';'
    data_white = pd.read_csv(file_paths[0], delimiter=";")
    data_red = pd.read_csv(file_paths[1], delimiter=";")

    # Aggiungi la colonna is_red (0 per bianco, 1 per rosso)
    data_white["is_red"] = 0
    data_red["is_red"] = 1

    # Unisci i due dataset
    data = pd.concat([data_white, data_red], ignore_index=True)
    return data
