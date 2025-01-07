import pandas as pd


def ingest_data(file_paths):
    """
  Carica i dati dei vini bianchi e rossi e aggiunge la colonna target.

  Args:
      file_paths (list): Lista di percorsi ai file CSV dei vini bianchi e rossi.

  Returns:
      pd.DataFrame: Il DataFrame combinato con la colonna target "is_red".
  """

    data_white = pd.read_csv(file_paths[0], delimiter=";")
    data_red = pd.read_csv(file_paths[1], delimiter=";")

    data_white["is_red"] = 0
    data_red["is_red"] = 1

    data = pd.concat([data_white, data_red], ignore_index=True)
    return data
