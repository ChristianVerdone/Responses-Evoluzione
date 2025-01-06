import pandas as pd


def transform_data(data):
    """Esempio di trasformazione dei dati (potrebbe essere personalizzato).

  Args:
      data: Dataframe Pandas contenente i dati da trasformare.

  Returns:
      Dataframe Pandas con i dati trasformati.
  """

    data = data.dropna()  # Rimuovi valori nulli
    return data
