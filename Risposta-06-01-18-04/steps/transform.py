import pandas as pd


def transform_data(data):
    """
  Esegue operazioni di trasformazione sui dati (es. rimozione di valori nulli).

  Args:
      data (pd.DataFrame): DataFrame contenente i dati da trasformare.

  Returns:
      pd.DataFrame: DataFrame con le trasformazioni applicate.
  """

    data = data.dropna()  # Rimuovi valori nulli
    return data
