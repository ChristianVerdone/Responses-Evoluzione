import pandas as pd


def transform_data(data):
    """Esempio di trasformazioni personalizzate (sostituisci con la tua logica).

  Args:
      data (pandas.DataFrame): DataFrame contenente i dati.

  Returns:
      pandas.DataFrame: DataFrame con le trasformazioni applicate.
  """
    # Esempio di rimozione valori nulli
    data = data.dropna()
    return data
