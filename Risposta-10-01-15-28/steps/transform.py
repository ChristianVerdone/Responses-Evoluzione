def transform_data(data):
    """
    Esegue la trasformazione dei dati.

    Args:
      data (pandas.DataFrame): DataFrame contenente i dati da trasformare.

    Returns:
      pandas.DataFrame: DataFrame contenente i dati trasformati.
    """
    data = data.dropna()
    return data
