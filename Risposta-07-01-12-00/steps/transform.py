def transform_data(data):
    """
    Esegue trasformazioni sui dati.

    Args:
        data (pd.DataFrame): Dataset originale.

    Returns:
        pd.DataFrame: Dataset trasformato.
    """
    data = data.dropna()  # Rimuovi valori nulli
    return data
