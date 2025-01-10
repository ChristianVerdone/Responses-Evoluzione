def transform_data(data):
    """
    Applica trasformazioni sui dati, come la rimozione di valori nulli.
    """
    return data.dropna()
