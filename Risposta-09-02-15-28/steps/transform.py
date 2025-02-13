def transform_data(data):
    """Rimuove valori nulli."""
    return data.dropna()

def transform_step(data):
    return transform_data(data)