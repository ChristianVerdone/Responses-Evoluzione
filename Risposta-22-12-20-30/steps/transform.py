# transform.py
def transform_data(data):
    """
    Esegue trasformazioni personalizzate sui dati.
    
    Args:
        data (pd.DataFrame): Dataset di input.

    Returns:
        pd.DataFrame: Dataset trasformato.
    """
    data = data.dropna()
    return data
