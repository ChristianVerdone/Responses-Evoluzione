def transform_data(data):
    """
    Esegue trasformazioni personalizzate sui dati.

    :param data: DataFrame contenente i dati da trasformare
    :return: DataFrame contenente i dati trasformati
    """
    data = data.dropna()
    return data